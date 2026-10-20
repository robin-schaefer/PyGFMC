# `pygfmc.wavefunction` — guiding wavefunctions for PyGFMC

This subpackage provides the guiding (trial) wavefunctions ψ_T used by the GFMC
engine ([`pygfmc.engine`](../engine/README.md)), plus optional NetKet-based
training for them.

```python
from pygfmc.wavefunction import (JastrowSpinHalf, JastrowSpin, JastrowBoseHubbard, ClusterCPS,
                                 ClusterMLP, ExactGroundState, train_with_netket, make_hilbert)
```

Importing `pygfmc.wavefunction` never imports NetKet, optax or flax; those are only
needed to *train* a guide (see [Training with NetKet](#training-with-netket)).

## Contents

- [Role of the guide in GFMC](#role-of-the-guide-in-gfmc)
- [The guide protocol](#the-guide-protocol)
- [Writing a custom guide](#writing-a-custom-guide)
- [Usage example](#usage-example)
- [Module reference](#module-reference)
  - [`jastrow_spin_half.py` — `JastrowSpinHalf`](#jastrow_spin_halfpy--jastrowspinhalf)
  - [`jastrow_spin.py` — `JastrowSpin`](#jastrow_spinpy--jastrowspin)
  - [`jastrow_bose_hubbard.py` — `JastrowBoseHubbard`](#jastrow_bose_hubbardpy--jastrowbosehubbard)
  - [`cluster_cps.py` — `ClusterCPS`](#cluster_cpspy--clustercps)
  - [`cluster_mlp.py`](#cluster_mlppy)
  - [`exact_ground_state.py`](#exact_ground_statepy)
  - [`generic_netket_training.py`](#generic_netket_trainingpy)
  - [`_optional.py`, `_seed.py`, `_tied_params.py`, `_layout.py`](#_optionalpy-_seedpy-_tied_paramspy-_layoutpy)
- [Training with NetKet](#training-with-netket)

## Role of the guide in GFMC

GFMC projects onto the ground state |ψ_0⟩ by repeatedly applying the
propagator `G = λ − H` (or its continuous-time limit `exp(−τH)`) to a
population of walkers, each a basis configuration `x`. Without a guide the walk
moves from `x` to `x'` with a rate proportional to `|H_{x'x}|`, independent of
where the ground-state amplitude actually sits, and the walker weights
fluctuate wildly.

**Importance sampling.** With a guide ψ_T the walk samples the product
ψ_T(x)ψ_0(x) instead of ψ_0(x). The transition rates become

    H_{x'x} · ψ_T(x') / ψ_T(x),

so the engine needs only *ratios* of guide amplitudes between a configuration
and its successors — never the normalisation of ψ_T. The per-walker weight
update is governed by the **local energy**

    e_L(x) = Σ_{x'} H_{x'x} ψ_T(x') / ψ_T(x).

**The mixed estimator.** The energy is measured as

    E = ⟨ψ_T|H|ψ_0⟩ / ⟨ψ_T|ψ_0⟩ = weighted average of e_L(x),

which equals the exact ground-state energy for **any** ψ_T with non-zero
overlap with ψ_0, because H acting to the right on ψ_0 gives E_0. The guide
therefore does not change the answer, only its statistical quality:

- The better ψ_T approximates ψ_0, the smaller the fluctuations of `e_L` and
  of the walker weights, hence the lower the variance and the slower the loss
  of walkers to branching. If ψ_T = ψ_0 exactly, `e_L(x) = E_0` for every
  configuration and the energy variance vanishes (see `ExactGroundState`).
- For observables that do not commute with H the mixed estimator
  ⟨ψ_T|O|ψ_0⟩ is biased; the engine removes this bias by forward walking (the
  pure estimator), whose statistics again improve with a better guide.

**Positivity.** For a sign-problem-free Hamiltonian (all off-diagonal
`H_{x'x} ≤ 0`) the ground state can be chosen non-negative, and any strictly
positive ψ_T gives a valid, sign-free walk projecting onto that same ψ_0. All
guides here are real and positive by construction: they are parametrised by
`log ψ_T`.

## The guide protocol

A guide is any object implementing **at least one** of the following methods
(in order of preference; the engine uses the first one it finds). The
interface is written down as `pygfmc.Guide` (`pygfmc/protocols.py`), a
`typing.Protocol` used in the type hints and the API documentation; a class
satisfies it by having the methods, not by inheriting from it.

| Method | Returns | Notes |
|---|---|---|
| `log_diff_from_operator(config, op, cache=None, state=None)` | `(op.M_total,)` array of `log ψ_T(x'_k) − log ψ_T(x)` in `op`'s transition layout (entry 0 is the diagonal, 0) | Preferred: operator-specific fast paths avoid materialising the `(M_total, N)` successor tensor. `state` is passed only to guides in the incremental mode (below). Each shipped guide accepts a fixed set of operator classes (table under *Operator layout* below) and raises `TypeError` for any other. |
| `log_diff_from_configs(config, new_configs, cache=None)` | `(n,)` array of `log ψ_T(new_configs[k]) − log ψ_T(config)` | The engine materialises `new_configs` via `op.get_rates(config, return_configs=True)`. |
| `log_psi(config, cache=None)` | `log ψ_T(config)` | Accepts a single `(N,)` configuration or a `(n, N)` batch. The engine forms the ratios by `vmap`. |

Optionally a guide provides `build_cache(op)`, returning a (pytree) cache that
the engine computes once per operator before the jitted loop and passes back to
`log_diff_from_operator`. Configuration-based fallbacks receive no cache.
The shipped guides use it to build index tables once for their fast
`log_diff_from_operator` path: `JastrowSpinHalf` keeps the per-edge partner
tables `(edge_other, edge_J)` of the operator's edges there, `JastrowSpin` and
`JastrowBoseHubbard` the transition tables and pair couplings of the operator,
`ClusterCPS` and `ClusterMLP` their cluster index tables.

**Incremental mode (optional).** A guide whose `incremental` attribute is True
also provides `init_state(config, cache=None)` and
`update_state(state, config, sites, deltas, cache=None)`, and its
`log_diff_from_operator` accepts `state=`. The engine then keeps a per-walker
state, updates it after every move from the Hamiltonian's `move(config, idx)`
(the sites changed and their increments), carries it through reconfiguration
next to the rates, and rebuilds it (with the rates) every 100 chain steps,
counted from the start of the chain, so that rounding cannot accumulate and
a split run reproduces a contiguous one. `JastrowSpin` and `JastrowBoseHubbard` keep their weighted
neighbour sums this way (constructor flag `incremental=True`): a rate
evaluation then costs the gathers at the moved sites instead of a sum over all
pairs. A run in this mode is not bit-identical to the same run without it.

Further conventions:

- **Configurations** are float arrays of per-site values from the
  Hamiltonian's `local_hilbert_space` — for spin-1/2 that is `±1`, not `0/1`.
- **Pytrees.** A guide must be a registered JAX pytree
  (`jax.tree_util.register_pytree_node`) so it can be passed into jitted code.
  Arrays (in particular trainable parameters) go into the *children*; the
  *aux data* must be hashable and comparable — Python ints, floats, tuples,
  dtypes, never raw arrays.
- **HDF5.** The shipped guides round-trip through `guide.save(path, **attrs)`
  and `Class.load(path, ...)`; extra keyword arguments to `save` are stored as
  scalar attributes of the guide's HDF5 group (read them with `h5py`).
  `save` overwrites `path`.
- **Operator layout.** `log_diff_from_operator` and `build_cache` index their
  output by the operator's transition layout, so each guide accepts only the
  operator classes whose layout it knows and raises `TypeError` (naming the
  requirement and the class it got) for any other, from both methods. The
  engine calls `guide.build_cache(op)` (through `engine.evolution.guide_cache`) for the Hamiltonian and every
  off-diagonal measurement operator before the run, so an unsupported
  operator fails before any step is taken.

  | Guide | Operators accepted by `log_diff_from_operator` / `build_cache` |
  |---|---|
  | `JastrowSpinHalf`, `ClusterCPS`, `ClusterMLP` | `SpinHalf` (Hamiltonian or measurement operator, incl. `StructureFactor`) and any operator exposing the `SpinHalf` layout `[diagonal, n_hx_static single-site flips, n_jpm_static edge flips on (edge_i, edge_j)]` |
  | `JastrowSpin` | `Spin`; any other operator falls back to its materialised successors |
  | `JastrowBoseHubbard` | `BoseHubbard`; any other operator falls back to its materialised successors |
  | `ExactGroundState` | has no `log_diff_from_operator`: the engine materialises the successors of any operator |

  For `ToricCode`, `PlaquetteModel` or a user-defined operator with its own
  layout, use a guide exposing only `log_diff_from_configs`/`log_psi`, e.g.
  `ExactGroundState` or a custom guide as in the next section.

## Writing a custom guide

The easiest route for a new ansatz is `log_psi` alone: it works with every
Hamiltonian, and with the generic trainer `pygfmc.wavefunction.train_with_netket`.
The engine forms the ratios `log ψ_T(x') − log ψ_T(x)` from it by `vmap`. A
complete minimal guide and a short GFMC run with it (run on CPU):

```python
import jax, jax.numpy as jnp
from pygfmc import run_gfmc
from pygfmc.walkers import prepare_walkers_sz
from pygfmc.hamiltonian import SpinHalf, chain, exact_diagonalization

class NNGuide:
    """log psi(s) = -a * sum_<ij> s_i s_j  -- implements log_psi only."""
    def __init__(self, edges, a):
        self.a = jnp.asarray(a, dtype=jnp.float32)     # trainable (float) leaf
        self.edges = jnp.asarray(edges)                # fixed (integer) leaf
    def log_psi(self, config, cache=None):             # (N,) or (n, N)
        s_i = config[..., self.edges[:, 0]]
        s_j = config[..., self.edges[:, 1]]
        return -self.a * jnp.sum(s_i * s_j, axis=-1)

def _unflatten(aux, children):
    g = object.__new__(NNGuide)
    g.a, g.edges = children
    return g

jax.tree_util.register_pytree_node(
    NNGuide,
    lambda g: ((g.a, g.edges), None),     # arrays are children; aux is hashable
    _unflatten)

h = SpinHalf.create(chain(8).edges, J_x=-1.0, J_y=-1.0, J_z=1.0)
guide = NNGuide(h.edges, 0.3)
walkers = prepare_walkers_sz(256, h.N, jax.random.PRNGKey(0), sz=0)
walkers, energy, obs, stats, raw = run_gfmc(
    walkers, 1000, h, 0.2, guide, [2, 4, 8],
    rng_key=jax.random.PRNGKey(1), warmup_steps=200, verbosity=0)
print(energy)                          # {2: -3.651765557214167, 4: -3.6518906392805186, 8: -3.651928974472916}
print(exact_diagonalization(h)[0][0])  # -3.6510934089371774
```

Pass it as the `guide` argument of `pygfmc.run` / `pygfmc.run_gfmc` like any
built-in guide (this one is the same state as `JastrowSpinHalf(h.N, h.edges,
beta=0.3)`). Once it works, add `log_diff_from_operator` (and `build_cache`)
for speed and memory: only the parts of `log ψ_T` touched by a transition need
recomputing.

## Usage example

Build a Jastrow guide for a spin-1/2 XXZ ring, evaluate it, and round-trip it
through HDF5 (run on CPU with `pygfmc` installed, or with `PYTHONPATH`
pointing at the folder containing `pygfmc/`):

```python
import h5py
import jax.numpy as jnp
from pygfmc.hamiltonian import SpinHalf, chain
from pygfmc.wavefunction import JastrowSpinHalf

L = 8
lat = chain(L)                                   # periodic ring
h = SpinHalf.create(lat.edges, J_x=-1.0, J_y=-1.0, J_z=1.0)

guide = JastrowSpinHalf(h.N, h.edges, beta=0.3)             # log psi = -0.3 sum_<ij> s_i s_j
config = jnp.array([1., -1.] * (L // 2))            # Neel state, spins are +-1

print(guide.log_psi(config))                        # 2.4
print(guide.log_psi(jnp.stack([config, -config])))  # [2.4 2.4]

# log psi(x') - log psi(x) in h's transition layout [diag, edge flips...]
cache = guide.build_cache(h)
print(guide.log_diff_from_operator(config, h, cache=cache))
# [ 0.  -1.2 -1.2 -1.2 -1.2 -1.2 -1.2 -1.2 -1.2]

guide.save("jastrow.h5", E_vmc=-3.65)
g2 = JastrowSpinHalf.load("jastrow.h5")
with h5py.File("jastrow.h5", "r") as f:
    print(dict(f["jastrow_spin_half"].attrs))
# {'E_vmc': np.float64(-3.65), 'N': np.int64(8), 'h_mode': 'none', 'tied_beta': np.int64(1)}
print(jnp.allclose(g2.log_psi(config), guide.log_psi(config)))  # True
```

## Module reference

`pygfmc/wavefunction/__init__.py` re-exports exactly `JastrowSpinHalf`, `JastrowSpin`,
`JastrowBoseHubbard`, `ClusterCPS`, `ClusterMLP`, `ExactGroundState`,
`train_with_netket` and `make_hilbert`. The other names below (cache named
tuples, NetKet helpers) are imported from their module, e.g.
`from pygfmc.wavefunction.generic_netket_training import make_netket_operator`.

In the signatures below `dtype` defaults to `jnp.float32` unless stated.

### `jastrow_spin_half.py` — `JastrowSpinHalf`

Two-body Jastrow guide for spin-1/2 (`SpinHalf`, configurations `x_i = ±1`):

    log ψ_T(x) = −Σ_{(i,j) ∈ pairs} β_ij x_i x_j + Σ_i h_i x_i ,

stored internally as `J = −β`. Transition ratios are computed from per-site
partner tables in O(degree) per transition.

- **`JastrowSpinHalf(N, pairs, beta, h=None, dtype=jnp.float32)`** — `pairs` is an
  `(n_pairs, 2)` site list (order within a pair irrelevant); `beta` (required)
  is a scalar (one tied coupling) or a length-`n_pairs` array, and `.beta`
  returns it in the order of `pairs`, also after `save`/`load`; `h` a
  scalar, an `(N,)` array or `None`. A uniform Ising-type guide is
  `JastrowSpinHalf(h.N, h.edges, beta=β)`. Attributes `N`, `beta`, `h`. The guide
  keeps the parameters in the form they were given, and `train_with_netket`
  trains exactly those: a scalar `beta` stays one tied coupling, `h=None`
  stays without a field, a scalar `h` stays one field. `save`/`load`
  preserve the form.
  - `log_psi(config, cache=None)` — `log ψ_T` for `(N,)` or `(n, N)` input.
  - `log_diff_from_operator(config, op, cache=None)` — ratios in a `SpinHalf`
    layout; `cache` from `build_cache(op)` selects the table-lookup path.
  - `log_diff_from_configs(config, new_configs, cache=None)` — generic path
    via `log_psi`.
  - `build_cache(op)` — returns a `JastrowOpCache` of per-edge partner tables
    (memoised per operator edge layout).
  - `save(path, group='jastrow_spin_half', **attrs)` / `load(path, hamiltonian=None,
    group='jastrow_spin_half', dtype=...)` — HDF5 round trip; with `hamiltonian` given,
    `load` zero-pads `h` onto a larger system.
  - `from_jax(dtype, N, site_other, site_J, site_valid_mask, h, pair_site,
    pair_slot, site_pair)` (classmethod) — rebuild from internal arrays,
    bypassing `__init__` (used by pytree unflatten); `pair_site`/`pair_slot`
    locate each constructor pair in the per-site tables and `site_pair` maps
    each table entry back to its pair, so both endpoints read one coupling.
- **`JastrowOpCache(edge_other, edge_J)`** — named tuple returned by
  `JastrowSpinHalf.build_cache`, both `(n_edges_op, 2, max_deg)`; padded partner
  slots carry `J = 0`, so no validity mask is needed.

### `jastrow_spin.py` — `JastrowSpin`

Jastrow guide for spin-S models (`Spin`, configurations `x_i ∈ {−S, …, S}`):

    log ψ_T(x) = −Σ_{(i,j) ∈ pairs} β_ij x_i x_j + Σ_i h_i x_i − Σ_i d_i x_i² .

The single-ion term `d` weights `|x_i|` against each other (`d > 0` favours
small `|x_i|`); for spin 1/2 it is a constant.

- **`JastrowSpin(N, pairs, beta, h=None, d=None, dtype=jnp.float32,
  incremental=False)`** —
  `beta` a scalar (tied) or a length-`n_pairs` array; `h` and `d` each a
  scalar (tied), an `(N,)` array or `None` (absent). Parameters keep the form
  they were given, through training and `save`/`load`. Properties `beta`
  `(n_pairs,)`, `h` and `d` `(N,)` (zeros when absent). `incremental=True`
  switches on the incremental mode (see the protocol section).
  - `log_psi(config, cache=None)` — `(N,)` or `(n, N)` input.
  - `log_diff_from_operator(config, op, cache=None, state=None)` — fast path
    for `Spin` operators (`S⁺S⁻`, `S⁺S⁺` and single-site `S^±` blocks), O(1)
    per transition from the weighted neighbour sums (`state`, when the engine
    carries them; rebuilt otherwise), without building successor
    configurations; any other operator falls back to `get_rates` configs.
  - `save(path, group='jastrow_spin', **attrs)` / `load(path,
    group='jastrow_spin', dtype=...)`.

### `jastrow_bose_hubbard.py` — `JastrowBoseHubbard`

Jastrow guide for bosons (`BoseHubbard`, occupations `x_i`), in the same
convention as `JastrowSpin`:

    log ψ_T(x) = −Σ_{(i,j) ∈ pairs} β_ij x_i x_j + Σ_i h_i x_i − Σ_i d_i x_i² .

`β` correlates neighbouring densities, the on-site (Gutzwiller) term `d > 0`
suppresses multiple occupancy, and the fugacity `h` sets the density when the
particle number is not conserved (e.g. with a pair term `eta`); at fixed
particle number it is a constant and can be left out.

- **`JastrowBoseHubbard(N, pairs, beta, h=None, d=None, dtype=jnp.float32,
  incremental=False)`** —
  same arguments and properties as `JastrowSpin`: `beta` a scalar (tied) or a
  length-`n_pairs` array; `h` and `d` each a scalar (tied), an `(N,)` array or
  `None` (absent). Parameters keep the form they were given, through training
  and `save`/`load`. `incremental=True` switches on the incremental mode.
  - `log_psi(config, cache=None)` — `(N,)` or `(n, N)` input.
  - `log_diff_from_operator(config, op, cache=None, state=None)` — fast path
    for `BoseHubbard` operators (hopping, pair term, single-site `b^†`/`b`),
    from the carried neighbour sums `state` when given; other operators fall
    back to `get_rates` configs.
  - `save(path, group='jastrow_bose_hubbard', **attrs)` / `load(path,
    group='jastrow_bose_hubbard', dtype=...)`.

### `cluster_cps.py` — `ClusterCPS`

General cluster **Correlator Product State**: a lookup table per cluster
pattern, summed over clusters and over cluster groups,

    log ψ_T(σ) = Σ_g Σ_{p ∈ group g} T_g[type_g(p), orbit_g(σ_p)] ,

where `σ_p` (the spins of cluster `p`, encoded as an integer in `[0, 2^m)`) is
reduced to its orbit under the group's slot permutations and, optionally, the
global spin flip. Groups may have different cluster sizes. A spin flip toggles
one bit of each affected cluster's index, so `log_diff_from_operator` updates
indices by arithmetic (`idx ± 2^slot`) without building any successor
configuration.

- **`ClusterCPS(groups, amplitudes=None, init='zero', seed=None,
  dtype=jnp.float32)`** — `groups` (required) is a list of group specs, each
  either the list `[clusters, symmetries, parity]` (all three required;
  `symmetries` may be `None`) or a dict

  ```python
  dict(clusters=arr,          # (n_clusters, m) int array
       symmetries=autos,      # optional (n_auto, m) slot permutations
       parity=True,           # optional, default True
       shared=True,           # optional: one table per group (else one per cluster)
       cluster_type=types)    # optional explicit table id per cluster
  ```

  These five are the only accepted keys (there are no aliases); **any other
  key raises `ValueError`**. `amplitudes` is a **top-level** argument: a list
  with one `(n_types, M_g)` (or `(M_g,)`) table per group. Without it the
  tables are initialised by `init` (`'zero'` is the uniform state
  `log ψ_T = 0`; `'gaussian'` has std 0.01, seeded by `seed`, drawn when
  `None`). A bare `(n_clusters, m)` array as `groups` builds a one-group guide
  with the dict defaults (its single table may then be passed without the
  list). Attributes `groups`, `n_groups`, `N`, `n_params`.

  ```python
  # correct: trained tables go in the top-level list
  ClusterCPS(groups=[dict(clusters=c, symmetries=autos)], amplitudes=[table])
  ```

  - `log_psi(configs, cache=None)` — `(N,)` or `(..., N)` input.
  - `log_diff_from_operator(config, op, cache=None)` — SpinHalf layout;
    index arithmetic with a cache, successor materialisation without.
  - `log_diff_from_configs(config, new_configs, cache=None)`.
  - `build_cache(op)` — `ClusterCPSCache` (one `ClusterCPSGroupCache` per
    group). It bakes in the current amplitudes; rebuild after changing them.
  - `save(path, group='cps', **attrs)` / `load(path, group='cps',
    dtype=...)` — nested per-group HDF5 layout (one `group_<i>` subgroup per
    cluster group); `load` reads exactly what `save` writes.
- **`ClusterCPSGroup`**, **`ClusterCPSGroupCache`**, **`ClusterCPSCache`** — named tuples holding one
  group's arrays, one group's cache, and the full cache.

### `cluster_mlp.py`

Per-cluster neural network guide:

    log ψ_T(σ) = Σ_g Σ_{c ∈ clusters_g} MLP_g(σ_c) ,

with one tanh MLP (arbitrary depth, linear output) shared by all clusters of a
group. `log_diff_from_operator` re-evaluates only the clusters touched by each
transition.

- **`ClusterMLP(groups, params=None, seed=None, dtype=...,
  N=None)`** — `groups = [[clusters, hidden], ...]`, `clusters` an
  `(n_clusters_g, k_g)` array and `hidden` a list of hidden widths (`[]` or
  `None` for a linear model; a bare int is rejected). `groups` is the only way
  to specify the architecture. `params` is one list of `(W_i, b_i)` layer
  pairs per group (the last layer is the linear output) whose shapes must
  match `groups`; it is initialised Gaussian with std 0.01 (seeded by
  `seed`, drawn when `None`) when omitted. `N` defaults to the largest site
  index + 1 (pass it if trailing sites belong to no cluster).
  - `log_psi(config, cache=None)`, `log_diff_from_configs(...)`,
    `log_diff_from_operator(config, op, cache=None)`.
  - `build_cache(op)` — `ClusterMLPCache` of site→cluster and edge→cluster
    tables (memoised).
  - `save(path, group='cluster_mlp', **attrs)` / `load(path,
    group='cluster_mlp', dtype=...)` — one `group_<g>` subgroup per
    group holding its clusters and `W<i>`/`b<i>` layer datasets; `load`
    rebuilds each group's hidden widths from the stored layer shapes.
  - property `n_params`.
- **`ClusterMLPCache(site_to_clusters, pair_to_clusters)`** — named tuple.

### `exact_ground_state.py`

- **`ExactGroundState(hamiltonian, log_amp=None, dtype=...,
  local_hilbert_space=None, N=None)`** — ψ_T(x) = |ψ_0(x)| from exact
  diagonalisation (`pygfmc.hamiltonian.exact.exact_diagonalization`), for any
  Hamiltonian implementing the protocol. The guide stores a `d^N` vector, so it
  is a diagnostic for small systems: with ψ_T = ψ_0 the local energy is the
  constant E_0 and the mixed estimator is already exact, so any remaining
  variance or p-dependence points at the sampler. Attribute `E0`. The
  diagonalisation returns one lowest eigenvector: for a degenerate ground
  state (e.g. across sectors the Hamiltonian does not connect, such as the
  flippability sectors of `PlaquetteModel` or the `B_p` sectors of
  `ToricCode`) that is an arbitrary vector of the degenerate subspace, which
  can vanish on part of the basis; no degeneracy check is made. For a run in
  one sector, supply that sector's ground state via `log_amp=`.
  - `log_diff_from_configs(config, new_configs=None, cache=None)` — the **only**
    ratio routine (no `log_diff_from_operator`, no `build_cache`).
  - `basis_index(config)` — index into the amplitude vector (site 0 most
    significant).
  - An externally computed ψ_0 is passed as `log_amp = log|ψ_0|` (length
    `d^N`, basis order of `basis_index`); `E0` is then NaN.
  - `save(path, group='exact_ground_state', **attrs)` /
    `load(path, hamiltonian, group='exact_ground_state', dtype=...)` —
    **`load` requires the Hamiltonian** and the `family`, `N` and `E0`
    attributes that `save` writes (a file without them raises `ValueError`); it
    checks the class name and `N` against the Hamiltonian (`ValueError` on a
    mismatch), restores a stored `local_hilbert_space`, and restores `E0` from
    the file.

### `generic_netket_training.py`

The only training entry point: any guide exposing `log_psi` on any
Hamiltonian exposing `get_rates` (needs the `netket` extra).

- **`train_with_netket(hamiltonian, guide, local_hilbert_space=None, N=None,
  sampler=None, initial_states=None, n_iter=100, learning_rate=0.02,
  n_samples=1024, n_chains=16, n_discard_per_chain=8, sr=False,
  diag_shift=1e-3, qgt=None, seed=0, show_progress=False,
  return_history=False)`** — returns a new guide of the same class (and the
  per-iteration energies with `return_history=True`). Every floating-point
  pytree leaf of the guide is trained, starting from its current value;
  integer/boolean leaves stay fixed. See
  [Training with NetKet](#training-with-netket).

It also exports one helper, `make_hilbert(local_hilbert_space, N, total=None)`:
a NetKet Hilbert space whose local states are exactly the given (evenly spaced)
values; `total` restricts it to configurations whose values sum to `total`. Use
it to build a sampler on PyGFMC's own encoding, e.g.
`nk.sampler.MetropolisLocal(make_hilbert(H.local_hilbert_space, N))` — NetKet's
own spin-1 space stores `2m`, not `m`.

The other helpers are module-level functions, not exported by `pygfmc.wavefunction`:

- `make_netket_operator(hamiltonian, hilbert, is_hermitian=True)` — wrap a
  `get_rates` Hamiltonian as a NetKet `DiscreteJaxOperator`.
- `make_guide_model(guide, param_dtype=jnp.float32)` — `flax.linen` module around
  `guide.log_psi`; returns `(model, treedef, leaves, trainable)`.
- `rebuild_guide(treedef, leaves, trainable, params)` — put trained
  parameters back into a guide.

### `_optional.py`, `_seed.py`, `_tied_params.py`, `_layout.py`

- `_layout.require_spin_half_layout(op, guide_name)` — raises `TypeError`
  unless `op` exposes `n_hx_static`, `n_jpm_static`, `edge_i` and `edge_j`
  (the `SpinHalf` transition layout); called by `log_diff_from_operator` and
  `build_cache` of `JastrowSpinHalf`, `ClusterCPS` and `ClusterMLP`.
- `_tied_params.write_tied_params(grp, params)` / `read_tied_params(grp, names)`
  — the HDF5 form of the optional scalar-or-array guide parameters (`beta`,
  `h`, `d`: absent, tied, or per element), shared by `JastrowSpin` and
  `JastrowBoseHubbard`.
- `_optional.import_netket()` — returns the `netket` module or raises
  `ImportError` naming the fix (`pip install "pygfmc[netket]"`). Every
  NetKet import in the package goes through it.
- `_seed.resolve_seed(seed, *, announce=False, context='guide')` — returns
  `int(seed)`, or draws a fresh seed from `os.urandom` when `seed is None`
  (printed only with `announce=True`); used by the random initialisers of
  `ClusterCPS` and `ClusterMLP`, which keep the drawn seed as `guide.seed`.

## Training with NetKet

Training is optional: install it with

```bash
pip install "pygfmc[netket]"         # netket, optax, flax
pip install -e "./codes[netket]"     # the same from source, in the root of a clone
```

Running GFMC with an already-trained guide needs none of these packages; a
training call without them raises an `ImportError` that names the install
command.

**`train_with_netket(hamiltonian, guide, ...)`** works for any
guide with `log_psi` and any Hamiltonian with `get_rates` (including
user-defined ones). It builds the NetKet Hilbert space from
`hamiltonian.local_hilbert_space` (so NetKet samples are PyGFMC
configurations, no re-encoding), wraps the rates as a NetKet operator, and
trains the guide's floating-point pytree leaves by VMC with
`optax.sgd(learning_rate)`, starting from the guide's current parameters.
The input guide is not modified. Training runs in float32: it sets
`NETKET_ENABLE_X64=0` (importing `pygfmc` does the same) so that importing
NetKet does not switch on x64, restores the caller's `jax_enable_x64` on exit,
and casts the trained parameters back to their original dtypes.

The default sampler is `MetropolisExchange`, which swaps the values of two
sites along the Hamiltonian's edges and so conserves `Σ_i x_i`. It samples the
sector of `initial_states` when given, else the balanced one (`S^z_tot = 0`,
or one particle per site). For a Hamiltonian that does not conserve this sum
(a transverse field, a pair term), pass a ready-made `sampler=`, e.g.
`nk.sampler.MetropolisLocal(make_hilbert(H.local_hilbert_space, N))`, whose
Hilbert space then replaces `local_hilbert_space`/`N`. Seed the chains inside
a constrained manifold with `initial_states` of shape `(n, N)` (important for,
e.g., spin ice).

```python
from pygfmc.hamiltonian import SpinHalf, chain
from pygfmc.wavefunction import JastrowSpinHalf, train_with_netket

h = SpinHalf.create(chain(8).edges, J_x=-1.0, J_y=-1.0, J_z=1.0)
guide = JastrowSpinHalf(h.N, h.edges, beta=0.1)

trained, hist = train_with_netket(h, guide, n_iter=50, n_samples=512,
                                  seed=1, return_history=True)
print(type(trained).__name__, hist[0], hist[-1])   # JastrowSpinHalf -3.227231740951538 -3.6117634773254395
print(float(trained.beta[0]))                      # 0.2496270388364792 (started at 0.1)
```

The snippet was run with NetKet 3.21 on Python 3.13 (CPU, ~27 s wall
including imports and compilation); the exact
numbers depend on the NetKet/JAX versions. The trainer
defaults to `sr=False`; to use stochastic reconfiguration with NetKet 3.21
pass `sr=True, qgt=nk.optimizer.qgt.QGTJacobianDense`.

### HDF5 attributes

The guides' `save` methods write their scalar metadata through
`pygfmc.walkers._h5io.write_attrs`, the helper `Walker.save` uses as well
(`None` values skipped, float scalars stored as Python `float`).
