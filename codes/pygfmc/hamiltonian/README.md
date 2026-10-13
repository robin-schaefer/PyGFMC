# `pygfmc.hamiltonian` — models, lattices, observables and exact diagonalization

This subpackage provides everything the GFMC engine
([`pygfmc.engine`](../engine/README.md)) needs to know about
the physical system:

- **Hamiltonians and operators** that obey a common *rate protocol*
  (`SpinHalf`, `Spin`, `BoseHubbard`, `PlaquetteModel`, `ToricCode`), which
  double as measurement operators;
- **lattice factories** that produce edge lists and geometric metadata
  (`chain`, `pyrochlore`, `kagome`, or a hand-built `Lattice`);
- **structure-factor measurement operators**: `StructureFactor` (an
  off-diagonal `SpinHalf` operator) and `SzSzStructureFactor` (a diagonal
  observable that is not a Hamiltonian), both built from site positions, and
  `is_valid_k` to check a wavevector against the periodic vectors;
- **exact diagonalization** of any object that implements the protocol
  (`exact_diagonalization`), used as the reference the Monte Carlo is checked
  against.

All model classes are immutable `NamedTuple`s registered as JAX pytrees, so
they can be passed straight into jitted/vmapped code.

```python
from pygfmc.hamiltonian import (SpinHalf, Spin, BoseHubbard, PlaquetteModel, ToricCode,
                                StructureFactor, SzSzStructureFactor, is_valid_k,
                                exact_diagonalization,
                                Lattice, chain, pyrochlore, kagome)
```

## The operator protocol

A configuration `config` is a 1-D float array of length `N`. Spin-1/2 models
store `σ_i = ±1` (so `S^z_i = σ_i / 2`), `Spin` stores `m_i ∈ {-S, ..., S}`,
`BoseHubbard` stores occupations `n_i ∈ {0, ..., n_max}`.

### Required

| Member | Meaning |
|---|---|
| `M_total` | static Python int: number of transitions **including** the diagonal (`n_off + 1`) |
| `get_rates(config, return_configs=False)` | returns `(rates, new_configs)`. `rates` has shape `(M_total,)`: `rates[0]` is the diagonal element `H[x, x]`, `rates[k]` for `k ≥ 1` the matrix element `H[y_k, x]` to successor `y_k`. `new_configs` is `None` unless `return_configs=True`, in which case it is `(M_total, N)` with `new_configs[0] == config`. |
| `generate_state(config, idx)` | the successor configuration for transition `idx` (`idx == 0` returns `config`). This is the hot path: production never materializes the `(M_total, N)` tensor. |

`run_gfmc` checks the three required members up front. The protocol is written
down as `pygfmc.Operator` (`pygfmc/protocols.py`), a `typing.Protocol` used in
the type hints and the API documentation; a class satisfies it by having the
members, not by inheriting from it. The matrix defined by
the protocol is `H[x, x] = rates[0]`, `H[y_k, x] += rates[k]`. The classes must
be JAX pytrees whose array data are children; auxiliary (static) data must be
hashable, so never put raw arrays there.

### Optional

| Member | Used by |
|---|---|
| `n_off` | number of off-diagonal slots (informational; the engine itself decides from `M_total > 1`) |
| `diagonal_weight(config)` | diagonal measurement operators (`diag_operators`); a diagonal observable needs **only** this method. May return a scalar or a vector. |
| `N`, `local_hilbert_space` | defaults for `exact_diagonalization` and the `ExactGroundState` guide, which index the basis as a digit string in `local_hilbert_space` order (site 0 most significant) |
| `d` | local Hilbert-space dimension; when present, `exact_diagonalization` cross-checks it against `len(local_hilbert_space)` (as it checks `N`) and raises on a mismatch (`Spin` provides it) |
| `move(config, idx)` | the sites changed by transition `idx` and their increments, `(sites, deltas)` of shape `(2,)` each, with `generate_state(config, idx) == config.at[sites].add(deltas)`; needed only by a guide in the incremental mode (`Spin` and `BoseHubbard` provide it) |

### Sign conventions

GFMC is sign-problem free (SPF) when every off-diagonal matrix element of the
propagated Hamiltonian is **≤ 0**. Each `create` enforces its class's
convention unless `sign_check=False`.

A **measurement** operator is never propagated; instead a successor is drawn
by inverse CDF over its rates, so an off-diagonal measurement operator needs
**all rates ≥ 0** (the opposite sign). Build such operators with positive
couplings and `sign_check=False` where the SPF check would object; split a
mixed-sign operator into two. A purely diagonal operator (`M_total == 1`, or
one used through `diagonal_weight`) has no sign restriction.

### Writing a custom operator

The example below (run as shown) defines `O = Σ_i S^x_i` as a measurement
operator and evaluates it exactly on the ground state of a transverse-field
Ising chain.

```python
import jax, jax.numpy as jnp
from typing import NamedTuple
from pygfmc.hamiltonian import SpinHalf, chain, exact_diagonalization

class TotalSx(NamedTuple):
    """O = sum_i S^x_i: one off-diagonal slot per site, rate 1/2 (>= 0)."""
    N: int

    @property
    def M_total(self):
        return self.N + 1

    @property
    def n_off(self):
        return self.N

    def get_rates(self, config, return_configs=False):
        rates = jnp.full(self.M_total, 0.5, dtype=config.dtype).at[0].set(0.0)
        if not return_configs:
            return rates, None
        flips = jnp.where(jnp.eye(self.N, dtype=bool), -config, config)
        return rates, jnp.concatenate([config[None], flips])

    def generate_state(self, config, idx):
        site = jnp.clip(idx - 1, 0, self.N - 1)
        return jnp.where(idx == 0, config, config.at[site].set(-config[site]))

jax.tree_util.register_pytree_node(
    TotalSx, lambda o: ((), o.N), lambda N, _: TotalSx(N))

lat = chain(8)
h = SpinHalf.create(lat.edges, J_z=-1.0, h_x=1.0)    # ferromagnetic TFIM
E, _, off = exact_diagonalization(h, local_hilbert_space=(1.0, -1.0),
                                  off_diag_operators={"Sx": TotalSx(h.N)})
print(E[0], off["Sx"][0])     # -4.2545411175...  3.7344172909...
```

## Models at a glance

| Class | Hamiltonian | Parameters | SPF condition (checked by `create`) | `M_total` |
|---|---|---|---|---|
| `SpinHalf` | `Σ_<ij> (J_x SˣSˣ + J_y SʸSʸ + J_z SᶻSᶻ) − Σ_i (h_z Sᶻ + h_x Sˣ)` | `edges, J_x, J_y, J_z, h_z, h_x, N` | `J_pm = (J_x+J_y)/4 ≤ 0`, `J_pmpm = (J_x−J_y)/4 ≤ 0`, `h_x ≥ 0` | `1 + [N if h_x≠0] + [n_edges if J_pm or J_pmpm ≠ 0]` |
| `Spin` (any S) | `Σ_<ij> [J_z SᶻSᶻ + J_pm (S⁺S⁻+S⁻S⁺) + J_pmpm (S⁺S⁺+S⁻S⁻)] + Σ_i [h_z Sᶻ + h_pm (S⁺+S⁻)]` | `edges, S, J_z, J_pm, J_pmpm, h_z, h_pm, N` | `J_pm ≤ 0`, `J_pmpm ≤ 0`, `h_pm ≤ 0` | `1 + 2n_edges·[J_pm≠0] + 2n_edges·[J_pmpm≠0] + 2N·[h_pm≠0]` |
| `BoseHubbard` | `−t Σ (b†b + h.c.) − η Σ (b†b† + bb) + U/2 Σ n(n−1) − μ Σ n + h_pm Σ (b† + b)` | `edges, t, eta, U, mu, h_pm, N, n_max` | `t ≥ 0`, `η ≥ 0`, `h_pm ≤ 0` | `1 + 2n_edges·[t≠0] + 2n_edges·[η≠0] + 2N·[h_pm≠0]` |
| `PlaquetteModel` | `−g Σ_p flip_p + mu Σ_p \|flippable_p⟩⟨flippable_p\|` | `plaquettes, g, mu` | `g ≥ 0` (a measurement operator needs `g ≤ 0`) | `1 + N_plaquette` |
| `ToricCode` | `−A_s Σ_v Π σˣ − B_p Σ_p Π σᶻ − h_z Σ σᶻ − h_x Σ σˣ` | `Lx, Ly, A_s, B_p, h_z, h_x` | `A_s ≥ 0`, `h_x ≥ 0` (elementwise) | `1 + Lx·Ly·[A_s≠0] + N·[h_x≠0]` |

The same spin-1/2 Hamiltonian is a `SpinHalf` or a `Spin(S=0.5)` with `Spin.J_z = SpinHalf.J_z`, `Spin.J_pm = (J_x+J_y)/4`, `Spin.J_pmpm = (J_x−J_y)/4`, `Spin.h_z = −SpinHalf.h_z`, `Spin.h_pm = −h_x/2` (the fields enter `SpinHalf` with a minus sign, and `h_pm (S⁺+S⁻) = 2 h_pm Sˣ`); configurations are `σ = ±1` for `SpinHalf` and `m = σ/2` for `Spin`. The rate layouts differ: `Spin` has two edge slots per edge (`S⁺S⁻` and `S⁻S⁺`, likewise for `J_pmpm`) and two site slots per site where `SpinHalf` has one, so `M_total` differs, while the diagonal and the per-edge / per-site sums of the off-diagonal rates agree.

`_common.py` holds what the classes share: `edges_array(edges)` validates an
edge list and infers the site count, and `two_site_update(config, state_idx,
edge_i, edge_j, N, edge_blocks, site_blocks)` is the scalar two-site
configuration update behind `Spin.generate_state` and
`BoseHubbard.generate_state`.

Every coupling may be a scalar (uniform) or an array (per edge, per site, per
plaquette or per star, as appropriate); a wrong length raises. Terms whose
coupling is identically zero are dropped from the rate layout. The shift `lam`
passed to the discrete GFMC methods must exceed the largest diagonal element
visited; the module docstrings give sufficient bounds per class.

## `SpinHalf.py`

**`class SpinHalf`** — spin-1/2 XYZ model in fields. The fields
enter with a minus sign, `−Σ_i (h_z Sᶻ_i + h_x Sˣ_i)` (as in `ToricCode`), so
`h_x ≥ 0` is sign-problem free. Stores `J_pm = (J_x+J_y)/4`,
`J_pmpm = (J_x−J_y)/4`, `J_z`, `−h_z` (attribute `h_z`) and `−h_x/2` (attribute
`h_x_half`); static flags gate inactive terms under JIT. Rate layout:
`rates[0]` diagonal, then `N` single-site flips (`−h_x/2` each), then one slot per edge (`J_pm` if the two
spins are anti-parallel, `J_pmpm` if parallel).

- `SpinHalf.create(edges, J_x=0, J_y=0, J_z=0, h_z=None, h_x=None, N=None, dtype=jnp.float32, sign_check=True)` — build the model. `J_pmpm > 0` raises (it is never fixed by silently swapping `J_x ↔ J_y`, which is not a symmetry when `h_x ≠ 0`); `h_x < 0` raises; `J_pm > 0` raises (all three only with `sign_check=True`). `N` is inferred from the edges and per-site arrays unless given (give it for field-only operators with `edges=[]`). A measurement operator for `+Σ_i Sˣ_i` needs non-negative rates, i.e. `create([], N=N, h_x=-1.0, sign_check=False)`.
- `diagonal_weight(config)` — the scalar `rates[0]` without building the full rate vector.
- `get_rates(config, return_configs=False)`, `generate_state(config, idx)` — protocol.
- `edges`, `n_hx`, `n_jpm`, `n_off`, `M_total` — derived sizes (static Python ints; `edges` is an `(n_edges, 2)` numpy array).
- `local_hilbert_space` — `(1.0, -1.0)`, so `σ = +1` is basis digit 0.

**`StructureFactor(positions, k, channel="SpSm", dtype=jnp.float32)`**
— returns a `SpinHalf` measurement operator
`O = Σ_{i<j} [1 + cos(k·(r_i − r_j))] (A_i B_j + h.c.) = AB(0) + AB(k)`,
off-diagonal only. It takes the geometry the way `SzSzStructureFactor` does:
`positions` is `(N, d)` (so `N` is the number of rows) and `k` one `(d,)`
wavevector; the sum runs over every pair `i < j`. The displacement is
`r_i − r_j` of the positions as given, with no periodic wrap: for a `k` in the
reciprocal lattice of the system (check with `is_valid_k`) every periodic image
gives the same `cos`, so the image the positions encode does not matter.
`channel` must be exactly one of the off-diagonal `"SpSm"`
(`S⁺S⁻ + S⁻S⁺ = 2(SˣSˣ + SʸSʸ)`), `"SpSp"` (`S⁺S⁺ + S⁻S⁻`) and
`"SpSm+SpSp"` (`4SˣSˣ`), or the diagonal `"SzSz"` (`2SᶻSᶻ`); there are no
aliases. `"SzSz"` returns an operator with `M_total = 1` for `diag_operators`
and warns: it equals `SzSzStructureFactor(positions, [0, k])` summed over its
two wavevectors (same `AB(0) + AB(k)` structure), which `SzSzStructureFactor`
computes for many `k` at once and without the shift. Any other name raises
`ValueError`, with an explanation for `YY` (`SpSm − SpSp`), `S·S` (`SpSm` plus
`SzSz`, measured separately) and mixed channels. Pairs whose weight is zero (to within `1e-12`) are dropped,
so `M_total − 1` can be smaller than `N(N−1)/2`. The operator is wide (one slot
per pair), so for large `N` pass `off_diag_batch_size` to `run_gfmc` to apply
it in walker chunks (bit-identical, lower peak memory). Subtract the separately measured k = 0 operator to obtain `S(k)`.

Example (run as shown): the transverse correlator
`S(q) = Σ_{i≠j} cos(q·dr_ij) ⟨S⁺_i S⁻_j⟩` at `q = 0` and at the smallest
nonzero `k` of an L = 8 ring, evaluated exactly on the Heisenberg ground state.

```python
import numpy as np
from pygfmc.hamiltonian import (SpinHalf, StructureFactor, chain,
                                exact_diagonalization, is_valid_k)

L = 8
lat = chain(L)
h = SpinHalf.create(lat.edges, J_x=-1.0, J_y=-1.0, J_z=1.0)

positions = np.arange(L, dtype=float)[:, None]       # (N, 1)
k = np.array([2 * np.pi / L])
print(is_valid_k(k, [[L]]))                          # True: commensurate with the ring

O_k = StructureFactor(positions, k)                  # AB(0) + AB(k), every pair i<j
O_0 = StructureFactor(positions, np.zeros(1))        # 2 AB(0)
print(O_k.M_total, O_0.M_total)                      # 25 29: pairs at distance 4 have weight 0

E, _, off = exact_diagonalization(h, off_diag_operators={"k": O_k, "0": O_0})
S0 = off["0"][0] / 2
Sk = off["k"][0] - S0
print(S0, Sk)                                        # 9.7672...  1.2217...
```

(With `J_x = J_y = −1` the Heisenberg model is written in the
sublattice-rotated frame, so `k = 0` here is the antiferromagnetic wavevector
of the unrotated chain.)

## `spin.py`

**`class Spin`** — general spin-S model (`2S` a non-negative integer). Rate
layout (blocks present only when their coupling is nonzero): `J_pm` forward
`S⁺_i S⁻_j` and backward `S⁻_i S⁺_j` (n_edges each), `J_pmpm` forward
`S⁺_i S⁺_j` and backward `S⁻_i S⁻_j`, then `h_pm·S⁺_i` and `h_pm·S⁻_i` (N each).
Forbidden moves at `m = ±S` carry zero rate; their successor configurations lie
outside the physical basis and are never sampled.

- `Spin.create(edges, S=0.5, J_z=0.0, J_pm=0.0, J_pmpm=0.0, h_z=None, h_pm=None, N=None, dtype=jnp.float32, sign_check=True)` — build the model. `h_pm` is one amplitude multiplying both `S⁺` and `S⁻` (keeps H Hermitian).
- `diagonal_weight`, `get_rates`, `generate_state` — protocol.
- `edges`, `n_edges`, `n_off`, `M_total`, `d` (= 2S+1) — derived sizes.
- `local_hilbert_space` — `(−S, ..., S)`.

Sufficient shift: `lam ≥ |J_z|·n_edges·S² + |h_z|·N·S`.

## `bose_hubbard.py`

**`class BoseHubbard`** — Bose-Hubbard model with hard occupation cutoff
`n_max`. Rate layout: forward hop `i→j` and backward hop `j→i` (n_edges each,
rate `−t√(n_i(n_j+1))` etc.), pair creation / annihilation (n_edges each), then
`b†_i` and `b_i` (N each, amplitude `h_pm`). Moves beyond `n_max` or below 0
have rate 0.

- `BoseHubbard.create(edges, t=0.0, eta=0.0, U=0.0, mu=0.0, h_pm=None, N=None, n_max=4, dtype=jnp.float32, sign_check=True)` — build the model. A nonzero `h_pm` always contributes both the `b†` and the `b` block.
- `diagonal_weight` (`U/2 Σ n(n−1) − μ Σ n`), `get_rates`, `generate_state` — protocol.
- `edges`, `n_edges`, `n_off`, `M_total` — derived sizes.
- `local_hilbert_space` — `(0, ..., n_max)`.

Sufficient shift: `lam ≥ N · max_{0≤n≤n_max} [U/2·n(n−1) − μ·n]` (the maximum over the occupation covers repulsive and attractive `U` alike).

## `plaquette.py`

**`class PlaquetteModel`** — ring exchange / Rokhsar–Kivelson model on
arbitrary plaquettes, `H = −g Σ_p flip_p + mu Σ_p |flippable_p⟩⟨flippable_p|`.
A plaquette is flippable when its spins alternate around the loop in the order
given; the flip reverses every spin of the plaquette. `rates[0] = Σ_p mu_p·[flippable_p]`,
`rates[p+1] = −g_p` if plaquette `p` is flippable, else 0.

- `PlaquetteModel.create(plaquettes, g=None, mu=None, dtype=jnp.float32, sign_check=True)` — `plaquettes` is `(N_plaquette, plaquette_size)`. There is no `N` argument: the site count is `max index + 1`, and every site from 0 to `N−1` must appear in some plaquette (a gap raises). Plaquettes must have even length: an odd loop cannot alternate around its closure, so its "flip" would change `Sz_tot` and is no ring exchange, and `create` raises on it whatever `sign_check`.
- `diagonal_weight`, `get_rates`, `generate_state` — protocol; `n_off`, `M_total`.
- `local_hilbert_space` — `(1.0, -1.0)`; `exact_diagonalization` needs nothing else.

## `toric.py`

**`class ToricCode`** — Kitaev toric code in fields on an `Lx × Ly` torus. Sites
are the `2·Lx·Ly` lattice edges: horizontal edge `h(x,y)` at index
`x + Lx·y`, vertical edge `v(x,y)` at `Lx·Ly + x + Lx·y`. Rates: diagonal, then
one star move per vertex (`−A_s`), then single-site `σˣ` flips (`−h_x`); the
off-diagonal rates are configuration-independent. Star moves conserve every
`B_p`, so start walkers in the intended plaquette sector (e.g. all `+1`).

- `ToricCode.create(Lx, Ly, A_s=1.0, B_p=1.0, h_z=0.0, h_x=0.0, dtype=jnp.float32, sign_check=True)` — `A_s`, `B_p` scalar or `(Lx·Ly,)`; `h_z`, `h_x` scalar or `(N,)`.
- `diagonal_weight`, `get_rates`, `generate_state` — protocol; `n_off`, `M_total`.
- `local_hilbert_space` — `(1.0, -1.0)`, as `SpinHalf`.

Shift: worst case `lam ≥ |B_p|·Lx·Ly + |h_z|·N`; for equilibrated walkers a
smaller value (e.g. `Lx·Ly`) is often enough, but check the diagonal statistics.

## `lattice.py`

**`class Lattice(name, N, edges, J_x=None, J_y=None, J_z=None, extras={})`** —
frozen dataclass. `J_x/J_y/J_z` are optional suggested couplings (only
`chain` sets them). `pyrochlore` fills `extras` with `hexagons`,
`type_per_hexagon`, `positions`, `supercell_dims`, `primitive_vectors`,
`sublattice_offsets`; `kagome` with `positions`, `primitive_vectors`,
`supercell_dims`, `sublattice_offsets`, `torus`, `hexagons` and
`hexagon_centers`. For any other graph, build a `Lattice(name, N, edges)`
directly.

Factories:

- `chain(n_sites, periodic=True)` — nearest-neighbour chain (suggested `J_x = J_y = −1`, `J_z = 1`); the closing bond is added only for `n_sites > 2`.
- `pyrochlore(nx, ny, nz, periodic=True)` — pyrochlore with the 4-site fcc primitive cell, `N = 4·nx·ny·nz`, `3N` edges (periodic needs all dims ≥ 2). With open boundaries only the hexagons fully inside the lattice are listed.
- `kagome(Lx, Ly, periodic=True)` — kagome lattice of `Lx × Ly` cells of a triangular Bravais lattice (`a1 = (2, 0)`, `a2 = (1, √3)`, nearest-neighbour distance 1), three sites per cell, `N = 3·Lx·Ly`, `2N` edges (periodic needs `Lx, Ly ≥ 2`). Site index `3·(x + Lx·y) + b`. One hexagon per cell in `extras['hexagons']`, counter-clockwise from angle 0; with open boundaries only the hexagons fully inside the lattice.

Every factory takes `periodic` (default `True`); `periodic=False` gives open boundaries.

Any other graph (square, triangular, ladder, ...) is a `Lattice(name, N, edges)`
built from your own edge list; the models only need the edges.

## `observables.py`

**`class SzSzStructureFactor(positions, k_vectors, dtype=jnp.float32)`** — a
diagonal observable (only `diagonal_weight`) returning one value per
wavevector: `S^zz(k) = Σ_{i<j} 2cos(k·dr_ij) S^z_i S^z_j`, computed as
`(|Σ_i e^{ik·r_i} σ_i|² − N)/4` in `O(N·n_k)`. The `i = j` terms are dropped.
Exact only for `k` in the reciprocal lattice of the simulation cell; `k` is not
validated here (use `is_valid_k`). `positions` is `(N, d)` and `k_vectors` one `(d,)` wavevector or
an `(n_k, d)` array; the attribute `k_vectors` (always `(n_k, d)`) is stored
for labelling only. `diagonal_weight` returns `(n_k,)` for one configuration
and `(n, n_k)` for a batch; in a GFMC run `obs[name][p]` is an `(n_k,)`
array.

**`is_valid_k(k_vectors, lattice_vectors, atol=1e-8)`** — whether
`k·L_j / 2π` is an integer (within `atol`) for every row `L_j` of the `(d, d)`
`lattice_vectors`, the periodic vectors of the whole system (e.g.
`supercell_dims[:, None] * primitive_vectors`, or kagome's `extras['torus']`),
not the primitive vectors of the unit cell. Returns a `bool` for one `(d,)`
wavevector and a boolean `(n_k,)` array for `(n_k, d)`. Exactly for these `k`
both structure factors are independent of the periodic image the positions
encode.

## `exact.py`

**`exact_diagonalization(hamiltonian, local_hilbert_space=None, N=None, diag_operators=None, off_diag_operators=None, number_of_states=1, return_states=False)`**
— builds the full sparse matrix from `get_rates` alone (so it works for any
protocol object, including `PlaquetteModel` and user classes; rates and the
successor-to-basis-index mapping are computed on the device in batched
chunks, and only integer indices come back to the host), solves for the
lowest `number_of_states` eigenpairs in float64 (dense `eigh` for bases of
at most 16 states or when the whole spectrum is requested, sparse Lanczos
otherwise) and evaluates observables on each. Returns `(energies, diag_obs, off_diag_obs)` (plus `states` of shape
`(dim, number_of_states)` if `return_states`). Diagonal operators are measured
via `diagonal_weight`, off-diagonal ones via their own `get_rates`.
`local_hilbert_space` (default `hamiltonian.local_hilbert_space`) fixes the
basis-digit order of the returned states. Raises beyond `2**24` basis states.
Couplings stored in float32 limit the accuracy to ~1e-7 relative; build the
model with `dtype=jnp.float64` for a tighter reference.

## Example

```python
import numpy as np
from pygfmc.hamiltonian import SpinHalf, chain, exact_diagonalization

lat = chain(10)                       # periodic ring, J_x = J_y = -1, J_z = 1
h = SpinHalf.create(lat.edges, J_x=lat.J_x, J_y=lat.J_y, J_z=lat.J_z)
print(h.N, h.M_total)                    # 10 11: 1 diagonal + 10 edge slots

config = np.array([1.0, -1.0] * 5)       # Neel state, spins are +-1 floats
rates, _ = h.get_rates(config)
print(rates[:3])                         # [-2.5 -0.5 -0.5]: H_diag, then J_pm per edge

print(exact_diagonalization(h)[0][0])    # -4.515446354492...: (energies, diag_obs, off_diag_obs)

# Protocol-only ED with a diagonal observable (nearest-neighbour Sz Sz)
zz = SpinHalf.create(lat.edges, J_z=1.0)
E, diag, off = exact_diagonalization(h, diag_operators={"ZZ": zz})
print(E[0], diag["ZZ"][0] / len(lat.edges))   # -4.515446354492...  -0.150514878...
```
