# `pygfmc.walkers`: walkers, initializers and reconfiguration

A GFMC simulation is a population of **walkers**: each is a basis
configuration with a log weight, and together they represent the projected
state. This subpackage holds the `Walker` pytree, the functions that prepare an
initial population, and the systematic resampling (`reconfiguration`) that is
applied after every step to keep the population size fixed. The engine in
[`pygfmc.engine`](../engine/README.md) builds on it; this subpackage does not
import the engine.

```python
from pygfmc.walkers import Walker, prepare_walkers_sz, reconfiguration
```

## `walker.py`: the `Walker` pytree and the initializers

- **`Walker(config, log_weight, e_loc)`**: `NamedTuple` pytree holding a
  batch (or a single walker): configurations `(N_walkers, N_sites)`, log
  weights and local energies `(N_walkers,)`.
  - `Walker.save(path, rng_key=None, group="walkers", **attrs)` writes the
    walkers (config compacted to `int8` when lossless), the RNG key and scalar
    attributes to HDF5.
  - `Walker.load(path, group="walkers", dtype=None, config_dtype=None)`
    returns `(walker, rng_key, avg_log_weight)` (`None` for missing items).
    `dtype=None` uses JAX's default float, like the initializers.
- **Initializers** (all return a `Walker` batch with zero log weights):
  - **`prepare_walkers_random(N_walkers, N_sites, rng_key, states=(-1.0, 1.0))`**:
    every site drawn independently from `states`; for non-conserving models.
  - **`prepare_walkers_sz(N_walkers, N_sites, rng_key, sz=0, states=(-1.0, 1.0))`**:
    random permutations of a multiset whose values sum exactly to `sz`
    (2 S^z for ±1 spins, S^z for S=1 with `(-1, 0, 1)`, particle number for
    bosons). Use it to start inside a conserved sector (`sz=0` for a
    balanced ±1 start).
  - **`prepare_walkers_from_configs(configs)`**: walkers at given
    configurations, an `(N_walkers, N_sites)` array (e.g. all walkers in a
    fixed topological sector of the toric code, `jnp.ones((N_walkers, N_sites))`).

## `reconfiguration.py`: systematic resampling

- **`reconfiguration(walkers, rng_key)`**:
  systematic resampling. Returns `(walkers, log_G_n, E_n, indices)`
  with `log_G_n` the log of the mean weight, `E_n` the weighted local energy
  and `indices[i]` the parent of new walker `i`; the new walkers have zero
  log weights.
- `systematic_parents(w, psi)`: the parent lookup of systematic resampling
  for normalised weights `w` and offset `psi`, shared with the multi-device
  variant: parent `k` for `cdf[k-1] <= u < cdf[k]`, capped at the last
  walker of positive weight, so a zero-weight walker is never selected
  (including at `psi = 0` or when rounding leaves the cdf short of 1).

The multi-device variant (`reconfiguration_global`, which resamples across all
devices) is part of the engine's multi-GPU driver, `pygfmc/engine/multi_gpu.py`.

## `_h5io.py`: internal helper

`write_attrs(group, attrs)` writes scalar metadata as HDF5 group attributes
(`None` values skipped, float scalars stored as Python `float`). It serves
`Walker.save` and the `save` methods of the guides in `pygfmc.wavefunction`.
