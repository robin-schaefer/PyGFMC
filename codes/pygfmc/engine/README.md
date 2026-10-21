# `pygfmc.engine`: the GFMC engine

`pygfmc.engine` is the core of PyGFMC. It implements Green's Function Monte
Carlo for any Hamiltonian that follows the rate protocol of the
[`pygfmc.hamiltonian`](../hamiltonian/README.md) subpackage, optionally
importance-sampled with a guiding wavefunction from the
[`pygfmc.wavefunction`](../wavefunction/README.md) subpackage. Its public
functions are re-exported by the top-level `pygfmc` package, so
`from pygfmc import run` and `from pygfmc.engine import run` are the same.
It provides:

- three propagators (`single_step`, `multi_step`, `cont_time`),
- the multi-device reconfiguration (the single-device one is in
  [`pygfmc.walkers`](../walkers/README.md)),
- windowed estimators for the energy, diagonal observables (ancestry-traced
  pure estimator) and off-diagonal observables (forward-walking pure estimator),
- sampler and lineage diagnostics (`statistics`),
- the run drivers (`run`, `run_gfmc`, `run_gfmc_pmap`),
- HDF5 storage, resumable continuation and re-estimation from stored series.

Everything is written in JAX; walkers, Hamiltonians and guides are pytrees.

## Algorithm in brief

Each production step does:

1. **Evolve** every walker (vmapped) by one step of the importance-sampled
   propagator: `step_size` elementary sub-steps of `(lam - H)` for the discrete
   methods, or imaginary time `beta = step_size` of `exp(-beta H)` for
   `cont_time`. Each walker accumulates a log weight.
2. **Reconfigure**: systematic resampling proportional to the weights. The
   mean weight `G_n` and the weighted local energy `E_n` of the step are
   recorded, and the parent map is kept.
3. **Diagonal observables**: evaluated on the ancestors `m = p/2` steps back,
   traced through a ring buffer of parent maps.
4. **Off-diagonal observables** (every `off_diag_stride`-th step): apply the
   operator to the ensemble and propagate the result along a shadow chain for
   `max(p)/2` steps (`forward_walk_operator`).

After the loop, windowed estimates are formed for every `p` in `p_array`:
`<E>_p = sum_i G_p(i) E(i+p) / sum_i G_p(i)` with `G_p(i)` the product of `p`
consecutive step weights, and analogous centred windows for the operators.

## Conventions you must know

- **`method` and `step_size`.** `method` is `"single_step"`, `"multi_step"`
  or `"cont_time"`; the **default is `"cont_time"`** for `run`, `run_gfmc` and
  `run_gfmc_pmap`. For the discrete methods `step_size` is an integer number of
  sub-steps and the keyword `lam` is required, with `lam >= max_x H_xx` and finite `lam`.
  Under `cont_time`, `step_size` is the imaginary time per step (any finite positive
  real number) and `lam` is not used (its default is `None`). `multi_step` batches
  diagonal holds with a geometric draw; `cont_time` is its `lam -> infinity`
  limit.
- **Argument order.** Every engine function that evolves walkers takes
  `(walkers, hamiltonian, step_size, guide, method, rng_key, avg_log_weight,
  lam=None, ...)`; the drivers take `(walkers, n_steps, hamiltonian,
  step_size, guide, p_array, method="cont_time", rng_key=None, lam=None,
  ...)`.
- **`avg_log_weight`** is the normalisation shift subtracted per sub-step
  (discrete) or per unit imaginary time (`cont_time`); the two scales differ,
  so a value from one method does not transfer to another.
- **Projection depth.** `tau = (p/2) * step_size` per side of the bra-ket.
  For continuous time this is imaginary time; for discrete methods it counts
  powers of `(lam - H)` and cannot be compared directly with continuous time.
  The two sides are equal for even windows; diagonal-only runs also allow odd windows.
- **`p_array`** must contain positive integers; with off-diagonal operators
  every `p` must be even.
- **Return tuple.** `run`/`run_gfmc` return
  `(walkers, energy, obs, stats, raw)`; with `return_raw=False` the 4-tuple
  `(walkers, energy, obs, stats)`. `energy` is `{p: float}`, `obs` is
  `{name: {p: value}}`.
- **`raw`** is a plain `dict` of the per-step series: `log_weights` (the log
  step weights `log G_n`), `energy`,
  `diag` (`{name: (n_steps, [n_vals,] n_p)}`), `off_diag` and `off_diag_norm`
  (`{name: (n_meas, n_p)}`; `off_diag_norm` holds the log of the shadow-chain
  weight products), `off_diag_steps` (the 0-based step of each measured row,
  present whenever there is off-diagonal data, on both the single- and the
  multi-device path), `additional_diag` (continuation lead-in; single-device
  only) and the metadata `p_array`, `m_array`, `diag_operators`,
  `off_diag_operators`. Entries without data are `None`. The last axis of
  every operator series is positional in `p_array`.
- **`stats`** holds the end-of-run `statistics()` output (sampler health,
  offspring distribution, `ancestor_m`, `ancestor_fraction`,
  `effective_sample_size` = Kish ESS, a count in `[1, N]`,
  `ancestor_correlation` = correlation of a walker's local energy with its
  ancestor's), timings,
  `avg_log_weight`, `autocorrelation/*` and the run's inputs under
  `parameters/*`.
- **Resume via `rng_key`.** `rng_key` is the chain position, not only a seed:
  the stream advances `key_{t+1} = split(key_t)` once per step. A run returns
  its end position as `stats['parameters/rng_key_next']`; passing it back as
  `rng_key` (with the returned walkers and `stats['avg_log_weight']`, and
  no warm-up; the default) continues the chain bit-identically. `Walker.save(path,
  rng_key=...)` / `Walker.load` checkpoint it, and `write_to_HDF5` /
  `run_from_HDF5` do the whole thing for you. Re-passing an old key replays
  the same random numbers, which silently correlates the new data with the
  old.
- **Warm-up: automatic without `avg_log_weight`.** `warmup_steps` defaults
  to `None` for `run`, `run_gfmc` and `run_gfmc_pmap`: 1000 steps
  (`DEFAULT_WARMUP_STEPS`) when no `avg_log_weight` is given, and none when
  one is, which is right when continuing from an equilibrated ensemble with
  `stats['avg_log_weight']` of the previous run (`run_from_HDF5` does this).
  A fresh start of a large system may need more, e.g. `warmup_steps=10000`.
  An explicit `warmup_steps=0` without `avg_log_weight` raises `ValueError`.
  With
  `warmup_steps > 0` the driver runs a 4-stage step-size ramp
  (`step_size/8 → /4 → /2 → step_size`) that equilibrates the walkers and,
  unless `avg_log_weight` is supplied (then it is kept fixed), measures
  `avg_log_weight`. Two details of the ramp are fixed by the stored runs:
  `run_gfmc`'s warm-up rebuilds an incremental guide's carried state before
  every 50-step chunk of a stage, while production and `run_gfmc_pmap`'s
  warm-up rebuild it every 100 chain steps; and for the discrete methods
  with `step_size < 8` two ramp stages can floor to the same sub-step count
  and, with equal step counts, then draw the same key stream (they start
  from different ensembles, so the stages are correlated but not biased).
- **Repeated calls do not recompile.** `run_gfmc`'s warm-up and production
  scans are module-level compiled functions keyed on the structure of the
  Hamiltonian, guide, operators and run settings, so a continuation call (or
  any later call with the same structure and shapes) reuses the compiled
  code, and block-wise runs pay compilation once, not per block. (An object
  that cannot be used as a cache key falls back to compiling per call.)
  `avg_log_weight` enters the scans as a 0-d array of the weight dtype
  whether it was pinned as a Python float or passed back from
  `stats['avg_log_weight']`, so neither changes the compiled signature.
- **Printed output (`verbosity`, 0–10).** `0` is silent. `1` prints a
  progress line (step, it/s, number of distinct configurations); `3` adds the
  energy per `p`; `8` adds the first diagonal and first off-diagonal
  observable (re-estimated over the whole series at every progress line, so
  it costs host time) and the warm-up lines; `10` prints all observables. The
  progress line itself computes only the energy estimate, and in both
  drivers its host-side processing overlaps with the devices computing the
  next chunk (the results are identical to an unpipelined loop). The
  end-of-run statistics report, the one-line positivity-check report and the
  `p_array` deduplication notice print for any `verbosity >= 1`. Both
  backends default to `verbosity=7` and `number_of_outputs=100`. The
  progress lines of the two drivers differ in content: `run_gfmc` prints
  `step`, it/s and the number of distinct configurations (plus the energy
  per `p` from `verbosity >= 3` and operator estimates from `8`), while
  `run_gfmc_pmap` prints `step`, it/s and `n_devices` (plus the energy per
  `p` from `verbosity >= 3`; no configuration count, no operator
  estimates). `write_to_HDF5` prints one line per call unless
  `verbose=False`.
- **`off_diag_stride`.** Each off-diagonal measurement costs a shadow chain of
  `max(p)/2` steps, so measuring every `X`-th step cuts the cost ~`X`-fold at
  little statistical loss. The stride counts branching steps; pick the
  imaginary time between measurements and convert with `step_size`.
- **Off-diagonal operators** must implement the operator protocol
  (`pygfmc.Operator`: `M_total`, `get_rates`, and `generate_state` when
  `M_total > 1`), and their guide-transformed rates must be non-negative.
  Every `p` must be even when an operator has an off-diagonal transition
  (`M_total > 1`); a purely diagonal operator (`M_total == 1`) routed through
  `off_diag_operators` accepts odd `p` too, and at `p = 1` (shadow depth
  `m = 0`) its estimate is the un-propagated mean, the same number the
  diagonal estimator gives at `p = 1`.
  **Diagonal operators** need only `diagonal_weight(config)` (scalar or
  vector-valued).
- **Determinism.** Importing `pygfmc` (which any import of `pygfmc.engine`
  does first) sets
  `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0` in
  `XLA_FLAGS` (unless already set) so GPU runs are bit-reproducible; set
  `PYGFMC_NONDETERMINISTIC=1` before import to opt out.

## Minimal example

Tested on CPU with `PYTHONPATH=<path to codes>`:

```python
import jax
from pygfmc import run, evaluate_observables
from pygfmc.walkers import prepare_walkers_sz
from pygfmc.hamiltonian import SpinHalf, chain, exact_diagonalization
from pygfmc.wavefunction import JastrowSpinHalf

# L=8 Heisenberg antiferromagnet on a ring, in the sublattice-rotated
# (sign-problem-free) form J_x = J_y = -1, J_z = +1.
lat = chain(8)
H = SpinHalf.create(lat.edges, J_x=-1.0, J_y=-1.0, J_z=1.0)
guide = JastrowSpinHalf(H.N, H.edges, beta=0.3)
walkers = prepare_walkers_sz(512, H.N, jax.random.PRNGKey(0), sz=0)

# fresh walkers: a warm-up (1000 steps by default without avg_log_weight; here 500)
walkers, energy, obs, stats, raw = run(
    walkers, 2000, H, step_size=0.2, guide=guide,
    p_array=[2, 4, 8, 16], method="cont_time",
    rng_key=jax.random.PRNGKey(1), warmup_steps=500, verbosity=0)
print(energy)                              # {2: -3.65148..., 4: -3.65151..., 8: -3.65153..., 16: -3.65157...}
print(exact_diagonalization(H)[0][0])      # -3.6510934...
print(stats["effective_sample_size"])      # Kish ESS at m = p/2: [447.3 416.1 330.2 236.6]

# continue the same chain: equilibrated walkers, so no warm-up (the default);
# the compiled scan from the first call is reused
walkers, energy, obs, stats, raw = run(
    walkers, 2000, H, 0.2, guide, [2, 4, 8, 16],
    rng_key=stats["parameters/rng_key_next"],
    avg_log_weight=stats["avg_log_weight"], verbosity=0)
print(energy)                              # {2: -3.65180..., 4: -3.65189..., 8: -3.65198..., 16: -3.65204...}

# blocked-jackknife errors from the in-memory series
E, O, E_err, O_err = evaluate_observables(raw, n_blocks=20)   # E_err[p] ~ 0.0016
```

## Modules

### `__init__.py`

Re-exports the public API (the deterministic XLA flags are set by the
top-level `pygfmc/__init__.py`, which also re-exports these same names): `statistics` (`estimators.py`);
`run_gfmc` (`driver.py`); `write_to_HDF5`, `run_from_HDF5`,
`evaluate_observables`, `load_raw_from_HDF5`, `check_run_compatible` and
`resume_from_HDF5` (`io.py`); `run`
(`multi_gpu.py`). Everything else listed below (including
`evolve_walkers`, `forward_walk_operator`,
`estimate_average_log_branch_weight` and `run_gfmc_pmap`) is a module-level
function, imported from its module (e.g.
`from pygfmc.engine.evolution import evolve_walkers`).

### `evolution.py`: propagators and forward walking

- `require_operator_protocol(op, role="hamiltonian")`: raise `TypeError`
  unless `op` has `M_total`, `get_rates` and (if `M_total > 1`)
  `generate_state` (`pygfmc.Operator`).
- `require_guide_protocol(guide)`: raise `TypeError` unless `guide` is None
  or has one of `log_diff_from_operator`, `log_diff_from_configs`, `log_psi`
  (`pygfmc.Guide`). Both drivers call the two checks before any step.
- `sample_index(cdf, weights, key)`: draw a transition slot from
  `cdf = cumsum(weights)`, returning only slots with `weights > 0`. Every
  propagator and the operator application sample through it: a plain
  `searchsorted(cdf, uniform·cdf[-1])` can land on a zero-weight slot,
  because XLA's parallel `cumsum` can rise by an ulp across one and
  `uniform·total` can round up to `total` (seen as NaN energies with
  `BoseHubbard`, whose forbidden moves carry zero rate).
- `guide_cache(guide, op)`: the guide's `build_cache(op)`, or `None` if it
  has none.
- `transition_rates(config, op, guide, return_configs=False, cache=None, state=None)`:
  `(rates, new_configs)` with the guide ratio folded in; `rates[0]` is
  `H_xx`, `sum(rates)` is the local energy. Dispatches to the guide's
  `log_diff_from_operator`, `log_diff_from_configs` or `log_psi`; `state`
  (the walker's guide state of the incremental mode) is passed on as
  `state=` to `log_diff_from_operator` when given.
- `require_lam(method, lam)` returns the shift validated for `method`: a float for
  the discrete methods (`ValueError` when missing), `None` for `cont_time`.
- `evolve_walker_single_step(walker, hamiltonian, n_bra, guide, rng_key, avg_log_weight, lam, cache=None, rates=None, state=None, return_rates=False)`:
  one walker, `n_bra` sub-steps of `(lam - H)`, one draw per sub-step.
- `evolve_walker_multi_step(walker, hamiltonian, n_bra, guide, rng_key, avg_log_weight, lam, cache=None, rates=None, state=None, return_rates=False)`:
  same distribution, diagonal holds batched into geometric draws.
- `evolve_walker_cont_time(walker, hamiltonian, beta, guide, rng_key, avg_log_weight, lam=None, cache=None, rates=None, state=None, return_rates=False)`:
  continuous imaginary time `beta`: exponential hold times, weight
  `exp(-e_L tau)`; `lam` is unused. The loop carries the rates of the
  current configuration and only ends on a truncated hold, so the local
  energy of the final configuration comes out of the loop without a further
  rate evaluation.
- **Carried rates and guide state.** Every propagator ends a step with the
  rates of its final configuration and the next step starts from that
  configuration after reconfiguration has copied it, so the drivers, the
  warm-up, the shadow chains and `statistics()` can carry the pair
  `(rates, state)` across the step boundary (`evolve_carried(walkers,
  carried, ...)` takes the pair as `carried` and returns `(walkers,
  carried)`, built on `evolve_walkers(..., rates=, state=,
  return_rates=True)`; `gather_carried` reorders it with the resampling
  indices) instead of
  evaluating the rates again. Carrying the rates is **off by default**
  (`driver._RATES_CARRY_MAX_BYTES = 0`): on a CPU it measured no gain (a
  60-site spin-1 chain with 2048 walkers ran 0.86 to 1.09 times as fast),
  and returning the final rates materialises them before the local-energy
  sum, which moves `e_loc` by one float32 ulp for some models. Set
  `PYGFMC_RATES_CARRY_MAX_BYTES` to a byte limit to carry the
  `(N_walkers, M_total)` array up to that size, e.g. to measure it on a GPU;
  configurations and weights are identical either way. On
  several devices the carried rates travel through the all_gather of the
  global reconfiguration, `M_total` floats per walker next to the `N_sites`
  of the configuration, so there the saved evaluation competes with the
  collective traffic; `PYGFMC_PMAP_CARRY_RATES=0` keeps the rates local to
  a step (the guide state is still carried). `state`
  is the per-walker state of a guide in the incremental mode (see the
  wavefunction README): `initial_carried` builds both, `refresh_carried_at`
  rebuilds the pair every `_STATE_REFRESH_STEPS` (100) chain steps, `require_incremental_support`
  checks that the guide and the Hamiltonian provide what the mode needs.
- **`evolve_walkers(walkers, hamiltonian, step_size, guide, method, rng_key, avg_log_weight, lam=None, cache=None, rates=None, state=None, return_rates=False)`**:
  validate `step_size` and `lam` for `method` and evolve the whole batch
  (jitted, vmapped). Returns the updated `Walker` batch, or the tuple
  `(walkers, rates, state)` with `return_rates=True`.
- `apply_operator_one(config, key, op, guide, cache_op)`: the operator
  application of one walker, `(new_config, total_weight)`; shared by the
  single- and the multi-device forward walk.
- **`forward_walk_operator(walkers, op, h_main, step_size, guide, method, rng_key, avg_log_weight, m_array, lam=None, cache_h=None, cache_op=None, shadow_key=None, op_batch_size=None)`**:
  apply an operator to the ensemble and propagate a shadow chain for
  `max(m_array)` steps. Returns `(val, log_norm)`, each `(len(m_array),)`:
  the mean operator weight traced through the shadow parent maps, and the log
  of the shadow chain's accumulated weight (`m = 0` is the un-propagated
  mean with `log_norm = 0`). `op_batch_size` chunks the operator application
  over walkers (bit-identical, saves memory).

### `estimators.py`: estimators and diagnostics

- **`statistics(walkers, hamiltonian, step_size, guide, method, rng_key, avg_log_weight=None, lam=None, M=10, quiet=False, m_array=None, cache=None)`**:
  run `max(M, max(m_array))` branching steps and return `(walkers, stats)`
  (`avg_log_weight=None` estimates the shift from the walkers; the
  diagnostics are invariant to it up to rounding):
  sampler health over the first `M` steps (unique configs, diagonal extremes,
  `p_d` and `hold_steps_*` for the discrete methods or `escape_rate_*`,
  `hold_time_*` and `hops_per_step_*` for `cont_time`, mixing rates,
  `H_diag_max_observed`), offspring fractions every step (`offspring_k0` to
  `offspring_k3` and `offspring_k4plus`), `survival_rate`, and, with
  `m_array`, `ancestor_fraction`, `effective_sample_size` and
  `ancestor_correlation` (Pearson correlation between each walker's local
  energy at depth m and its step-0 ancestor's: 1 at m = 0, about 0 once the
  population has forgotten its start) at each depth.
  Raises if the discrete propagator loses positivity (`lam` too small). The
  pass runs on the whole population on one device; hand it a slice of a very
  large ensemble.
- `estimate_average_log_branch_weight(walkers, hamiltonian, guide, method="cont_time", lam=None, cache=None)`:
  initial `avg_log_weight`: `mean(-e_L)` (`cont_time`) or
  `mean(log(lam - e_L))` (discrete, `lam` required).
- `count_unique_configs(walkers)`: number of distinct configurations.
- `check_off_diag_operator_positivity(walkers, off_diag_operators, guide, caches=None, verbose=True)`:
  raise `ValueError` if any operator rate is negative on the first 100
  configurations.
- `autocorrelation(series, c=5.0)`: integrated autocorrelation time with
  a self-consistent window based on the absolute correlation envelope;
  the returned `(tau_int, window)` uses the signed sum, including anticorrelation.
- `autocorrelation_stats(E_np, log_weights_np, diag_np=None, diag_names=None, off_diag_np=None, off_diag_names=None, off_diag_stride=1)`:
  the `autocorrelation/*` entries of `stats` (in branching steps):
  `autocorrelation/energy_estimator`, `autocorrelation/log_weights`,
  `autocorrelation/diag/<name>` and `autocorrelation/off_diag/<name>`.
  Both drivers record them.
- `dense_and_names(series, names=None, what="off_diag_np")`: convert a
  `{name: array}` mapping to `(dense, names)` with a positional operator axis.

### `driver.py`: single-device driver

- **`run_gfmc(walkers, n_steps, hamiltonian, step_size, guide, p_array, method="cont_time", rng_key=None, lam=None, diag_operators=None, off_diag_operators=None, return_raw=True, avg_log_weight=None, warmup_steps=None, verbosity=7, number_of_outputs=100, off_diag_stride=1, off_diag_batch_size=None)`**:
  warm-up (1000 steps by default without `avg_log_weight`, none with it), production scan, windowed estimation
  and end-of-run `statistics`. Returns `(walkers, energy, obs, stats, raw)` (4-tuple without
  `raw` when `return_raw=False`). Without `rng_key` a fresh key is drawn from `secrets.randbits(32)` (printed at `verbosity >= 1`, stored in `stats['parameters/rng_key']`).
  With `return_raw=False` the off-diagonal measurements are restricted to the
  steps usable by every `p` (equal windows): the per-call numbers are then
  final and comparable across `p`. With the raw series kept, every step is
  measured and the windows can span run boundaries on re-estimation.
  Before anything is walked the plan checks that `max(p) < n_steps`, that
  with off-diagonal operators `n_steps > max(p) + max(p)//2`, and that with
  equal windows the band `[max(p), n_steps - max(p)//2 - 1]` contains a
  stride step (`ValueError` otherwise); `off_diag_batch_size` must be an
  integer `>= 1` dividing the walker count (the per-device shard in
  `run_gfmc_pmap`).
- `LINEAGE_COST_WARN_FRAC`: warn when the end-of-run lineage pass adds more
  than this fraction of `n_steps`.
- The driver is a short orchestration of module-level stages shared with
  `run_gfmc_pmap`: `_measurement_plan` (window grid, measurement gating),
  `_warmup_schedule` and `_warmup` (the ramp), `_positivity_probe`,
  `_production` (the chunked, pipelined scan loop), `_run_stats` and
  `_raw_series` (the `stats` and `raw` dicts); `run_gfmc_pmap` has its own
  `_production_pmap` loop over the pmapped step chunk. Every message is
  printed through `pygfmc._output.emit`, gated on `verbosity`.

### `multi_gpu.py`: multi-device driver and dispatcher

- **`run(walkers, n_steps, hamiltonian, step_size, guide, p_array, method="cont_time", rng_key=None, lam=None, **kwargs)`**:
  the recommended entry point. Same arguments and return value as
  `run_gfmc` (plus `n_devices`, default `jax.device_count()`); uses
  `run_gfmc_pmap` when more than one device is used, unless the run has
  `diag_operators`; those fall back to `run_gfmc`. Off-diagonal
  measurement gating (`off_diag_stride > 1`, `return_raw=False`) works on
  both backends and measures the same steps.
- `run_gfmc_pmap(walkers, n_steps, hamiltonian, step_size, guide, p_array, method="cont_time", rng_key=None, lam=None, diag_operators=None, off_diag_operators=None, avg_log_weight=None, warmup_steps=None, verbosity=7, number_of_outputs=100, n_devices=None, return_raw=True, off_diag_stride=1, off_diag_batch_size=None)`:
  walkers sharded over devices, global reconfiguration. Energy and
  off-diagonal operators only: `diag_operators` raise `NotImplementedError`.
  Off-diagonal measurement gating (`off_diag_stride`, the equal windows of
  `return_raw=False`) measures the same steps as `run_gfmc`, and
  `raw['off_diag_steps']` labels each stored row. Warm-up, the positivity
  check, `stats` (including `parameters/*` and `parameters/rng_key_next`)
  and continuation follow `run_gfmc`, but the chain differs from the
  single-device chain at equal seed (per-device key folding, global
  resampling).
- `shard_walkers(walkers, n_devices=None)` / `unshard_walkers(walkers)`:
  add/remove the leading device axis (walker count must divide evenly).
- `reconfiguration_global(walkers, rng_key, axis_name="devices", aux=None)`:
  systematic resampling across all devices (inside `pmap`), returning
  `(walkers, log_G_n, E_n[, aux])`; optionally carries per-walker `aux` data
  through the resampling.
- `forward_walk_operator_global(walkers, op, h_main, step_size, guide, method, rng_key, avg_log_weight, m_array, lam=None, axis_name="devices", cache_h=None, cache_op=None, op_batch_size=None, shadow_key=None)`:
  multi-device `forward_walk_operator` (inside `pmap`).

### Reproducibility: what is guaranteed

- **Same inputs, same bits.** A run with the same walkers, parameters and
  `rng_key` reproduces its configurations, weights, energies and series bit
  for bit on the same backend (the deterministic-XLA flags are on by
  default; see the package README). Results differ between CPU and GPU.
- **Precision.** The package computes in float32; weights follow
  `jax_enable_x64` (float64 when it is on). Guide checkpoints keep the
  parameters' own dtype, so a float64 guide loads back as float64.
- **RNG keys.** The drivers and `Walker.save` accept legacy `PRNGKey` arrays
  and typed Threefry `jax.random.key` keys with the default Threefry PRNG;
  both are stored as uint32 key data and load back as legacy keys, which
  continue the same stream. Other implementations are rejected before running
  or writing rather than producing unusable checkpoints.
- **Continuation.** `run_from_HDF5` re-walks exactly the steps a contiguous
  run would: the walkers, the production-final key and the diagonal lead-in
  come from the file, so a run split over several blocks equals the
  unsplit run (`test_split_run_matches_contiguous_for_equal_and_unequal_pieces`).
- **Incremental guide mode.** A run with `incremental=True` is not
  bit-identical to the same run without it (the neighbour sums are updated
  instead of rebuilt), but it is reproducible in itself: the carried state
  and rates are rebuilt every `_STATE_REFRESH_STEPS` (100) chain steps,
  counted from the start of the chain and passed to a continuation as its
  step offset, so neither `number_of_outputs` nor a split changes the
  result (`test_incremental_mode_does_not_depend_on_the_output_chunking`).
- **Carried rates** (off by default). Carrying the rates across a step
  leaves every configuration and weight unchanged; `e_loc` can differ by one
  float32 ulp from the default run, because the final rates are
  materialised instead of fused into the local-energy sum.
- **Population extinction.** A step on which every walker's weight is zero
  raises a `ValueError` naming the step instead of propagating NaN.

### `io.py`: HDF5 storage, continuation and re-estimation

A file holds one continuous Markov chain: fixed parameters in `/parameters`,
the newest walker ensemble (with its RNG key) in `/current_walkers`, one
`/run_<n>` group per call, and the per-step series appended to `/raw`.

Reading goes through the internal `_StoredChain` view (one open handle: run
indices, fixed and per-run parameters, statistics, the diagonal lead-in,
the walker source), which `run_from_HDF5`, `resume_from_HDF5`,
`load_raw_from_HDF5` and `check_run_compatible` share. `evaluate_observables`
validates its input through `_normalized_raw`, which turns a raw dict (from
`run()`, from a file, or assembled by hand) into numpy arrays with the
operator names and the p grid resolved.

The raw series are written with the `shuffle` filter plus gzip level 1, in
chunks of whole steps (~1 MiB each). This is lossless and readable by any HDF5
tool; appending to an existing file keeps that file's
existing dataset settings.

- **`write_to_HDF5(filename, stats, walkers=None, energy=None, obs=None, raw=None, walker_filename=None, verbose=True)`**:
  append one run. Validates the `MUST_MATCH` parameters and content hashes
  (Hamiltonian, guide, operators) and the RNG stream (a replayed key raises)
  before writing. Returns the run index; prints one line unless
  `verbose=False`.
- **`check_run_compatible(filename, stats=None, **parameters)`**: the same
  validation *before* a run: pass the objects you are about to use
  (`hamiltonian=`, `guide=`, `diag_operators=`, `off_diag_operators=`) and
  the plain parameters (`n_walkers=`, `step_size=`, ...) and a mismatch
  raises immediately instead of after the walk. A missing file passes.
- **`resume_from_HDF5(filename, hamiltonian=None, guide=None, diag_operators=None, off_diag_operators=None, walker_filename=None, dtype=None, config_dtype=None)`**:
  the pieces of a continuation as `run_gfmc` keyword arguments
  (`walkers`, `rng_key`, `avg_log_weight`, `warmup_steps=0`), verified
  against the file when the objects are given. `run_from_HDF5` builds on it.
- **`run_from_HDF5(filename, hamiltonian, guide=None, diag_operators=None, off_diag_operators=None, n_steps=None, walker_filename=None, **run_kwargs)`**:
  continue a stored chain in one call (no warm-up, stored key and
  `avg_log_weight`, same `lam`/`step_size`/`p_array`/`method`); returns what
  `run` returns, with the diagonal lead-in spliced in so the result matches an
  uninterrupted run.
- **`load_raw_from_HDF5(filename)`**: the whole stored chain's raw series in
  `run`'s `raw` layout, with `p_array`, `m_array` and operator names.
- **`evaluate_observables(raw, p_array=None, n_blocks=None)`**:
  re-estimate from a `raw` dict (in memory or loaded). The energy can be
  re-windowed at any `p_array`; operators only at the stored grid. With
  `n_blocks`, returns `(energy, obs, energy_err, obs_err)` from a blocked
  jackknife (use blocks longer than both `2 tau_int` and `p`, and about 20 or
  more blocks). A trailing remainder is discarded with a warning so point
  estimates and errors use the same equal-block dataset. This is the one public estimator: the drivers use the same
  internal windowed estimator (`estimators._estimate_observables`) at run time,
  so `evaluate_observables` reproduces their numbers exactly when no samples
  are discarded. A hand-built `raw` (for
  example a slice of the series) works too; it then needs `p_array=` (or a
  `raw['p_array']` entry naming the grid its operator arrays sit on), and its
  off-diagonal rows are placed by `raw['off_diag_steps']`, which must be
  present whenever `raw['off_diag']` is.
- `pytree_fingerprint(obj)`: content hash of a pytree (Hamiltonian, guide,
  operator); used to detect changed objects behind unchanged names.
- `diag_mapping_to_dense(diag)`: `{name: array}` to a dense array, or `None`
  if the widths differ.
- `MUST_MATCH`, `MAY_DIFFER`: which parameters must agree across the runs of
  one file, and which may vary.
- `FORMAT`, `FORMAT_VERSION`: the `format`/`format_version` file attributes
  (`"pygfmc"`, `2`) that `write_to_HDF5` stamps on a new file.
