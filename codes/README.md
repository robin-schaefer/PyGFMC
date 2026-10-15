# PyGFMC

PyGFMC is a Green's Function Monte Carlo (GFMC) code for quantum lattice models, written in [JAX](https://github.com/jax-ml/jax) so it runs on CPUs and GPUs (including several GPUs at once). It projects the ground state out of a trial state by imaginary-time evolution, with importance sampling by a guiding wavefunction. It covers:

- three propagators: `single_step`, `multi_step` and continuous-time `cont_time` (the default);
- mixed estimators for the energy, and pure (forward-walking) estimators for diagonal and off-diagonal observables;
- spin-1/2, spin-S, Bose-Hubbard, plaquette ring-exchange and toric-code Hamiltonians on arbitrary graphs;
- Jastrow guides for spin-1/2, spin-S and bosons, cluster-CPS and cluster-MLP guides, optionally trained by VMC through NetKet;
- HDF5 checkpointing (lossless compressed series) with bit-identical continuation, and re-estimation with blocked-jackknife error bars.

- **Documentation:** https://robin-schaefer.github.io/PyGFMC/
- **Repository:** https://github.com/robin-schaefer/PyGFMC
- **Paper:** *PyGFMC: GPU-accelerated Green's Function Monte Carlo*, available [here](https://robin-schaefer.github.io/PyGFMC/paper.html)

## Installation

From [PyPI](https://pypi.org/project/pygfmc/):

```bash
pip install pygfmc            # GFMC only
pip install "pygfmc[netket]"  # + training guides with NetKet (VMC)
```

From source, in this folder of a clone of the [repository](https://github.com/robin-schaefer/PyGFMC):

```bash
pip install .                 # GFMC only
pip install '.[netket]'       # + training guides with NetKet (VMC)
pip install '.[test]'          # + pytest
```

For GPU support, install the CUDA build of JAX first, following the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Dependencies

| Group | Packages | Needed for |
|---|---|---|
| core | `jax`, `numpy`, `scipy`, `h5py` | everything: GFMC runs, Hamiltonians, guides, exact diagonalization, HDF5 I/O |
| `netket` extra | `netket`, `optax`, `flax` | only for **training** guides by VMC (`pygfmc.wavefunction.train_with_netket`) |
| `test` extra | `pytest` | the test suite |

Python >= 3.11. Tested with Python 3.13.9, jax 0.9.2, numpy 2.3.5, scipy 1.16.3, h5py 3.15.1, flax 0.12.5, netket 3.21.0 and optax 0.2.6.

The NetKet imports are inside the training function, so `import pygfmc` works without NetKet. A training call without it raises an `ImportError` that says how to install it. Guides trained elsewhere can be saved to HDF5 and loaded for GFMC.

## Package layout

The code is one package, `pygfmc`, with four subpackages: one for each
ingredient of a simulation (walkers, Hamiltonian, guide) and the engine that
runs it. The everyday engine API (`run`, `run_gfmc`, `statistics` and the
HDF5 functions) is re-exported at the top level, so `from pygfmc import run`
works; walkers, Hamiltonians and guides are imported from their subpackages
(`from pygfmc.walkers import prepare_walkers_sz`,
`from pygfmc.hamiltonian import SpinHalf`, `from pygfmc.wavefunction import JastrowSpinHalf`).
Importing `pygfmc` also enables deterministic XLA GPU kernels (set
`PYGFMC_NONDETERMINISTIC=1` before the import to opt out).
All console output of the package goes through `pygfmc._output.emit`, gated
on the `verbosity` argument of the drivers (`verbosity=0` is silent); set
`pygfmc._output.stream` to redirect it to a file-like object.

```
pygfmc/
├── __init__.py      # deterministic XLA flags, top-level API, __version__
├── _output.py       # every printed message passes through emit(), gated on verbosity
├── protocols.py     # Operator, DiagonalOperator, Guide: the interfaces as typing.Protocol classes
├── _precision.py    # float_dtype(): float64 when jax_enable_x64 is on, else float32
├── walkers/         # Walker population, initializers, reconfiguration
├── engine/          # the GFMC engine
├── hamiltonian/     # models, lattices, exact diagonalization
└── wavefunction/    # guiding wavefunctions
```

| Folder | Role | Detailed documentation |
|---|---|---|
| `pygfmc/walkers/` | The `Walker` population (a JAX pytree with HDF5 `save`/`load`), the initializers `prepare_walkers_random`, `prepare_walkers_sz`, `prepare_walkers_from_configs`, and `reconfiguration` (systematic resampling). | [`pygfmc/walkers/README.md`](https://github.com/robin-schaefer/PyGFMC/blob/main/codes/pygfmc/walkers/README.md) |
| `pygfmc/engine/` | The GFMC engine: the propagators, reconfiguration (resampling), the estimators and diagnostics, the drivers `run` / `run_gfmc` (and the multi-device `run_gfmc_pmap` behind `run`), and HDF5 storage, continuation and re-estimation. | [`pygfmc/engine/README.md`](https://github.com/robin-schaefer/PyGFMC/blob/main/codes/pygfmc/engine/README.md) |
| `pygfmc/hamiltonian/` | Hamiltonians and measurement operators that follow the rate protocol `pygfmc.Operator` (`M_total`, `get_rates`, `generate_state`), lattice factories, structure-factor measurement operators, and exact diagonalization for reference values. | [`pygfmc/hamiltonian/README.md`](https://github.com/robin-schaefer/PyGFMC/blob/main/codes/pygfmc/hamiltonian/README.md) |
| `pygfmc/wavefunction/` | Guiding (trial) wavefunctions, which follow the guide protocol `pygfmc.Guide` (`log_diff_from_operator` / `log_diff_from_configs` / `log_psi`), plus optional VMC training through NetKet. | [`pygfmc/wavefunction/README.md`](https://github.com/robin-schaefer/PyGFMC/blob/main/codes/pygfmc/wavefunction/README.md) |

**Each subpackage has its own `README.md`.** They document every module, all public functions and classes with their signatures, the protocols for writing your own Hamiltonian or guide, and the conventions (step size, estimator windows, return values, resuming a run). This file gives an overview only.

## Quick start

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

# fresh, unequilibrated walkers: ask for a warm-up explicitly (default is none)
walkers, energy, obs, stats, raw = run(
    walkers, 2000, H, step_size=0.2, guide=guide,
    p_array=[2, 4, 8, 16], method="cont_time",
    rng_key=jax.random.PRNGKey(1), warmup_steps=500, verbosity=0)
print(energy)                              # dict p -> energy: {2: -3.65148..., 4: -3.65151..., 8: -3.65153..., 16: -3.65157...}
print(exact_diagonalization(H)[0][0])      # -3.6510934...

# continue the same Markov chain (bit-identical to one longer run); the
# walkers are equilibrated now, so no warm-up, and the compiled scan is reused
walkers, energy, obs, stats, raw = run(
    walkers, 2000, H, 0.2, guide, [2, 4, 8, 16],
    rng_key=stats["parameters/rng_key_next"],
    avg_log_weight=stats["avg_log_weight"], verbosity=0)

# blocked-jackknife error bars from the returned raw series
E, O, E_err, O_err = evaluate_observables(raw, n_blocks=20)   # E_err[p] ~ 0.0016
```

**Warm-up is off by default (`warmup_steps=0`).** That is right when you
continue from an equilibrated ensemble (walkers returned by a previous run or
loaded from a checkpoint). When you start from freshly prepared walkers, pass
a warm-up explicitly, e.g. `warmup_steps=10000` (the example uses 500 because
the system is tiny); otherwise the first production steps are not
equilibrated and `avg_log_weight` stays at its initial estimate.

## Example notebooks

The repository's [`notebooks/`](https://github.com/robin-schaefer/PyGFMC/blob/main/notebooks/README.md) folder holds runnable
examples, stored with the outputs of a full run:

| Notebook | Model | What it shows |
|---|---|---|
| `bose_hubbard_chain.ipynb` | Bose-Hubbard chain, 50 sites | a guide trained with NetKet, then the ground-state energy |
| `transverse_field_ising_chain.ipynb` | critical spin-1 transverse-field Ising chain, 12 sites | every code block of the documentation page *Running a simulation* |
| `xxz_chain.ipynb` | XXZ chain with random couplings and a field, 20 sites | the paper's example: a Jastrow guide trained with NetKet, observables, the raw per-step series, the exact ground state as guide |
| `n_body_model.ipynb` | four-site n-body model with a field, 12 sites | writing your own Hamiltonian and guiding wavefunction, and training the guide with NetKet |
| `kagome_xxz.ipynb` | XXZ model on the kagome lattice, 18 and 108 sites | a symmetry-reduced cluster guide: training, storing, loading, and its effect on GFMC |
| `toric_code.ipynb` | toric code, 10×10 and 3×3 tori | the exact energy without a field, and the convergence with fields against exact diagonalization |
| `ring_model_pyrochlore.ipynb` | ring exchange on the pyrochlore lattice, 128 sites | sweep from pure ring exchange to the Rokhsar-Kivelson point |

## Conventions at a glance

- Configurations are stored as floats: ±1 for spin 1/2 (not 0/1), `m ∈ {−S, …, S}` for spin S, occupations `0, …, n_max` for bosons.
- Everything runs in float32 by default; importing `pygfmc` sets `NETKET_ENABLE_X64=0`, so a later `import netket` does not switch JAX to float64.
- `method` is one of `"single_step"`, `"multi_step"` or `"cont_time"` (the default). With `cont_time`, `step_size` is the imaginary time per branching step (any positive float) and no shift is needed. With the discrete methods, `step_size` is an integer number of sub-steps of `(lam - H)`, and the keyword `lam` (the shift, `lam >= max_x H_xx`) is required.
- `run` / `run_gfmc` return `(walkers, energy, obs, stats, raw)`. Access energy as `energy[p]` and observables as `obs[name][p]`. For even windows, the projection depth on each side of the estimator is `tau = (p/2) * step_size`.
- `rng_key` is the position of the chain. Passing `stats["parameters/rng_key_next"]` back continues the run exactly. Repeated `run_gfmc` calls reuse the compiled scans, so a continuation does not recompile.
- `warmup_steps` defaults to 0 (no warm-up); pass e.g. `warmup_steps=10000` for a fresh start.
- `verbosity=0` prints nothing; the default progress line reports the energy per `p`, and operator estimates are printed only at `verbosity >= 8`.
- Off-diagonal measurement operators must have non-negative rates, and all `p` must be even when one of them has an off-diagonal transition (`M_total > 1`).

## Full list of public functions and classes

These are the names each subpackage exports, grouped by the module that defines them. The subfolder READMEs also list the methods of every class (`create`, `get_rates`, `log_psi`, `save`/`load`, ...) and the module-level functions that are not exported (e.g. `pygfmc.engine.multi_gpu.run_gfmc_pmap`). Everything listed under `pygfmc.engine` is also importable directly from `pygfmc`.

### `pygfmc.walkers`

**`pygfmc/walkers/walker.py`**

| Name | Kind | Description |
|---|---|---|
| `prepare_walkers_random` | function | Initialize walkers by drawing every site independently from `states`. |
| `prepare_walkers_sz` | function | Initialize walkers in a fixed-charge sector drawn from `states`. |
| `prepare_walkers_from_configs` | function | Initialize walkers at user-supplied configurations, an `(N_walkers, N_sites)` array. |
| `Walker` | class | Walker population (or a single walker) as a JAX pytree. |

**`pygfmc/walkers/reconfiguration.py`**

| Name | Kind | Description |
|---|---|---|
| `reconfiguration` | function | Systematic resampling of the walker population. |

### `pygfmc.engine`

**`pygfmc/engine/estimators.py`**

| Name | Kind | Description |
|---|---|---|
| `statistics` | function | Run branching steps and collect diagnostic statistics. |

**`pygfmc/engine/driver.py`**

| Name | Kind | Description |
|---|---|---|
| `run_gfmc` | function | Run the GFMC production loop on a single device. |

**`pygfmc/engine/io.py`**

| Name | Kind | Description |
|---|---|---|
| `evaluate_observables` | function | Re-evaluate the energy and the observables from raw per-step series. |
| `load_raw_from_HDF5` | function | Read a file's raw series back in the layout `run` returns them in. |
| `run_from_HDF5` | function | Continue a stored chain in one call, and return what `run_gfmc` returns. |
| `check_run_compatible` | function | Validate a planned run against a stored file *before* walking. |
| `resume_from_HDF5` | function | Assemble the `run_gfmc` arguments that continue a stored chain. |
| `write_to_HDF5` | function | Append one GFMC run to an HDF5 file as a self-contained group. |

**`pygfmc/engine/multi_gpu.py`**

| Name | Kind | Description |
|---|---|---|
| `run` | function | Auto-dispatch GFMC: run_gfmc_pmap on >1 devices, run_gfmc otherwise. |

### `pygfmc.hamiltonian`

**`pygfmc/hamiltonian/bose_hubbard.py`**

| Name | Kind | Description |
|---|---|---|
| `BoseHubbard` | class | Bose-Hubbard model; see the module docstring for H and the rate layout. |

**`pygfmc/hamiltonian/exact.py`**

| Name | Kind | Description |
|---|---|---|
| `exact_diagonalization` | function | Lowest eigenpairs of `hamiltonian`, with observables on each. |

**`pygfmc/hamiltonian/lattice.py`**

| Name | Kind | Description |
|---|---|---|
| `chain` | function | Nearest-neighbour chain of `n_sites` sites. |
| `kagome` | function | Kagome lattice of `Lx` x `Ly` unit cells (O(N)). |
| `Lattice` | class | A finite lattice: a site count, an edge list and optional metadata. |
| `pyrochlore` | function | Pyrochlore lattice built from explicit FCC unit-cell edge templates (O(N)). |

**`pygfmc/hamiltonian/observables.py`**

| Name | Kind | Description |
|---|---|---|
| `SzSzStructureFactor` | class | `S^{zz}(\vec k)` for one or many wavevectors, from positions. |
| `is_valid_k` | function | Whether each wavevector lies in the reciprocal lattice of the system. |

**`pygfmc/hamiltonian/plaquette.py`**

| Name | Kind | Description |
|---|---|---|
| `PlaquetteModel` | class | Plaquette ring-exchange / Rokhsar–Kivelson Hamiltonian. |

**`pygfmc/hamiltonian/spin.py`**

| Name | Kind | Description |
|---|---|---|
| `Spin` | class | Spin-S Hamiltonian; see the module docstring for H and the rate layout. |

**`pygfmc/hamiltonian/SpinHalf.py`**

| Name | Kind | Description |
|---|---|---|
| `SpinHalf` | class | Spin-1/2 XYZ model with longitudinal and transverse fields. |
| `StructureFactor` | function | The k-resolved correlator `AB(0) + AB(k)` as ONE `SpinHalf`. |

**`pygfmc/hamiltonian/toric.py`**

| Name | Kind | Description |
|---|---|---|
| `ToricCode` | class | Toric code Hamiltonian on an Lx × Ly torus. |

### `pygfmc.wavefunction`

**`pygfmc/wavefunction/cluster_cps.py`**

| Name | Kind | Description |
|---|---|---|
| `ClusterCPS` | class | Cluster-CPS guide composed from one or more patch groups. |

**`pygfmc/wavefunction/cluster_mlp.py`**

| Name | Kind | Description |
|---|---|---|
| `ClusterMLP` | class | Per-cluster MLP guide for SpinHalf, with one shared tanh MLP per cluster group. |

**`pygfmc/wavefunction/exact_ground_state.py`**

| Name | Kind | Description |
|---|---|---|
| `ExactGroundState` | class | Guiding wavefunction with amplitudes from the exact ground state. |

**`pygfmc/wavefunction/generic_netket_training.py`**

| Name | Kind | Description |
|---|---|---|
| `make_hilbert` | function | A NetKet Hilbert space whose local states are exactly these values. |
| `train_with_netket` | function | Train `guide` on `hamiltonian` with NetKet VMC and return a new guide. |

**`pygfmc/wavefunction/jastrow_bose_hubbard.py`**

| Name | Kind | Description |
|---|---|---|
| `JastrowBoseHubbard` | class | Jastrow guide for boson occupations, with an on-site (Gutzwiller) term. |

**`pygfmc/wavefunction/jastrow_spin_half.py`**

| Name | Kind | Description |
|---|---|---|
| `JastrowSpinHalf` | class | Jastrow guiding wavefunction with sparse two-body couplings. |

**`pygfmc/wavefunction/jastrow_spin.py`**

| Name | Kind | Description |
|---|---|---|
| `JastrowSpin` | class | Jastrow guide for spin-S configurations, with a single-ion term. |

## License

Apache-2.0, see [`LICENSE`](https://github.com/robin-schaefer/PyGFMC/blob/main/codes/LICENSE) and [`NOTICE`](https://github.com/robin-schaefer/PyGFMC/blob/main/codes/NOTICE).
