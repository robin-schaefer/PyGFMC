# PyGFMC

[![Tests](https://github.com/robin-schaefer/PyGFMC/actions/workflows/tests.yml/badge.svg)](https://github.com/robin-schaefer/PyGFMC/actions/workflows/tests.yml)
[![Docs](https://github.com/robin-schaefer/PyGFMC/actions/workflows/docs.yml/badge.svg)](https://robin-schaefer.github.io/PyGFMC/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](codes/LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](codes/pyproject.toml)

**Py**thon **G**reen's **F**unction **M**onte **C**arlo (**PyGFMC**) is a [JAX](https://github.com/jax-ml/jax)-based implementation of Green's function Monte Carlo for ground states of sign-problem-free quantum lattice models. It

* runs efficiently on **GPUs**: nearly **three orders of magnitude** higher throughput than on CPUs, scalable across multiple GPUs;
* provides access to **diagonal** and **off-diagonal** observables;
* accepts **arbitrary variational wavefunctions** as guides, including neural quantum states, which can be trained with [NetKet](https://www.netket.org);
* comes with several built-in Hamiltonians and guiding wavefunctions.

## Docs

The documentation (installation, usage, examples and API reference) is online at **https://robin-schaefer.github.io/PyGFMC/**.

## Paper

The accompanying paper, **PyGFMC: GPU-accelerated Green's Function Monte Carlo**, is available [here](https://robin-schaefer.github.io/PyGFMC/paper.html). 

## Installation

From [PyPI](https://pypi.org/project/pygfmc/):

```bash
pip install pygfmc                  # GFMC only
pip install "pygfmc[netket]"        # + training guides with NetKet
```

## Quick start

```python
import numpy as np
import jax
from pygfmc import run
from pygfmc.walkers import prepare_walkers_sz
from pygfmc.hamiltonian import SpinHalf, chain
from pygfmc.wavefunction import JastrowSpinHalf

L = 100
lat = chain(L)
H = SpinHalf.create(lat.edges, J_x=-1.0, J_y=-1.0, J_z=1.0)   # Heisenberg ring, sign-free frame
guide = JastrowSpinHalf(H.N, H.edges, beta=0.3)
walkers = prepare_walkers_sz(1024, H.N, jax.random.PRNGKey(0), sz=0)

walkers, energy, obs, stats, raw = run(
    walkers, 3000, H, step_size=0.2, guide=guide, p_array=[4, 16, 64, 128],
    rng_key=jax.random.PRNGKey(1), warmup_steps=2000, verbosity=10)
print(energy[128] / L, 0.25 - np.log(2))   # ≈ -0.44324 (stochastic), -0.44315 (L → ∞)
# exact for L = 100 (Bethe ansatz): -0.4432296
```

## License

Apache-2.0, see [`codes/LICENSE`](codes/LICENSE) and [`codes/NOTICE`](codes/NOTICE).
