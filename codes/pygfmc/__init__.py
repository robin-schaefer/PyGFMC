"""PyGFMC — Green's Function Monte Carlo in JAX.

Subpackages: :mod:`pygfmc.walkers` (walker population, initializers,
reconfiguration), :mod:`pygfmc.engine` (propagation, estimators, I/O),
:mod:`pygfmc.hamiltonian` (models and lattices) and :mod:`pygfmc.wavefunction`
(guiding wavefunctions).  The everyday engine API is re-exported here; the
walker functions are imported from :mod:`pygfmc.walkers`.
"""
import os as _os

# --------------------------------------------------------------------------
# Deterministic GPU kernels, ON BY DEFAULT.
#
# Without these flags a byte-identical invocation can produce a DIFFERENT
# Markov chain on GPU.  The cause is not the RNG (JAX's counter-based PRNG has
# no hidden state and PRNGKey(seed) gives identical bits on any device) but
# floating-point summation order: XLA may combine partial sums in
# thread-completion order, float addition is not associative, and a 1-ULP
# shift in a weight normalisation can move a systematic-resampling boundary or
# a searchsorted threshold.  One flipped discrete decision is then amplified by
# the chain.
#
# This does NOT change the physics: any trajectory produced by the correct
# kernel is a valid sample, equivalent to a different seed, and the ~1e-7
# relative FP noise sits orders of magnitude below the statistical error.
# What the flags buy is REPRODUCIBILITY (bit-identical reruns and resumes).
# Their throughput cost has not been measured carefully; it is expected to be
# small, since the hot path is gathers/scatters rather than GEMMs.  Set
# PYGFMC_NONDETERMINISTIC=1 to disable them.
#
# NOT set here: JAX_DEFAULT_MATMUL_PRECISION=highest.  TF32 is deterministic
# run-to-run on fixed hardware (it affects cross-hardware agreement, not the
# problem above) and forcing highest costs time on matmul-heavy guides such as
# ClusterMLP.  Set it yourself if you need bitwise agreement across GPU models.
#
# XLA_FLAGS is read lazily at backend initialisation, not at `import jax`, so
# this works even if jax was imported earlier -- provided no JAX computation
# has run yet.
# --------------------------------------------------------------------------
def _enable_deterministic_xla():
    if _os.environ.get("PYGFMC_NONDETERMINISTIC", "") not in ("", "0"):
        return
    existing = _os.environ.get("XLA_FLAGS", "")
    # append only what the caller has not already specified, so an explicit
    # user setting always wins
    for flag in ("--xla_gpu_deterministic_ops=true",
                 "--xla_gpu_autotune_level=0"):
        name = flag.split("=")[0]
        if name not in existing:
            existing = (existing + " " + flag).strip()
    _os.environ["XLA_FLAGS"] = existing


_enable_deterministic_xla()

# PyGFMC runs in float32.  ``import netket`` switches on jax_enable_x64
# process-wide unless NETKET_ENABLE_X64=0, and float64 walkers then meet a
# float32 Hamiltonian in the GFMC scan.  NetKet's experimental sharding leaks a
# named mesh into JAX's config that breaks later ``jax.pmap`` calls.  Importing
# pygfmc BEFORE netket avoids both; explicit settings by the user win.
# Newer NetKet releases read NETKET_SHARDING (the experimental name is
# deprecated), so both are set.
_os.environ.setdefault("NETKET_ENABLE_X64", "0")
_os.environ.setdefault("NETKET_EXPERIMENTAL_SHARDING", "0")
_os.environ.setdefault("NETKET_SHARDING", "0")

def _read_version():
    """The installed distribution's version, else the one in the source
    tree's ``pyproject.toml`` (the single place the number is written)."""
    from importlib.metadata import version, PackageNotFoundError
    try:
        return version("pygfmc")
    except PackageNotFoundError:          # running from a source tree
        pass
    import re
    pyproject = _os.path.join(_os.path.dirname(_os.path.dirname(
        _os.path.abspath(__file__))), "pyproject.toml")
    try:
        with open(pyproject, encoding="utf-8") as f:
            match = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.M)
        if match:
            return match.group(1)
    except OSError:
        pass
    return "0+unknown"


__version__ = _read_version()

# The subpackages are imported only AFTER the XLA flags are set.
from . import walkers, engine, hamiltonian, wavefunction
from .protocols import Operator, DiagonalOperator, Guide
from .engine import (statistics, run_gfmc, run, write_to_HDF5, run_from_HDF5,
                     evaluate_observables, load_raw_from_HDF5,
                     check_run_compatible, resume_from_HDF5)

__all__ = [
    "walkers", "engine", "hamiltonian", "wavefunction", "__version__",
    "Operator", "DiagonalOperator", "Guide",
    "statistics", "run_gfmc", "run",
    "write_to_HDF5", "run_from_HDF5", "evaluate_observables",
    "load_raw_from_HDF5", "check_run_compatible", "resume_from_HDF5",
]
