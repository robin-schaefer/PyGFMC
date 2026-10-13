"""The shift ``lam``: a keyword the discrete methods require and ``cont_time`` ignores.

``require_lam`` is the single validation point; the drivers, ``statistics``,
``evolve_walkers`` and ``estimate_average_log_branch_weight`` all go through
it.  A continuous-time run must give the same bits whatever ``lam`` is, and a
continuous-time file (``lam`` stored as NaN) must resume through
``run_from_HDF5``.
"""
import numpy as np
import pytest
import jax

from pygfmc import run, run_gfmc, statistics, write_to_HDF5, run_from_HDF5
from pygfmc.engine.estimators import estimate_average_log_branch_weight
from pygfmc.engine.evolution import evolve_walkers, require_lam
from pygfmc.engine.io import _read_from_HDF5
from pygfmc.hamiltonian import SpinHalf, chain
from pygfmc.walkers import prepare_walkers_sz
from pygfmc.wavefunction import JastrowSpinHalf

_H = SpinHalf.create(chain(8).edges, J_x=-1.0, J_y=-1.0, J_z=1.0)
_G = JastrowSpinHalf(_H.N, _H.edges, beta=0.3)


def _walkers():
    return prepare_walkers_sz(64, 8, jax.random.PRNGKey(0), sz=0)


def test_require_lam():
    """cont_time returns None whatever is passed; the discrete methods need a float."""
    assert require_lam("cont_time", None) is None
    assert require_lam("cont_time", 5.0) is None
    assert require_lam("multi_step", 4) == 4.0 and isinstance(require_lam("single_step", 4.5), float)
    for method in ("multi_step", "single_step"):
        with pytest.raises(ValueError, match="needs the GFMC shift"):
            require_lam(method, None)
    with pytest.raises(ValueError, match="Unknown method"):
        require_lam("leapfrog", 1.0)


@pytest.mark.parametrize("entry", ["run", "run_gfmc", "statistics", "evolve_walkers",
                                   "estimate_average_log_branch_weight"])
@pytest.mark.parametrize("method", ["multi_step", "single_step"])
def test_discrete_methods_without_lam_raise_before_any_work(entry, method):
    """Every entry point refuses a discrete method without ``lam`` with the same message."""
    w = _walkers()
    key = jax.random.PRNGKey(1)
    with pytest.raises(ValueError, match="needs the GFMC shift"):
        if entry == "run":
            run(w, 10, _H, 2, _G, [2], method=method, rng_key=key, verbosity=0)
        elif entry == "run_gfmc":
            run_gfmc(w, 10, _H, 2, _G, [2], method=method, rng_key=key, verbosity=0)
        elif entry == "statistics":
            statistics(w, _H, 2, _G, method, key, 1.0, M=2, quiet=True)
        elif entry == "evolve_walkers":
            evolve_walkers(w, _H, 2, _G, method, key, 1.0)
        else:
            estimate_average_log_branch_weight(w, _H, _G, method=method)


def test_cont_time_ignores_lam_bit_for_bit():
    """lam=None, lam=5.0 and lam=-3.0 give identical continuous-time runs; stats record what was passed."""
    outs = {}
    for lam in (None, 5.0, -3.0):
        outs[lam] = run_gfmc(_walkers(), 30, _H, 0.2, _G, [2, 4], lam=lam,
                             rng_key=jax.random.PRNGKey(2), warmup_steps=8, verbosity=0)
    ref = outs[None]
    for lam, out in outs.items():
        assert np.array_equal(np.asarray(out[0].config), np.asarray(ref[0].config)), lam
        assert np.array_equal(np.asarray(out[4]["energy"]), np.asarray(ref[4]["energy"])), lam
        assert np.array_equal(np.asarray(out[4]["log_weights"]),
                              np.asarray(ref[4]["log_weights"])), lam
        stored = float(out[3]["parameters/lam"])
        assert np.isnan(stored) if lam is None else stored == lam


def test_discrete_lam_is_recorded_and_changes_the_chain():
    """For a discrete method lam is physics: it is stored and a different lam gives a different chain."""
    a = run_gfmc(_walkers(), 30, _H, 2, _G, [2], method="multi_step", lam=4.0,
                 rng_key=jax.random.PRNGKey(3), warmup_steps=8, verbosity=0)
    b = run_gfmc(_walkers(), 30, _H, 2, _G, [2], method="multi_step", lam=6.0,
                 rng_key=jax.random.PRNGKey(3), warmup_steps=8, verbosity=0)
    assert float(a[3]["parameters/lam"]) == 4.0 and float(b[3]["parameters/lam"]) == 6.0
    assert not np.array_equal(np.asarray(a[4]["energy"]), np.asarray(b[4]["energy"]))


def test_cont_time_file_resumes_through_run_from_HDF5(tmp_path):
    """A continuous-time chain stores lam as NaN and continues bit-identically."""
    path = str(tmp_path / "ct.h5")
    zz = SpinHalf.create(chain(8).edges, J_z=1.0)
    ops = {"zz": zz}
    full = run_gfmc(_walkers(), 60, _H, 0.2, _G, [2, 4], rng_key=jax.random.PRNGKey(4),
                    diag_operators=ops, warmup_steps=8, verbosity=0)
    first = run_gfmc(_walkers(), 30, _H, 0.2, _G, [2, 4], rng_key=jax.random.PRNGKey(4),
                     diag_operators=ops, warmup_steps=8, verbosity=0)
    write_to_HDF5(path, first[3], first[0], first[1], first[2], first[4], verbose=False)
    assert np.isnan(float(_read_from_HDF5(path)["parameters"]["lam"]))

    second = run_from_HDF5(path, _H, _G, ops, n_steps=30, verbosity=0)
    write_to_HDF5(path, second[3], second[0], second[1], second[2], second[4],
                  verbose=False)
    joined = np.concatenate([np.asarray(first[4]["energy"]), np.asarray(second[4]["energy"])])
    assert np.array_equal(joined, np.asarray(full[4]["energy"]))
    assert np.array_equal(np.asarray(second[0].config), np.asarray(full[0].config))
    assert str(_read_from_HDF5(path, run=1)["parameters"]["method"]) == "cont_time"


def test_lam_stays_a_keyword():
    """``lam`` comes after ``rng_key``: a ``lam`` placed positionally right after the
    Hamiltonian lands in the ``step_size`` slot, and the call is rejected."""
    with pytest.raises(TypeError, match="got multiple values for argument 'step_size'"):
        run_gfmc(_walkers(), 30, _H, 4.0, step_size=2, guide=_G, p_array=[2],
                 method="multi_step", rng_key=jax.random.PRNGKey(0), verbosity=0)
