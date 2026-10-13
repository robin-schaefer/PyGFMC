"""``verbosity=0`` prints nothing, and the explicit ``off_diag_equal_windows``.

The positivity report, the p_array deduplication notice and the end-of-run
statistics report are all gated on ``verbosity``; ``write_to_HDF5`` takes a
``verbose`` flag.  Equal-window gating defaults to ``not return_raw`` and can
be requested explicitly together with the raw series.  The step-count
checks of a gated run are in ``test_validation_messages.py``.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from pygfmc import run_gfmc, write_to_HDF5, check_run_compatible
from pygfmc.engine.estimators import dense_and_names
from pygfmc.hamiltonian import SpinHalf, chain
from pygfmc.walkers import prepare_walkers_from_configs

_L = 8
_EDGES = chain(_L).edges
_H = SpinHalf.create(_EDGES, J_z=-4.0, h_x=1.5)
_XTOT = SpinHalf.create(_EDGES, h_x=-2.0, sign_check=False)
_ZZ = SpinHalf.create(_EDGES, J_z=1.0)


def _walkers(n=64):
    return prepare_walkers_from_configs(jnp.ones((n, _L), dtype=jnp.float32))


def test_verbosity_zero_is_silent_even_with_duplicates_and_operators(capsys, tmp_path):
    out = run_gfmc(_walkers(), 30, _H, 1, None, [2, 2, 4], method="single_step", lam=15.0,
                   rng_key=jax.random.PRNGKey(0), warmup_steps=8,
                   diag_operators={"zz": _ZZ}, off_diag_operators={"x": _XTOT},
                   verbosity=0)
    assert sorted(out[1]) == [2, 4], "duplicates are dropped silently"
    write_to_HDF5(str(tmp_path / "a.h5"), out[3], out[0], out[1], out[2], out[4],
                  verbose=False)
    assert capsys.readouterr().out == ""


def test_verbosity_one_reports_positivity_and_statistics(capsys):
    run_gfmc(_walkers(), 30, _H, 1, None, [2, 2], method="single_step", lam=15.0,
             rng_key=jax.random.PRNGKey(0), warmup_steps=8,
             off_diag_operators={"x": _XTOT}, verbosity=1)
    text = capsys.readouterr().out
    assert "passed positivity check" in text
    assert "dedup'd p_array" in text
    assert "[statistics]" in text
    assert "#########" not in text.split("[run_gfmc]")[0], "no banner before the run"


def test_write_to_HDF5_prints_one_line_by_default(capsys, tmp_path):
    out = run_gfmc(_walkers(), 20, _H, 1, None, [2], method="single_step", lam=15.0,
                   rng_key=jax.random.PRNGKey(0), warmup_steps=4, verbosity=0)
    path = str(tmp_path / "a.h5")
    write_to_HDF5(path, out[3], out[0], out[1], out[2], out[4])
    assert capsys.readouterr().out.strip() == f"[write_to_HDF5] wrote {path}:/run_0"


@pytest.mark.parametrize("stride", [1, 2])
def test_explicit_equal_windows_with_raw_returned(stride):
    """off_diag_equal_windows=True keeps raw and gates the rows to the band every p can use."""
    p_array, n_steps = [4, 8], 60
    kw = dict(method="single_step", lam=15.0, rng_key=jax.random.PRNGKey(3), warmup_steps=8,
              avg_log_weight=jnp.float32(1.0), off_diag_operators={"x": _XTOT},
              off_diag_stride=stride, verbosity=0)
    plain = run_gfmc(_walkers(), n_steps, _H, 1, None, p_array, **kw)
    gated = run_gfmc(_walkers(), n_steps, _H, 1, None, p_array,
                     off_diag_equal_windows=True, **kw)
    assert len(gated) == 5, "raw is still returned"
    assert int(gated[3]["parameters/off_diag_equal_windows"]) == 1
    assert int(plain[3]["parameters/off_diag_equal_windows"]) == 0
    steps = np.asarray(gated[4]["off_diag_steps"])
    p_max, m_max = max(p_array), max(p_array) // 2
    band = [k for k in range(p_max, n_steps - m_max) if (k + 1) % stride == 0]
    assert steps.tolist() == band
    # the main chain is untouched and the kept rows equal the ungated ones
    for key in ("energy", "log_weights"):
        assert np.array_equal(np.asarray(plain[4][key]), np.asarray(gated[4][key]))
    plain_steps = np.asarray(plain[4]["off_diag_steps"]).tolist()
    od_plain, _ = dense_and_names(plain[4]["off_diag"])
    od_gated, _ = dense_and_names(gated[4]["off_diag"])
    assert np.array_equal(od_gated, od_plain[[plain_steps.index(k) for k in band]])
    # the largest window never used the dropped rows, so its estimate is unchanged
    assert float(gated[2]["x"][p_max]) == float(plain[2]["x"][p_max])


def test_equal_windows_default_follows_return_raw():
    """The default is ``not return_raw``."""
    kw = dict(method="single_step", lam=15.0, rng_key=jax.random.PRNGKey(4), warmup_steps=4,
              off_diag_operators={"x": _XTOT}, verbosity=0)
    with_raw = run_gfmc(_walkers(), 40, _H, 1, None, [4, 8], **kw)
    without = run_gfmc(_walkers(), 40, _H, 1, None, [4, 8], return_raw=False, **kw)
    assert int(with_raw[3]["parameters/off_diag_equal_windows"]) == 0
    assert int(without[3]["parameters/off_diag_equal_windows"]) == 1


def test_equal_windows_is_a_must_match_parameter(tmp_path):
    """Flipping the gating on an existing file is refused."""
    path = str(tmp_path / "a.h5")
    kw = dict(method="single_step", lam=15.0, rng_key=jax.random.PRNGKey(5), warmup_steps=4,
              off_diag_operators={"x": _XTOT}, verbosity=0)
    out = run_gfmc(_walkers(), 40, _H, 1, None, [4, 8], **kw)
    write_to_HDF5(path, out[3], out[0], out[1], out[2], out[4], verbose=False)
    check_run_compatible(path, off_diag_equal_windows=0)
    with pytest.raises(ValueError, match="off_diag_equal_windows"):
        check_run_compatible(path, off_diag_equal_windows=1)
