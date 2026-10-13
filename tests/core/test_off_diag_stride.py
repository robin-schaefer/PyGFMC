"""run_gfmc's off_diag_stride and off_diag_batch_size.

Skipping off-diagonal measurements must not perturb the main chain, and every
measurement that IS taken must equal the stride=1 run's value at the same
step.  Chunking the operator application over walkers must be a total no-op.
Equal-window gating is checked in ``test_silence_and_gating.py``, the
batch-size validation in ``test_validation_messages.py``.
"""
import jax
import jax.numpy as jnp
import numpy as np

from pygfmc import run_gfmc
from pygfmc.walkers import prepare_walkers_from_configs
from pygfmc.hamiltonian import SpinHalf
from pygfmc.engine.estimators import dense_and_names

_L       = 10
_N_STEPS = 60
_P_ARRAY = [4, 8]
_STRIDES = [1, 2, 3, 6]


def _run_tfim(stride, batch_size=None):
    """One deterministic TFIM run measuring Sum_i X_i; returns (E, obs, stats, raw).

    The shift is pinned, so the warm-up only equilibrates and every run with the
    same key walks the same main chain.
    """
    edges = np.asarray([(i, (i + 1) % _L) for i in range(_L)], dtype=np.int32)
    h = SpinHalf.create(edges, J_x=0.0, J_y=0.0, J_z=-4.0, h_x=1.5)
    # measurement-only Sum_i X_i: positive rates, never drives the branching
    xtot = SpinHalf.create(edges, J_x=0.0, J_y=0.0, J_z=0.0, h_x=-2.0,
                           sign_check=False)
    walkers = prepare_walkers_from_configs(jnp.ones((64, _L), dtype=jnp.float32))
    out = run_gfmc(
        walkers, _N_STEPS, h, 1, None, _P_ARRAY,
        method="single_step", lam=15.0, rng_key=jax.random.PRNGKey(3),
        off_diag_operators={"Xtot": xtot},
        avg_log_weight=jnp.float32(1.0), verbosity=0,
        off_diag_stride=stride,
        off_diag_batch_size=batch_size, warmup_steps=100)
    _, results, obs, stats, raw = out
    return results, obs, stats, raw


def test_off_diag_stride_consistency():
    """Strides 1/2/3/6: bit-identical main chain, compact rows equal to stride-1 values."""
    runs = {s: _run_tfim(s) for s in _STRIDES}
    _, _, _, raw1 = runs[1]
    assert set(raw1['off_diag']) == {'Xtot'}, "off_diag must be keyed by name"
    off1, _ = dense_and_names(raw1['off_diag'])   # (n_steps, 1, n_p)
    assert off1.shape[0] == _N_STEPS
    assert not np.any(np.isnan(off1))

    for s in _STRIDES:
        results_s, _, stats_s, raw_s = runs[s]
        assert np.array_equal(raw_s['energy'],      raw1['energy'])
        assert np.array_equal(raw_s['log_weights'], raw1['log_weights'])
        assert results_s == runs[1][0]

        off_s, _ = dense_and_names(raw_s['off_diag'])
        assert off_s.shape == (_N_STEPS // s,) + off1.shape[1:], (
            f"stride={s}: wrong off_diag shape {off_s.shape}")
        assert not np.any(np.isnan(off_s)), f"stride={s}: NaN in compact array"

        # row r is the measurement at step (r+1)*s - 1
        steps = (np.arange(off_s.shape[0]) + 1) * s - 1
        assert np.array_equal(np.asarray(raw_s['off_diag_steps']), steps)
        assert np.array_equal(off_s, off1[steps]), (
            f"stride={s}: measured forward-walk values differ from stride=1")
        assert int(np.ravel(stats_s['parameters/off_diag_stride'])[0]) == s


def test_off_diag_batch_size_is_a_noop():
    """off_diag_batch_size is BIT-IDENTICAL (per-walker keys from absolute indices), so assert ==."""
    ref_E, ref_obs, _, ref_raw = _run_tfim(1)
    od, _ = dense_and_names(ref_raw["off_diag"])
    assert np.any(od != 0) and np.all(np.isfinite(od))   # non-vacuous

    for bs in (8, 32, 64):        # 64 walkers; 64 takes the plain-vmap branch
        E, obs, _, raw = _run_tfim(1, batch_size=bs)
        for key in ("off_diag", "off_diag_norm", "energy", "log_weights"):
            a, _ = dense_and_names(ref_raw[key])
            b, _ = dense_and_names(raw[key])
            assert a.shape == b.shape, f"{key} shape, batch_size={bs}"
            assert np.array_equal(a, b), (
                f"{key} differs at off_diag_batch_size={bs}: "
                f"max|delta|={np.max(np.abs(a - b)):.3e}")
        assert E == ref_E, f"energy differs at batch_size={bs}"
        for name in ref_obs:
            for pv in ref_obs[name]:
                assert ref_obs[name][pv] == obs[name][pv], (
                    f"{name} p={pv}, batch_size={bs}")
