"""run_gfmc_pmap should agree with run_gfmc on the same TFIM problem.

On a single device the pmap path still exercises every branch (sharding,
all_gather, global reconfiguration, off-diagonal forward walking) but not real
parallelism; tolerances allow for the different RNG paths.  The genuinely
multi-device checks are in test_gfmc_multidevice.py.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from pygfmc.hamiltonian import SpinHalf
from pygfmc import run, run_gfmc
from pygfmc.engine.estimators import dense_and_names
from pygfmc.engine.multi_gpu import run_gfmc_pmap
from pygfmc.wavefunction import JastrowSpinHalf
from core.walkers import neel_walkers

L         = 20
J_Z       = 1.0
HX        = J_Z / 2
BETA      = 0.2
N_BRA     = 10
N_WALKERS = 128
WARMUP    = 300
N_STEPS   = 400
P_ARRAY   = [4, 8]
DTYPE     = jnp.float32
EDGES     = [(i, (i + 1) % L) for i in range(L)]
LAM       = J_Z * len(EDGES) / 4

ATOL_E   = 0.02   # per-site energy tolerance
ATOL_OBS = 0.05   # per-site observable tolerance


def _h():
    return SpinHalf.create(EDGES, J_x=0.0, J_y=0.0, J_z=J_Z, h_x=HX, dtype=DTYPE)


def _h_sx():
    return SpinHalf.create(EDGES, J_x=0.0, J_y=0.0, J_z=0.0, h_x=-1.0,
                           sign_check=False, dtype=DTYPE)


def _run_sx(runner, **kwargs):
    """TFIM + Jastrow guide measuring Sum_i S^x with ``runner`` (5-tuple)."""
    h = _h()
    return runner(
        neel_walkers(N_WALKERS, L, dtype=DTYPE), N_STEPS, h, N_BRA, lam=LAM,
        guide=JastrowSpinHalf(h.N, h.edges, beta=BETA), p_array=P_ARRAY,
        method='single_step', rng_key=jax.random.PRNGKey(0),
        warmup_steps=WARMUP, off_diag_operators={'H_sx': _h_sx()},
        number_of_outputs=1, verbosity=0, **kwargs)


@pytest.fixture(scope="module")
def pmap_ref():
    """The pmap run of the problem: (energy, obs, raw).  The tests only read it."""
    _, res_p, obs_p, _, raw_p = _run_sx(run_gfmc_pmap)
    return res_p, obs_p, raw_p


@pytest.fixture(scope="module")
def both(pmap_ref):
    """(serial energy, serial obs, pmap energy, pmap obs) for the same problem."""
    _, res_s, obs_s, _, _ = _run_sx(run_gfmc)
    res_p, obs_p, _ = pmap_ref
    return res_s, obs_s, res_p, obs_p


def test_pmap_energy_matches_serial(both):
    """Windowed energies agree between the serial and pmap backends."""
    res_s, _, res_p, _ = both
    for p in P_ARRAY:
        assert np.isclose(res_s[p] / L, res_p[p] / L, atol=ATOL_E), (
            f'energy mismatch at p={p}: serial={res_s[p]/L}, pmap={res_p[p]/L}')


def test_pmap_off_diagonal_matches_serial(both):
    """Off-diagonal forward-walking estimator: same formula in both paths."""
    _, obs_s, _, obs_p = both
    for p in P_ARRAY:
        assert np.isclose(obs_s['H_sx'][p] / L, obs_p['H_sx'][p] / L,
                          atol=ATOL_OBS), (
            f'<Sx>/L mismatch at p={p}: '
            f'serial={obs_s["H_sx"][p]/L}, pmap={obs_p["H_sx"][p]/L}')


def test_run_dispatcher_matches_run_gfmc():
    """`run` on a single device reproduces `run_gfmc` exactly (same path, same RNG)."""
    kwargs = dict(guide=None, p_array=P_ARRAY, method='single_step',
                  warmup_steps=50, number_of_outputs=1, verbosity=0)
    _, res_direct, _, _, raw_direct = run_gfmc(
        neel_walkers(N_WALKERS, L, dtype=DTYPE), 100, _h(), N_BRA, lam=LAM,
        rng_key=jax.random.PRNGKey(0), **kwargs)
    out = run(neel_walkers(N_WALKERS, L, dtype=DTYPE), 100, _h(), N_BRA, lam=LAM,
              rng_key=jax.random.PRNGKey(0), **kwargs)
    assert len(out) == 5, "run() must return the 5-tuple (return_raw default)"
    _, res_disp, _, _, raw_disp = out

    if jax.device_count() == 1:
        for p in P_ARRAY:
            assert res_direct[p] == res_disp[p]
        assert np.array_equal(raw_direct['energy'], raw_disp['energy'])
        assert np.array_equal(raw_direct['log_weights'], raw_disp['log_weights'])
    else:
        for p in P_ARRAY:
            assert np.isclose(res_direct[p] / L, res_disp[p] / L, atol=ATOL_E)


def test_pmap_rejects_diag_operators():
    """diag_operators is not supported on the pmap path and must raise."""
    h_ising = SpinHalf.create(EDGES, J_x=0.0, J_y=0.0, J_z=J_Z, dtype=DTYPE)
    with pytest.raises(NotImplementedError, match='diag_operators'):
        run_gfmc_pmap(
            neel_walkers(N_WALKERS, L, dtype=DTYPE), 1, _h(), N_BRA, lam=LAM,
            guide=None, p_array=P_ARRAY, method='single_step',
            rng_key=jax.random.PRNGKey(0),
            diag_operators={'H_ising': h_ising},
            warmup_steps=1, number_of_outputs=1, verbosity=0)


def test_pmap_off_diag_batch_size_is_a_noop(pmap_ref):
    """off_diag_batch_size is bit-identical on the pmap path's separate operator application."""
    res_ref, obs_ref, raw_ref = pmap_ref
    assert set(raw_ref['off_diag']) == {'H_sx'}, "off_diag is keyed by name"
    od, _ = dense_and_names(raw_ref['off_diag'])
    assert np.any(od != 0) and np.all(np.isfinite(od))   # non-vacuous

    # the batch size divides the PER-DEVICE shard, not the global count
    shard = N_WALKERS // jax.device_count()
    for bs in (b for b in (8, 16, 32) if shard % b == 0):
        _, res, obs, _, raw = _run_sx(run_gfmc_pmap, off_diag_batch_size=bs)
        for key in ('off_diag', 'off_diag_norm', 'energy'):
            a, _ = dense_and_names(raw_ref[key])
            b, _ = dense_and_names(raw[key])
            assert a.shape == b.shape, f'{key} shape, batch_size={bs}'
            assert np.array_equal(a, b), (
                f'{key} differs at off_diag_batch_size={bs}: '
                f'max|delta|={np.max(np.abs(a - b)):.3e}')
        for pv in P_ARRAY:
            assert res_ref[pv] == res[pv], f'energy p={pv}, batch_size={bs}'
            assert obs_ref['H_sx'][pv] == obs['H_sx'][pv], (
                f'H_sx p={pv}, batch_size={bs}')


def test_pmap_off_diag_batch_size_must_divide_shard():
    """A batch size that does not divide the PER-DEVICE shard is rejected."""
    with pytest.raises(ValueError, match='must divide'):
        _run_sx(run_gfmc_pmap, off_diag_batch_size=7)   # divides no shard of 128


def test_pmap_return_raw_false_without_off_diag():
    """Without off-diagonal operators pmap accepts return_raw=False, returns the 4-tuple and records rng keys."""
    key = jax.random.PRNGKey(0)
    out = run_gfmc_pmap(
        neel_walkers(N_WALKERS, L, dtype=DTYPE), 20, _h(), N_BRA, lam=LAM,
        guide=None, p_array=P_ARRAY, method='single_step', rng_key=key,
        warmup_steps=8, return_raw=False, n_devices=1,
        number_of_outputs=1, verbosity=0)
    assert len(out) == 4
    stats = out[3]
    assert np.array_equal(stats['parameters/rng_key'], np.asarray(key))
    assert 'parameters/rng_key_next' in stats
    assert int(stats['parameters/off_diag_equal_windows']) == 1
