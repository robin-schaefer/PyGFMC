"""Guide tests for the four documented wavefunctions.

``JastrowSpinHalf``, ``ClusterCPS``, ``ClusterMLP`` and ``ExactGroundState`` are each
checked for initialization, HDF5 round-trip, generic NetKet training (not for
``ExactGroundState``), GFMC energy convergence against ED, and consistency of
the ratio routines with ``log_psi``.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.guides import GUIDE_NAMES, L, build_guide
from pygfmc import run_gfmc
from pygfmc.walkers import prepare_walkers_random
from pygfmc.wavefunction import ClusterCPS, ClusterMLP, ExactGroundState, JastrowSpinHalf

GUIDES = GUIDE_NAMES
TRAINABLE = ["Jastrow", "ClusterCPS", "ClusterMLP"]
HDF5_GROUP = {"Jastrow": "jastrow_spin_half", "ClusterCPS": "cps", "ClusterMLP": "cluster_mlp",
              "ExactGroundState": "exact_ground_state"}
CLASSES = {"Jastrow": JastrowSpinHalf, "ClusterCPS": ClusterCPS, "ClusterMLP": ClusterMLP,
           "ExactGroundState": ExactGroundState}


def _random_cfg(rng, shape=L):
    return jnp.asarray(rng.choice([-1.0, 1.0], size=shape), dtype=jnp.float32)


def _log_psi(guide, cfg):
    return float(np.ravel(guide.log_psi(cfg))[0])


def _ratios(guide, h, cfg):
    """Amplitude ratios in the operator layout via log_diff_from_configs."""
    _, new_configs = h.get_rates(cfg, return_configs=True)
    return np.asarray(guide.log_diff_from_configs(cfg, new_configs, None)).reshape(-1)


def _require_netket():
    """Skip when NetKet is absent or fails on import."""
    from pygfmc.wavefunction._optional import import_netket
    try:
        import_netket()
    except ImportError as e:
        pytest.skip(str(e))


def test_default_sampler_is_refused_when_the_sum_is_not_conserved(h):
    """The default MetropolisExchange conserves S^z_tot; h_x breaks it, so it raises."""
    _require_netket()
    from pygfmc.hamiltonian import SpinHalf
    from pygfmc.wavefunction import train_with_netket

    with pytest.raises(ValueError, match="MetropolisLocal"):
        train_with_netket(h, build_guide("Jastrow", h), n_iter=1)
    anisotropic = SpinHalf.create(np.asarray(h.edges), J_x=-1.0, J_y=-0.5, J_z=1.0)
    with pytest.raises(ValueError, match="changes the sum of the site values"):
        train_with_netket(anisotropic, build_guide("Jastrow", h), n_iter=1)


@pytest.mark.parametrize("name", GUIDES)
def test_initialization(name, h):
    """The guide builds and evaluates amplitudes through whatever routines it provides."""
    guide = build_guide(name, h)
    rng = np.random.default_rng(0)
    cfg = _random_cfg(rng)
    assert any(hasattr(guide, m) for m in
               ("log_psi", "log_diff_from_configs", "log_diff_from_operator"))

    if hasattr(guide, "log_psi"):
        single = np.asarray(guide.log_psi(cfg))
        assert single.shape in ((), (1,)) and np.isfinite(single).all()
        batch = _random_cfg(rng, (5, L))
        out = np.asarray(guide.log_psi(batch)).reshape(-1)
        assert out.shape == (5,)
        np.testing.assert_allclose(out, [_log_psi(guide, b) for b in batch], atol=1e-5)

    ratios = _ratios(guide, h, cfg)
    assert ratios.shape == (h.M_total,) and np.isfinite(ratios).all()


@pytest.mark.parametrize("name", TRAINABLE)
def test_hdf5_roundtrip(name, h, tmp_path):
    """save / load reproduces the amplitudes (ExactGroundState: test_exact_ground_state.py)."""
    guide = build_guide(name, h)
    path = str(tmp_path / f"{name}.h5")
    guide.save(path)
    loaded = CLASSES[name].load(path, group=HDF5_GROUP[name])

    rng = np.random.default_rng(1)
    for _ in range(4):
        cfg = _random_cfg(rng)
        np.testing.assert_allclose(_ratios(loaded, h, cfg), _ratios(guide, h, cfg),
                                   atol=1e-6)
        if hasattr(guide, "log_psi"):
            np.testing.assert_allclose(_log_psi(loaded, cfg), _log_psi(guide, cfg),
                                       atol=1e-6)


@pytest.mark.parametrize("name", TRAINABLE)
def test_train_with_netket_generic(name, h):
    """The generic trainer returns a usable float32 guide of the same class, with moved parameters."""
    _require_netket()
    import netket as nk
    from pygfmc.wavefunction import make_hilbert, train_with_netket

    guide = build_guide(name, h)
    x64_before = jax.config.jax_enable_x64
    # h_x breaks S^z_tot, so the sampler must change single sites
    sampler = nk.sampler.MetropolisLocal(make_hilbert(h.local_hilbert_space, h.N), n_chains=8)
    trained, history = train_with_netket(
        h, guide, sampler=sampler, n_iter=3, n_samples=256, n_discard_per_chain=2,
        seed=0, return_history=True)

    assert type(trained) is type(guide)
    assert len(history) == 3 and np.all(np.isfinite(history))
    assert jax.config.jax_enable_x64 == x64_before, "NetKet leaked jax_enable_x64"
    assert np.isfinite(_log_psi(trained, _random_cfg(np.random.default_rng(2))))

    before = jax.tree_util.tree_leaves(guide)
    after = jax.tree_util.tree_leaves(trained)
    assert len(before) == len(after)
    moved = 0
    for b, a in zip(before, after):
        b, a = np.asarray(b), np.asarray(a)
        assert b.dtype == a.dtype, f"leaf dtype changed {b.dtype} -> {a.dtype}"
        if np.issubdtype(b.dtype, np.floating):
            moved += int(not np.array_equal(b, a))
        else:            # integer/bool leaves are structure (e.g. index arrays)
            assert np.array_equal(b, a), f"non-floating leaf of dtype {b.dtype} modified"
    assert moved > 0, "no floating-point parameter changed, so nothing trained"


@pytest.mark.parametrize("name", GUIDES)
def test_gfmc_energy_convergence(name, h, e0):
    """A short run reproduces E0 within 0.02/site, and the deepest window is worse by at most 0.01/site."""
    guide = build_guide(name, h)
    p_array = [2, 8]
    walkers = prepare_walkers_random(1024, L, jax.random.PRNGKey(0))
    _, energy, _, _, _ = run_gfmc(
        walkers, 400, h, lam=float(L / 4.0 + 3.0), step_size=1, guide=guide,
        p_array=p_array, method="single_step", rng_key=jax.random.PRNGKey(1),
        warmup_steps=200, verbosity=0)
    assert sorted(energy.keys()) == sorted(p_array)
    e = np.array([float(np.ravel(energy[p])[0]) for p in p_array])
    assert np.isfinite(e).all()
    err = np.abs(e - e0) / L
    assert err[-1] < 0.02, f"{name}: E(p={p_array[-1]}) = {e[-1]:.6f} vs E0 = {e0:.6f}"
    assert err[-1] <= err[0] + 0.01, f"{name}: deeper window got worse: {err}"


@pytest.mark.parametrize("name", TRAINABLE)
def test_log_diff_consistency(name, h):
    """log_diff_from_operator, log_diff_from_configs and log_psi agree (ExactGroundState has only one)."""
    guide = build_guide(name, h)
    rng = np.random.default_rng(3)
    pre = guide.build_cache(h) if hasattr(guide, "build_cache") else None
    for _ in range(3):
        cfg = _random_cfg(rng)
        _, new_configs = h.get_rates(cfg, return_configs=True)
        from_op = np.asarray(guide.log_diff_from_operator(cfg, h, pre)).reshape(-1)
        from_cfg = np.asarray(guide.log_diff_from_configs(cfg, new_configs, None)).reshape(-1)
        # slot 0 is the diagonal; some guides leave it untouched
        np.testing.assert_allclose(from_op[1:], from_cfg[1:], atol=1e-4,
                                   err_msg=f"{name}: operator vs configs")
        if hasattr(guide, "log_psi"):
            base = _log_psi(guide, cfg)
            expected = np.array([_log_psi(guide, c) - base for c in new_configs])
            np.testing.assert_allclose(from_cfg[1:], expected[1:], atol=1e-4,
                                       err_msg=f"{name}: configs vs log_psi")
