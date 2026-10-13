"""Walker ensembles: each initializer must produce what it promises.

For a charge-conserving Hamiltonian no move can leave the starting sector, so
a wrong-sector ensemble converges smoothly to the wrong energy; the
initializer is the only place that choice is made.
"""
import numpy as np
import pytest
import jax

import jax.numpy as jnp

from pygfmc.walkers import (prepare_walkers_from_configs, prepare_walkers_random,
                           prepare_walkers_sz)

_LADDERS = [(-1, 1), (-0.5, 0.5), (-1, 0, 1), (0, 1, 2)]


def test_prepare_walkers_random_is_iid():
    """Every site an independent coin flip: +-1 values, magnetization varies."""
    N_walkers, N_sites = 256, 12
    w = prepare_walkers_random(N_walkers, N_sites, jax.random.PRNGKey(3))
    cfg = np.asarray(w.config)
    assert cfg.shape == (N_walkers, N_sites)
    assert set(np.unique(cfg)) == {-1.0, 1.0}
    assert np.std(cfg.sum(axis=1)) > 0.5
    assert np.all(np.asarray(w.log_weight) == 0.0)


@pytest.mark.parametrize("states", _LADDERS)
def test_prepare_walkers_random_draws_only_from_states(states):
    """Every entry is one of ``states`` -- and all of them do occur."""
    w = prepare_walkers_random(512, 10, jax.random.PRNGKey(7), states=states)
    cfg = np.asarray(w.config)
    assert cfg.shape == (512, 10)
    seen = set(np.unique(cfg).tolist())
    assert seen <= set(float(v) for v in states), f"foreign value in {seen}"
    assert seen == set(float(v) for v in states), f"unused states: {seen}"


@pytest.mark.parametrize("states,sz", [((-1, 1), 4), ((-1, 1), 0),
                                        ((-0.5, 0.5), 2.0), ((-0.5, 0.5), -3.0),
                                        ((-1, 0, 1), 3), ((-1, 0, 1), -7),
                                        ((0, 1, 2), 10), ((0, 1, 2), 0)])
def test_prepare_walkers_sz_sums_exactly(states, sz):
    """sz is the SUM over sites and must be exact for EVERY walker (it pins a sector)."""
    N_sites = 10
    w = prepare_walkers_sz(128, N_sites, jax.random.PRNGKey(11), sz=sz,
                           states=states)
    cfg = np.asarray(w.config)
    assert cfg.shape == (128, N_sites)
    assert set(np.unique(cfg).tolist()) <= set(float(v) for v in states)
    sums = cfg.sum(axis=1)
    assert np.allclose(sums, sz), f"sums range [{sums.min()}, {sums.max()}], want {sz}"


@pytest.mark.parametrize("states,sz", [((-1, 1), 3),          # parity: N even, sz odd
                                        ((-1, 1), 99),         # above the range
                                        ((0, 1, 2), -1),       # below the range
                                        ((-0.5, 0.5), 0.25)])  # off the 1/2 ladder
def test_prepare_walkers_sz_rejects_unreachable(states, sz):
    """An impossible request must raise, not return a wrong-sum ensemble."""
    with pytest.raises(ValueError):
        prepare_walkers_sz(4, 10, jax.random.PRNGKey(13), sz=sz, states=states)


@pytest.mark.parametrize("init", ["random", "sz"])
def test_prepare_walkers_dtype_follows_states(init):
    """The config dtype comes from ``states``; weights are always floating point."""
    for states, kind in (((-1, 0, 1), "i"), ((-0.5, 0.5), "f")):
        if init == "random":
            w = prepare_walkers_random(16, 8, jax.random.PRNGKey(17), states=states)
        else:
            sz = 2 if kind == "i" else 1.0
            w = prepare_walkers_sz(16, 8, jax.random.PRNGKey(17), sz=sz, states=states)
        assert np.dtype(w.config.dtype).kind == kind, (
            f"states={states} gave config dtype {w.config.dtype}")
        # width follows jax_enable_x64, so assert the kind only
        assert np.dtype(w.log_weight.dtype).kind == "f"
        assert np.dtype(w.e_loc.dtype).kind == "f"
        assert set(np.unique(np.asarray(w.config)).tolist()) <= set(states)


def test_prepare_walkers_from_configs_keeps_the_configurations():
    """The given (N_walkers, N_sites) array is used as is, dtype included."""
    cfg = np.random.default_rng(0).choice([-1, 1], size=(64, 10)).astype(np.int8)
    w = prepare_walkers_from_configs(cfg)
    assert w.config.dtype == jnp.int8
    np.testing.assert_array_equal(np.asarray(w.config), cfg)
    assert w.log_weight.shape == w.e_loc.shape == (64,)
    assert np.all(np.asarray(w.log_weight) == 0.0)
    assert np.all(np.asarray(w.e_loc) == 0.0)


@pytest.mark.parametrize("shape", [(10,), (4, 5, 2)])
def test_prepare_walkers_from_configs_rejects_non_2d(shape):
    """A single (N_sites,) configuration or a 3-D array raises."""
    with pytest.raises(ValueError, match="N_walkers, N_sites"):
        prepare_walkers_from_configs(np.ones(shape))
