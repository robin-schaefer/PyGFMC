"""Walker checkpoints for every configuration encoding, and the load dtype.

``Walker.save`` compacts integer-valued configurations to ``int8`` and keeps
any other encoding as is; ``Walker.load`` casts back and, by default, gives the
weights JAX's default float like the initializers do.
"""
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pygfmc.walkers import Walker, prepare_walkers_sz


@pytest.mark.parametrize("states,sz,stored", [
    ((-1.0, 1.0), 0, np.int8),             # spin-1/2 encoding
    ((-1.0, 0.0, 1.0), 1, np.int8),        # spin 1
    ((0.0, 1.0, 2.0), 8, np.int8),         # bosons
    ((-0.5, 0.5), 1.0, np.float32),        # physical S^z: not integer, kept as given
    ((-1.5, -0.5, 0.5, 1.5), 2.0, np.float32),
])
def test_save_compacts_when_lossless_and_loads_back_exactly(tmp_path, states, sz, stored):
    w = prepare_walkers_sz(32, 8, jax.random.PRNGKey(0), sz=sz, states=states)
    w = Walker(config=w.config, log_weight=jnp.linspace(-1, 1, 32), e_loc=jnp.arange(32.0))
    path = str(tmp_path / "w.h5")
    key = jax.random.PRNGKey(7)
    w.save(path, rng_key=key, avg_log_weight=0.25, note="x")
    with h5py.File(path, "r") as f:
        assert f["walkers/config"].dtype == stored
        assert f["walkers"].attrs["avg_log_weight"] == 0.25
        assert f["walkers"].attrs["note"] == "x"
    loaded, key2, alw = Walker.load(path)
    np.testing.assert_array_equal(np.asarray(loaded.config), np.asarray(w.config))
    np.testing.assert_array_equal(np.asarray(loaded.log_weight), np.asarray(w.log_weight))
    np.testing.assert_array_equal(np.asarray(loaded.e_loc), np.asarray(w.e_loc))
    assert np.array_equal(np.asarray(key2), np.asarray(key))
    assert float(alw) == 0.25


def test_load_without_key_or_alw_returns_none(tmp_path):
    w = prepare_walkers_sz(8, 6, jax.random.PRNGKey(1), sz=0)
    path = str(tmp_path / "w.h5")
    w.save(path)
    _, key, alw = Walker.load(path)
    assert key is None and alw is None


def test_load_dtype_follows_jax_default_float(tmp_path, x64):
    w = prepare_walkers_sz(8, 6, jax.random.PRNGKey(2), sz=0)
    path = str(tmp_path / "w.h5")
    w.save(path, avg_log_weight=1.0)
    loaded, _, alw = Walker.load(path)
    assert loaded.log_weight.dtype == jnp.zeros(()).dtype == loaded.config.dtype
    assert alw.dtype == loaded.log_weight.dtype
    with x64():
        loaded64, _, alw64 = Walker.load(path)
        assert loaded64.log_weight.dtype == jnp.float64 == loaded64.config.dtype
        assert alw64.dtype == jnp.float64
        narrow, _, _ = Walker.load(path, dtype=jnp.float32, config_dtype=jnp.int8)
        assert narrow.log_weight.dtype == jnp.float32 and narrow.config.dtype == jnp.int8
