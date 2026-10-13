"""Walker population round-trips through HDF5, and a resumed run is exact.

A run stopped halfway, checkpointed, reloaded and continued must reproduce the
configurations of an uninterrupted run exactly, not just statistically.
"""
import jax
import numpy as np
import jax.numpy as jnp

from pygfmc.hamiltonian import SpinHalf, chain
from pygfmc.walkers import Walker
from pygfmc.engine.evolution import evolve_walkers
from pygfmc.walkers import reconfiguration
from pygfmc.engine.estimators import estimate_average_log_branch_weight
from core.walkers import balanced_walkers

_N_TOTAL = 300
_N_HALF = _N_TOTAL // 2
_DTYPE = jnp.float32


def _production_loop(walkers, h, lam, n_bra, avg_log_weight, n_steps, rng_key):
    def step_fn(carry, _):
        w, key = carry
        key, k1, k2 = jax.random.split(key, 3)
        w = evolve_walkers(w, h, n_bra, None, "single_step", k1, avg_log_weight, lam=lam)
        w, _, _, _ = reconfiguration(w, k2)
        return (w, key), None
    (walkers, rng_key), _ = jax.lax.scan(step_fn, (walkers, rng_key), None, length=n_steps)
    return walkers, rng_key


def test_walker_save_load_resume(tmp_path):
    """Checkpointed walkers + RNG key reload bit-exactly and the resumed run matches."""
    lattice = chain(16, periodic=True)
    h   = SpinHalf.create(lattice.edges, -1.0, -1/3, 0.0, h_z=0.0, h_x=0.7, dtype=_DTYPE)
    lam = 0.1

    rng = jax.random.PRNGKey(7)
    rng, init_key, warmup_key, prod_key = jax.random.split(rng, 4)
    walkers_init = balanced_walkers(256, lattice.N, init_key, dtype=_DTYPE)
    avg_log_weight = estimate_average_log_branch_weight(walkers_init, h, None, method="single_step", lam=lam)

    walkers_full, _ = _production_loop(walkers_init, h, lam, 10, avg_log_weight, _N_TOTAL, prod_key)
    walkers_mid, rng_mid = _production_loop(walkers_init, h, lam, 10, avg_log_weight, _N_HALF, prod_key)

    path = str(tmp_path / "checkpoint.h5")
    walkers_mid.save(path, rng_key=rng_mid)
    walkers_loaded, rng_loaded, _ = Walker.load(path, dtype=_DTYPE)

    assert jnp.array_equal(walkers_mid.config,     walkers_loaded.config)
    assert jnp.array_equal(walkers_mid.log_weight, walkers_loaded.log_weight)
    assert jnp.array_equal(walkers_mid.e_loc,      walkers_loaded.e_loc)
    assert jnp.array_equal(rng_mid, rng_loaded)

    walkers_resumed, _ = _production_loop(walkers_loaded, h, lam, 10, avg_log_weight, _N_HALF, rng_loaded)
    assert jnp.array_equal(walkers_full.config, walkers_resumed.config)


def test_systematic_resampling_never_selects_a_zero_weight_walker():
    """The parent lookup takes ``cdf[k-1] <= u < cdf[k]``: a zero-weight walker
    is never chosen, at ``psi == 0`` for a leading zero, for a trailing zero,
    and when rounding leaves the cdf short of the largest ``u``."""
    from pygfmc.walkers.reconfiguration import systematic_parents
    f32 = jnp.float32
    assert systematic_parents(jnp.asarray([0.0, 1.0], f32), f32(0.0)).tolist() == [1, 1]
    assert systematic_parents(jnp.asarray([0.0, 0.5, 0.5, 0.0], f32), f32(0.0)).tolist() == [1, 1, 2, 2]
    short = jnp.asarray([0.25, 0.25, 0.4999999, 0.0], f32)         # cdf[-1] < 1
    parents = systematic_parents(short, f32(0.9999999))
    assert int(parents[-1]) == 2 and not np.any(np.asarray(short)[np.asarray(parents)] == 0)
    # through reconfiguration: a -inf log weight is a dead walker
    w = Walker(config=jnp.asarray([[1.0], [-1.0], [1.0]]),
               log_weight=jnp.asarray([-jnp.inf, 0.0, -jnp.inf]), e_loc=jnp.zeros(3))
    for seed in range(20):
        _, _, _, idx = reconfiguration(w, jax.random.PRNGKey(seed))
        assert idx.tolist() == [1, 1, 1]


def test_typed_prng_keys_are_checkpointed_as_key_data(tmp_path):
    """``Walker.save`` and the drivers accept ``jax.random.key`` keys and
    store their raw key data; the stored key loads back as a legacy key."""
    from pygfmc import run_gfmc
    from pygfmc.walkers.walker import key_data
    key = jax.random.key(0)
    legacy = jax.random.PRNGKey(0)
    np.testing.assert_array_equal(key_data(key), key_data(legacy))
    np.testing.assert_array_equal(key_data(legacy), np.asarray(legacy, dtype=np.uint32))
    w = balanced_walkers(8, 4, jax.random.PRNGKey(1))
    w.save(str(tmp_path / "typed.h5"), rng_key=key)
    loaded, stored_key, _ = Walker.load(str(tmp_path / "typed.h5"))
    np.testing.assert_array_equal(np.asarray(stored_key), key_data(key))
    H = SpinHalf.create(chain(4).edges, J_x=-1.0, J_y=-1.0, J_z=1.0)
    _, _, _, st, _ = run_gfmc(w, 12, H, 0.2, None, [2], rng_key=key, verbosity=0)
    assert st["parameters/rng_key"].dtype == np.uint32
