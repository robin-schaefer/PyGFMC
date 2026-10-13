"""Deterministic walker initializers used as test setup.

``balanced_walkers`` permutes a balanced +-1 configuration per walker and
``neel_walkers`` splits the population between the two Neel states.  Tests
whose expected numbers depend on the initial ensemble build it here, so the
ensemble is fixed by this file rather than by a library default.
"""
import jax
import jax.numpy as jnp

from pygfmc.walkers import Walker


def balanced_walkers(n_walkers, n_sites, rng_key, dtype=jnp.float32):
    """Random permutations of a balanced +-1 base, one key per walker."""
    base = jnp.array([-1] * (n_sites // 2) + [1] * (n_sites - n_sites // 2),
                     dtype=dtype)
    keys = jax.random.split(rng_key, n_walkers)
    configs = jax.vmap(lambda k: jax.random.permutation(k, base))(keys)
    zeros = jnp.zeros(n_walkers, dtype=dtype)
    return Walker(config=configs, log_weight=zeros, e_loc=zeros)


def neel_walkers(n_walkers, n_sites, dtype=jnp.float32):
    """Half the walkers on each Neel state (the first half on the one with site 0 up)."""
    neel_a = jnp.array([1 if i % 2 == 0 else -1 for i in range(n_sites)],
                       dtype=dtype)
    n_a = n_walkers // 2
    configs = jnp.concatenate([jnp.tile(neel_a, (n_a, 1)),
                               jnp.tile(-neel_a, (n_walkers - n_a, 1))], axis=0)
    zeros = jnp.zeros(n_walkers, dtype=dtype)
    return Walker(config=configs, log_weight=zeros, e_loc=zeros)
