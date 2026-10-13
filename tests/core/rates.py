"""The operator matrix rebuilt from ``get_rates`` alone.

``rebuild_matrix`` assembles ``H[y, x]`` column by column from the rate tensor
over a full basis, in the index convention of ``reference.py``, so a test can
compare it with the independent Kronecker construction or assert a property
(Hermiticity, signs) of what the propagators actually sample.  On the way it
checks that ``generate_state(config, k)`` equals the successor tensor at every
configuration and slot, and that no slot of nonzero rate leaves the basis.
"""
import itertools

import jax
import jax.numpy as jnp
import numpy as np

from core.reference import reference_basis_index


def all_configs(values, N):
    """Every configuration of N sites drawn from ``values``, ``(len(values)**N, N)``."""
    return np.array(list(itertools.product(values, repeat=N)), dtype=np.float64)


def rebuild_matrix(h, configs, dtype=jnp.float32, name=None):
    """Dense ``H`` (float64) assembled from ``h.get_rates`` over ``configs``."""
    name = name or type(h).__name__
    dim = len(configs)
    cfg_jax = jnp.asarray(configs, dtype=dtype)
    indices = reference_basis_index(h, configs)

    rates, new_cfgs = jax.vmap(lambda c: h.get_rates(c, return_configs=True))(cfg_jax)
    rates = np.asarray(rates, dtype=np.float64)
    new_cfgs = np.asarray(new_cfgs, dtype=np.float64)
    assert rates.shape == (dim, h.M_total), rates.shape

    for k in range(h.M_total):
        gen = np.asarray(jax.vmap(lambda c, k=k: h.generate_state(c, k))(cfg_jax),
                         dtype=np.float64)
        bad = np.flatnonzero(np.any(gen != new_cfgs[:, k, :], axis=1))
        assert bad.size == 0, (
            f"{name}: generate_state disagrees with get_rates at slot k={k} for "
            f"{bad.size}/{dim} configurations, e.g. {configs[bad[0]]}")

    succ_idx = reference_basis_index(h, new_cfgs)            # (dim, M_total)
    # Forbidden moves (S+ at m=+S, a boson past n_max) leave the basis; that is
    # safe only if such a slot carries zero rate, so check it.
    off = np.zeros_like(succ_idx, dtype=bool)
    off[:, 1:] = True
    in_range = (succ_idx >= 0) & (succ_idx < dim)
    leaks = off & ~in_range & (np.abs(rates) > 0)
    if leaks.any():
        x, k = np.argwhere(leaks)[0]
        raise AssertionError(
            f"{name}: slot k={k} leaves the basis for config {configs[x]} "
            f"yet carries rate {rates[x, k]!r}")

    H = np.zeros((dim, dim))
    np.add.at(H, (indices, indices), rates[:, 0])
    use = off & in_range
    cols = np.broadcast_to(indices[:, None], succ_idx.shape)[use]
    np.add.at(H, (succ_idx[use], cols), rates[use])
    return H
