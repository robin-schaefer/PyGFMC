"""Helpers shared by the Hamiltonian classes: edge-list validation and the
scalar two-site configuration update used by ``generate_state``."""
import numpy as np
import jax.numpy as jnp


def edges_array(edges):
    """Validate an edge list.

    Returns ``(edges_arr, n_edges, N_min)``: the ``(n_edges, 2)`` int64 array
    (empty for no edges), the edge count, and the smallest site count the
    edges imply (``max index + 1``, or 0 without edges). Each edge must join
    two distinct sites with nonnegative integer indices. Self-edges, malformed
    shapes and fractional indices raise ``ValueError``. ``[]`` and arrays of
    shape ``(0, 2)`` are accepted.
    """
    raw = np.asarray(edges)
    if raw.shape == (0,):
        return np.zeros((0, 2), dtype=np.int64), 0, 0
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("edges must have shape (n_edges, 2).")
    if (raw.dtype.kind not in "iuf"
            or not np.all(np.isfinite(raw))
            or np.any(raw < 0)
            or np.any(raw >= 2**63)
            or np.any(raw != np.floor(raw))):
        raise ValueError("edge indices must be nonnegative integers fitting in int64.")
    edges_arr = raw.astype(np.int64)
    if np.any(edges_arr[:, 0] == edges_arr[:, 1]):
        raise ValueError("self-edges are not supported; edge endpoints must differ.")
    return edges_arr, len(edges_arr), int(edges_arr.max()) + 1 if edges_arr.size else 0


def two_site_move(state_idx, edge_i, edge_j, N, edge_blocks, site_blocks, dtype):
    """The (at most two) sites changed by transition ``state_idx`` and their
    increments, as ``(sites, deltas)`` of shape ``(2,)`` each.

    The off-diagonal index ``state_idx - 1`` is laid out in consecutive blocks.
    An edge block ``(base, d_i, d_j)`` covers ``[base, base + 2 n_edges)``:
    its first half adds ``(d_i, d_j)`` to the sites of edge ``k``, its second
    half the negated increments (the reverse transition).  A site block
    ``(base, delta)`` covers ``[base, base + N)`` and adds ``delta`` to site
    ``k`` (the second slot then repeats the site with increment 0).  The
    diagonal ``state_idx == 0`` gives zero increments.
    """
    off_idx = state_idx - 1
    n_e = edge_i.shape[0]
    idx_dtype = edge_i.dtype if n_e > 0 else jnp.int32
    site_i = jnp.zeros((), dtype=idx_dtype)
    site_j = jnp.zeros((), dtype=idx_dtype)
    d_i = jnp.zeros((), dtype=dtype)
    d_j = jnp.zeros((), dtype=dtype)

    def pick(cond, i, di, j, dj):
        nonlocal site_i, site_j, d_i, d_j
        site_i = jnp.where(cond, i, site_i)
        site_j = jnp.where(cond, j, site_j)
        d_i = jnp.where(cond, jnp.asarray(di, dtype), d_i)
        d_j = jnp.where(cond, jnp.asarray(dj, dtype), d_j)

    for base, di_fwd, dj_fwd in edge_blocks:
        # forward transitions [base, base+n_e), backward [base+n_e, base+2n_e)
        e_fwd = jnp.clip(off_idx - base,       0, max(n_e - 1, 0))
        e_bwd = jnp.clip(off_idx - base - n_e, 0, max(n_e - 1, 0))
        pick((off_idx >= base) & (off_idx < base + n_e),
             edge_i[e_fwd], di_fwd, edge_j[e_fwd], dj_fwd)
        pick((off_idx >= base + n_e) & (off_idx < base + 2 * n_e),
             edge_i[e_bwd], -di_fwd, edge_j[e_bwd], -dj_fwd)
    for base, delta in site_blocks:
        site = jnp.clip(off_idx - base, 0, N - 1)
        pick((off_idx >= base) & (off_idx < base + N), site, delta, site, 0)
    return jnp.stack([site_i, site_j]), jnp.stack([d_i, d_j])


def two_site_update(config, state_idx, edge_i, edge_j, N, edge_blocks, site_blocks):
    """Apply transition ``state_idx`` when every transition changes at most two
    sites (the layout of :func:`two_site_move`).  The target sites and
    increments are selected as scalars and the configuration is updated once,
    instead of building one candidate configuration per block.
    ``state_idx == 0`` returns ``config`` unchanged.
    """
    sites, deltas = two_site_move(state_idx, edge_i, edge_j, N, edge_blocks,
                                  site_blocks, config.dtype)
    out = config.at[sites[0]].add(deltas[0]).at[sites[1]].add(deltas[1])
    return jnp.where(state_idx == 0, config, out)
