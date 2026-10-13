"""Shared by the Jastrow guides: HDF5 storage of the optional,
scalar-or-per-element parameters (``beta``, ``h``, ``d``), and the per-site
neighbour tables of the incremental mode.

Each parameter is None (absent), a scalar tied across all pairs or sites, or
an array.  The datasets hold the values of the present parameters; the mode
of every parameter is an attribute ``<name>_mode`` in ``none``/``tied``/``array``.
"""
import numpy as np


def param_mode(p):
    """``"none"`` for None, ``"tied"`` for a scalar, ``"array"`` otherwise."""
    return "none" if p is None else ("tied" if np.ndim(p) == 0 else "array")


def write_tied_params(grp, params):
    """Write ``params`` (name -> None | scalar | array) as datasets in their
    own floating dtype (a float64 parameter under ``jax_enable_x64`` stays
    float64) and return the ``<name>_mode`` attributes to store with them."""
    for name, p in params.items():
        if p is not None:
            grp.create_dataset(name, data=np.atleast_1d(np.asarray(p)))
    return {f"{name}_mode": param_mode(p) for name, p in params.items()}


def read_tied_params(grp, names):
    """Read the parameters written by :func:`write_tied_params`: None, a
    Python float for a tied parameter, or an array in the stored dtype."""
    out = {}
    for name in names:
        mode = grp.attrs[f"{name}_mode"]
        mode = mode.decode() if isinstance(mode, bytes) else str(mode)
        if mode == "none":
            out[name] = None
        else:
            vals = np.asarray(grp[name][()])
            out[name] = float(vals[0]) if mode == "tied" else vals
    return out


def neighbour_tables(pairs, N):
    """Per-site neighbour tables of a pair list, padded to the largest degree.

    Returns int32 ``nbr_site`` and ``nbr_pair`` of shape ``(N, max_deg)`` --
    the other site and the pair index of every pair a site belongs to -- and
    the float32 ``nbr_mask`` that is 1 on real entries and 0 on the padding
    (which points at the site itself and pair 0, so a masked scatter-add is a
    no-op).  A pair that appears twice is listed twice.
    """
    import jax.numpy as jnp
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    lists = [[] for _ in range(N)]
    for k, (i, j) in enumerate(pairs):
        lists[i].append((j, k))
        lists[j].append((i, k))
    max_deg = max((len(nb) for nb in lists), default=0)
    max_deg = max(max_deg, 1)
    site = np.repeat(np.arange(N, dtype=np.int64)[:, None], max_deg, axis=1)
    pair = np.zeros((N, max_deg), dtype=np.int64)
    mask = np.zeros((N, max_deg), dtype=np.float32)
    for s_, nb in enumerate(lists):
        for col, (j, k) in enumerate(nb):
            site[s_, col], pair[s_, col], mask[s_, col] = j, k, 1.0
    return (jnp.asarray(site, dtype=jnp.int32), jnp.asarray(pair, dtype=jnp.int32),
            jnp.asarray(mask, dtype=jnp.float32))
