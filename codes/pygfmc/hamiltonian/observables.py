"""Measurement operators that are not Hamiltonians.

A diagonal observable needs only ``diagonal_weight(config)`` -- not the
``M_total``/``get_rates``/``generate_state`` protocol a Hamiltonian or an
off-diagonal operator must implement -- so it is cheap both to write and to
measure: ``run_gfmc`` evaluates it every step and carries it through the
ancestry ring buffer, with no shadow chain.
"""
import numpy as np
import jax
import jax.numpy as jnp


def is_valid_k(k_vectors, lattice_vectors, atol=1e-5):
    r"""Whether each wavevector lies in the reciprocal lattice of the system.

    A wavevector is valid when ``k . L_j = 2 pi x integer`` for every periodic
    vector ``L_j`` of the system.  Exactly then every periodic image of a pair
    carries the same phase: shifting a site to a periodic image,
    ``r -> r + sum_j m_j L_j`` with integers ``m_j``, changes ``k . (r_i - r_j)``
    by a multiple of ``2 pi`` and leaves ``cos(k . (r_i - r_j))`` unchanged.
    The direct displacements ``r_i - r_j`` of the positions as given therefore
    suffice, without choosing periodic images, so
    :class:`SzSzStructureFactor` and
    :func:`~pygfmc.hamiltonian.StructureFactor` do not depend on which image
    the positions encode.

    Args:
        k_vectors:       One wavevector ``(d,)`` or several ``(n_k, d)``.
        lattice_vectors: ``(d, d)``, rows the periodic vectors of the whole
                         system, e.g. ``supercell_dims[:, None] *
                         primitive_vectors`` or kagome's ``extras['torus']``.
                         They are the vectors the boundary identifies, not the
                         primitive vectors of the unit cell.
        atol:            Absolute tolerance on ``k . L_j / (2 pi)`` being an
                         integer.  The default 1e-5 accepts wavevectors built
                         in float32 (e.g. with ``jax.numpy``), whose rounding
                         error is about 1e-6.

    Returns:
        ``bool`` for a single ``(d,)`` wavevector, a boolean ``(n_k,)`` array
        otherwise.

    Example:
        >>> L = np.array([[8.0]])                        # L = 8 ring
        >>> is_valid_k(2 * np.pi * np.arange(3)[:, None] / 8, L)
        array([ True,  True,  True])
        >>> is_valid_k([1.0], L)
        False
    """
    lat = np.asarray(lattice_vectors, dtype=np.float64)
    if lat.ndim != 2 or lat.shape[0] != lat.shape[1]:
        raise ValueError(
            f"lattice_vectors must be (d, d), rows the periodic vectors; got "
            f"{lat.shape}.")
    k = np.asarray(k_vectors, dtype=np.float64)
    single = k.ndim == 1
    k = np.atleast_2d(k)
    if k.ndim != 2 or k.shape[1] != lat.shape[1]:
        raise ValueError(
            f"k_vectors must be (d,) or (n_k, d) with d = {lat.shape[1]}; got "
            f"{np.shape(k_vectors)}.")
    m = k @ lat.T / (2 * np.pi)                         # (n_k, d)
    ok = np.all(np.abs(m - np.round(m)) <= atol, axis=1)
    return bool(ok[0]) if single else ok


class SzSzStructureFactor:
    r""":math:`S^{zz}(\vec k)` for one or many wavevectors, from positions.

    The observable is the pair sum

    .. math::
        S^{zz}(\vec k) = \sum_{i<j} 2\cos(\vec k\cdot\vec{dr}_{ij})
                         S^z_i S^z_j ,

    evaluated through the equivalent single sum

    .. math::
        \rho_{\vec k} = \sum_i e^{i\vec k\cdot\vec r_i}\sigma^z_i , \qquad
        |\rho_{\vec k}|^2 = \sum_{ij}\cos(\vec k\cdot(\vec r_i-\vec r_j))
                            \sigma^z_i\sigma^z_j ,

    so that (with :math:`\sigma^z=\pm1`, :math:`S^z=\sigma^z/2`)
    ``diagonal_weight`` returns :math:`(|\rho_{\vec k}|^2 - N)/4`.  The
    ``i == j`` terms are always dropped: they are the constant :math:`N/4`
    (``sigma^2 = 1``), carry no ``k`` dependence, and would shift every
    wavevector by the same amount.

    **Why the single sum.**  It is an exact algebraic rewriting, holding
    configuration by configuration, and costs ``O(N n_k)`` per walker instead
    of the pair sum's ``O(N^2)``; measuring many ``k`` in one operator is
    therefore nearly free compared with a run that measures nothing.

    **It is exact only for ``k`` in the reciprocal lattice of the simulation
    cell**, i.e. ``k . L = 2 pi x integer`` for every supercell vector ``L``,
    which is exactly the set of wavevectors a finite periodic cell has.  There
    every periodic image of a pair carries the same phase, so the result
    agrees with a pair sum built from minimum-image or absolute displacements.
    Off that grid the minimum-image pair sum is not a structure factor (it can
    even be negative, whereas :math:`|\rho_k|^2 \geq 0`).

    **Supplying valid wavevectors is the caller's job; nothing here checks
    it** (use :func:`is_valid_k`).  The positions alone cannot establish validity, because folded
    coordinates do not record the boundary identification (``x = 0..7`` is
    equally an L=8 ring, a half-empty L=16 cell or an open chain).  Generate
    the grid rather than typing it::

        cell = lat.extras['supercell_dims'][:, None] * lat.extras['primitive_vectors']
        ks = 2 * np.pi * np.asarray(ms) @ np.linalg.inv(cell).T   # ms: integer rows

    (``pyrochlore`` ships both ``positions`` and that
    cell in ``extras``.)

    **``diagonal_weight`` returns one value per wavevector**, which the GFMC
    pipeline carries as a trailing axis: ``raw['diag'][name]`` is
    ``(n_steps, n_k, n_p)``, ``obs[name][p]`` is an ``(n_k,)`` array and the
    HDF5 dataset ``run_<n>/observables/<name>`` is ``(n_p, n_k)``.  The ``k``
    axis carries no labels -- store ``op.k_vectors`` yourself, e.g. as
    ``stats['parameters/k_vectors']`` before ``write_to_HDF5`` (it is not a
    ``MUST_MATCH`` key, so it cannot break a later append).
    ``stats['autocorrelation/diag/<name>']`` is resolved per wavevector,
    shape ``(n_k, n_p)``.  The operator may share a run with diagonal
    operators of any other width; memory of its ancestry buffer grows
    linearly in ``n_k``.

    Args:
        positions:    Site coordinates, shape ``(N, d)``.  Row ``i`` is the
                      position of the site at index ``i`` in ``config``.
        k_vectors:    One wavevector ``(d,)`` or several ``(n_k, d)``.
        dtype:        Dtype of the stored phase tables (default
                      ``jnp.float32``).  They are always *built* in float64,
                      since ``cos``/``sin`` of a large ``k.r`` is where the
                      precision is lost, not the contraction.

    Example:
        >>> ks = 2 * np.pi * np.arange(L) / L                  # allowed grid
        >>> sf = SzSzStructureFactor(positions, ks[:, None])
        >>> *_, obs, stats, raw = run_gfmc(..., diag_operators={"Szz": sf})
        >>> obs["Szz"][p]                                      # (n_k,)
    """

    def __init__(self, positions, k_vectors, dtype=jnp.float32):
        r = np.atleast_2d(np.asarray(positions, dtype=np.float64))
        k = np.atleast_2d(np.asarray(k_vectors, dtype=np.float64))
        if r.ndim != 2 or k.ndim != 2:
            raise ValueError(
                f"positions must be (N, d) and k_vectors (n_k, d); got "
                f"{r.shape} and {k.shape}.")
        if k.shape[1] != r.shape[1]:
            raise ValueError(
                f"k_vectors have dimension {k.shape[1]} but positions have "
                f"{r.shape[1]}; a wavevector needs one component per spatial "
                "dimension of the lattice.")
        kr = k @ r.T                                    # (n_k, N), float64
        self.cos = jnp.asarray(np.cos(kr), dtype=dtype)
        self.sin = jnp.asarray(np.sin(kr), dtype=dtype)
        # the i == j terms, dropped always: a constant N/4 with no k dependence
        self.offset = jnp.asarray(0.25 * r.shape[0], dtype=dtype)
        self.k_vectors = jnp.asarray(k, dtype=dtype)    # for labelling only


    def diagonal_weight(self, config):
        """``(n_k,)`` for a single ``(N,)`` config, ``(n, n_k)`` for a batch."""
        return 0.25 * ((config @ self.cos.T) ** 2
                       + (config @ self.sin.T) ** 2) - self.offset


def _unflatten(children):
    obj = object.__new__(SzSzStructureFactor)
    obj.cos, obj.sin, obj.offset, obj.k_vectors = children
    return obj


jax.tree_util.register_pytree_node(
    SzSzStructureFactor,
    # The phase tables are arrays, so they are CHILDREN: an array in the aux
    # data is unhashable and breaks JIT.  __init__
    # recomputes the tables from positions, so unflattening bypasses it.
    lambda o: ((o.cos, o.sin, o.offset, o.k_vectors), ()),
    lambda _, ch: _unflatten(ch))
