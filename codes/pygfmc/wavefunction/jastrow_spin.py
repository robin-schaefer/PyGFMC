"""Jastrow guiding wavefunction for spin-S models.

    log ψ_T(x) = -Σ_{(i,j) ∈ pairs} β_ij x_i x_j  +  Σ_i h_i x_i  -  Σ_i d_i x_i²

``x_i ∈ {-S, ..., S}`` is the magnetic quantum number of site ``i`` in the
configuration ``x`` of a :class:`~pygfmc.hamiltonian.Spin` walker.  ``β`` is
the pair coupling (as in :class:`~pygfmc.wavefunction.JastrowSpinHalf`), ``h`` a
one-body field, and ``d`` a single-ion term that weights the values of
``|x_i|`` against each other; for spin 1/2 it is a constant.

Every ``Spin`` move changes at most two sites, ``a`` and ``b``, by ``δ_a`` and
``δ_b``, so the ratio needs only those sites and the weighted neighbour sums
``W_a = Σ_{pairs ∋ a} β x_other``:

    Δ = -[δ_a W_a + δ_b W_b + δ_a δ_b B_ab]
        + h_a δ_a + h_b δ_b
        - d_a [(x_a+δ_a)² - x_a²] - d_b [(x_b+δ_b)² - x_b²]

with ``B_ab`` the summed coupling of the pairs joining ``a`` and ``b``.
:meth:`JastrowSpin.log_diff_from_operator` evaluates this for all transitions
of a ``Spin`` operator at once, without building the successor configurations.
"""

import numpy as np
from typing import NamedTuple
import h5py
import jax
import jax.numpy as jnp
from ..hamiltonian._common import edges_array
from ._tied_params import write_tied_params, read_tied_params, neighbour_tables

from ..hamiltonian import Spin


def _param(value, n, dtype, name):
    """A parameter in the form it was given: None, a 0-d (tied) or an (n,) array."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return jnp.asarray(float(arr), dtype=dtype)
    if arr.shape != (n,):
        raise ValueError(f"{name} must be a scalar or have shape ({n},), got {arr.shape}")
    return jnp.asarray(arr, dtype=dtype)


class JastrowSpinOpCache(NamedTuple):
    """Walker-independent tables of :meth:`JastrowSpin.log_diff_from_operator`
    for one :class:`~pygfmc.hamiltonian.Spin` operator: the sites ``a``,
    ``b`` and changes ``da``, ``db`` of every transition in the operator's
    layout, and the summed guide coupling ``pair_beta`` of each transition's
    pair.  Built once per run by :meth:`JastrowSpin.build_cache`."""
    a: jax.Array
    b: jax.Array
    da: jax.Array
    db: jax.Array
    pair_beta: jax.Array
    nbr_site: jax.Array
    nbr_pair: jax.Array
    nbr_mask: jax.Array


class JastrowSpin:
    r"""Jastrow guide for spin-S configurations.

    .. math::
        \log\psi_T(x) = -\sum_{(i,j)} \beta_{ij} x_i x_j + \sum_i h_i x_i
                        - \sum_i d_i x_i^2

    Args:
        N:     number of sites.
        pairs: ``(n_pairs, 2)`` site pairs of the two-body term, e.g. the
               Hamiltonian edges.
        beta:  a scalar (one coupling shared by all pairs) or an array of
               length ``n_pairs``.  ``beta < 0`` favours aligned neighbours.
        h:     one-body field: a scalar (shared), an ``(N,)`` array, or None
               (no field).
        d:     single-ion term: a scalar (shared), an ``(N,)`` array, or None
               (no term).  ``d > 0`` favours small ``|x_i|``.
        dtype: JAX dtype of the amplitudes.
        incremental: carry the weighted neighbour sums ``W_i = Σ_j β_ij x_j``
               of every walker through the run and update them after each
               move (two sites at most) instead of rebuilding them from all
               pairs at every rate evaluation.  The sums are rebuilt every 100
               chain steps, so float rounding cannot accumulate.  The run is
               then no longer bit-identical to the default, which rebuilds
               the sums every time; the physics is unchanged.

    The guide keeps every parameter in the form it was given, and
    :func:`~pygfmc.wavefunction.train_with_netket` trains exactly those: a
    scalar stays a single tied value, ``None`` stays absent.
    """

    def __init__(self, N, pairs, beta, h=None, d=None, dtype=jnp.float32,
                 incremental=False):
        pairs, _, n_min = edges_array(pairs)      # shape, integer and self-pair checks as for edges
        if n_min > int(N):
            raise ValueError(f"pair indices must lie in [0, {N}).")
        n_pairs = len(pairs)
        self.N, self.dtype = int(N), dtype
        self.pair_i = jnp.asarray(pairs[:, 0], dtype=jnp.int32)
        self.pair_j = jnp.asarray(pairs[:, 1], dtype=jnp.int32)
        # sorted (min, max) keys, to look up the pairs joining two sites
        keys = np.minimum(pairs[:, 0], pairs[:, 1]) * N + np.maximum(pairs[:, 0], pairs[:, 1])
        order = np.argsort(keys, kind="stable")
        self.key_order = jnp.asarray(order, dtype=jnp.int32)
        self.sorted_keys = jnp.asarray(keys[order], dtype=jnp.int32)
        if np.ndim(beta) == 0:
            self._beta = jnp.asarray(float(np.asarray(beta)), dtype=dtype)
        else:
            self._beta = _param(beta, n_pairs, dtype, "beta")
        self._h = _param(h, self.N, dtype, "h")
        self._d = _param(d, self.N, dtype, "d")
        self.incremental = bool(incremental)
        self._pairs_np = pairs            # host copy for the neighbour tables

    # ── parameters ──────────────────────────────────────────────────────

    @property
    def beta(self):
        """Coupling of every pair, ``(n_pairs,)``, in constructor order."""
        return jnp.broadcast_to(self._beta, self.pair_i.shape).astype(self.dtype)

    @property
    def h(self):
        """One-body field per site, ``(N,)`` (zeros when constructed with ``h=None``)."""
        if self._h is None:
            return jnp.zeros(self.N, dtype=self.dtype)
        return jnp.broadcast_to(self._h, (self.N,))

    @property
    def d(self):
        """Single-ion term per site, ``(N,)`` (zeros when constructed with ``d=None``)."""
        if self._d is None:
            return jnp.zeros(self.N, dtype=self.dtype)
        return jnp.broadcast_to(self._d, (self.N,))

    # ── amplitudes ──────────────────────────────────────────────────────

    def log_psi(self, config, cache=None):
        """``log ψ_T`` of one ``(N,)`` configuration or an ``(n, N)`` batch."""
        x = jnp.asarray(config).astype(self.dtype)
        if x.ndim == 2:
            return jax.vmap(self.log_psi)(x)
        return (-jnp.sum(self.beta * x[self.pair_i] * x[self.pair_j])
                + jnp.sum(self.h * x) - jnp.sum(self.d * x * x))

    def _weighted_neighbour_sums(self, x):
        b = self.beta
        return (jnp.zeros(self.N, self.dtype)
                .at[self.pair_i].add(b * x[self.pair_j])
                .at[self.pair_j].add(b * x[self.pair_i]))

    def _pair_coupling(self, a, b):
        """Summed beta of the pairs joining sites ``a`` and ``b`` (0 if none)."""
        key = jnp.minimum(a, b) * self.N + jnp.maximum(a, b)
        lo = jnp.searchsorted(self.sorted_keys, key, side="left")
        hi = jnp.searchsorted(self.sorted_keys, key, side="right")
        csum = jnp.concatenate([jnp.zeros(1, self.dtype),
                                jnp.cumsum(self.beta[self.key_order])])
        return csum[hi] - csum[lo]

    def _transition_tables(self, op):
        """Sites ``a``, ``b`` and changes ``da``, ``db`` of every transition
        of the :class:`~pygfmc.hamiltonian.Spin` operator ``op``, in its
        layout (entry 0 is the diagonal; a single-site move has ``b = a``,
        ``db = 0``)."""
        zi, zf = jnp.zeros(1, jnp.int32), jnp.zeros(1, self.dtype)
        a, b, da, db = [zi], [zi], [zf], [zf]
        ei, ej = op.edge_i.astype(jnp.int32), op.edge_j.astype(jnp.int32)
        ones = jnp.ones(op.n_edges, self.dtype)
        blocks = []
        if op.n_jpm_off > 0:
            blocks += [(+1, -1), (-1, +1)]          # S+_i S-_j, S-_i S+_j
        if op.n_jpmpm_off > 0:
            blocks += [(+1, +1), (-1, -1)]          # S+_i S+_j, S-_i S-_j
        for sa, sb in blocks:
            a.append(ei)
            b.append(ej)
            da.append(sa * ones)
            db.append(sb * ones)
        if op.n_hpm_off > 0:
            sites = jnp.arange(op.N, dtype=jnp.int32)
            for s in (+1, -1):                      # S+_i, S-_i
                a.append(sites)
                b.append(sites)
                da.append(s * jnp.ones(op.N, self.dtype))
                db.append(jnp.zeros(op.N, self.dtype))
        return (jnp.concatenate(a), jnp.concatenate(b),
                jnp.concatenate(da), jnp.concatenate(db))

    def build_cache(self, op):
        """The walker-independent tables of :meth:`log_diff_from_operator`
        for a :class:`~pygfmc.hamiltonian.Spin` operator, as a
        :class:`JastrowSpinOpCache` (None for any other operator).  Memoised
        per operator layout and parameter values, so repeated calls with the
        same operator skip the rebuild."""
        if not isinstance(op, Spin):
            return None
        key = (np.asarray(op.edge_i).tobytes(), np.asarray(op.edge_j).tobytes(),
               op.n_jpm_off, op.n_jpmpm_off, op.n_hpm_off, op.N,
               np.asarray(self.beta).tobytes())
        if not hasattr(self, "_cache"):
            self._cache = {}
        if self._pairs_np is None:
            self._pairs_np = np.stack([np.asarray(self.pair_i), np.asarray(self.pair_j)], axis=1)
        hit = self._cache.get(key)
        if hit is None:
            a, b, da, db = self._transition_tables(op)
            hit = JastrowSpinOpCache(a, b, da, db, self._pair_coupling(a, b),
                                     *neighbour_tables(self._pairs_np, self.N))
            self._cache[key] = hit
        return hit

    # ── incremental neighbour sums (the per-walker state) ─────────────────
    def init_state(self, config, cache=None):
        """The per-walker state of the incremental mode: the weighted
        neighbour sums ``W_i = Σ_j β_ij x_j`` of ``config``, shape ``(N,)``."""
        return self._weighted_neighbour_sums(jnp.asarray(config).astype(self.dtype))

    def update_state(self, state, config, sites, deltas, cache=None):
        """``state`` after the move that adds ``deltas`` to ``sites`` (shape
        ``(2,)`` each, as :meth:`pygfmc.hamiltonian.Spin.move` returns):
        every neighbour ``j`` of a moved site ``s`` gets ``β_sj Δ_s``.  Needs
        the operator's :meth:`build_cache` result, which holds the neighbour
        tables."""
        if self.pair_i.shape[0] == 0:        # no pairs: the sums stay zero
            return state
        beta = self.beta
        for k in range(2):
            s_k, d_k = sites[k], jnp.asarray(deltas[k], self.dtype)
            w = beta[cache.nbr_pair[s_k]] * cache.nbr_mask[s_k]
            state = state.at[cache.nbr_site[s_k]].add(w * d_k)
        return state

    def log_diff_from_operator(self, config, op, cache=None, state=None):
        """``log ψ_T(x') - log ψ_T(x)`` for every transition of ``op``.

        Fast path for :class:`~pygfmc.hamiltonian.Spin` operators (entry 0,
        the diagonal, is 0), from the tables of ``cache`` when it is the
        operator's :meth:`build_cache` result and built on the fly otherwise.
        ``state`` are the walker's neighbour sums from :meth:`init_state` /
        :meth:`update_state` (rebuilt here when None).  Any other operator
        falls back to its successor configurations.
        """
        if not isinstance(op, Spin):
            _, new_configs = op.get_rates(config, return_configs=True)
            return self.log_psi(new_configs) - self.log_psi(config)
        if isinstance(cache, JastrowSpinOpCache):
            a, b, da, db, pair_beta = cache[:5]
        else:
            a, b, da, db = self._transition_tables(op)
            pair_beta = self._pair_coupling(a, b)

        x = jnp.asarray(config).astype(self.dtype)
        W = self._weighted_neighbour_sums(x) if state is None else state
        xa, xb = x[a], x[b]
        h, d = self.h, self.d
        pair = da * W[a] + db * W[b] + da * db * pair_beta
        single = (h[a] * da + h[b] * db
                  - d[a] * ((xa + da) ** 2 - xa ** 2) - d[b] * ((xb + db) ** 2 - xb ** 2))
        return -pair + single

    # ── HDF5 save / load ────────────────────────────────────────────────

    def save(self, path, group="jastrow_spin", **attrs):
        """Write the guide to an HDF5 file (created or overwritten).

        Stores the pairs, per-pair ``beta``, ``h`` and ``d``, and the form of
        each parameter, so :meth:`load` rebuilds the same parametrisation.
        Extra keyword arguments are stored as scalar group attributes.
        """
        from ..walkers._h5io import write_attrs
        pairs = np.stack([np.asarray(self.pair_i), np.asarray(self.pair_j)], axis=1)
        with h5py.File(path, "w") as f:
            grp = f.require_group(group)
            grp.create_dataset("pairs", data=pairs.astype(np.int32))
            modes = write_tied_params(grp, {"beta": self._beta, "h": self._h, "d": self._d})
            write_attrs(grp, dict(N=self.N, incremental=int(self.incremental),
                                  **modes, **attrs))

    @classmethod
    def load(cls, path, group="jastrow_spin", dtype=jnp.float32):
        """Load a guide written by :meth:`save`."""
        with h5py.File(path, "r") as f:
            grp = f[group]
            pairs = np.asarray(grp["pairs"][()], dtype=np.int64)
            N = int(grp.attrs["N"])
            incremental = bool(int(grp.attrs.get("incremental", 0)))
            args = read_tied_params(grp, ("beta", "h", "d"))
        return cls(N, pairs, args["beta"], h=args["h"], d=args["d"], dtype=dtype,
                   incremental=incremental)


# ── pytree registration ─────────────────────────────────────────────────

def _flatten(obj):
    # integer leaves are structure; _beta, _h, _d are the parameters in the
    # form they were given (None is an empty subtree, i.e. not trained)
    children = (obj.pair_i, obj.pair_j, obj.key_order, obj.sorted_keys,
                obj._beta, obj._h, obj._d)
    # the incremental flag enters the (static) aux only when set, so a
    # default guide keeps the tree, and the content hash, it always had
    aux = (obj.N, obj.dtype) + ((True,) if obj.incremental else ())
    return children, aux


def _unflatten(aux, children):
    obj = object.__new__(JastrowSpin)
    obj.N, obj.dtype = aux[:2]
    obj.incremental = len(aux) > 2
    obj._pairs_np = None              # rebuilt from the pairs by build_cache
    (obj.pair_i, obj.pair_j, obj.key_order, obj.sorted_keys,
     obj._beta, obj._h, obj._d) = children
    return obj


jax.tree_util.register_pytree_node(JastrowSpin, _flatten, _unflatten)
