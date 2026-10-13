"""Jastrow guiding wavefunction.

ψ_T(x) = exp(Σ_{(i,j) ∈ pairs} J_ij x_i x_j  +  Σ_i h_i x_i)

The two-body coefficients J_ij and the optional one-body fields h_i are
arbitrary; the only structural requirement is the user-supplied pair list.
A uniform Ising-type guide is the special case ``JastrowSpinHalf(h.N, h.edges,
beta=β)``, i.e. pairs = Hamiltonian edges and J_ij = -β.

For any flip set S, only pairs that "straddle" S (exactly one endpoint in S)
contribute to log(ψ'/ψ).  The log-amp ratio is therefore

    log(ψ'/ψ) = -2 Σ_{(a,b): |{a,b}∩S|=1} J_ab x_a x_b - 2 Σ_{a∈S} h_a x_a

For the SpinHalf transition layout this evaluates to:
  - h_x site flip at k:        -2 x_k (Σ_b J_{k,b} x_b)              - 2 h_k x_k
  - J_pm edge flip on (i,j):   -2 x_i Σ_{b≠j} J_{i,b} x_b - 2 x_j Σ_{b≠i} J_{j,b} x_b
                               - 2 (h_i x_i + h_j x_j)
"""

from typing import NamedTuple

import numpy as np
import h5py
import jax
import jax.numpy as jnp
from ..hamiltonian._common import edges_array
from ._layout import require_spin_half_layout


class JastrowOpCache(NamedTuple):
    """Precomputed external-partner tables for a (Jastrow, operator) pair.

    Built by ``JastrowSpinHalf.build_cache(op)``; consumed by
    ``JastrowSpinHalf.log_diff_from_operator(config, op, cache=cache)`` for the
    cached fast path.
    """
    edge_other:      jnp.ndarray   # (n_edges_op, 2, max_deg) int
    edge_J:          jnp.ndarray   # (n_edges_op, 2, max_deg) float


class JastrowSpinHalf:
    """Jastrow guiding wavefunction with sparse two-body couplings.

    ψ_T(x) = exp(Σ_{(i,j) ∈ pairs} J_ij x_i x_j  +  Σ_i h_i x_i)

    The constructor is Hamiltonian-agnostic: pass ``N`` (number of sites),
    a ``(n_pairs, 2)`` pair list and the couplings ``beta``.  No reference to
    a ``SpinHalf`` operator is kept on the guide.  The coupling follows the
    convention ``log ψ = -Σ_<ij> β_ij x_i x_j + Σ_i h_i x_i`` and is stored
    internally as ``J = -beta`` (exposed as the attribute ``beta``).

    Args:
        N:     int, total number of sites in the system.
        pairs: (n_pairs, 2) int array of site index pairs (i, j).
               Order within a pair does not matter (canonicalised to (min, max)
               internally).
        beta:  required; a scalar (one tied coupling for every pair) or an
               array of length ``n_pairs`` (one coupling per pair).
        h:     one-body fields — a scalar (one tied field on every site), an
               (N,) float array (one per site), or None (no field).
        dtype: JAX dtype for amplitudes.

    The guide keeps the parameters in the form they were given, and
    :func:`~pygfmc.wavefunction.train_with_netket` trains exactly those: a
    scalar ``beta`` stays a single coupling, ``h=None`` stays without a field,
    a scalar ``h`` stays a single field.

    Operators: :meth:`log_diff_from_operator` and :meth:`build_cache` index
    their output by the ``SpinHalf`` transition layout, so they accept a
    :class:`~pygfmc.hamiltonian.SpinHalf` (Hamiltonian or measurement
    operator, e.g. :func:`~pygfmc.hamiltonian.StructureFactor`) or any
    operator exposing ``n_hx_static``, ``n_jpm_static``, ``edge_i`` and
    ``edge_j``; any other operator (``ToricCode``, ``PlaquetteModel``, ...)
    raises ``TypeError``.  :meth:`log_psi` and :meth:`log_diff_from_configs`
    take configurations of any spin-1/2 operator.
    """

    def __init__(self, N, pairs, beta, h=None, dtype=jnp.float32):
        N = int(N)
        if pairs is None:
            raise ValueError("pairs must be provided (n_pairs, 2) int array)")
        pairs_np, _, n_min = edges_array(pairs)   # shape and integer checks as for edges
        if n_min > N:
            raise ValueError(f"pair indices must lie in [0, {N}).")
        pairs_np = pairs_np.astype(np.int32)
        # canonicalise (a,b) to (min,max) so adjacency lookup is symmetric
        pairs_np = np.sort(pairs_np, axis=1)
        n_pairs  = pairs_np.shape[0]

        # beta is a scalar (tied across all pairs) or an array of length
        # n_pairs (the broadcast rejects any other length).  Paper convention:
        # log psi = -sum_<ij> beta_ij sz_i sz_j + sum_i h_i sz_i, stored
        # internally as J = -beta.
        J_np = -np.broadcast_to(np.asarray(beta, dtype=dtype), (n_pairs,))

        # h mirrors beta: a scalar is a uniform field on every site, an array
        # gives one field per site.  broadcast_to (rather than a bare asarray)
        # is what admits the scalar; the explicit shape check below still
        # rejects a wrong-length array, which broadcasting would otherwise
        # accept silently for length-1 input.
        if h is None:
            h_np = np.zeros(N, dtype=dtype)
        else:
            h_in = np.asarray(h, dtype=dtype)
            if h_in.ndim > 1 or (h_in.ndim == 1 and h_in.shape != (N,)):
                raise ValueError(
                    f"h must be a scalar or have shape ({N},), got {h_in.shape}")
            h_np = np.broadcast_to(h_in, (N,)).copy()

        # ── per-site adjacency: site k → list of (partner, J) ────────────
        # pair_site/pair_slot record each input pair's slot in the per-site
        # tables, so ``save`` writes the pairs back in constructor order.
        site_partners = [[] for _ in range(N)]
        pair_site = np.zeros(n_pairs, dtype=np.int32)
        pair_slot = np.zeros(n_pairs, dtype=np.int32)
        for p_idx, (a, b) in enumerate(pairs_np):
            pair_site[p_idx] = a
            pair_slot[p_idx] = len(site_partners[a])
            site_partners[a].append((int(b), p_idx))
            site_partners[b].append((int(a), p_idx))
        max_deg = max((len(s) for s in site_partners), default=0) or 1

        # site_pair[k, slot] is the pair behind each table entry, so both
        # endpoints of a pair read the SAME coupling: a pair's J is stored
        # once, and training cannot make the two copies drift apart.
        site_other = -np.ones((N, max_deg), dtype=np.int32)
        site_pair  = np.zeros((N, max_deg), dtype=np.int32)
        for k in range(N):
            for slot, (other, p_idx) in enumerate(site_partners[k]):
                site_other[k, slot] = other
                site_pair[k, slot]  = p_idx
        site_valid_mask = site_other >= 0

        # ── store ────────────────────────────────────────────────────────
        self.dtype = dtype
        self.N     = N
        self.site_other      = jnp.array(np.clip(site_other, 0, None), dtype=jnp.int32)
        self.site_valid_mask = jnp.array(site_valid_mask)
        self._pair_site      = jnp.asarray(pair_site)
        self._pair_slot      = jnp.asarray(pair_slot)
        self._site_pair      = jnp.asarray(site_pair)
        # The parameters are stored in the form they were given, because the
        # pytree's floating-point leaves are what ``train_with_netket`` trains:
        # a scalar beta is ONE tied coupling (a 0-d leaf), an array one coupling
        # per pair; h=None is no field at all (no leaf), a scalar one tied
        # field, an array one field per site.  ``site_J`` and ``h`` are derived.
        if np.ndim(beta) == 0:
            self._J = jnp.asarray(-float(np.asarray(beta)), dtype=dtype)
        else:
            self._J = jnp.array(J_np, dtype=dtype)       # one J per pair
        if h is None:
            self._h = None
        elif np.ndim(h) == 0:
            self._h = jnp.asarray(float(np.asarray(h)), dtype=dtype)
        else:
            self._h = jnp.array(h_np, dtype=dtype)

    @property
    def site_J(self):
        """Couplings ``J = -beta`` per site and partner slot, ``(N, max_deg)``."""
        J = self._J if self._J.ndim == 0 else self._J[self._site_pair]
        return jnp.where(self.site_valid_mask, J, 0).astype(self.dtype)

    @property
    def h(self):
        """One-body field per site, ``(N,)`` (zeros when constructed with ``h=None``)."""
        if self._h is None:
            return jnp.zeros(self.N, dtype=self.dtype)
        return jnp.broadcast_to(self._h, (self.N,))

    @property
    def tied_beta(self):
        """True when ``beta`` was a single coupling shared by all pairs."""
        return self._J.ndim == 0

    @property
    def beta(self):
        """Per-pair coupling in the paper's convention, ``beta = -J``.

        Read from the stored couplings (a tied scalar is broadcast to every
        pair), so it follows NetKet training or a ``tree_map`` and survives a
        pytree round trip / jit.  The order is the order of ``pairs`` given to
        the constructor.
        """
        n_pairs = self._pair_site.shape[0]
        return -jnp.broadcast_to(self._J, (n_pairs,))


    # ── HDF5 save / load ────────────────────────────────────────────────

    def save(self, path, group='jastrow_spin_half', **attrs):
        """Write the Jastrow couplings to an HDF5 file.

        Stores the pair list (in constructor order), per-pair J, the
        one-body field h, and N, so :meth:`load` rebuilds the same guide with
        ``beta`` in the same order.
        Any extra keyword arguments are written as scalar HDF5 attributes on
        the group, e.g. ``E_vmc=...``, ``E_vmc_err=...``.

        Args:
            path:  HDF5 file path (created or overwritten).
            group: HDF5 group name to write under.
            **attrs: extra metadata stored as group attributes.
        """
        # _pair_site/_pair_slot locate each constructor pair's entry in the
        # per-site tables, so the pairs come back in their original order.
        a       = np.asarray(self._pair_site)
        slot    = np.asarray(self._pair_slot)
        b       = np.asarray(self.site_other)[a, slot]
        pairs   = np.stack([a, b], axis=1).astype(np.int32)
        J_pairs = -np.asarray(self.beta)          # in the guide's own dtype

        from ..walkers._h5io import write_attrs
        with h5py.File(path, 'w') as f:
            g = f.require_group(group)
            g.create_dataset('pairs', data=pairs)
            g.create_dataset('J',     data=J_pairs)
            g.create_dataset('h',     data=np.asarray(self.h))
            h_mode = ('none' if self._h is None
                      else 'tied' if self._h.ndim == 0 else 'sites')
            write_attrs(g, dict(N=self.N, tied_beta=int(self.tied_beta),
                                h_mode=h_mode, **attrs))

    @classmethod
    def load(cls, path, hamiltonian=None, group='jastrow_spin_half', dtype=jnp.float32):
        """Load a Jastrow saved with :meth:`save`.

        Args:
            path:        HDF5 file path written by :meth:`save`.
            hamiltonian: optional ``SpinHalf``.  If provided, ``N`` is taken as
                         the largest of ``saved_N``, ``hamiltonian.N`` and the
                         highest edge index + 1, and ``h`` is zero-padded when
                         the new system is larger; otherwise ``N=saved_N``.
            group:       HDF5 group name to read from.
            dtype:       JAX dtype for the rebuilt guide.
        """
        with h5py.File(path, 'r') as f:
            g = f[group]
            pairs = np.asarray(g['pairs'][()], dtype=np.int32)
            J     = np.asarray(g['J'][()])
            h     = np.asarray(g['h'][()])
            saved_N = int(g.attrs['N'])
            tied_beta = bool(g.attrs.get('tied_beta', 0))
            h_mode = g.attrs.get('h_mode', 'sites')
            if isinstance(h_mode, bytes):
                h_mode = h_mode.decode()

        if hamiltonian is None:
            N = saved_N
        else:
            edges_max = int(np.asarray(hamiltonian.edges).max(initial=-1)) + 1
            N = max(saved_N, edges_max, int(hamiltonian.N))
            if h.shape == (saved_N,) and saved_N < N:
                h_full = np.zeros(N, dtype=h.dtype)
                h_full[:saved_N] = h
                h = h_full
        if h.shape != (N,):
            raise ValueError(
                f"saved h has shape {h.shape} but resolved N={N}"
            )
        beta = float(-J[0]) if (tied_beta and J.size) else -J
        h = None if h_mode == 'none' else float(h[0]) if h_mode == 'tied' else h
        return cls(N=N, pairs=pairs, beta=beta, h=h, dtype=dtype)


    @classmethod
    def from_jax(cls, dtype, N,
                 site_other, site_J, site_valid_mask, h,
                 pair_site, pair_slot, site_pair):
        """Rebuild a guide from its internal arrays (used by pytree unflatten).

        Bypasses ``__init__``.  ``pair_site``/``pair_slot`` locate each pair's
        coupling in ``site_J`` (they define ``beta``).
        """
        obj = object.__new__(cls)
        obj.dtype = dtype
        obj.N     = N
        obj.site_other      = site_other
        obj._J              = site_J      # 0-d (tied) or (n_pairs,)
        obj.site_valid_mask = site_valid_mask
        obj._h              = h           # None, 0-d (tied) or (N,)
        obj._pair_site = pair_site
        obj._pair_slot = pair_slot
        obj._site_pair = site_pair
        return obj

    # ── log-amplitude ratios ────────────────────────────────────────────

    def _site_log_diff(self, config):
        """log(ψ'/ψ) for each single-site flip k, shape (N,)."""
        weighted_mf = jnp.sum(
            jnp.where(self.site_valid_mask,
                      self.site_J * config[self.site_other], 0),
            axis=-1,
        )
        return -2.0 * config * weighted_mf - 2.0 * self.h * config

    def log_psi(self, config, cache=None):
        """log ψ_T(config) = Σ_pairs J_ij x_i x_j + Σ_i h_i x_i.

        Accepts a single ``(N,)`` configuration or a ``(n, N)`` batch, matching
        the guide protocol's third routine.  ``cache`` is accepted for
        protocol parity and unused: the sum is already O(N·max_deg) and there
        is nothing operator-specific to cache.

        Evaluated through ``site_other``/``site_J``, which store each pair from
        both endpoints -- hence the 1/2 on the pair term.

        GFMC itself never calls this; it uses the ratio routines
        (``log_diff_from_operator``/``log_diff_from_configs``).  It exists for
        diagnostics and for testing those ratios against an explicit amplitude.
        """
        config = jnp.asarray(config)
        if config.ndim == 2:
            return jax.vmap(lambda c: self.log_psi(c))(config)
        config = config.astype(self.dtype)
        pair_term = 0.5 * jnp.sum(
            jnp.where(self.site_valid_mask,
                      self.site_J * config[:, None] * config[self.site_other],
                      0.0)
        )
        field_term = jnp.sum(self.h * config)
        return pair_term + field_term

    def _edge_log_diff_for_edges(self, config, edge_i, edge_j):
        """log(ψ'/ψ) for each edge flip in (edge_i, edge_j), shape (n_edges,).

        Uses ψ_T's own site-partner tables (``site_other``/``site_J``) and
        rebuilds the per-edge external-partner mask on the fly so we don't
        need a materialised ``(n_edges, N)`` new_configs array.
        """
        i_partners = self.site_other[edge_i]                 # (n_edges, max_deg)
        j_partners = self.site_other[edge_j]
        i_J        = self.site_J[edge_i]
        j_J        = self.site_J[edge_j]
        i_valid    = self.site_valid_mask[edge_i]
        j_valid    = self.site_valid_mask[edge_j]

        # exclude the bond endpoints from each other (the i↔j bond itself)
        i_excl = (i_partners == edge_j[:, None])
        j_excl = (j_partners == edge_i[:, None])
        i_mask = i_valid & ~i_excl
        j_mask = j_valid & ~j_excl

        i_mf = jnp.sum(jnp.where(i_mask, i_J * config[i_partners], 0.0), axis=-1)
        j_mf = jnp.sum(jnp.where(j_mask, j_J * config[j_partners], 0.0), axis=-1)

        sz_i = config[edge_i]
        sz_j = config[edge_j]
        pair_term = sz_i * i_mf + sz_j * j_mf
        h_term    = self.h[edge_i] * sz_i + self.h[edge_j] * sz_j
        return -2.0 * pair_term - 2.0 * h_term

    # ── public guide protocol ───────────────────────────────────────────


    def log_diff_from_configs(self, config, new_configs, cache=None):
        """``log ψ(x′) − log ψ(x)`` for each row of ``new_configs``.

        Returns shape ``(new_configs.shape[0],)``.  Generic: works for any
        ``new_configs`` regardless of how they were produced.

        ``cache`` is accepted for protocol parity and unused.
        """
        base = self.log_psi(config)
        return jax.vmap(self.log_psi)(new_configs) - base

    def log_diff_from_operator(self, config, op, cache=None):
        """Return log-amplitude ratios indexed by a SpinHalf operator's layout.

        Layout matches ``op.get_rates`` of a ``SpinHalf``:
        ``[diag (=0), op.n_hx site flips at sites 0..N-1, op.n_jpm op-edge flips]``.
        No successor configurations are materialised.

        ``cache=None`` (default) rebuilds the per-edge external-partner
        mask on the fly via ``_edge_log_diff_for_edges``.  Pass a
        ``JastrowOpCache`` (obtained from ``self.build_cache(op)``) to use the
        cached fast path instead.

        ``op`` must expose the SpinHalf layout (``n_hx_static``,
        ``n_jpm_static``, ``edge_i``, ``edge_j``); any other operator raises
        ``TypeError``.
        """
        require_spin_half_layout(op, type(self).__name__)
        M = 1 + op.n_hx_static + op.n_jpm_static
        out = jnp.zeros(M, dtype=self.dtype)
        if op.n_hx_static > 0:
            out = out.at[1 : 1 + self.N].set(self._site_log_diff(config))
        if op.n_jpm_static > 0:
            if cache is not None:
                edge_part = self._edge_log_diff_from_cache(
                    config, op.edge_i, op.edge_j, cache)
            else:
                edge_part = self._edge_log_diff_for_edges(
                    config, op.edge_i, op.edge_j)
            out = out.at[1 + op.n_hx_static :].set(edge_part)
        return out

    def build_cache(self, op):
        """Build and return the external-partner cache for ``op``'s edges.

        Returns a ``JastrowOpCache`` pytree of shape-``(n_edges_op, 2, max_deg)``
        ``(edge_other, edge_J)`` arrays — the result of
        joining ``self.site_other`` / ``self.site_J`` with ``op.edge_i`` /
        ``op.edge_j`` and masking out the i↔j bond endpoints themselves.
        Pass the returned cache to ``log_diff_from_operator(..., cache=...)``.

        Results are memoised by the content of ``op``'s edge layout and of
        the couplings ``J`` (which ``edge_J`` bakes in), so repeated calls
        with the same operator (e.g. once from ``run_gfmc`` and again from
        ``statistics()``) skip the rebuild, while a guide whose couplings
        changed (e.g. a trained copy) gets its own tables.

        ``op`` must expose the SpinHalf layout (``n_hx_static``,
        ``n_jpm_static``, ``edge_i``, ``edge_j``); any other operator raises
        ``TypeError``.
        """
        require_spin_half_layout(op, type(self).__name__)
        edge_i_np = np.asarray(op.edge_i)
        edge_j_np = np.asarray(op.edge_j)

        # Memoise by the operator's edge layout and the guide's couplings.
        if not hasattr(self, '_cache'):
            self._cache = {}
        cache_key = (edge_i_np.tobytes(), edge_j_np.tobytes(),
                     np.asarray(self._J).tobytes())
        hit = self._cache.get(cache_key)
        if hit is not None:
            return hit

        site_other      = np.asarray(self.site_other)
        site_J          = np.asarray(self.site_J)
        site_valid_mask = np.asarray(self.site_valid_mask)

        i_partners = site_other[edge_i_np]                       # (n_edges, max_deg)
        j_partners = site_other[edge_j_np]
        i_J        = site_J[edge_i_np]
        j_J        = site_J[edge_j_np]
        i_valid    = site_valid_mask[edge_i_np]
        j_valid    = site_valid_mask[edge_j_np]

        # exclude the bond endpoints from each other (the i↔j bond itself)
        i_excl = i_partners == edge_j_np[:, None]
        j_excl = j_partners == edge_i_np[:, None]
        i_mask = i_valid & ~i_excl
        j_mask = j_valid & ~j_excl

        edge_other      = np.stack([i_partners, j_partners], axis=1)
        # padded slots carry J = 0, so they drop out of the sum without a mask
        edge_J          = np.where(np.stack([i_mask, j_mask], axis=1),
                                   np.stack([i_J,    j_J   ], axis=1), 0)

        out = JastrowOpCache(
            edge_other      = jnp.asarray(edge_other, dtype=jnp.int32),
            edge_J          = jnp.asarray(edge_J,     dtype=self.dtype),
        )
        self._cache[cache_key] = out
        return out


    def _edge_log_diff_from_cache(self, config, edge_i, edge_j, cache):
        """Cached edge-log-diff kernel (mirrors ``_edge_log_diff_for_edges``)."""
        weighted_mf = jnp.sum(
            cache.edge_J * config[cache.edge_other],
            axis=-1,
        )                                                    # (n_edges, 2)
        sz_i = config[edge_i]
        sz_j = config[edge_j]
        pair_term = sz_i * weighted_mf[:, 0] + sz_j * weighted_mf[:, 1]
        h_term    = self.h[edge_i] * sz_i + self.h[edge_j] * sz_j
        return -2.0 * pair_term - 2.0 * h_term


# ── pytree registration ─────────────────────────────────────────────────

def _jastrow_flatten(obj):
    # _pair_site/_pair_slot are integer index leaves (structure, not
    # parameters); they carry ``beta`` through the round trip.
    # _J and _h are the parameters exactly as given (see __init__); a None
    # _h is an empty subtree, so a guide without a field has no h leaf.
    children = (
        obj.site_other, obj._J, obj.site_valid_mask,
        obj._h, obj._pair_site, obj._pair_slot, obj._site_pair,
    )
    aux = (obj.dtype, obj.N)
    return children, aux


def _jastrow_unflatten(aux, children):
    dtype, N = aux
    site_other, J, site_valid_mask, h, pair_site, pair_slot, site_pair = children
    return JastrowSpinHalf.from_jax(
        dtype, N, site_other, J, site_valid_mask, h,
        pair_site=pair_site, pair_slot=pair_slot, site_pair=site_pair,
    )


jax.tree_util.register_pytree_node(JastrowSpinHalf, _jastrow_flatten, _jastrow_unflatten)
