"""General per-cluster MLP neural quantum state for SpinHalf.

    log ψ(x) = Σ_g Σ_{c ∈ clusters_g} MLP_g(x_c)

where each cluster group `g` is a list of clusters of uniform size `k_g`,
and `MLP_g : R^{k_g} → R` is a shared tanh MLP of arbitrary depth (linear
output layer) applied to every cluster's spin vector and summed over
clusters.  Different groups may have different cluster sizes, depths and
widths.

Smart log_diff_from_operator
----------------------------
For each SpinHalf transition (which flips a known set of sites `S`):

    Δ log ψ = Σ_g  Σ_{c ∈ affected_g(S)}  [MLP_g(x'_c) − MLP_g(x_c)]

where `affected_g(S)` is the set of clusters in group `g` containing at
least one site in `S`.  Only clusters whose `x_c` actually changed need
the MLP re-evaluated.  ``build_cache(op)`` builds the site→affected and
edge→affected lookup tables for the operator's transition layout.

Training
--------
Train with :func:`pygfmc.wavefunction.train_with_netket` (NetKet, optional).
"""

from typing import NamedTuple

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from ._layout import require_spin_half_layout


def _tanh(x):
    return jnp.tanh(x)


# ── guide cache ─────────────────────────────────────────────────────────────

class ClusterMLPCache(NamedTuple):
    """Affected-cluster lookup tables for the smart log_diff path.

    site_to_clusters: tuple of (N, max_aff_g) int32 arrays, one per group g.
                      ``[a, k]`` is a cluster-index in group g containing site
                      a, or -1 for padding.
    pair_to_clusters: tuple of (n_jpm, max_aff_pair_g) int32 arrays, one per
                      group g.  ``[e, k]`` is a cluster-index in group g
                      containing edge_i[e] *or* edge_j[e], or -1 for padding.
                      (Empty tuple if op has n_jpm=0.)
    """
    site_to_clusters: tuple
    pair_to_clusters: tuple


# ── core class ───────────────────────────────────────────────────────────────

def _parse_mlp_groups(groups):
    """Split ``[[clusters, hidden], ...]`` into parallel lists.

    Both fields are required, for the same reason as in ClusterCPS: a shorter
    form would leave the call site guessing.
    """
    if isinstance(groups, dict) or not isinstance(groups, (list, tuple)):
        raise ValueError(
            f"groups must be a list of [clusters, hidden] specs; "
            f"got {type(groups).__name__}")
    if len(groups) == 0:
        raise ValueError("groups must contain at least one cluster group")
    # a single spec passed unwrapped would otherwise look like two groups
    if (len(groups) == 2 and not isinstance(groups[0], (list, tuple))
            and np.ndim(groups[0]) == 2
            and (groups[1] is None or np.ndim(groups[1]) <= 1)):
        raise ValueError(
            "groups looks like one [clusters, hidden] spec passed unwrapped; "
            "wrap it in a list of groups: ClusterMLP([[clusters, hidden]])")

    clusters, hidden = [], []
    for g_idx, spec in enumerate(groups):
        if not isinstance(spec, (list, tuple)) or len(spec) != 2:
            raise ValueError(
                f"group {g_idx}: a group spec must be a 2-field list "
                f"[clusters, hidden]; got {type(spec).__name__}"
                + (f" of length {len(spec)}" if isinstance(spec, (list, tuple))
                   else ""))
        c, h = spec
        c = np.asarray(c)
        if c.ndim != 2:
            raise ValueError(
                f"group {g_idx}: clusters must be 2-D (n_clusters, L); "
                f"got shape {c.shape}")
        # hidden: this group's widths only -- a list of ints, one per hidden
        # layer.  A bare integer is deliberately NOT accepted: it means the
        # same as ``[w]`` but leaves the depth implicit at the call site, and
        # the list form makes the number of layers readable directly from the
        # spec.  ``np.ndim`` (rather than ``np.isscalar``) also catches a 0-d
        # array, which would otherwise fall through and fail on iteration.
        if h is None:
            h = []
        elif np.ndim(h) == 0:
            raise ValueError(
                f"group {g_idx}: hidden must be a list of layer widths, not a "
                f"bare integer; pass [{int(h)}] for a single hidden layer of "
                f"width {int(h)}")
        else:
            h = [int(x) for x in h]
        clusters.append(c)
        hidden.append(h)
    return clusters, hidden


_INIT_STD = 0.01   # std of the Gaussian weight initialisation


def _init_group_params(dims, rng):
    """Weights/biases for one group's network, layer dims ``dims``.

    Weights are always Gaussian (std ``_INIT_STD``), biases zero.  There is
    deliberately no all-zero option: with W=b=0 every MLP output is 0 AND
    d(out)/dW vanishes, so only the final bias can move -- and a constant
    shift of log psi is just a normalisation.  Such a guide could never train
    (unlike ClusterCPS, whose zero-initialised lookup entries are independent
    parameters with their own gradients).
    """
    layers = []
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        W = rng.normal(0, _INIT_STD, (d_in, d_out)).astype(np.float32)
        layers.append((W, np.zeros(d_out, dtype=np.float32)))
    return layers


class ClusterMLP:
    """Per-cluster MLP guide for SpinHalf, with one shared tanh MLP per cluster group.

    The log-amplitude is summed over groups ``g`` and their clusters ``c``:

        log ψ(x) = Σ_g Σ_{c ∈ clusters_g}  MLP_g( x[clusters[g][c]] )

    Args:
        groups:      list of ``[clusters, hidden]`` specs, one per group:
                     ``clusters`` is an ``(n_clusters_g, k_g)`` int array (all
                     clusters of a group share the size ``k_g``) and
                     ``hidden`` a list of hidden-layer widths (``[]`` or
                     ``None`` for a linear model; a bare int is rejected).
        params:      optional list, one entry per group, each a list of
                     ``(W_i, b_i)`` pairs for layers ``i = 0 … n_layers-1``.
                     The last layer is the output (no activation).  If
                     omitted, weights are drawn from a Gaussian of std
                     0.01 and the biases are zero; if given, the layer
                     shapes must match ``groups``.
        seed:        RNG seed for the initialisation; ``None`` draws one,
                     kept as ``guide.seed`` (``None`` on a guide rebuilt
                     from its pytree, e.g. after training).
        dtype:       JAX dtype.
        N:           number of sites; defaults to the largest site index + 1
                     (pass it if trailing sites belong to no cluster).

    Operators: :meth:`log_diff_from_operator` and :meth:`build_cache` index
    their output by the ``SpinHalf`` transition layout, so they accept a
    :class:`~pygfmc.hamiltonian.SpinHalf` (Hamiltonian or measurement
    operator) or any operator exposing ``n_hx_static``, ``n_jpm_static``,
    ``edge_i`` and ``edge_j``; any other operator (``ToricCode``,
    ``PlaquetteModel``, ...) raises ``TypeError``.  :meth:`log_psi` and
    :meth:`log_diff_from_configs` take configurations of any spin-1/2
    operator.
    """

    # ── construction ────────────────────────────────────────────────────

    def __init__(self, groups, params=None,
                 seed=None, dtype=jnp.float32, N=None):
        self.dtype = dtype

        # ── group specs, mirroring ClusterCPS's ``groups`` ─────────────────
        # Each group is a mandatory 2-field [clusters, hidden]: the
        # ARCHITECTURE.  ``params`` carries the weights, exactly as
        # ``amplitudes`` does for ClusterCPS, and is initialised when omitted.
        clusters, hidden = _parse_mlp_groups(groups)

        # ── N: derived from the clusters unless given ──────────────────────
        # Every site index that appears must be < N, and build_cache() builds
        # per-site tables of length N, so an explicit N is still accepted for
        # lattices whose trailing sites sit in no cluster (e.g. diluted ones),
        # where max(index)+1 undercounts.
        n_derived = max(int(c.max()) for c in clusters) + 1
        if N is None:
            N = n_derived
        elif int(N) < n_derived:
            raise ValueError(
                f"N={int(N)} is smaller than the largest site index in "
                f"clusters ({n_derived - 1})")
        self.N = int(N)

        # ── parameters: supplied, or initialised from the architecture ─────
        if params is None:
            from ._seed import resolve_seed
            seed = resolve_seed(seed, context="ClusterMLP")
            rng = np.random.default_rng(seed)
            params = [
                _init_group_params([int(c.shape[1])] + list(h) + [1], rng)
                for c, h in zip(clusters, hidden)
            ]

        self.seed = seed     # the drawn seed reproduces a seed=None run

        if len(clusters) != len(params):
            raise ValueError(
                f"clusters ({len(clusters)}) and params ({len(params)}) "
                f"must have the same length"
            )
        self.n_groups = len(clusters)

        self.clusters = tuple(
            jnp.asarray(c, dtype=jnp.int32) for c in clusters
        )
        self.k_g   = tuple(int(c.shape[1]) for c in self.clusters)
        self.n_c_g = tuple(int(c.shape[0]) for c in self.clusters)

        # Normalise each group's params to a list of (W, b) jnp-array pairs.
        def _normalise(p):
            return [(jnp.asarray(W, dtype=dtype),
                     jnp.asarray(b, dtype=dtype)) for W, b in p]

        self.weights_g = tuple(_normalise(p) for p in params)  # ((W,b), …) per group

        # Supplied params must match the architecture; otherwise a mismatched
        # pair would silently build a different network than the one requested.
        for g, (c, h, wg) in enumerate(zip(clusters, hidden, self.weights_g)):
            want = [int(c.shape[1])] + list(h) + [1]
            got = [int(wg[0][0].shape[0])] + [int(W.shape[1]) for W, _ in wg]
            if want != got:
                raise ValueError(
                    f"group {g}: params describe layer dims {got}, but "
                    f"clusters/hidden imply {want}")

        # Derived shape info
        self.n_layers_g = tuple(len(wg) for wg in self.weights_g)
        # hidden widths = all layers except the last (output) layer
        self.hidden_g   = tuple(
            tuple(int(W.shape[1]) for W, _ in wg[:-1])
            for wg in self.weights_g
        )


    @property
    def n_params(self):
        """Total number of weights and biases over all groups."""
        return sum(W.size + b.size
                   for wg in self.weights_g for W, b in wg)

    # ── MLP kernels ─────────────────────────────────────────────────────

    def _mlp_g(self, g):
        """Return a callable ``f: (k_g,) → scalar`` for group g."""
        layers = self.weights_g[g]
        def f(x):
            for i, (W, b) in enumerate(layers):
                x = x @ W + b
                if i < len(layers) - 1:
                    x = _tanh(x)
            return jnp.squeeze(x)
        return f

    # ── log ψ ───────────────────────────────────────────────────────────


    def log_psi(self, config, cache=None):
        """log ψ(x) = Σ_g Σ_{c ∈ clusters_g} MLP_g(x[clusters[g][c]]).

        ``config`` may be a single ``(N,)`` config or a batch
        ``(n_configs, N)``; the return is a scalar or ``(n_configs,)``
        accordingly (same convention as :meth:`ClusterCPS.log_psi`).
        ``cache`` is accepted for protocol parity but unused — the
        cluster cache only accelerates :meth:`log_diff_from_operator`."""
        if config.ndim == 2:
            return jax.vmap(lambda c: self.log_psi(c))(config)
        σ = config.astype(self.dtype)
        total = jnp.array(0.0, dtype=self.dtype)
        for g in range(self.n_groups):
            sigma_c = σ[self.clusters[g]]                      # (n_c_g, k_g)
            y = jax.vmap(self._mlp_g(g))(sigma_c)              # (n_c_g,)
            total = total + jnp.sum(y)
        return total

    # ── guide protocol ──────────────────────────────────────────────────

    def log_diff_from_configs(self, config, new_configs, cache=None):
        """``log ψ(x') − log ψ(x)`` for each row of new_configs.

        Brute-force path: runs full log_psi per row. Memory: O(M·N) for
        new_configs.  Use ``log_diff_from_operator(..., cache=cache)``
        for the memory- and compute-light path when M can be huge.
        """
        base = self.log_psi(config)
        return jax.vmap(self.log_psi)(new_configs) - base

    def log_diff_from_operator(self, config, op, cache=None):
        """Log-amplitude ratios indexed by ``op``'s transition layout.

        Layout matches ``op.get_rates`` of a ``SpinHalf``:
          [diag (=0), n_hx site flips at sites 0..N-1, n_jpm op-edge flips]

        With ``cache=None``: falls back to
        ``log_diff_from_configs(config, op.get_rates(config, return_configs=True)[1])``.
        With a ``ClusterMLPCache`` from ``self.build_cache(op)``: uses the
        affected-cluster lookup to recompute MLPs only for clusters whose
        ``x_c`` actually changed under each transition.  ``op`` must expose
        the SpinHalf layout (``n_hx_static``, ``n_jpm_static``, ``edge_i``,
        ``edge_j``) either way; any other operator raises ``TypeError``.
        """
        require_spin_half_layout(op, type(self).__name__)
        if cache is None:
            _, new_configs = op.get_rates(config, return_configs=True)
            return self.log_diff_from_configs(config, new_configs)

        σ = config.astype(self.dtype)

        # Base per-cluster MLP outputs (one (n_c_g,) array per group)
        y_base = []
        for g in range(self.n_groups):
            sigma_c = σ[self.clusters[g]]
            y_g = jax.vmap(self._mlp_g(g))(sigma_c)
            y_base.append(y_g)

        M = 1 + op.n_hx_static + op.n_jpm_static
        out = jnp.zeros(M, dtype=self.dtype)

        # ── h_x: single-site flips ─────────────────────────────────────
        if op.n_hx_static > 0:
            def site_delta(site_a):
                # Flip only within the gathered cluster arrays, as the edge
                # path does: σ.at[site_a] under a vmap over all N sites would
                # materialise an (N, N) array per walker.
                total = jnp.array(0.0, dtype=self.dtype)
                for g in range(self.n_groups):
                    aff = cache.site_to_clusters[g][site_a]   # (max_aff_g,) padded
                    mask = aff >= 0
                    aff_safe = jnp.where(mask, aff, 0)
                    site_ids = self.clusters[g][aff_safe]     # (max_aff_g, k_g)
                    sigma_c = σ[site_ids]
                    sigma_c_new = jnp.where(site_ids == site_a, -sigma_c, sigma_c)
                    y_new = jax.vmap(self._mlp_g(g))(sigma_c_new)
                    y_old = y_base[g][aff_safe]
                    delta = jnp.where(mask, y_new - y_old, 0.0)
                    total = total + jnp.sum(delta)
                return total

            site_deltas = jax.vmap(site_delta)(jnp.arange(self.N))
            out = out.at[1 : 1 + self.N].set(site_deltas)

        # ── J_pm: edge flips ──────────────────────────────────────────
        if op.n_jpm_static > 0:
            edge_i = jnp.asarray(op.edge_i, dtype=jnp.int32)
            edge_j = jnp.asarray(op.edge_j, dtype=jnp.int32)
            def edge_delta(e):
                # Avoid creating a full (N,) σ_new — flip only within cluster arrays.
                i = edge_i[e]
                j = edge_j[e]
                total = jnp.array(0.0, dtype=self.dtype)
                for g in range(self.n_groups):
                    aff = cache.pair_to_clusters[g][e]
                    mask = aff >= 0
                    aff_safe = jnp.where(mask, aff, 0)
                    site_ids   = self.clusters[g][aff_safe]          # (T, k)
                    sigma_c    = σ[site_ids]                         # (T, k)
                    flip       = (site_ids == i) | (site_ids == j)
                    sigma_c_new = jnp.where(flip, -sigma_c, sigma_c)
                    y_new = jax.vmap(self._mlp_g(g))(sigma_c_new)
                    y_old = y_base[g][aff_safe]
                    delta = jnp.where(mask, y_new - y_old, 0.0)
                    total = total + jnp.sum(delta)
                return total

            edge_deltas = jax.vmap(edge_delta)(jnp.arange(op.n_jpm_static))
            out = out.at[1 + op.n_hx_static :].set(edge_deltas)

        return out

    def build_cache(self, op):
        """Build a ``ClusterMLPCache`` for fast ``log_diff_from_operator(op)``.

        For each group g, computes:
          - ``site_to_clusters[g]`` of shape ``(N, max_aff_g)`` — for each
            site, the cluster indices of group g containing that site
            (padded with -1 to a uniform max length).
          - ``pair_to_clusters[g]`` of shape ``(n_jpm, max_aff_pair_g)`` —
            for each op-edge ``(edge_i[e], edge_j[e])``, the union of
            cluster indices containing either endpoint.

        Memoised by the byte content of ``op``'s edge layout.  ``op`` must
        expose the SpinHalf layout (``n_hx_static``, ``n_jpm_static``,
        ``edge_i``, ``edge_j``); any other operator raises ``TypeError``.
        """
        require_spin_half_layout(op, type(self).__name__)
        if not hasattr(self, '_cache'):
            self._cache = {}
        ei = np.asarray(op.edge_i) if op.n_jpm_static > 0 else np.zeros(0, dtype=np.int32)
        ej = np.asarray(op.edge_j) if op.n_jpm_static > 0 else np.zeros(0, dtype=np.int32)
        key = (ei.tobytes(), ej.tobytes(), int(op.n_hx_static),
               int(op.n_jpm_static))
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        site_to_clusters = []
        for g in range(self.n_groups):
            clusters_g = np.asarray(self.clusters[g])
            site_to_cs = [[] for _ in range(self.N)]
            for c_idx, sites in enumerate(clusters_g):
                for s in sites:
                    site_to_cs[int(s)].append(c_idx)
            max_aff = max((len(x) for x in site_to_cs), default=0)
            padded = np.full((self.N, max_aff), -1, dtype=np.int32)
            for s, cs in enumerate(site_to_cs):
                padded[s, :len(cs)] = cs
            site_to_clusters.append(jnp.asarray(padded, dtype=jnp.int32))

        pair_to_clusters = []
        if op.n_jpm_static > 0:
            for g in range(self.n_groups):
                site_map = np.asarray(site_to_clusters[g])
                pair_aff = []
                for e in range(len(ei)):
                    aff_i = set(int(c) for c in site_map[int(ei[e])] if c >= 0)
                    aff_j = set(int(c) for c in site_map[int(ej[e])] if c >= 0)
                    pair_aff.append(sorted(aff_i | aff_j))
                max_aff = max((len(x) for x in pair_aff), default=0)
                padded = np.full((len(ei), max_aff), -1, dtype=np.int32)
                for e, cs in enumerate(pair_aff):
                    padded[e, :len(cs)] = cs
                pair_to_clusters.append(jnp.asarray(padded, dtype=jnp.int32))

        cache = ClusterMLPCache(
            site_to_clusters=tuple(site_to_clusters),
            pair_to_clusters=tuple(pair_to_clusters),
        )
        self._cache[key] = cache
        return cache


    # ── HDF5 ────────────────────────────────────────────────────────────

    def save(self, path, group='cluster_mlp', **attrs):
        """Write clusters + per-group variable-depth MLP params to HDF5."""
        from ..walkers._h5io import write_attrs
        with h5py.File(path, 'w') as f:
            root = f.require_group(group)
            root.attrs['n_groups'] = self.n_groups
            for g in range(self.n_groups):
                sub = root.require_group(f'group_{g}')
                sub.create_dataset('clusters', data=np.asarray(self.clusters[g]))
                sub.attrs['n_layers'] = self.n_layers_g[g]
                sub.attrs['k_g']      = self.k_g[g]
                sub.attrs['n_c_g']    = self.n_c_g[g]
                for i, (W, b) in enumerate(self.weights_g[g]):
                    sub.create_dataset(f'W{i}', data=np.asarray(W))
                    sub.create_dataset(f'b{i}', data=np.asarray(b))
            base = dict(N=self.N, n_groups=self.n_groups,
                        k_g=list(self.k_g))
            base.update(attrs)
            write_attrs(root, base)

    @classmethod
    def load(cls, path, group='cluster_mlp', dtype=jnp.float32):
        """Load a guide written by :meth:`save`."""
        with h5py.File(path, 'r') as f:
            root     = f[group]
            n_groups = int(root.attrs['n_groups'])
            N        = int(root.attrs['N'])
            groups, params = [], []
            for g in range(n_groups):
                sub = root[f'group_{g}']
                n_layers = int(sub.attrs['n_layers'])
                layers = [(np.asarray(sub[f'W{i}'][()]),
                           np.asarray(sub[f'b{i}'][()]))
                          for i in range(n_layers)]
                # hidden widths = output width of every layer but the last
                groups.append([np.asarray(sub['clusters'][()]),
                               [int(W.shape[1]) for W, _ in layers[:-1]]])
                params.append(layers)
        return cls(groups, params=params, N=N, dtype=dtype)


# ── pytree registration ──────────────────────────────────────────────────────

def _flatten(obj):
    # Children: the true storage — per-group [(W, b), …] layer lists (nested
    # pytree, so any depth flattens correctly) and the cluster index arrays.
    children = (obj.weights_g, obj.clusters)
    aux = (obj.dtype, obj.N, obj.n_groups, obj.k_g, obj.n_c_g, obj.hidden_g,
           obj.n_layers_g)
    return children, aux


def _unflatten(aux, children):
    dtype, N, n_groups, k_g, n_c_g, hidden_g, n_layers_g = aux
    weights_g, clusters = children
    o = object.__new__(ClusterMLP)
    o.dtype     = dtype
    o.N         = N
    o.n_groups  = n_groups
    o.k_g       = k_g
    o.n_c_g     = n_c_g
    o.hidden_g  = hidden_g
    o.n_layers_g = n_layers_g
    o.weights_g = weights_g
    o.clusters  = clusters
    o.seed      = None    # not a pytree field; only the constructed guide knows it
    return o


jax.tree_util.register_pytree_node(ClusterMLP, _flatten, _unflatten)
