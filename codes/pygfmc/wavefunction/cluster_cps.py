"""General (multi-group) cluster Correlator Product State guide.

``ClusterCPS`` is a sum of CPS factors (cluster groups) with independent patch
sizes, symmetry reductions, parity and table sharing:

    log psi(sigma) = sum_g sum_p T_g[type_g(p), orbit_g(sigma[p])]

where ``orbit_g`` maps the binary encoding of the spins on patch ``p`` to its
orbit under group ``g``'s slot permutations (and optionally the global spin
flip), so symmetry-related patterns share one table entry.

The constructor takes ``groups`` as a list of cluster-group specifications.  The
list form has exactly three entries, all required:

    [clusters, symmetries, parity]

where ``clusters`` has shape ``(n_clusters, L)``, ``symmetries`` is an
``(n_auto, L)`` array of slot permutations or ``None``, and ``parity`` is a
bool.  Shorter list forms are rejected (they would be ambiguous); use a dict
spec to get defaults.

Dictionary specs are also accepted so callers can express parameter sharing:

    {
        "clusters": clusters,
        "symmetries": autos,          # optional, default None
        "parity": True,               # optional, default True
        "shared": False,              # optional, default True
        "cluster_type": types,        # optional explicit table id per patch
    }

Any other key raises ``ValueError``.  In particular the trained tables are
NOT part of a group spec: ``amplitudes`` is a top-level ``ClusterCPS``
argument taking a list with one table per group.

With ``shared=True`` all patches in a group share one amplitude table.  With
``shared=False`` each patch receives its own table.  Passing
``cluster_type`` gives explicit control.
"""

from typing import NamedTuple, Optional

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from ._layout import require_spin_half_layout

# ── orbit-mapping helper for symmetry-reduced lookup ─────────────────────────

def _compute_orbit_mapping(m, automorphisms, use_parity=False):
    """Reduce the ``2^m`` patch patterns by a closed permutation group.

    ``automorphisms`` is a ``(N_auto, m)`` int array; row ``i`` is a permutation
    of the patch slots; applying it sends bit pattern ``b`` to ``b[autos[i]]``.

    Assumes the rows form a closed group under composition; the identity is
    auto-prepended if not already present.

    Algorithm:
        (i)   enumerate the 2^m bit patterns                     (2^m, m)
        (ii)  apply every automorphism                           (2^m, N_auto, m)
        (iii) compute integer index of each permuted pattern     (2^m, N_auto)
        (iv)  minimum over the symmetry axis                     (2^m,)
        (v)   collect unique values → M = # orbits = # params
        (vi)  build idx_to_param[p] ∈ [0, M)                     (2^m,)

    If ``use_parity=True``, also apply the global spin flip x → −x to the
    permuted-bits array (equivalently ``bits → 1 − bits``), compute its
    minimum the same way, and take the overall minimum of the two, i.e.,
    enlarge the equivalence to ``automorphisms ∪ (automorphisms · parity)``,
    a group of size ``2 · N_auto``.

    Returns:
        idx_to_param: ``(2^m,)`` int32: orbit id of each pattern.
        param_to_rep: ``(M,)``   int32: the canonical (smallest) pattern in
                      each orbit, in orbit-id order.
        M: int, the number of distinct orbits / variational parameters per type.
    """
    autos = np.asarray(automorphisms, dtype=np.int64)
    if autos.ndim != 2 or autos.shape[1] != m:
        raise ValueError(
            f"automorphisms must have shape (N_auto, m={m}); got {autos.shape}"
        )
    if not np.array_equal(np.sort(autos, axis=1),
                          np.broadcast_to(np.arange(m), autos.shape)):
        raise ValueError(
            "every row of automorphisms must be a permutation of 0..m-1"
        )
    identity = np.arange(m, dtype=np.int64)
    if not any(np.array_equal(row, identity) for row in autos):
        autos = np.concatenate([identity[None, :], autos], axis=0)

    size   = 1 << m
    # int32 holds indices up to 2^31 − 1; safe for m ≤ 30. Promote to int64
    # if a user ever asks for a huge patch.
    idx_dt = np.int32 if m <= 30 else np.int64
    # bits are 0/1; int8 is enough (and shrinks the (2^m, m) buffer ~8×).
    bits   = ((np.arange(size, dtype=idx_dt)[:, None]
               >> np.arange(m,   dtype=idx_dt)[None, :]) & 1).astype(np.int8)
    powers = (1 << np.arange(m, dtype=idx_dt))                         # (m,)

    # We never materialise the (2^m, N_auto, m) tensor `bits_perm`. Instead
    # we do the matmul identity
    #     indices_perm[p, i] = Σ_j bits[p, autos[i, j]] · powers[j]
    #                        = Σ_k bits[p, k]            · Pinv[i, k]
    # with   Pinv[i, k] = powers[argsort(autos[i])][k]
    # (i.e. Pinv[i] is the auto-permuted powers vector). That gives
    # `indices_perm` directly as a 2D matmul, and chunking over the auto
    # axis keeps even this 2D buffer bounded.
    #
    # Parity is handled WITHOUT a flipped 3D tensor: with
    #   flipped_indices = (size − 1) − indices_perm
    # we have
    #   min_i flipped_indices[p, i] = (size − 1) − max_i indices_perm[p, i]
    # so we just track the running MAX alongside the running min.
    N_auto    = autos.shape[0]
    orbit_min = np.full(size, size, dtype=idx_dt)
    orbit_max = np.zeros(size, dtype=idx_dt) if use_parity else None
    # Aim for a per-chunk `sub_idx` buffer below ~64 MB.
    chunk = max(1, min(N_auto, (64 * 1024 * 1024) // (size * idx_dt(0).nbytes)))
    for start in range(0, N_auto, chunk):
        sub_autos = autos[start : start + chunk].astype(idx_dt, copy=False)
        Pinv      = powers[np.argsort(sub_autos, axis=1)]               # (k, m)
        sub_idx   = (bits @ Pinv.T).astype(idx_dt, copy=False)          # (2^m, k)
        np.minimum(orbit_min, sub_idx.min(axis=-1), out=orbit_min)
        if use_parity:
            np.maximum(orbit_max, sub_idx.max(axis=-1), out=orbit_max)

    if use_parity:
        orbit_rep = np.minimum(orbit_min, (size - 1) - orbit_max)
    else:
        orbit_rep = orbit_min

    unique_reps  = np.unique(orbit_rep)
    M            = int(unique_reps.size)
    idx_to_param = np.searchsorted(unique_reps, orbit_rep).astype(np.int32)
    param_to_rep = unique_reps.astype(np.int32)
    return idx_to_param, param_to_rep, M


class ClusterCPSGroup(NamedTuple):
    """Arrays and static sizes of one cluster group of a :class:`ClusterCPS`.

    ``amplitudes`` has shape ``(n_types, M)``; ``idx_to_param`` maps each of the
    ``2^m`` patch patterns to its orbit id in ``[0, M)``.
    """
    clusters: jnp.ndarray
    amplitudes: jnp.ndarray
    cluster_type: jnp.ndarray
    idx_to_param: jnp.ndarray
    param_to_rep: jnp.ndarray
    powers: jnp.ndarray
    automorphisms: Optional[jnp.ndarray]
    use_parity: bool
    n_patches: int
    m: int
    n_types: int
    M: int


class ClusterCPSGroupCache(NamedTuple):
    """Per-group tables for the index-arithmetic ``log_diff_from_operator``.

    ``amp_full`` is the ``(n_types, 2^m)`` table with the orbit map inlined;
    the ``site_*``/``pair_*`` arrays list, per flipped site / operator edge,
    the affected patches, their table ids and the ``2^slot`` of each flipped
    site in them (0 at padding).
    """
    amp_full: jnp.ndarray
    site_patch: jnp.ndarray
    site_type: jnp.ndarray
    site_pow: jnp.ndarray
    pair_patch: jnp.ndarray
    pair_type: jnp.ndarray
    pair_pow_i: jnp.ndarray
    pair_pow_j: jnp.ndarray


class ClusterCPSCache(NamedTuple):
    """Cache returned by :meth:`ClusterCPS.build_cache`: one
    :class:`ClusterCPSGroupCache` per group."""
    groups: tuple


def _parse_group_spec(spec):
    """Return ``(clusters, automorphisms, use_parity, nty, shared)``."""
    cluster_type = None
    shared = True

    if isinstance(spec, dict):
        if "clusters" not in spec:
            raise ValueError("group spec dict must contain 'clusters'")
        clusters = spec["clusters"]

        # Reject unknown keys.  Silently dropping them would let
        # `dict(clusters=…, automorphisms=…, amplitudes=…)` look like it
        # supplies trained parameters while `amplitudes` (a TOP-LEVEL __init__
        # argument, one entry per group) is thrown away -- `init="zero"` would
        # win and the guide would be the UNIFORM STATE (log psi == 0
        # everywhere) with no error.  A typo in any other key would fail the
        # same way.
        _known = {"clusters", "symmetries", "parity", "cluster_type",
                  "shared"}
        _bad = set(spec) - _known
        if _bad:
            hint = ""
            if "amplitudes" in _bad:
                hint = (" -- 'amplitudes' is a TOP-LEVEL ClusterCPS argument "
                        "taking a LIST with one table per group: "
                        "ClusterCPS(groups=[{...}], amplitudes=[table])")
            raise ValueError(
                f"unknown key(s) in ClusterCPS group spec: {sorted(_bad)}"
                f"{hint}. Known keys: {sorted(_known)}")

        automorphisms = spec.get("symmetries")
        # default True, matching the single-group class: parity (sigma -> -sigma)
        # is a symmetry of every sign-problem-free model this guide targets and
        # halves the table.  Set parity=False per group for a Hamiltonian that
        # breaks it (any longitudinal field), since imposing a symmetry the
        # Hamiltonian lacks biases the guide just as breaking one does.
        use_parity = bool(spec.get("parity", True))
        cluster_type = spec.get("cluster_type")
        shared = bool(spec.get("shared", True))
    else:
        # A list spec must give all three fields: a short form would be
        # ambiguous ([clusters, x] -- is x a symmetry array or a parity
        # flag?), and an implicit parity default could disagree with the dict
        # form's.  Pass a dict if you want defaults.
        if not isinstance(spec, (list, tuple)):
            raise ValueError(
                "a cluster group must be a [clusters, symmetries, parity] list "
                "or a dict; got a bare "
                f"{type(spec).__name__}.  Write [clusters, None, False] to "
                "state 'no symmetries, no parity reduction' explicitly."
            )
        if len(spec) != 3:
            raise ValueError(
                "a list group spec must give all three fields explicitly: "
                f"[clusters, symmetries, parity]; got {len(spec)}.  Pass "
                "symmetries=None for no cluster automorphisms and choose "
                "parity deliberately (True halves the table and is only "
                "correct if the Hamiltonian is parity symmetric)."
            )
        clusters, automorphisms, use_parity = spec[0], spec[1], spec[2]
        if not isinstance(use_parity, (bool, np.bool_)):
            raise ValueError(
                "the third field of a group spec is parity and must be a bool; "
                f"got {type(use_parity).__name__}"
            )
        use_parity = bool(use_parity)

    clusters = np.asarray(clusters, dtype=np.int64)
    if clusters.ndim != 2:
        raise ValueError(f"clusters must be 2-D; got {clusters.shape}")
    if clusters.shape[0] == 0:
        raise ValueError("each patch group must contain at least one patch")

    if automorphisms is not None:
        automorphisms = np.asarray(automorphisms, dtype=np.int32)

    if cluster_type is not None:
        cluster_type = np.asarray(cluster_type, dtype=np.int32)
        if cluster_type.shape != (clusters.shape[0],):
            raise ValueError(
                f"cluster_type shape {cluster_type.shape} must equal "
                f"({clusters.shape[0]},)"
            )

    return clusters, automorphisms, use_parity, cluster_type, shared


def _initial_amplitudes(n_types, M, init, rng, dtype=np.float32):
    if init == "zero":
        return np.zeros((n_types, M), dtype=dtype)
    if init == "gaussian":
        return (rng.normal(size=(n_types, M)) * 0.01).astype(dtype)
    raise ValueError(f"init must be 'zero' or 'gaussian'; got {init!r}")


def _normalise_amplitudes(amplitudes, n_types, M, group_idx, dtype=np.float32):
    amp = np.asarray(amplitudes, dtype=dtype)
    if amp.ndim == 1:
        if n_types != 1:
            raise ValueError(
                f"group {group_idx}: 1-D amplitudes only valid for n_types=1; "
                f"got n_types={n_types}"
            )
        amp = amp[None, :]
    if amp.shape != (n_types, M):
        raise ValueError(
            f"group {group_idx}: amplitudes must have shape {(n_types, M)}, "
            f"got {amp.shape}"
        )
    return amp


class ClusterCPS:
    """Cluster-CPS guide composed from one or more patch groups.

    Args:
        groups: list of patch-group specs (see the module docstring), or a
            single ``(n_patches, m)`` cluster array for a one-group guide.
        amplitudes: optional LIST of amplitude tables, one per group.  Each
            table has shape ``(n_types, M_g)`` (or ``(M_g,)`` if the group has
            one table) after that group's symmetry and parity reduction.  If
            omitted, amplitudes are initialised according to ``init``.  When
            ``groups`` is a bare cluster array a single table may be passed
            without the list.
        init: ``'zero'`` (default; the uniform state ``log psi == 0``) or
            ``'gaussian'`` (std 0.01) when ``amplitudes`` is omitted.
        seed: RNG seed for Gaussian initialisation; ``None`` draws one,
              kept as ``guide.seed`` (``None`` on a guide rebuilt from its
              pytree, e.g. after training).
        dtype: JAX dtype for amplitude arrays.

    Attributes: ``groups`` (tuple of :class:`ClusterCPSGroup`), ``n_groups``,
    ``N`` (largest site index + 1), ``n_params``.

    Operators: :meth:`log_diff_from_operator` and :meth:`build_cache` index
    their output by the ``SpinHalf`` transition layout, so they accept a
    :class:`~pygfmc.hamiltonian.SpinHalf` (Hamiltonian or measurement
    operator) or any operator exposing ``n_hx_static``, ``n_jpm_static``,
    ``edge_i`` and ``edge_j``; any other operator (``ToricCode``,
    ``PlaquetteModel``, ...) raises ``TypeError``.  :meth:`log_psi` and
    :meth:`log_diff_from_configs` take configurations of any spin-1/2
    operator.
    """

    def __init__(self, groups, amplitudes=None, init="zero", seed=None,
                 dtype=jnp.float32):
        # a bare (n_patches, m) array is a one-group spec with the defaults
        arr = np.asarray(groups, dtype=object) if isinstance(groups, (list, tuple)) else np.asarray(groups)
        if not isinstance(groups, (list, tuple)) and arr.ndim == 2:
            groups = [{"clusters": np.asarray(groups, dtype=np.int32)}]
            # a bare (n_types, M) or (M,) table is the one group's table
            if amplitudes is not None and not isinstance(amplitudes, (list, tuple)):
                amplitudes = [amplitudes]
        specs = list(groups)
        if not specs:
            raise ValueError("groups must contain at least one cluster group")
        # a single 3-field spec passed unwrapped would otherwise be read as
        # three separate groups
        if (len(specs) == 3 and isinstance(specs[2], (bool, np.bool_))
                and not isinstance(specs[0], dict)):
            raise ValueError(
                "groups looks like one [clusters, symmetries, parity] spec "
                "passed unwrapped; wrap it in a list of groups: "
                "ClusterCPS([[clusters, symmetries, parity]])"
            )

        if amplitudes is not None:
            if len(amplitudes) != len(specs):
                raise ValueError(
                    f"amplitudes must have one entry per patch group "
                    f"({len(specs)}); got {len(amplitudes)}"
                )

        # ``seed=None`` draws from the machine.  Resolved only when it is
        # actually consumed (gaussian init); the drawn seed is kept as
        # ``self.seed``, the record of an otherwise unreproducible run.
        if amplitudes is None and init == "gaussian":
            from ._seed import resolve_seed
            seed = resolve_seed(seed, context="ClusterCPS")
        self.seed = seed
        rng = np.random.default_rng(seed)
        groups = []
        for group_idx, spec in enumerate(specs):
            nbh, autos, use_parity, nty, shared = _parse_group_spec(spec)
            n_patches, m = int(nbh.shape[0]), int(nbh.shape[1])

            if nty is None:
                nty = (np.zeros(n_patches, dtype=np.int32) if shared
                       else np.arange(n_patches, dtype=np.int32))
            if nty.min() < 0:
                raise ValueError(f"group {group_idx}: cluster_type contains negative ids")
            n_types = int(nty.max()) + 1

            if autos is None and not use_parity:
                idx_to_param = np.arange(1 << m, dtype=np.int32)
                param_to_rep = idx_to_param.copy()
                M = 1 << m
            else:
                autos_for_M = (np.arange(m, dtype=np.int32)[None, :]
                               if autos is None else autos)
                idx_to_param, param_to_rep, M = _compute_orbit_mapping(
                    m, autos_for_M, use_parity=use_parity)

            if amplitudes is None:
                amp = _initial_amplitudes(n_types, M, init, rng, dtype)
            else:
                amp = _normalise_amplitudes(amplitudes[group_idx], n_types, M, group_idx, dtype)

            groups.append(ClusterCPSGroup(
                clusters=jnp.asarray(nbh, dtype=jnp.int32),
                amplitudes=jnp.asarray(amp, dtype=dtype),
                cluster_type=jnp.asarray(nty, dtype=jnp.int32),
                idx_to_param=jnp.asarray(idx_to_param, dtype=jnp.int32),
                param_to_rep=jnp.asarray(param_to_rep, dtype=jnp.int32),
                powers=jnp.asarray(np.array([1 << j for j in range(m)], dtype=np.int32),
                                   dtype=jnp.int32),
                automorphisms=(jnp.asarray(autos, dtype=jnp.int32)
                               if autos is not None else None),
                use_parity=bool(use_parity),
                n_patches=n_patches,
                m=m,
                n_types=n_types,
                M=int(M),
            ))

        self.groups = tuple(groups)
        self.dtype = dtype
        self.n_groups = len(groups)
        self.N = max(int(np.asarray(g.clusters).max()) for g in groups) + 1

    @property
    def n_params(self):
        """Total number of amplitude-table entries over all groups."""
        return int(sum(np.asarray(g.amplitudes).size for g in self.groups))

    def _group_patch_indices(self, group, config):
        sigma = config.astype(jnp.int32)
        bits = (sigma + 1) >> 1
        x = bits[group.clusters]
        return jnp.sum(x * group.powers, axis=-1)


    def log_psi(self, configs, cache=None):
        """Evaluate ``log psi`` for one config ``(N,)`` or a batch ``(..., N)``.

        ``cache`` is accepted for protocol parity and ignored.
        """
        del cache

        def _one(config):
            total = jnp.array(0.0, dtype=self.dtype)
            for group in self.groups:
                idx = self._group_patch_indices(group, config)
                param_idx = group.idx_to_param[idx]
                values = group.amplitudes[group.cluster_type, param_idx]
                total = total + jnp.sum(values).astype(self.dtype)
            return total

        configs = jnp.asarray(configs)
        if configs.ndim == 1:
            return _one(configs)
        flat = configs.reshape((-1, configs.shape[-1]))
        out = jax.vmap(_one)(flat)
        return out.reshape(configs.shape[:-1])

    def log_diff_from_configs(self, config, new_configs, cache=None):
        """``log psi(sigma') - log psi(sigma)`` for each row of ``new_configs``."""
        base = self.log_psi(config, cache)
        return self.log_psi(new_configs, cache) - base

    def build_cache(self, op):
        """Build affected-patch caches for SpinHalf-like site and edge flips.

        ``op`` must expose the SpinHalf layout (``n_hx_static``,
        ``n_jpm_static``, ``edge_i``, ``edge_j`` and optionally ``N``); any
        other operator raises ``TypeError``.  Returns a
        :class:`ClusterCPSCache`.  The cache bakes in the current amplitudes,
        so rebuild it after changing them.  Not memoised.
        """
        require_spin_half_layout(op, type(self).__name__)
        op_N = int(getattr(op, "N", self.N))
        ei = np.asarray(op.edge_i) if op.n_jpm_static > 0 else np.zeros(0, dtype=np.int32)
        ej = np.asarray(op.edge_j) if op.n_jpm_static > 0 else np.zeros(0, dtype=np.int32)

        caches = []
        for group in self.groups:
            nbh = np.asarray(group.clusters)
            nty = np.asarray(group.cluster_type)

            site_to_patches_list = [[] for _ in range(op_N)]
            for patch_idx in range(nbh.shape[0]):
                for site in nbh[patch_idx]:
                    site_i = int(site)
                    if site_i >= op_N:
                        raise ValueError(
                            f"patch site {site_i} lies outside operator N={op_N}"
                        )
                    site_to_patches_list[site_i].append(patch_idx)

            max_aff = max((len(items) for items in site_to_patches_list), default=0)
            site_patch = np.zeros((op_N, max_aff), dtype=np.int32)
            site_pow = np.zeros((op_N, max_aff), dtype=np.int32)
            for site, patch_ids in enumerate(site_to_patches_list):
                for k, patch_idx in enumerate(patch_ids):
                    slot = int(np.where(nbh[patch_idx] == site)[0][0])
                    site_patch[site, k] = patch_idx
                    site_pow[site, k] = 1 << slot
            site_type = nty[site_patch] if max_aff else np.zeros((op_N, 0), dtype=np.int32)

            if op.n_jpm_static > 0:
                pair_aff = []
                for edge_idx in range(len(ei)):
                    aff_i = set(site_to_patches_list[int(ei[edge_idx])])
                    aff_j = set(site_to_patches_list[int(ej[edge_idx])])
                    pair_aff.append(sorted(aff_i | aff_j))
                max_aff_pair = max((len(items) for items in pair_aff), default=0)
                pair_patch = np.zeros((len(ei), max_aff_pair), dtype=np.int32)
                pair_pow_i = np.zeros((len(ei), max_aff_pair), dtype=np.int32)
                pair_pow_j = np.zeros((len(ei), max_aff_pair), dtype=np.int32)
                for edge_idx, patch_ids in enumerate(pair_aff):
                    for k, patch_idx in enumerate(patch_ids):
                        pair_patch[edge_idx, k] = patch_idx
                        slot_i = np.where(nbh[patch_idx] == int(ei[edge_idx]))[0]
                        slot_j = np.where(nbh[patch_idx] == int(ej[edge_idx]))[0]
                        if slot_i.size:
                            pair_pow_i[edge_idx, k] = 1 << int(slot_i[0])
                        if slot_j.size:
                            pair_pow_j[edge_idx, k] = 1 << int(slot_j[0])
                pair_type = (nty[pair_patch] if max_aff_pair
                             else np.zeros((len(ei), 0), dtype=np.int32))
            else:
                pair_patch = np.zeros((0, 0), dtype=np.int32)
                pair_type = np.zeros((0, 0), dtype=np.int32)
                pair_pow_i = np.zeros((0, 0), dtype=np.int32)
                pair_pow_j = np.zeros((0, 0), dtype=np.int32)

            amp_full = jnp.take(group.amplitudes, group.idx_to_param, axis=1)
            caches.append(ClusterCPSGroupCache(
                amp_full=amp_full,
                site_patch=jnp.asarray(site_patch, dtype=jnp.int32),
                site_type=jnp.asarray(site_type, dtype=jnp.int32),
                site_pow=jnp.asarray(site_pow, dtype=jnp.int32),
                pair_patch=jnp.asarray(pair_patch, dtype=jnp.int32),
                pair_type=jnp.asarray(pair_type, dtype=jnp.int32),
                pair_pow_i=jnp.asarray(pair_pow_i, dtype=jnp.int32),
                pair_pow_j=jnp.asarray(pair_pow_j, dtype=jnp.int32),
            ))

        return ClusterCPSCache(groups=tuple(caches))


    def log_diff_from_operator(self, config, op, cache=None):
        """Log-amplitude ratios in the SpinHalf transition layout.

        Layout matches ``op.get_rates``: ``[diag (=0), n_hx site flips,
        n_jpm edge flips]``.  With a :class:`ClusterCPSCache` from
        :meth:`build_cache` each affected patch index is updated by index
        arithmetic (``idx_new = idx_old ± 2^slot``) without materialising any
        successor configuration; with ``cache=None`` the successors are built
        via ``op.get_rates(config, return_configs=True)``.  ``op`` must expose
        the SpinHalf layout either way; any other operator raises
        ``TypeError``.
        """
        require_spin_half_layout(op, type(self).__name__)
        if cache is None:
            _, new_configs = op.get_rates(config, return_configs=True)
            return self.log_diff_from_configs(config, new_configs)

        M = 1 + op.n_hx_static + op.n_jpm_static
        out = jnp.zeros(M, dtype=self.dtype)
        b_base = ((config.astype(jnp.int32) + 1) >> 1)

        for group, cache in zip(self.groups, cache.groups):
            amp_full = cache.amp_full
            idx_base = self._group_patch_indices(group, config)
            y_base = amp_full[group.cluster_type, idx_base]

            if op.n_hx_static > 0:
                def site_delta(site):
                    patches = cache.site_patch[site]
                    typ = cache.site_type[site]
                    pow_ = cache.site_pow[site]
                    sign = 1 - 2 * b_base[site]
                    idx_new = idx_base[patches] + sign * pow_
                    return jnp.sum(amp_full[typ, idx_new] - y_base[patches]).astype(self.dtype)

                site_deltas = jax.vmap(site_delta)(jnp.arange(op.n_hx_static))
                out = out.at[1 : 1 + op.n_hx_static].add(site_deltas)

            if op.n_jpm_static > 0:
                edge_i = jnp.asarray(op.edge_i, dtype=jnp.int32)
                edge_j = jnp.asarray(op.edge_j, dtype=jnp.int32)

                def edge_delta(edge):
                    patches = cache.pair_patch[edge]
                    typ = cache.pair_type[edge]
                    pow_i = cache.pair_pow_i[edge]
                    pow_j = cache.pair_pow_j[edge]
                    idx_new = (
                        idx_base[patches]
                        + (1 - 2 * b_base[edge_i[edge]]) * pow_i
                        + (1 - 2 * b_base[edge_j[edge]]) * pow_j
                    )
                    return jnp.sum(amp_full[typ, idx_new] - y_base[patches]).astype(self.dtype)

                edge_deltas = jax.vmap(edge_delta)(jnp.arange(op.n_jpm_static))
                out = out.at[1 + op.n_hx_static :].add(edge_deltas)

        return out

    def save(self, path, group="cps", **attrs):
        """Write the guide to ``path`` (overwritten) under ``group``.

        Each cluster group goes to a subgroup ``group_<i>`` (``clusters``,
        ``amplitudes``, ``cluster_type``, orbit tables, optional
        ``automorphisms``); ``n_groups``, ``N``, ``n_params`` and any extra
        ``**attrs`` are written as attributes of ``group``.
        """
        from ..walkers._h5io import write_attrs

        with h5py.File(path, "w") as h5:
            root = h5.require_group(group)
            for idx, patch_group in enumerate(self.groups):
                g = root.require_group(f"group_{idx}")
                g.create_dataset("clusters", data=np.asarray(patch_group.clusters))
                g.create_dataset("amplitudes", data=np.asarray(patch_group.amplitudes))
                g.create_dataset("cluster_type", data=np.asarray(patch_group.cluster_type))
                g.create_dataset("idx_to_param", data=np.asarray(patch_group.idx_to_param))
                g.create_dataset("param_to_rep", data=np.asarray(patch_group.param_to_rep))
                if patch_group.automorphisms is not None:
                    g.create_dataset("automorphisms", data=np.asarray(patch_group.automorphisms))
                write_attrs(g, {
                    "m": patch_group.m,
                    "n_patches": patch_group.n_patches,
                    "n_types": patch_group.n_types,
                    "M": patch_group.M,
                    "use_parity": bool(patch_group.use_parity),
                    "shared": bool(patch_group.n_types == 1),
                })
            base = {
                "n_groups": self.n_groups,
                "N": self.N,
                "n_params": self.n_params,
            }
            base.update(attrs)
            write_attrs(root, base)


    @classmethod
    def load(cls, path, group="cps", dtype=jnp.float32):
        """Load a guide written by :meth:`save` from HDF5 group ``group``."""
        with h5py.File(path, "r") as h5:
            root = h5[group]
            n_groups = int(root.attrs["n_groups"])
            specs = []
            amplitudes = []
            for idx in range(n_groups):
                g = root[f"group_{idx}"]
                spec = {
                    "clusters": np.asarray(g["clusters"][()]),
                    "symmetries": (np.asarray(g["automorphisms"][()])
                                   if "automorphisms" in g else None),
                    "parity": bool(g.attrs["use_parity"]),
                    "cluster_type": np.asarray(g["cluster_type"][()]),
                }
                specs.append(spec)
                amplitudes.append(np.asarray(g["amplitudes"][()]))
        return cls(specs, amplitudes=amplitudes, dtype=dtype)


# ── pytree registration ──────────────────────────────────────────────────────
# Required so ClusterCPS can be passed as a traced argument to the jitted GFMC
# evolution.  Array fields of each
# ClusterCPSGroup are children; static ints/bools and the optional
# automorphism table (encoded as hashable bytes) live in aux.

def _ccpsmulti_flatten(obj):
    children = tuple((g.clusters, g.amplitudes, g.cluster_type,
                      g.idx_to_param, g.param_to_rep, g.powers)
                     for g in obj.groups)
    autos_keys = tuple(
        None if g.automorphisms is None
        else (tuple(g.automorphisms.shape),
              np.asarray(g.automorphisms, dtype=np.int32).tobytes())
        for g in obj.groups)
    static = tuple((g.use_parity, g.n_patches, g.m, g.n_types, g.M)
                   for g in obj.groups)
    aux = (obj.dtype, obj.n_groups, obj.N, autos_keys, static)
    return children, aux


def _ccpsmulti_unflatten(aux, children):
    dtype, n_groups, N, autos_keys, static = aux
    groups = []
    for arrs, akey, st in zip(children, autos_keys, static):
        nbh, amp, nty, i2p, p2r, powers = arrs
        # numpy, not jnp: unflatten runs inside traces, where jnp.asarray
        # would give a tracer that the flatten above cannot turn into bytes
        autos = (None if akey is None else
                 np.frombuffer(akey[1], dtype=np.int32)
                 .reshape(akey[0]).copy())
        use_parity, n_patches, m, n_types, M = st
        groups.append(ClusterCPSGroup(
            clusters=nbh, amplitudes=amp, cluster_type=nty,
            idx_to_param=i2p, param_to_rep=p2r, powers=powers,
            automorphisms=autos, use_parity=use_parity,
            n_patches=n_patches, m=m, n_types=n_types, M=M))
    o = object.__new__(ClusterCPS)
    o.groups   = tuple(groups)
    o.dtype    = dtype
    o.n_groups = n_groups
    o.N        = N
    o.seed     = None     # not a pytree field; only the constructed guide knows it
    return o


jax.tree_util.register_pytree_node(ClusterCPS, _ccpsmulti_flatten, _ccpsmulti_unflatten)
