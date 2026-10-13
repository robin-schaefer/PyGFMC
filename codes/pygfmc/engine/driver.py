"""Single-device GFMC driver: :func:`run_gfmc`."""
import jax
import os as _os
import time
from functools import partial
from types import SimpleNamespace
import jax.numpy as jnp
import numpy as np
from typing import Mapping, Optional, Sequence, Union

from .evolution import (require_step_size, forward_walk_operator,
                         require_operator_protocol, require_guide_protocol,
                         require_incremental_support, guide_cache, require_lam,
                         initial_carried, evolve_carried, gather_carried, rates_fit_budget,
                         refresh_carried, refresh_carried_at)
from ..protocols import Operator, DiagonalOperator, Guide
from ..walkers import Walker
from ..walkers.walker import key_data
from ..walkers import reconfiguration
from .io import pytree_fingerprint
from .._output import emit
from .estimators import (_estimate_observables, estimate_average_log_branch_weight,
                         statistics, check_off_diag_operator_positivity,
                         count_unique_configs, _validated_p_array,
                         autocorrelation_stats)
from .._validation import _validated_int

# Warn when the always-on lineage diagnostics' extra evolution exceeds this
# fraction of a call's production steps (see the statistics() call below).
LINEAGE_COST_WARN_FRAC = 0.05

# Donate the diagonal ring-buffer carry to the compiled scan: halves its peak
# memory and is bit-identical either way, so it is unconditional.  Private
# seam for testing, not a user-facing option.
_DONATE_DIAG_CARRY = True

# Tags for the side-streams derived off the caller's key (the production
# stream itself is the caller's rng_key, used untouched).
# Distinct constants, and derived by fold_in rather than by splitting, so they
# are disjoint from the production stream's split-chain.
_AUX_POSITIVITY = 10_001
_AUX_STATISTICS = 10_002
# Tag of the root of every non-production stream (``_aux_root`` below).
_AUX_ROOT = 9_000


def _warmup_stage_key(aux_root, n_warmup, step_):
    """Side-stream of one warm-up ramp stage, derived from the stage's own
    step size and length (disjoint from the production stream).

    Known limitation: for the discrete methods with ``step_size < 8`` the
    first ramp stages floor to the same sub-step count, and two stages of
    equal length then draw the SAME stream.  That correlates the two stages'
    moves; they start from different ensembles, so it is not a bias."""
    return jax.random.fold_in(
        jax.random.fold_in(aux_root, int(n_warmup)), int(step_ * 1000))


# ---- compiled scans reusable across run_gfmc calls ----
# A jax.jit created inside run_gfmc (or an eager lax.scan over a fresh
# closure) is a NEW function every call, so every call -- and every distinct
# scan length within it -- paid a full trace + XLA compile.  Instead the step
# functions are built by module-level factories and the scans below are
# module-level jits: the Hamiltonian, guide, caches, operators, lam and alw go
# in as pytree ARGUMENTS, split into their array leaves (traced) and
# everything else -- the treedefs, Python scalars, the static run spec,
# unregistered user objects -- which forms one hashable static argument.
# A later call with the same structure, spec and shapes then hits jit's own
# cache.  Anything unhashable, or a leaf that the step function uses as a
# Python value (a trace-time concretisation error), falls back to closing
# over the objects, which is recompiled per call.
_TRACE_ERRORS = (jax.errors.ConcretizationTypeError,
                 jax.errors.TracerIntegerConversionError,
                 jax.errors.TracerArrayConversionError)
_UNCACHEABLE = set()

# The rates of every walker's current configuration CAN be carried across
# the step boundary (through reconfiguration) so that a step starts from
# them instead of evaluating them again; see evolution._current_rates.  It
# is off by default: on a CPU the saved evaluation measured no gain (a
# 60-site spin-1 chain with 2048 walkers ran 0.86-1.09 times as fast), and
# returning the final rates materialises them before the local-energy sum,
# which moves ``e_loc`` by one float32 ulp for some models.  Set
# PYGFMC_RATES_CARRY_MAX_BYTES to a byte limit (e.g. 268435456) to carry the
# (N_walkers, M_total) float array up to that size, e.g. to measure it on a
# GPU.  The guide state of an incremental guide (evolution.initial_carried)
# is carried regardless; the two form the carried pair ``(rates, state)``.
_RATES_CARRY_MAX_BYTES = int(_os.environ.get("PYGFMC_RATES_CARRY_MAX_BYTES", 0))


def _initial_carried(walkers, hamiltonian, guide, cache_h):
    """The carried pair ``(rates, state)`` of ``walkers``, evaluated once
    here; the rates are None when carrying them would exceed
    ``_RATES_CARRY_MAX_BYTES``."""
    n_walkers = walkers.config.shape[0]
    with_rates = rates_fit_budget(n_walkers, hamiltonian,
                                 walkers.log_weight.dtype, _RATES_CARRY_MAX_BYTES)
    return initial_carried(walkers, hamiltonian, guide, cache_h, with_rates=with_rates)


def _is_dynamic_leaf(x):
    return isinstance(x, (jax.Array, np.ndarray, np.generic))


def _partition(tree):
    """(hashable static part, list of array leaves) of a pytree."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    # (type, value) so that 1, 1.0 and True -- equal and equal-hashing --
    # do not share a cache entry (they trace to different weak types).
    static = tuple(None if _is_dynamic_leaf(leaf) else (type(leaf), leaf)
                   for leaf in leaves)
    return (treedef, static), [leaf for leaf in leaves if _is_dynamic_leaf(leaf)]


def _combine(static, dynamic):
    treedef, stat = static
    it = iter(dynamic)
    return jax.tree_util.tree_unflatten(
        treedef, [next(it) if s is None else s[1] for s in stat])


@partial(jax.jit, static_argnums=(0, 1))
def _scan_cached(make_step_fn, static, dynamic, carry, xs):
    return jax.lax.scan(make_step_fn(*_combine(static, dynamic)), carry, xs)


# The carry is unpacked so that ONLY the internally created ring buffers
# (parent_buf, diag_buf) are donated, never the caller's walkers.
@partial(jax.jit, static_argnums=(0, 1), donate_argnums=(5, 6))
def _scan_cached_donated(make_step_fn, static, dynamic, walkers_, rates_,
                         parent_buf_, diag_buf_, buf_idx_, key_, xs):
    return jax.lax.scan(make_step_fn(*_combine(static, dynamic)),
                        (walkers_, rates_, parent_buf_, diag_buf_, buf_idx_, key_), xs)


def _run_cached_scan(make_step_fn, objs, carry, xs, donate=False):
    """lax.scan(make_step_fn(*objs), carry, xs) through a cross-call jit cache.

    With ``donate`` the carry must be the six-slot diagonal carry
    (walkers, rates, parent_buf, diag_buf, buf_idx, key); slots 2 and 3 are
    donated.
    """
    static, dynamic = _partition(objs)
    try:
        hash(static)
        cache_key = (make_step_fn, static)
        cacheable = cache_key not in _UNCACHEABLE
    except TypeError:
        cacheable = False
    if cacheable:
        try:
            if donate:
                return _scan_cached_donated(make_step_fn, static, dynamic,
                                            *carry, xs)
            return _scan_cached(make_step_fn, static, dynamic, carry, xs)
        except _TRACE_ERRORS:
            # raised while tracing, i.e. before anything ran or was donated
            _UNCACHEABLE.add(cache_key)
    step_fn = make_step_fn(*objs)
    if donate:
        return jax.jit(
            lambda w, r, pb, db, bi, k, xs_: jax.lax.scan(
                step_fn, (w, r, pb, db, bi, k), xs_),
            donate_argnums=(2, 3))(*carry, xs)
    return jax.lax.scan(step_fn, carry, xs)


def _traced_alw(avg_log_weight, dtype):
    """``avg_log_weight`` as a 0-d array of the walkers' weight ``dtype``,
    so that it is a traced argument of the cached scans rather than part
    of their static key, with ONE signature whether the caller pinned a
    Python float or passed back the array a warm-up measured: a pinned
    value that changes between calls does not force a recompile, and a
    continuation with ``stats['avg_log_weight']`` hits the compiled scans
    of the run that produced it.  The value is the one the step would see
    anyway (a Python float is rounded to the weight dtype on entry)."""
    return jnp.asarray(avg_log_weight, dtype=dtype)


def _make_warmup_step_fn(hamiltonian, guide, lam, avg_log_weight, cache_h,
                         step_, method):
    def step_fn(carry, rng):
        walkers, carried = carry
        k1, k2 = jax.random.split(rng)
        walkers, carried = evolve_carried(walkers, carried, hamiltonian, step_, guide,
                                          method, k1, avg_log_weight, lam=lam,
                                          cache=cache_h)
        walkers, log_G_n, E_n, indices = reconfiguration(walkers, k2)
        return (walkers, gather_carried(carried, indices)), (log_G_n, E_n)
    return step_fn


def _make_prod_step_fn(hamiltonian, guide, lam, avg_log_weight, cache_h,
                       diag_ops, off_ops, cache_off, step_, method, m_array,
                       N_walker, n_p, dtype, gate_off_diag,
                       off_diag_batch_size, diag_group_idx):
    """One production step.  ``diag_ops``/``off_ops``/``cache_off`` are
    tuples in the caller's operator order (a dict would be re-ordered by key
    on the way through the pytree machinery)."""
    n_diag_ops = len(diag_ops)
    n_off_ops = len(off_ops)
    m_max = max(m_array) if m_array else 0
    def step_fn(carry, xs):
        measure_flag, step_abs = xs
        if n_diag_ops > 0:
            walkers, carried, parent_buf, diag_buf, buf_idx, key = carry
        else:
            walkers, carried, key = carry
        carried = refresh_carried_at(step_abs, carried, walkers, hamiltonian,
                                     guide, cache_h)

        # key_{t+1}, sub_t = split(key_t).  The position IS the key: there
        # is no step index in the derivation, so nothing has to be told
        # where in the chain this call starts.  `key` after the step is
        # exactly what the next step (in this call or the next) consumes,
        # which is what makes a resume continue rather than replay.
        key, rng = jax.random.split(key)
        k1 = jax.random.fold_in(rng, 0)
        k2 = jax.random.fold_in(rng, 1)
        k3 = jax.random.fold_in(rng, 2)
        walkers, carried = evolve_carried(walkers, carried, hamiltonian, step_, guide,
                                          method, k1, avg_log_weight, lam=lam,
                                          cache=cache_h)
        walkers, log_G_n, E_n, indices = reconfiguration(walkers, k2)
        carried = gather_carried(carried, indices)

        if n_diag_ops > 0:
            per_op = [jax.vmap(op.diagonal_weight)(walkers.config)
                      for op in diag_ops]
            # per_op[k] shape: (N_walker,) for scalar ops,
            # or (N_walker, n_vals) for vector-valued ops.
            # Normalise to (N_walker, n_vals) by adding a trailing dim for scalars.
            per_op = [v if v.ndim > 1 else v[..., None] for v in per_op]
            # One stack -- and one ring buffer -- PER WIDTH GROUP.  Operators
            # need not agree on how many values they return: a scalar and a
            # structure factor over n_k wavevectors simply land in different
            # buffers.  Padding them into one array would work too, but the
            # buffer is (n_ops, m_max+1, N_walker, n_vals) and is the binding
            # memory constraint at large scale, so a scalar must not be
            # inflated to the widest operator's width: grouping costs
            # sum(width), padding n_ops*max(width).
            per_group = [jnp.stack([per_op[i] for i in idx], axis=0)
                         for idx in diag_group_idx]

            buf_size = m_max + 1
            parent_buf = parent_buf.at[buf_idx].set(indices)
            diag_buf   = tuple(buf.at[:, buf_idx].set(vals)
                               for buf, vals in zip(diag_buf, per_group))

            dm_vals = [m_array[0]] + [m_array[i + 1] - m_array[i]
                                       for i in range(len(m_array) - 1)]
            ancestor = jnp.arange(N_walker, dtype=jnp.int32)
            steps_back = 0
            diag_estimates = []
            for dm, m in zip(dm_vals, m_array):
                # lax.fori_loop, NOT a Python `for _ in range(dm)`.
                # An unrolled loop emits dm sequentially data-dependent
                # gathers into the scan body, and the XLA CPU backend
                # can crash while compiling such a chain once a single
                # gap dm between consecutive m values exceeds ~40.  A
                # fori_loop emits one loop instead of dm copies, so the
                # chain depth is bounded regardless of p.  There is
                # nothing to fuse across these iterations anyway (each
                # gather needs the previous ancestor).
                base = steps_back

                def _walk_one(k, anc):
                    map_idx = (buf_idx - (base + k)) % buf_size
                    return parent_buf[map_idx][anc]

                ancestor = jax.lax.fori_loop(0, dm, _walk_one, ancestor)
                steps_back += dm
                old_idx = (buf_idx - m) % buf_size
                # one gather per width group.  The ancestry walk above is
                # shared: it is operator-independent, so grouping costs
                # extra gathers only, not extra walking.
                diag_estimates.append(tuple(
                    jnp.mean(buf[:, old_idx, ancestor], axis=1)
                    for buf in diag_buf))          # (n_ops_g, n_vals_g)

            # per group: (n_ops_g, n_vals_g, n_p).  The trailing width axis
            # is kept even for scalar operators and trimmed off by name
            # after the scan, where the widths are known.
            diag_vals = tuple(
                jnp.stack([est[g] for est in diag_estimates], axis=-1)
                for g in range(len(diag_buf)))
            buf_idx = (buf_idx + 1) % buf_size

        if n_off_ops > 0:
            def _measure_off_diag(_):
                # The shadow chain consumes the main chain's own future
                # step keys: from the (near-identical) operator-applied
                # ensemble it then nearly replays the main future, so the
                # population-level weight noise cancels between the
                # estimator's numerator and denominator (correlated
                # sampling).  Keys are state-independent, so expectations
                # are unchanged; an independent shadow stream would give
                # a much larger ratio variance at large p*step_size.
                # `key` is already the NEXT step's key, so the shadow
                # running the same recurrence from it reproduces the main
                # chain's future exactly -- across call boundaries too,
                # since the key is checkpointed with the walkers.
                pairs = [
                    forward_walk_operator(
                        walkers, op, hamiltonian, step_, guide, method,
                        jax.random.fold_in(k3, i), avg_log_weight, m_array,
                        lam=lam,
                        cache_h=cache_h,
                        cache_op=c_op,
                        shadow_key=key,
                        op_batch_size=off_diag_batch_size,
                    )
                    for i, (op, c_op) in enumerate(zip(off_ops, cache_off))
                ]
                return (jnp.stack([f for f, _ in pairs]),
                        jnp.stack([n for _, n in pairs]))
            if not gate_off_diag:
                off_diag_vals, off_diag_norms = _measure_off_diag(None)
            else:
                # lax.cond with a scalar predicate executes only the taken
                # branch at runtime, so unmeasured steps genuinely skip the
                # m_max-deep shadow-chain evolution (the dominant cost).
                off_diag_vals, off_diag_norms = jax.lax.cond(
                    measure_flag,
                    _measure_off_diag,
                    lambda _: (jnp.full((n_off_ops, n_p), jnp.nan, dtype=dtype),
                               jnp.full((n_off_ops, n_p), jnp.nan, dtype=dtype)),
                    None,
                )

        outputs = (log_G_n, E_n)
        if n_diag_ops > 0:
            outputs = outputs + (diag_vals,)
        if n_off_ops > 0:
            outputs = outputs + (off_diag_vals, off_diag_norms)

        if n_diag_ops > 0:
            return (walkers, carried, parent_buf, diag_buf, buf_idx, key), outputs
        else:
            return (walkers, carried, key), outputs
    return step_fn


def _parameter_stats(*, n_steps, walkers, step_size, lam, p_array, m_array,
                     warmup_steps, verbosity, number_of_outputs,
                     off_diag_stride, off_diag_offset, off_diag_equal_windows,
                     diag_operators, off_diag_operators, hamiltonian, guide,
                     method, rng_key_in, rng_key_next):
    """The ``stats['parameters/*']`` entries of a run.

    Shared by :func:`run_gfmc` and :func:`pygfmc.engine.multi_gpu.run_gfmc_pmap` so
    both backends record the same inputs and the same RNG stream:
    ``parameters/rng_key`` is where the production stream started and
    ``parameters/rng_key_next`` where it ended, the key a continuation passes
    back as ``rng_key``.  ``walkers`` only supplies shapes and dtypes.
    """
    diag_names = list(diag_operators or {})
    off_diag_names = list(off_diag_operators or {})
    return {
        "parameters/n_steps":              int(n_steps),
        "parameters/n_walkers":            int(walkers.config.shape[0]),
        "parameters/n_sites":              int(walkers.config.shape[1]),
        "parameters/step_size":            float(step_size),
        "parameters/lam":                  (float(lam) if lam is not None
                                            else float("nan")),
        "parameters/p_array":              np.asarray(p_array, dtype=np.int64),
        "parameters/m_array":              np.asarray(m_array, dtype=np.int64),
        "parameters/warmup_steps":         int(warmup_steps),
        "parameters/verbosity":            int(verbosity),
        "parameters/number_of_outputs":    int(number_of_outputs),
        "parameters/off_diag_stride":      int(off_diag_stride),
        "parameters/off_diag_offset":      int(off_diag_offset),
        "parameters/off_diag_equal_windows": int(bool(off_diag_equal_windows)),
        "parameters/n_diag_operators":     len(diag_names),
        "parameters/n_off_diag_operators": len(off_diag_names),
        # The operator NAMES, not just how many.  Two runs measuring the
        # same number of different operators would otherwise concatenate
        # into one dataset whose rows mean different things per run --
        # silently, since the shapes agree.
        "parameters/diag_operators":       np.asarray(diag_names,
                                                      dtype="S64"),
        "parameters/off_diag_operators":   np.asarray(off_diag_names,
                                                      dtype="S64"),
        # Content hashes alongside the names.  A name is a label, and every
        # label here can stay fixed while the object behind it changes --
        # a retrained guide written back to the same path, an edited
        # coupling, an operator moved to a different bond.  See
        # pygfmc.engine.io.pytree_fingerprint for what the hash does and does not
        # cover.
        "parameters/hamiltonian_hash":     np.asarray(
            pytree_fingerprint(hamiltonian), dtype="S32"),
        "parameters/guide_hash":           np.asarray(
            pytree_fingerprint(guide), dtype="S32"),
        "parameters/diag_operator_hashes": np.asarray(
            [pytree_fingerprint(op)
             for op in (diag_operators or {}).values()], dtype="S32"),
        "parameters/off_diag_operator_hashes": np.asarray(
            [pytree_fingerprint(op)
             for op in (off_diag_operators or {}).values()], dtype="S32"),
        "parameters/has_guide":            int(guide is not None),
        "parameters/method":               np.asarray(str(method), dtype="S32"),
        "parameters/guide":                np.asarray(
            type(guide).__name__ if guide is not None else "none", dtype="S32"),
        # dtypes matter for reproducing a stored run: the configuration and
        # the weights are typed independently (an int8 config with float32
        # weights is legitimate), and the weight WIDTH follows
        # jax_enable_x64 rather than any argument.
        "parameters/config_dtype":         np.asarray(
            str(np.dtype(walkers.config.dtype)), dtype="S32"),
        "parameters/weight_dtype":         np.asarray(
            str(np.dtype(walkers.log_weight.dtype)), dtype="S32"),
        "parameters/jax_enable_x64":       int(jax.config.jax_enable_x64),
        "parameters/hamiltonian":          np.asarray(
            type(hamiltonian).__name__, dtype="S32"),
        # --- the RNG stream.  There is no base key + counter:
        # ``rng_key`` is where this run's production stream STARTED and
        # ``rng_key_next`` where it ended, so appending a run to a file
        # can assert rng_key == previous rng_key_next -- an equality
        # test that proves the file is ONE unbroken stream, which is
        # strictly stronger than checking that two step ranges do not
        # overlap.  ``rng_key_next`` is what to pass as rng_key to continue.
        "parameters/rng_key":              key_data(rng_key_in),
        "parameters/rng_key_next":         key_data(rng_key_next),
    }


# ---- pieces shared by run_gfmc and run_gfmc_pmap ----
# Both drivers validate the same inputs, plan the same measurement grid, run
# the same warm-up ramp and assemble the same ``stats`` and ``raw`` dicts; the
# helpers below are that common part, so the two backends cannot drift.

#: Side-stream steps the positivity probe evolves before testing the
#: off-diagonal operators (see _positivity_probe).
_POSITIVITY_PROBE_STEPS = 100


def _measurement_plan(n_steps, p_array, n_off_ops, off_diag_stride,
                      off_diag_equal_windows, return_raw):
    """The window grid and the off-diagonal measurement grid of one call.

    ``p_array`` is the validated, sorted grid; ``m_array = p // 2``.

    A measurement at step k (0-based) is usable by window p only if p past
    main-chain weights and m = p/2 future ones exist IN THIS CALL, so by
    default the early and late steps serve only the smaller p: each p rests
    on a different number of measurements, drawn from a different slice of
    the chain.  That confounds a p-scan -- part of the apparent p-dependence
    is then a different sample rather than deeper projection.

    There are two ways out, and which one is available depends on whether
    the raw series survive the call:
      return_raw=True  -> raw['log_weights'] and raw['off_diag'] come back,
                          write_to_HDF5 APPENDS them into one top-level
                          /raw group, and evaluate_observables re-estimates
                          over the whole concatenated chain -- windows may
                          then span run boundaries and every p ends up with
                          the same set.  Gating here would merely throw
                          measurements away that the re-estimation could
                          have used, once per call, irrecoverably.
      return_raw=False -> nothing is kept, the per-call windowed numbers are
                          final, and the only way to make them comparable
                          across p is to gate the measurements now.
    Hence the DEFAULT: gate exactly when the raw series are not returned.
    A caller who takes `raw` but relies on the per-call numbers passes
    off_diag_equal_windows=True explicitly.

    The measurement grid is CALL-local: every stride-th step of this call.
    Rows carry their own step index (raw['off_diag_steps']), so the
    estimator reads the step instead of deriving it, any grid is legible,
    and the absolute numbering across calls is the writer's business
    (write_to_HDF5 offsets it by the rows already in the file).
    """
    sorted_p = list(p_array)
    m_array = [p // 2 for p in sorted_p]
    if off_diag_equal_windows is None:
        off_diag_equal_windows = not bool(return_raw)
    equal_windows = bool(off_diag_equal_windows)
    p_max = sorted_p[-1] if sorted_p else 0
    m_max_od = p_max // 2
    # What the final estimator needs, checked before anything is walked: a
    # window p needs p past weights, and an off-diagonal window also
    # m = p // 2 future ones, within this call.
    if p_max >= n_steps:
        raise ValueError(
            f"every p value must be smaller than n_steps: got n_steps={n_steps} "
            f"and max(p)={p_max}")
    if n_off_ops > 0 and n_steps <= p_max + m_max_od:
        raise ValueError(
            f"off-diagonal operators need n_steps > 1.5*max(p) "
            f"(max(p) + max(p)//2 = {p_max + m_max_od}: every window p needs "
            f"p past and p//2 future steps in this call), got n_steps={n_steps}")
    gate_off_diag = n_off_ops > 0 and (off_diag_stride > 1 or equal_windows)
    if equal_windows and n_off_ops > 0:
        # first gated step at or after the lead-in
        off_diag_offset = next(
            (k for k in range(p_max, p_max + off_diag_stride)
             if (k + 1) % off_diag_stride == 0), p_max)
        # The band usable by every p, [p_max, n_steps - m_max - 1], must hold
        # a stride step, or the run would walk everything and measure nothing.
        if not any((k + 1) % off_diag_stride == 0
                   for k in range(p_max, n_steps - m_max_od)):
            raise ValueError(
                f"equal off-diagonal windows (off_diag_equal_windows=True, the "
                f"default for return_raw=False) measure only the steps "
                f"{p_max}..{n_steps - m_max_od - 1} (n_steps={n_steps} minus "
                f"the lead-in max(p)={p_max} and the tail max(p)//2={m_max_od}), "
                f"and none of them is a multiple of off_diag_stride={off_diag_stride} "
                f"(counted from 1); raise n_steps or lower the stride")
    else:
        off_diag_offset = off_diag_stride - 1
    return SimpleNamespace(
        p_array=sorted_p, m_array=m_array, p_max=p_max,
        m_max=max(m_array) if m_array else 0,
        n_off_ops=n_off_ops, off_diag_stride=off_diag_stride,
        equal_windows=equal_windows, gate_off_diag=gate_off_diag,
        off_diag_offset=off_diag_offset)


def _measured_steps(step0, chunk, n_steps, plan):
    """The 0-based steps of a chunk and which of them measure off-diagonal
    operators: those with (t + 1) % off_diag_stride == 0 (stride 1 -> every
    step), restricted with equal windows to the band usable by every p.  The
    lead-in/tail band is per call by construction: a call can only see its
    own past and future weights."""
    t = step0 + np.arange(chunk)
    measured = (t + 1) % plan.off_diag_stride == 0
    if plan.equal_windows and plan.n_off_ops > 0:
        measured &= (t >= plan.p_max) & (t <= n_steps - plan.p_max // 2 - 1)
    return t, measured


def _warmup_schedule(method, step_f, warmup_steps):
    """The ``(step size, step count)`` of each warm-up ramp stage.

    ``warmup_steps`` is split over four stages of growing step size: 1/8,
    1/4, 1/2 and the full ``step_f`` (sub-step counts floored to at least 1
    for the discrete methods).  Empty when ``warmup_steps == 0``.
    """
    if warmup_steps <= 0:
        return []
    base, extra = divmod(warmup_steps, 4)
    counts = [base + (i < extra) for i in range(4)]
    if method == "cont_time":
        sizes = [step_f / 8.0, step_f / 4.0, step_f / 2.0, step_f]
    else:
        n_bra = int(step_f)
        sizes = [float(max(1, n_bra // 8)), float(max(1, n_bra // 4)),
                 float(max(1, n_bra // 2)), float(n_bra)]
    return [(size, count) for size, count in zip(sizes, counts) if count > 0]


def _off_by_name(arr, names):
    """Dense ``(n_rows, n_ops, n_p)`` -> ``{name: (n_rows, n_p)}``.

    Keyed by operator name like ``raw['diag']``.  Unlike the diagonal series
    these always share a shape, so ``_estimate_observables`` and the HDF5
    writer rebuild the dense array at their own boundary.
    """
    if arr is None:
        return None
    return {name: np.asarray(arr)[:, i] for i, name in enumerate(names)}


def _lineage_warning(m_array, n_steps, verbosity, prefix, tail):
    """Warn when the end-of-run lineage pass is a real fraction of the call.

    The lineage diagnostics are not a separate thing: the SAME statistics()
    pass that measures sampler health also tracks the offspring distribution
    and the ancestor survival at this run's m_array, extending its evolution
    from M=10 to max(10, m_max) branching steps.  Those extra steps are the
    one cost of always-on lineage, and they are charged per driver CALL, not
    per production step -- negligible for a long run, but a self-looping
    runner with a deep p grid and short calls can spend a real fraction of
    its budget here, so say so rather than letting it hide in the
    wall-clock.
    """
    extra = max(0, max(m_array) - 10) if m_array else 0
    if extra > LINEAGE_COST_WARN_FRAC * n_steps:
        emit(f"{prefix} WARNING: lineage diagnostics add {extra} evolution "
             f"steps to this call's {n_steps} production steps "
             f"({100.0 * extra / n_steps:.1f}% > "
             f"{100.0 * LINEAGE_COST_WARN_FRAC:.0f}%){tail}",
             verbosity, flush=True)


def _run_stats(*, walkers, hamiltonian, step_f, guide, method, aux_root,
               avg_log_weight, lam, lam_given, cache_h, plan, n_steps,
               warmup_steps, verbosity, number_of_outputs, diag_operators,
               off_diag_operators, rng_key_in, rng_key_next, E_np, weights_np,
               diag_np, off_diag_np, production_runtime, wall_start,
               number_of_devices, avg_log_weight_given=None):
    """The ``stats`` dict of a run: the end-of-run :func:`statistics` pass,
    the run's own inputs (``parameters/*``), the autocorrelation times of
    every series and the timings.

    ``avg_log_weight`` is the shift as the production scans received it (a
    0-d array, :func:`_traced_alw`); ``avg_log_weight_given`` is the value
    before that conversion (the caller's pinned Python float, or the array
    the warm-up measured) and is what ``stats['avg_log_weight']`` records.

    ``stats`` is built unconditionally: it is the diagnostic record of the
    run and none of it depends on keeping the per-step series (``raw`` is
    what ``return_raw`` gates).  The statistics pass runs on ``walkers``
    without touching it -- pytrees are immutable and its own advanced copy
    is discarded -- so the walkers a driver returns stay the production-final
    state.
    """
    diag_names = list(diag_operators or {})
    off_diag_names = list(off_diag_operators or {})
    k_stat_end = jax.random.fold_in(aux_root, _AUX_STATISTICS)
    _, end_stats = statistics(walkers, hamiltonian, step_f, guide, method,
                              k_stat_end, avg_log_weight, lam=lam,
                              quiet=(verbosity < 1), m_array=plan.m_array,
                              cache=cache_h)
    total_wall_time = time.time() - wall_start
    stats = dict(end_stats)
    # The run's own inputs, so a returned stats dict -- and a file written
    # from it -- identifies the run that produced it.
    stats.update(_parameter_stats(
        n_steps=n_steps, walkers=walkers, step_size=step_f, lam=lam_given,
        p_array=plan.p_array, m_array=plan.m_array, warmup_steps=warmup_steps,
        verbosity=verbosity, number_of_outputs=number_of_outputs,
        off_diag_stride=plan.off_diag_stride,
        off_diag_offset=plan.off_diag_offset,
        off_diag_equal_windows=plan.equal_windows,
        diag_operators=diag_operators, off_diag_operators=off_diag_operators,
        hamiltonian=hamiltonian, guide=guide, method=method,
        rng_key_in=rng_key_in, rng_key_next=rng_key_next))
    # tau_int per series: converts steps into independent samples, and is
    # what bounds a usable off_diag_stride from below.  Cheap enough to do
    # unconditionally -- ~150 ms for a 1e6-step series, and the per-p
    # observable tables add ~30 ms each.
    stats.update(autocorrelation_stats(
        E_np, weights_np,
        diag_np=diag_np, diag_names=diag_names or None,
        off_diag_np=off_diag_np, off_diag_names=off_diag_names or None,
        off_diag_stride=plan.off_diag_stride))
    stats.update(it_per_sec=n_steps / production_runtime,
                 runtime=production_runtime,
                 total_wall_time=total_wall_time,
                 number_of_devices=number_of_devices,
                 avg_log_weight=float(avg_log_weight if avg_log_weight_given is None
                                      else avg_log_weight_given))
    return stats


def _raw_series(*, weights_np, E_np, diag_np, off_diag_np, off_diag_norm_np,
                off_diag_steps_np, additional_diag_np, plan, diag_names,
                off_diag_names):
    """The ``raw`` dict of a run: the per-step series plus the metadata that
    makes them self-describing (``evaluate_observables(raw)`` needs no
    further arguments, the same way it needs none for the dict
    ``load_raw_from_HDF5`` reads back from a file).

    The operator arrays are indexed by POSITION in the grid --
    ``off_diag[name][:, j]`` is the value at window ``p_array[j]`` -- and
    nothing in the array itself says so, hence the grid and the operator
    names travel with the series.  ``off_diag_steps`` is the LOCAL main-chain
    step each off-diagonal row was launched at, stored with the rows so the
    estimator reads the position instead of deriving it from a uniform grid;
    that is what frees the grid to be per-call and lets rows from several
    runs be concatenated.  ``additional_diag`` holds the diagonal values of
    the m_max steps FOLLOWING the run, traced with the run's ancestry
    history: the lead-in a continuation needs to fill its own first rows.
    """
    return dict(log_weights=weights_np, energy=E_np,
                diag=diag_np, off_diag=off_diag_np,
                off_diag_norm=off_diag_norm_np,
                off_diag_steps=off_diag_steps_np,
                additional_diag=additional_diag_np,
                p_array=list(plan.p_array), m_array=list(plan.m_array),
                diag_operators=list(diag_names) or None,
                off_diag_operators=list(off_diag_names) or None)


# ---- single-device stages ----

def _warmup_stage(walkers, carried, objs, dtype, aux_root, step_, n_warmup,
                  avg_log_weight):
    """One warm-up ramp stage: ``n_warmup`` steps of size ``step_``.

    Returns the walkers and their carried pair, the shift measured over the
    stage (``avg_log_weight`` plus the mean log branch weight per unit step)
    and the elapsed seconds.  A single jax.lax.scan over all n_warmup steps
    materialises the multi_step geometric-hold buffer (shape ~(max_k,
    N_walker, N_sites)) once per compiled scan body; at large N_walker with a
    tight lambda-margin (large max_k) this can exceed GPU memory even though
    each individual step is cheap.  So the scan is chunked with a host sync
    between chunks.
    """
    # One side-stream per ramp stage (disjoint from production).
    subkey = _warmup_stage_key(aux_root, n_warmup, step_)
    start = time.time()
    _CHUNK = 50
    remaining = n_warmup
    log_weight_sum = jnp.array(-jnp.inf, dtype=dtype)
    while remaining > 0:
        m = min(_CHUNK, remaining)
        subkey, k = jax.random.split(subkey)
        rngs = jax.random.split(k, m)
        carried = refresh_carried(carried, walkers, objs[0], objs[1], objs[4])
        (walkers, carried), (log_weights_chunk, _) = _run_cached_scan(
            _make_warmup_step_fn, objs, (walkers, carried), rngs)
        log_weight_sum = jnp.logaddexp(
            log_weight_sum, jax.scipy.special.logsumexp(log_weights_chunk))
        remaining -= m
    measured = avg_log_weight + (
        log_weight_sum - jnp.log(jnp.asarray(n_warmup, dtype=dtype))) / step_
    return walkers, carried, measured, time.time() - start


def _warmup(walkers, carried, hamiltonian, guide, method, step_f, lam, cache_h,
            avg_log_weight, warmup_steps, aux_root, dtype, verbosity):
    """Equilibrate the ensemble and measure ``avg_log_weight``.

    Those are two different jobs: the ramp always runs (subject to
    ``warmup_steps``), and only the shift UPDATE is conditional, so a pinned
    value is carried through untouched while the walkers are still
    equilibrated.  ``carried`` is the walkers' carried pair ``(rates, state)``.
    Returns ``(walkers, carried, avg_log_weight)``.
    """
    alw_pinned = avg_log_weight is not None
    if not alw_pinned:
        avg_log_weight = estimate_average_log_branch_weight(
            walkers, hamiltonian, guide, method=method, lam=lam, cache=cache_h)
    stages = _warmup_schedule(method, step_f, warmup_steps)
    for i, (step_, n_warmup) in enumerate(stages):
        objs = (hamiltonian, guide, lam, _traced_alw(avg_log_weight, dtype), cache_h,
                step_, method)
        walkers, carried, measured_alw, elapsed = _warmup_stage(
            walkers, carried, objs, dtype, aux_root, step_, n_warmup, avg_log_weight)
        label = f"{i + 1}/{len(stages)}"
        if method == "cont_time":
            emit(f"[gfmc] - [run_warmup {label}] - step={step_}, {n_warmup} "
                 f"steps @ {n_warmup / elapsed:.1f} it/s — avg_log_weight = "
                 f"{float(measured_alw):.6f}", verbosity, 8)
        else:
            emit(f"[gfmc] - [run_warmup {label}] - step={step_}, {n_warmup} "
                 f"steps @ {n_warmup / elapsed:.1f} it/s — average_b_x = "
                 f"{float(jnp.exp(measured_alw)):.6f}", verbosity, 8)
        if alw_pinned:
            # keep the caller's value; report the drift the ramp measured,
            # which is the honest diagnostic (a locked-on alw is
            # self-fulfilling, a FROZEN one shows its residual)
            emit(f"[gfmc] - [run_warmup {label}] - alw pinned at "
                 f"{float(avg_log_weight):.6f}; this stage would have "
                 f"measured {float(measured_alw):.6f}", verbosity, 8)
        else:
            avg_log_weight = measured_alw
    if not stages:
        emit("[gfmc] - [run_gfmc] - no warmup (warmup_steps=0), "
             f"avg_log_weight={float(avg_log_weight):.6f}", verbosity, 8)
    return walkers, carried, avg_log_weight


def _positivity_probe(walkers, carried, hamiltonian, guide, method, step_f, lam,
                      cache_h, cache_off, avg_log_weight, off_diag_operators,
                      aux_root, verbosity):
    """Test the off-diagonal operators on an ensemble evolved
    ``_POSITIVITY_PROBE_STEPS`` steps along a side-stream, so configurations
    typical of the walk are tested rather than the (possibly artificial)
    starting ones.  The side-stream leaves the production stream untouched."""
    subkey = jax.random.fold_in(aux_root, _AUX_POSITIVITY)
    rngs = jax.random.split(subkey, _POSITIVITY_PROBE_STEPS)
    (walkers_check, _), _ = _run_cached_scan(
        _make_warmup_step_fn,
        (hamiltonian, guide, lam, _traced_alw(avg_log_weight, walkers.log_weight.dtype),
         cache_h, step_f, method),
        (walkers, carried), rngs)
    check_off_diag_operator_positivity(
        walkers_check, off_diag_operators, guide, caches=cache_off,
        verbose=verbosity >= 1)


def _diag_ring_buffers(walkers, diag_operators, m_max, dtype):
    """The ancestry ring buffer and the per-width operator value buffers.

    Sized m_max+1 so that (buf_idx - m_max) % (m_max+1) points to the slot
    written m_max steps ago and not the current slot.  The operators are
    probed for how many values each returns (scalar -> 1) and grouped by
    that width: one ring buffer per distinct width, so operators of
    different widths can share a run without either padding (which would
    cost n_ops*max(width)) or a second run.

    Returns ``(parent_buf, diag_buf, buf_idx, group_widths, group_idx)``.
    """
    N_walker = walkers.config.shape[0]
    buf_size = m_max + 1
    identity_map = jnp.broadcast_to(jnp.arange(N_walker, dtype=jnp.int32),
                                    (buf_size, N_walker))
    parent_buf = identity_map + 0
    probe = [jax.vmap(op.diagonal_weight)(walkers.config)
             for op in diag_operators.values()]
    widths = [int(v.shape[1]) if v.ndim > 1 else 1 for v in probe]
    by_width = {}
    for i, w in enumerate(widths):
        by_width.setdefault(w, []).append(i)
    group_widths = sorted(by_width)                  # deterministic
    group_idx = [by_width[w] for w in group_widths]
    diag_buf = tuple(
        jnp.zeros((len(idx), buf_size, N_walker, w), dtype=dtype)
        for w, idx in zip(group_widths, group_idx))
    return parent_buf, diag_buf, jnp.int32(0), group_widths, group_idx


def _diag_by_name(chunks, names, group_widths, group_idx):
    """Per-width-group scan output -> ``{name: (n_steps, [n_vals], n_p)}``.

    ``raw['diag']`` is keyed by operator name rather than being one dense
    array with a positional operator axis, because the operators need not
    share a width.
    """
    if not chunks:
        return None
    found = {}
    for g, (w, idx) in enumerate(zip(group_widths, group_idx)):
        arr = np.concatenate([c[g] for c in chunks], axis=0)
        # arr: (n_steps, n_ops_g, w, n_p) -> per operator, width trimmed
        for pos, op_i in enumerate(idx):
            found[names[op_i]] = arr[:, pos, 0] if w == 1 else arr[:, pos]
    return {name: found[name] for name in names}


def _progress_report(*, step, n_steps, chunk, it_per_sec, walkers, p_array,
                     m_array, weights_chunks, E_chunks, diag_chunks,
                     off_diag_chunks, off_diag_norm_chunks,
                     off_diag_step_chunks, diag_names, off_diag_names,
                     group_widths, group_idx, verbosity):
    """The progress line after a chunk.  Nothing here is returned, so the
    driver skips it at verbosity 0; the energy estimate is computed only
    when it is printed (verbosity >= 3) and the operator estimates (re-run
    over the whole series so far) only at verbosity >= 8."""
    N_walker = walkers.config.shape[0]
    weights_np = np.concatenate(weights_chunks)
    E_np = np.concatenate(E_chunks)
    N_config = count_unique_configs(walkers)
    if verbosity >= 8 and (diag_names or off_diag_names):
        diag_np = _diag_by_name(diag_chunks, diag_names, group_widths, group_idx)
        off_diag_np = _off_by_name(
            np.concatenate(off_diag_chunks, axis=0) if off_diag_chunks else None,
            off_diag_names)
        off_diag_norm_np = (np.concatenate(off_diag_norm_chunks, axis=0)
                            if off_diag_norm_chunks else None)
        off_diag_steps_np = (np.concatenate(off_diag_step_chunks)
                             if off_diag_step_chunks else None)
        results, obs_partial = _estimate_observables(
            p_array, weights_np, E_np, cutoff=step - 1,
            off_diag_np=off_diag_np, op_names=off_diag_names or None,
            m_array=m_array, diag_op_np=diag_np,
            diag_op_names=diag_names or None,
            off_diag_steps=off_diag_steps_np,
            off_diag_norm_np=off_diag_norm_np)
    elif verbosity >= 3:
        # the energy estimate does not depend on the operator arguments
        # (they only add obs entries)
        results, obs_partial = _estimate_observables(
            p_array, weights_np, E_np, cutoff=step - 1)
    else:
        results, obs_partial = None, None    # neither is printed below
    line = (f"[gfmc] - [run_gfmc] - step: {step}/{n_steps}, {it_per_sec:.1f} "
            f"it/s, N_config = {N_config}/{N_walker} ({N_config / N_walker:.1%})")
    if verbosity >= 3:
        line += f" Energy[p]: {results}"
    emit(line, verbosity)
    if verbosity >= 8 and obs_partial:
        if verbosity >= 10:
            print_obs = obs_partial
        else:
            print_obs = {}
            if diag_names and diag_names[0] in obs_partial:
                print_obs[diag_names[0]] = obs_partial[diag_names[0]]
            if off_diag_names and off_diag_names[0] in obs_partial:
                print_obs[off_diag_names[0]] = obs_partial[off_diag_names[0]]
        if print_obs:
            emit(f"[gfmc] - [run_gfmc] - operators: {print_obs}", verbosity)


def _check_population(log_G_chunk, step0, E_chunk=None):
    """Raise when a step's mean branch weight is not finite: every walker
    then has zero weight (or one overflowed), the population is extinct and
    nothing downstream can be normalised.  Also raise when a step's energy
    ``E_chunk`` (given) is not finite while the weights are: a walker's
    local energy overflowed, which the final estimator would reject after
    the whole run had been walked."""
    bad = np.flatnonzero(~np.isfinite(log_G_chunk))
    if bad.size:
        raise ValueError(
            f"[gfmc] population extinct at step {step0 + int(bad[0])}: the mean "
            "branch weight of the step is not finite, i.e. every walker's "
            "weight is zero (or one overflowed).  The ensemble cannot be "
            "renormalised; check lam against the diagonal elements "
            "(discrete methods), the step size and the guide.")
    if E_chunk is not None:
        bad = np.flatnonzero(~np.isfinite(E_chunk))
        if bad.size:
            raise ValueError(
                f"[gfmc] non-finite energy at step {step0 + int(bad[0])}: the "
                "weighted mean of the local energies is not finite while the "
                "branch weights are, i.e. a walker's local energy overflowed "
                "(a guide ratio or a rate of its configuration is inf or NaN); "
                "check the guide and the Hamiltonian on the configurations "
                "the walk reaches.")


def _production(*, walkers, carried, step_offset, rng_key, prod_objs, n_steps,
                print_every, plan, diag_names, off_diag_names, ring,
                verbosity):
    """The production loop: ``n_steps`` steps in chunks of ``print_every``,
    pipelined, collecting the per-step series.

    The caller's ``rng_key`` IS the production stream -- used directly, not
    split off.  That is what makes the handed-back key resumable: split it
    here and the key returned to the caller would no longer be the one
    production continues from, so feeding it back would start a different
    stream.  Everything else that needs randomness -- the warm-up ramp, the
    positivity probe, the end-of-run statistics -- draws from tagged
    side-streams off the aux root instead, so none of them can shift or
    collide with the production sequence.

    ``carried`` is the walkers' carried pair ``(rates, state)``;
    ``step_offset`` is the number of steps the chain already holds (a
    continuation), so the fixed refresh schedule of an incremental guide
    (:data:`pygfmc.engine.evolution._STATE_REFRESH_STEPS`) counts absolute
    chain steps.

    Returns a namespace with the production-final ``walkers`` and
    ``rng_key_final``, the series ``weights_np``, ``E_np``, ``diag_np``,
    ``off_diag_np``, ``off_diag_norm_np``, ``off_diag_steps_np``, the
    diagonal lead-in ``additional_diag_np`` and the ``runtime``.
    """
    n_diag_ops = len(diag_names)
    n_off_ops = len(off_diag_names)
    if n_diag_ops > 0:
        parent_buf, diag_buf, buf_idx, group_widths, group_idx = ring
        carry = (walkers, carried, parent_buf, diag_buf, buf_idx, rng_key)
    else:
        group_widths, group_idx = [], []
        carry = (walkers, carried, rng_key)

    # ---- carry donation (unconditional; not a public option) ----
    # lax.scan is invoked eagerly below, so its carry is BOTH a parameter and
    # a result of the compiled computation.  JAX does not alias arguments to
    # outputs unless asked, so XLA would keep two full copies of the ancestry
    # ring buffer live, which can dominate device memory for many operators
    # and deep p grids.  Wrapping the scan in jax.jit with donate_argnums on
    # the ring-buffer slots lets XLA alias each buffer's input to its output,
    # so only one copy is ever allocated, at no measurable speed cost.
    # Donation is purely a buffer-lifetime directive -- no arithmetic and no
    # RNG consumption changes -- hence bit-identical either way, which is why
    # it needs no opt-out.  Only the internally created buffers are donated,
    # NEVER the caller's `walkers`, which must stay valid after the run.
    #
    # _DONATE_DIAG_CARRY is a module-private seam, not a knob: a test can A/B
    # the two allocation strategies to check that XLA has not silently
    # DECLINED the donation and turned it into a no-op.
    donate = _DONATE_DIAG_CARRY and n_diag_ops > 0

    def _run_scan(carry, xs):
        # module-level jit, reused across calls and scan lengths of equal
        # structure (see _run_cached_scan); donation of the ring buffers is
        # preserved: the carry is unpacked so they are their own arguments,
        # and `carry` is rebound by the caller, dropping invalidated handles.
        return _run_cached_scan(_make_prod_step_fn, prod_objs, carry, xs,
                                donate=donate)

    # Smallest cutoff at which a progress estimate is computable for EVERY p.
    # The progress call passes cutoff = step - 1, and the off-diagonal
    # estimator needs, for each p, at least one MEASURED step in
    # [p, cutoff - m - 1].  Measurements sit on the strided grid
    # off_diag_offset + r*off_diag_stride, so that interval has to be at
    # least `off_diag_stride` long: the dense-grid bound max(p) + max(m)
    # leaves the largest p with an empty window whenever off_diag_stride > 1
    # (e.g. p=8, m=4, stride=2 at cutoff 13 gives the single step 8, which is
    # off-grid).  The estimator treats an unusable window as an error, so the
    # guard has to be right.  The second term covers the equal-window gating
    # (return_raw=False), where the first measurement is delayed to
    # off_diag_offset.
    if n_off_ops:
        progress_window = max(
            plan.p_max + plan.m_max + plan.off_diag_stride,
            plan.off_diag_offset + plan.m_max + 1)
    else:
        progress_window = plan.p_max
    weights_chunks, E_chunks, diag_chunks = [], [], []
    off_diag_chunks, off_diag_norm_chunks, off_diag_step_chunks = [], [], []

    production_start = time.time()
    start = production_start
    last_step = 0

    def _dispatch(step0, carry):
        """Launch the scan of the chunk starting at step0 (asynchronously)."""
        chunk = min(print_every, n_steps - step0)
        t, measured_np = _measured_steps(step0, chunk, n_steps, plan)
        # int32 like the lead-in scan below: the index only feeds the
        # refresh schedule (step % _STATE_REFRESH_STEPS).
        xs = (jnp.asarray(measured_np),
              jnp.asarray(t + step_offset, dtype=jnp.int32))
        carry, scan_out = _run_scan(carry, xs)
        # the chunk's own final walkers and key travel with its outputs: by
        # the time the chunk is processed, `carry` already belongs to the
        # NEXT chunk.  Neither is donated, so the handles stay valid.
        return carry, (step0, chunk, t, measured_np, scan_out,
                       carry[0], carry[-1])

    # Pipelined: chunk k+1 is dispatched BEFORE chunk k's outputs are pulled
    # to the host and the progress line is computed, so the device is not
    # idle while the host works.  JAX dispatch is asynchronous and the carry
    # chain is still strictly sequential, so the device-side sequence -- and
    # every result -- is exactly that of the unpipelined loop.
    pending = None
    if n_steps > 0:
        carry, pending = _dispatch(0, carry)
    while pending is not None:
        (step0, chunk, t, measured_np, scan_out,
         walkers, rng_key_final) = pending    # key: what continues
        if step0 + chunk < n_steps:
            carry, pending = _dispatch(step0 + chunk, carry)
        else:
            pending = None

        out_idx = 0
        weights_chunks.append(np.asarray(scan_out[out_idx]))
        out_idx += 1
        E_chunks.append(np.asarray(scan_out[out_idx]))
        out_idx += 1
        _check_population(weights_chunks[-1], step0, E_chunks[-1])
        if n_diag_ops > 0:
            diag_chunks.append(tuple(np.asarray(a) for a in scan_out[out_idx]))
            out_idx += 1
        if n_off_ops > 0:
            # keep only the measured rows — the scan emits one slot per step
            # (lax.cond branches share a fixed shape), filler rows are dropped
            # here so raw['off_diag'] is compact: one row per measurement.
            gate = plan.gate_off_diag
            od_chunk = np.asarray(scan_out[out_idx])
            out_idx += 1
            off_diag_chunks.append(od_chunk[measured_np] if gate else od_chunk)
            # the LOCAL step each kept row belongs to.  Stored with the rows
            # so the estimator reads the step instead of deriving it.
            off_diag_step_chunks.append(
                (t[measured_np] if gate else t).astype(np.int64))
            on_chunk = np.asarray(scan_out[out_idx])
            out_idx += 1
            off_diag_norm_chunks.append(on_chunk[measured_np] if gate
                                        else on_chunk)
        del scan_out
        step = step0 + chunk

        if verbosity >= 1 and step - 1 > progress_window:
            elapsed = time.time() - start
            _progress_report(
                step=step, n_steps=n_steps, chunk=chunk,
                it_per_sec=(step - last_step) / elapsed, walkers=walkers,
                p_array=plan.p_array, m_array=plan.m_array,
                weights_chunks=weights_chunks, E_chunks=E_chunks,
                diag_chunks=diag_chunks, off_diag_chunks=off_diag_chunks,
                off_diag_norm_chunks=off_diag_norm_chunks,
                off_diag_step_chunks=off_diag_step_chunks,
                diag_names=diag_names, off_diag_names=off_diag_names,
                group_widths=group_widths, group_idx=group_idx,
                verbosity=verbosity)
            start = time.time()
            last_step = step

    # ---- diagonal lead-in for the NEXT run --------------------------------
    # The ancestry ring buffer is built per call, so a continuation's first
    # m = p/2 diagonal rows would trace back only to its own beginning and
    # come out as zeros.  Rather than checkpoint the buffer (which can be tens
    # of GB), walk m_max further steps HERE, where the buffer still holds this
    # run's history, and record their diagonal values.  The continuation
    # re-walks exactly these steps -- same ensemble, same key -- so its
    # walkers are bit-identical and only its trace is short; substituting
    # this block for its first rows reproduces the uninterrupted result
    # exactly.
    #
    # Two things this must not disturb.  The returned/checkpointed walkers
    # stay the production-final ones, from BEFORE these steps, since that is
    # where the continuation starts.  And ``n_steps`` keeps its production
    # meaning, so the recorded step range excludes them and the continuation
    # is free to re-walk their keys.  No off-diagonal measurement: the shadow
    # chains dominate the cost and the off-diagonal estimator needs no
    # lead-in (it only looks forward).
    additional_diag_np = None
    if n_diag_ops > 0 and plan.m_max > 0:
        _extra_carry, _extra_out = _run_scan(
            carry, (jnp.zeros(plan.m_max, dtype=bool),
                    jnp.arange(n_steps, n_steps + plan.m_max, dtype=jnp.int32) + step_offset))
        additional_diag_np = _diag_by_name(
            [tuple(np.asarray(a) for a in _extra_out[2])],
            diag_names, group_widths, group_idx)
        del _extra_carry, _extra_out

    return SimpleNamespace(
        walkers=walkers, rng_key_final=rng_key_final,
        weights_np=np.concatenate(weights_chunks),
        E_np=np.concatenate(E_chunks),
        diag_np=_diag_by_name(diag_chunks, diag_names, group_widths, group_idx),
        off_diag_np=_off_by_name(
            np.concatenate(off_diag_chunks, axis=0) if off_diag_chunks else None,
            off_diag_names),
        off_diag_norm_np=_off_by_name(
            np.concatenate(off_diag_norm_chunks, axis=0)
            if off_diag_norm_chunks else None, off_diag_names),
        off_diag_steps_np=(np.concatenate(off_diag_step_chunks)
                           if off_diag_step_chunks else None),
        additional_diag_np=additional_diag_np,
        runtime=time.time() - production_start)


# ---- the driver ----

def run_gfmc(walkers: Walker, n_steps: int, hamiltonian: Operator,
             step_size: Union[int, float], guide: Optional[Guide],
             p_array: Union[Sequence[int], np.ndarray], *, method: str = "cont_time",
             rng_key: Optional[jax.Array] = None, lam: Optional[float] = None,
             diag_operators: Optional[Mapping[str, DiagonalOperator]] = None,
             off_diag_operators: Optional[Mapping[str, Operator]] = None,
             return_raw: bool = True, avg_log_weight: Optional[float] = None,
             warmup_steps: int = 0, verbosity: int = 7,
             number_of_outputs: int = 100, off_diag_stride: int = 1,
             off_diag_batch_size: Optional[int] = None,
             off_diag_equal_windows: Optional[bool] = None,
             _step_offset: int = 0) -> tuple:
    """Run the GFMC production loop on a single device.

    Sequence:
      1. Warm-up: a 4-stage step-size ramp of ``warmup_steps`` total steps
         (skipped when ``warmup_steps=0``).  It equilibrates the ensemble and,
         unless ``avg_log_weight`` is supplied, also measures the
         normalisation shift ``avg_log_weight``.
      2. Production scan of n_steps branching steps (evolve -> reconfigure ->
         diagonal ancestry / off-diagonal forward walking), printing progress
         every n_steps // number_of_outputs steps.
      3. Windowed estimation over ``p_array`` and an end-of-run
         :func:`pygfmc.engine.estimators.statistics` pass (sampler health, offspring
         distribution, ancestor survival at m = p/2).

    Args:
        walkers:            Initial Walker batch.
        n_steps:            Number of production branching steps.
        hamiltonian:        Any object implementing the Hamiltonian protocol
                            (``M_total``, ``get_rates``, ``generate_state``)
                            — e.g. SpinHalf, Spin,
                            BoseHubbard, ToricCode, or PlaquetteModel.
        step_size:          Per-Markov-step size.  For the discrete methods
                            this is the integer count of imaginary-time
                            sub-steps (must be integer-valued); for
                            ``cont_time`` it is the imaginary-time length
                            $\\beta$ (any positive float).
        guide:              Guiding wavefunction or None.
        p_array:            Window sizes for the mixed/pure estimator
                            (deduplicated and sorted).  The projection depth
                            per side is tau = (p/2)·step_size.  When an
                            off-diagonal operator has an off-diagonal
                            transition (``M_total > 1``) all p values must
                            be even.
        method:             "single_step", "multi_step", or "cont_time"
                            (default "cont_time").  This choice sets the
                            meaning of ``step_size`` (integer sub-step count
                            for the discrete methods, imaginary time beta for
                            cont_time) and of ``avg_log_weight`` (log b per
                            sub-step vs. minus the mean local energy per unit
                            imaginary time); under cont_time ``lam`` has no
                            effect (it is the lambda -> infinity limit).
        rng_key:            JAX PRNGKey or typed Threefry key (required).
                            The default PRNG must be Threefry; unsupported key
                            implementations raise before running.
                            THIS IS THE CHAIN POSITION, not just a seed: the
                            production stream starts here and advances
                            key_{t+1} = split(key_t) once per step, with no
                            step index anywhere.  Pass back the key a
                            previous run returned
                            (``stats['parameters/rng_key_next']``, which
                            ``Walker.save(rng_key=...)`` checkpoints and
                            ``Walker.load`` returns) and the continuation is
                            bit-identical to one uninterrupted run -- that one
                            array is the whole resume state.  Everything else
                            needing randomness draws from tagged side-streams,
                            so it cannot shift the production sequence.
        lam:                GFMC shift λ, required by the discrete methods,
                            where it must satisfy λ ≥ max_x H_xx over the
                            reachable configurations (ValueError when
                            omitted).  Unused by ``cont_time``, the default
                            method, so it defaults to ``None``.
        diag_operators:     dict  name → operator  of diagonal operators, or
                            None.  Each needs ``diagonal_weight(config)``
                            returning a scalar or an ``(n_vals,)`` vector.
                            Measured with the ancestry-traced (forward-walking)
                            pure estimator: the value at the ancestor m = p/2
                            steps back, weighted like the energy.
        off_diag_operators: dict  name → operator  of off-diagonal operators,
                            or None.  Each must implement the operator
                            protocol (:class:`pygfmc.Operator`), with
                            non-negative rates.  Measured via the
                            forward-walking pure estimator (one shadow chain
                            of depth max(p)/2 per operator per measurement).
                            Every p must be even when an operator has
                            ``M_total > 1``; a purely diagonal operator
                            (``M_total == 1``) accepts odd p as well.
        return_raw:         If True (default), append the ``raw`` dict to the
                            return tuple (see Returns).  Also sets the DEFAULT
                            of ``off_diag_equal_windows`` (below).
        off_diag_equal_windows: Restrict the off-diagonal measurements to the
                            band of steps usable by EVERY p: the first max(p)
                            steps become a lead-in and the last max(p)/2 a
                            tail, so each p rests on the identical set of
                            measurements and n_steps > 1.5*max(p) is REQUIRED
                            (ValueError otherwise).  ``None`` (default) means
                            ``not return_raw``: with the raw series returned
                            they can be stored and re-estimated over the whole
                            concatenated chain (``evaluate_observables``),
                            where windows span run boundaries and gating would
                            only throw measurements away; without them the
                            per-call windowed numbers are final and gating is
                            what makes them comparable across p.  Pass True or
                            False to decide explicitly (e.g. True when you take
                            ``raw`` but rely on the per-call numbers).
        avg_log_weight:     Pre-computed normalisation shift (e.g. from
                            ``stats['avg_log_weight']`` of a previous run).  When supplied it is kept
                            fixed; the warm-up ramp still runs (unless
                            ``warmup_steps=0``) but only equilibrates the
                            walkers.  When None it is estimated and refined
                            by the warm-up.
        warmup_steps:       Total warm-up steps split across 4 ramp stages
                            (step_size/8 → step_size/4 → step_size/2 →
                            step_size; integer division, floored at 1, for the
                            discrete methods).  Default 0: no warm-up, which is
                            right when continuing from an equilibrated ensemble
                            (e.g. walkers returned by a previous run).  For a
                            fresh start from unequilibrated walkers pass a
                            warm-up explicitly, e.g. ``warmup_steps=10000``;
                            otherwise the first steps are not equilibrated and
                            ``avg_log_weight`` stays at its initial estimate.
        verbosity:          Integer 0–10 controlling printed output:

                            * 0 — silent
                            * 1 — step/it/N_config progress (no energy)
                            * 3 — progress + Energy[p]
                            * 8 — progress + Energy[p] + first diag/off-diag
                              observable (re-estimated over the whole
                              series at every progress line, so it costs
                              host time) + warmup lines
                            * 10 — full output (all observables)

                            The end-of-run statistics report prints for any
                            verbosity >= 1.  Default 7.
        number_of_outputs:  How many progress lines to print during the production run
                            (default 100).  Actual cadence is n_steps // number_of_outputs.
        off_diag_stride:    Measure the off-diagonal operators only every
                            ``off_diag_stride``-th production step (default 1
                            = every step).  Each off-diagonal measurement
                            launches a fresh shadow Markov chain m_max =
                            max(p)/2 steps deep, so it dominates the per-step
                            cost by ~m_max× — but consecutive main-chain steps
                            are strongly correlated, so per-step measurement
                            mostly averages redundant values.  A stride ≈ the
                            observable's autocorrelation time keeps nearly
                            all the statistical information at ~1/stride of
                            the cost.  Only the measured steps are stored
                            (``raw['off_diag']`` has one row per measurement,
                            ``raw['off_diag_steps']`` gives each row's step).
                            Energy and diagonal observables are unaffected —
                            they are cheap and stay per-step.

                            The stride counts BRANCHING STEPS, so choose it
                            against ``step_size``: decide the imaginary time
                            between measurements and convert::

                                off_diag_stride = round(tau_between/step_size)

                            since ensemble decorrelation is a process in
                            imaginary time.  A stride copied between runs with
                            different ``step_size`` silently changes meaning.
                            A useful rule of thumb is tau_between ≈ tau_max/4
                            (i.e. stride ≈ m_max/4), bounded below by the
                            observable's autocorrelation time.
        off_diag_batch_size: optional walker-chunk size for APPLYING each
                            off-diagonal operator (not for the shadow chain,
                            whose cost is independent of the operator's
                            width).  None = apply to all walkers at once.  An
                            int cuts the peak (N_walkers, n_off, max_aff)
                            intermediate by off_diag_batch_size/N_walkers and
                            is BIT-IDENTICAL: the operator application is
                            per-walker independent and each walker's PRNG key
                            comes from its absolute index.  Must divide the
                            walker count.  Needed only for a WIDE operator
                            (e.g. a pair sum over all sites).  Costs
                            sequential passes, so leave it None unless memory
                            demands it.

        _step_offset:       Internal.  The number of steps the chain already
                            holds; ``run_from_HDF5`` passes it so that the
                            fixed refresh schedule of a guide in the
                            incremental mode counts absolute chain steps and
                            a split run reproduces a contiguous one.

    Returns:
        ``(walkers, energy, obs, stats, raw)`` by default (``return_raw=True``);
        with ``return_raw=False`` the 4-tuple ``(walkers, energy, obs, stats)``.

        * ``walkers``: Walker batch after the production run (the
          production-final state; the extra lead-in and statistics steps are
          not applied to it).
        * ``energy``: dict  p → float  energy estimates.
        * ``obs``: dict  name → dict  p → value  operator estimates (a float,
          or an ``(n_vals,)`` array for a vector-valued diagonal operator);
          empty when no operators are provided.

        ``stats`` is a dict of run diagnostics — the end-of-run
        statistics() dict (health stats, offspring_k*, ancestor_m,
        ancestor_fraction, effective_sample_size) plus:

        * ``stats['avg_log_weight']``: the normalisation shift used in
          production; pass it back to reuse it.
        * ``stats['it_per_sec']``: production throughput.
        * ``stats['runtime']``: wall-clock seconds for the production loop.
        * ``stats['total_wall_time']``: complete call time, including warm-up,
          final estimation, and diagnostics.
        * ``stats['number_of_devices']``: 1.
        * ``stats['autocorrelation/...']``: integrated autocorrelation times of
          the energy, log weights and every operator series, in branching
          steps.
        * ``stats['parameters/...']``: the run's inputs (n_steps, n_walkers,
          step_size, lam, method, p_array, m_array, off_diag_stride, operator
          names and content hashes, dtypes, ...) and the RNG stream:
          ``parameters/rng_key`` (where production started) and
          ``parameters/rng_key_next`` (where it ended — pass it as
          ``rng_key`` to continue).

        ``raw`` holds the per-step GFMC values and the metadata needed to
        re-estimate them (``evaluate_observables(raw)`` needs no further arguments):

        * ``raw['log_weights']``: (n_steps,) log branch weights log G_n.
        * ``raw['energy']``: (n_steps,) per-step energies E_n.
        * ``raw['diag']``: dict name -> (n_steps, n_p) for a scalar operator or
          (n_steps, n_vals, n_p) for a vector-valued one; None without
          diagonal operators.  Operators of different widths may share a run.
        * ``raw['additional_diag']``: same layout, for the m_max steps
          FOLLOWING this run, traced with this run's ancestry; the lead-in a
          continuation substitutes for its own first rows (see
          ``run_from_HDF5``).
        * ``raw['off_diag']``: dict name -> (n_meas, n_p) forward-walk values,
          one row per MEASURED step; None without off-diagonal operators.
        * ``raw['off_diag_norm']``: dict name -> (n_meas, n_p) log shadow-chain
          weight products (same rows).
        * ``raw['off_diag_steps']``: (n_meas,) 0-based production step (within
          this call) at which each row was measured.
        * ``raw['p_array']``, ``raw['m_array']``, ``raw['diag_operators']``,
          ``raw['off_diag_operators']``: the window grid and operator names;
          the last axis of every operator series is positional in
          ``p_array``.
    """
    wall_start = time.time()
    if rng_key is None:
        raise ValueError("run_gfmc requires rng_key (a JAX PRNG key).")
    if isinstance(rng_key, np.ndarray):
        # a key read back from stats or HDF5; as a JAX array it matches the
        # signature of the scans compiled by the run it continues
        rng_key = jnp.asarray(rng_key)
    key_data(rng_key)  # fail before walking if the stream cannot be checkpointed
    # The caller's key is the production stream; keep it so the stored
    # parameters identify the stream this run started from.  Every
    # NON-production random stream in this call hangs off aux_root.
    rng_key_in = rng_key
    aux_root = jax.random.fold_in(rng_key, _AUX_ROOT)
    N_walker = walkers.config.shape[0]
    if N_walker == 0:
        raise ValueError("At least one walker is required.")
    dtype = walkers.log_weight.dtype

    n_steps = _validated_int("n_steps", n_steps, 1)
    require_step_size(method, step_size)
    warmup_steps = _validated_int("warmup_steps", warmup_steps, 0)
    number_of_outputs = _validated_int(
        "number_of_outputs", number_of_outputs, 1)
    off_diag_stride = _validated_int("off_diag_stride", off_diag_stride, 1)
    if off_diag_batch_size is not None:
        off_diag_batch_size = _validated_int(
            "off_diag_batch_size", off_diag_batch_size, 1)
        if N_walker % off_diag_batch_size:
            raise ValueError(
                f"off_diag_batch_size={off_diag_batch_size} must divide "
                f"N_walkers={N_walker} (a remainder would compile a second shape "
                f"for the tail; keep one shape so throughput and memory stay "
                f"predictable)")
    lam_given = lam                   # recorded in stats as the caller gave it
    lam = require_lam(method, lam)
    require_operator_protocol(hamiltonian, role="hamiltonian")
    for _name, _op in (off_diag_operators or {}).items():
        require_operator_protocol(_op, role=f"off-diagonal operator {_name!r}")
    require_guide_protocol(guide)
    require_incremental_support(guide, hamiltonian)

    # Deduplicate + sort p_array.  Duplicates would cause the windowed
    # estimator to repeat the same computation under different dict keys,
    # silently wasting work; the engine assumes sorted, distinct values
    # downstream (m_array = p//2 from sorted_p, etc.).
    n_p_given = len(p_array) if p_array is not None else 0
    p_array = _validated_p_array(p_array)
    if len(p_array) != n_p_given:
        emit(f"[gfmc.run_gfmc] dedup'd p_array: {n_p_given - len(p_array)} "
             f"duplicate value(s) dropped, using {p_array}", verbosity)
    off_diag_names = list(off_diag_operators.keys()) if off_diag_operators else []
    diag_names     = list(diag_operators.keys()) if diag_operators else []
    n_off_ops  = len(off_diag_names)
    n_diag_ops = len(diag_names)

    collisions = set(diag_names) & set(off_diag_names)
    if collisions:
        raise ValueError(
            f"diag_operators and off_diag_operators share name(s) {sorted(collisions)}; "
            f"the returned obs dict is keyed by name, so colliding entries would "
            f"silently overwrite each other."
        )
    if off_diag_operators is not None:
        # The shadow chain is m = p // 2 steps deep, so an operator with an
        # off-diagonal transition needs even p; a purely diagonal one
        # (M_total == 1) is measured at m = p // 2 as well and accepts odd p.
        has_off_diag = any(int(op.M_total) > 1 for op in off_diag_operators.values())
        if has_off_diag:
            bad = [p for p in p_array if p % 2 != 0]
            if bad:
                raise ValueError(
                    f"All p values must be divisible by 2 when off_diag_operators have "
                    f"off-diagonal elements (M_total > 1); got {bad}"
                )
    plan = _measurement_plan(n_steps, sorted(p_array), n_off_ops,
                             off_diag_stride, off_diag_equal_windows, return_raw)
    # step_size is interpreted by evolve_walkers based on method; the warm-up
    # ramp ratios apply uniformly, so it is kept as a float, which works for
    # both integer-valued discrete step counts and continuous imaginary-time
    # lengths.
    step_f = float(step_size)

    # Precompute the guide's per-operator cache once, before entering the JIT
    # loop.  Used by transition_rates via cache=... so that it is not rebuilt
    # on every vmap'd call inside evolve_walkers / forward_walk_operator.  No
    # effect for guides without build_cache().
    cache_h   = guide_cache(guide, hamiltonian)
    cache_off = {name: guide_cache(guide, op)
                 for name, op in (off_diag_operators or {}).items()}

    carried = _initial_carried(walkers, hamiltonian, guide, cache_h)
    walkers, carried, avg_log_weight = _warmup(
        walkers, carried, hamiltonian, guide, method, step_f, lam, cache_h,
        avg_log_weight, warmup_steps, aux_root, dtype, verbosity)
    # One jit signature for the probe, the production scans and the
    # statistics pass whether the shift was pinned (a Python float) or
    # measured (an array); see _traced_alw.  stats['avg_log_weight'] records
    # the value as given.
    avg_log_weight_given = avg_log_weight
    avg_log_weight = _traced_alw(avg_log_weight, dtype)
    if off_diag_operators is not None:
        _positivity_probe(walkers, carried, hamiltonian, guide, method, step_f, lam,
                          cache_h, cache_off, avg_log_weight,
                          off_diag_operators, aux_root, verbosity)

    ring = (_diag_ring_buffers(walkers, diag_operators, plan.m_max, dtype)
            if n_diag_ops > 0 else None)
    # Operators as tuples in the caller's order; see _make_prod_step_fn.
    prod_objs = (hamiltonian, guide, lam, _traced_alw(avg_log_weight, dtype), cache_h,
                 tuple((diag_operators or {}).values()),
                 tuple((off_diag_operators or {}).values()),
                 tuple(cache_off[name] for name in off_diag_names),
                 step_f, method, list(plan.m_array), N_walker, len(plan.p_array),
                 dtype, plan.gate_off_diag, off_diag_batch_size,
                 ring[4] if n_diag_ops > 0 else [])
    prod = _production(
        walkers=walkers, carried=carried, step_offset=int(_step_offset),
        rng_key=rng_key, prod_objs=prod_objs,
        n_steps=n_steps,
        print_every=max(1, n_steps // max(1, number_of_outputs)), plan=plan,
        diag_names=diag_names, off_diag_names=off_diag_names, ring=ring,
        verbosity=verbosity)
    walkers = prod.walkers

    final_results, obs_results = _estimate_observables(
        plan.p_array, prod.weights_np, prod.E_np,
        off_diag_np=prod.off_diag_np, op_names=off_diag_names or None,
        m_array=plan.m_array,
        diag_op_np=prod.diag_np, diag_op_names=diag_names or None,
        off_diag_steps=prod.off_diag_steps_np,
        off_diag_norm_np=prod.off_diag_norm_np,
    )
    _lineage_warning(
        plan.m_array, n_steps, verbosity, "[gfmc] - [run_gfmc] -",
        ".  They are measured once per call, so raise n_steps (or lower "
        f"max(p_array), m_max={plan.m_max}) to amortise them.")
    stats = _run_stats(
        walkers=walkers, hamiltonian=hamiltonian, step_f=step_f, guide=guide,
        method=method, aux_root=aux_root, avg_log_weight=avg_log_weight,
        lam=lam, lam_given=lam_given, cache_h=cache_h, plan=plan,
        n_steps=n_steps, warmup_steps=warmup_steps, verbosity=verbosity,
        number_of_outputs=number_of_outputs, diag_operators=diag_operators,
        off_diag_operators=off_diag_operators, rng_key_in=rng_key_in,
        rng_key_next=prod.rng_key_final, E_np=prod.E_np,
        weights_np=prod.weights_np, diag_np=prod.diag_np,
        off_diag_np=prod.off_diag_np, production_runtime=prod.runtime,
        wall_start=wall_start, number_of_devices=1,
        avg_log_weight_given=avg_log_weight_given)
    if return_raw:
        raw = _raw_series(
            weights_np=prod.weights_np, E_np=prod.E_np, diag_np=prod.diag_np,
            off_diag_np=prod.off_diag_np, off_diag_norm_np=prod.off_diag_norm_np,
            off_diag_steps_np=prod.off_diag_steps_np,
            additional_diag_np=prod.additional_diag_np, plan=plan,
            diag_names=diag_names, off_diag_names=off_diag_names)
        return walkers, final_results, obs_results, stats, raw
    return walkers, final_results, obs_results, stats
