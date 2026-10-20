"""Multi-device GFMC via jax.pmap.

Walker population is sharded across devices: each device evolves its own shard
independently, communication only happens at the reconfiguration step.

  evolve (one step_size) on each device ←  embarrassingly parallel
                ↓
  reconfiguration with all_gather       ←  global weight collection + resample
                ↓                            (only at branching boundaries)
              (repeat)

Falls back to a single device automatically: jax.device_count() returns 1 on
CPU or single-GPU machines, so the same code path runs everywhere.

Supports energy and off-diagonal operators (forward-walking pure estimator
with global shadow-ensemble reconfiguration), including measurement gating
(``off_diag_stride > 1`` and the equal windows of ``return_raw=False``).  The
public ``run`` dispatcher falls back to the single-device backend only when
diagonal operators are requested.
"""
import os as _os
import time
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from typing import Any, Mapping, Optional, Sequence, Union

from ..walkers import Walker
from ..walkers.walker import key_data
from ..walkers.reconfiguration import systematic_parents
from ..protocols import Operator, DiagonalOperator, Guide
from .evolution import (require_step_size, require_operator_protocol,
                        require_guide_protocol, require_incremental_support,
                        guide_cache, require_lam, initial_carried,
                        evolve_carried, refresh_carried_at, rates_fit_budget,
                        apply_operator_one, _carry_rates_bytes)
from .estimators import (_estimate_observables,
                         estimate_average_log_branch_weight,
                         check_off_diag_operator_positivity,
                         _validated_p_array)
from .driver import (run_gfmc, _AUX_ROOT, _AUX_POSITIVITY, _check_population,
                     _POSITIVITY_PROBE_STEPS, _warmup_stage_key, _traced_alw,
                     _measurement_plan, _measured_steps, _warmup_schedule,
                     _require_warmup_or_alw, _default_warmup, _default_rng_key,
                     _off_by_name, _lineage_warning, _run_stats, _raw_series)
from .._validation import _validated_int
from .._output import emit


# ── walker sharding helpers ─────────────────────────────────────────────────

def shard_walkers(walkers, n_devices=None):
    """Reshape walker arrays for pmap: (N, ...) → (n_devices, N//n_devices, ...).

    Each leaf array gains a leading device axis.  N must be divisible by
    n_devices.
    """
    if n_devices is None:
        n_devices = jax.device_count()
    n_devices = _validated_int("n_devices", n_devices, 1)
    N = walkers.config.shape[0]
    if N == 0:
        raise ValueError("At least one walker is required.")
    if N % n_devices != 0:
        raise ValueError(f"N_walkers={N} not divisible by n_devices={n_devices}")
    N_per = N // n_devices
    return Walker(
        config=walkers.config.reshape(n_devices, N_per, *walkers.config.shape[1:]),
        log_weight=walkers.log_weight.reshape(n_devices, N_per),
        e_loc=walkers.e_loc.reshape(n_devices, N_per),
    )


def unshard_walkers(walkers):
    """Inverse of shard_walkers: (n_devices, N_per, ...) → (n_devices*N_per, ...)."""
    n_devices, N_per = walkers.log_weight.shape
    return Walker(
        config=walkers.config.reshape(n_devices * N_per, *walkers.config.shape[2:]),
        log_weight=walkers.log_weight.reshape(n_devices * N_per),
        e_loc=walkers.e_loc.reshape(n_devices * N_per),
    )


_PMAP_CARRY_RATES = _os.environ.get("PYGFMC_PMAP_CARRY_RATES", "1") not in ("0", "")


# ── global reconfiguration ──────────────────────────────────────────────────

def reconfiguration_global(walkers: Walker, rng_key: jax.Array,
                           axis_name: str = 'devices', aux: Any = None) -> tuple:
    """Reconfiguration that resamples globally across all devices.

    Each device contributes its walker shard.  Weights are gathered globally,
    a single systematic resampling is performed, and each device picks the
    slice of resampled walkers it owns.

    The rng_key must be identical on all devices so they agree on the random
    offset of the systematic resample.

    If aux is given (a pytree of arrays of shape (N_per_device, ...)), it is
    also gathered, remapped through the resampled indices, and the per-device
    slice is returned as a fourth element.  Useful for carrying per-walker
    auxiliary state through reconfiguration (forward-walking values, the
    carried rates).

    Must be called inside ``jax.pmap`` with the matching ``axis_name``.

    Returns:
        ``(walkers_shard, log_G_n, E_n[, aux_shard])`` with ``log_G_n`` the
        log of the global mean weight and ``E_n`` the global weighted local
        energy, as in :func:`pygfmc.walkers.reconfiguration`.
    """
    N_per = walkers.config.shape[0]

    log_w_all  = jax.lax.all_gather(walkers.log_weight, axis_name=axis_name)
    e_loc_all  = jax.lax.all_gather(walkers.e_loc,      axis_name=axis_name)
    config_all = jax.lax.all_gather(walkers.config,     axis_name=axis_name)
    aux_all = (jax.tree_util.tree_map(
        lambda a: jax.lax.all_gather(a, axis_name=axis_name), aux)
        if aux is not None else None)

    n_devices = log_w_all.shape[0]
    N_total   = n_devices * N_per

    log_w_flat  = log_w_all.reshape(N_total)
    e_loc_flat  = e_loc_all.reshape(N_total)
    config_flat = config_all.reshape(N_total, *walkers.config.shape[1:])

    log_G_n = logsumexp(log_w_flat)
    w       = jnp.exp(log_w_flat - log_G_n)
    E_n     = jnp.sum(e_loc_flat * w)
    log_avg_weight = log_G_n - jnp.log(jnp.asarray(
        N_total, dtype=walkers.log_weight.dtype))
    branch_weight = log_avg_weight

    # Systematic resampling — same on all devices because rng_key is replicated
    psi = jax.random.uniform(rng_key)
    indices = systematic_parents(w, psi)

    # Each device picks its slice of new walkers
    device_id = jax.lax.axis_index(axis_name)
    my_idx = jax.lax.dynamic_slice_in_dim(indices, device_id * N_per, N_per)

    new_walkers = Walker(
        config=config_flat[my_idx],
        log_weight=jnp.zeros(N_per, dtype=walkers.log_weight.dtype),
        e_loc=e_loc_flat[my_idx],
    )
    if aux is None:
        return new_walkers, branch_weight, E_n
    aux_new = jax.tree_util.tree_map(
        lambda a: a.reshape(N_total, *a.shape[2:])[my_idx], aux_all)
    return new_walkers, branch_weight, E_n, aux_new


# ── multi-device forward-walking estimator ──────────────────────────────────

def forward_walk_operator_global(walkers, op, h_main, step_size, guide, method,
                                 rng_key, avg_log_weight, m_array, lam=None,
                                 axis_name='devices', cache_h=None,
                                 cache_op=None, op_batch_size=None,
                                 shadow_key=None):
    """Multi-device version of forward_walk_operator.

    Each device evolves its own shadow walker shard; reconfiguration of the
    shadow ensemble is global (uses reconfiguration_global with aux=vals so
    per-walker operator values follow walker resampling correctly), and the
    mean of vals at each step uses pmean across devices.  Arguments and the
    ``(val, log_norm)`` return value are as in
    :func:`pygfmc.engine.evolution.forward_walk_operator`, except that
    ``op_batch_size`` must divide the per-device shard size.

    ``shadow_key``: optional PRNG key to start the shadow chain from,
    replicated across devices.  Pass the main chain's NEXT key and the shadow
    -- running the identical key recurrence -- replays the main chain's future
    draws from the (near-identical) operator-applied ensemble, so the
    population-level weight noise cancels between the pure estimator's
    numerator and denominator.  The sub-key
    derivation below therefore MUST match the main production step
    (fold_in 0 = evolve, 1 = reconfigure), exactly as
    evolution.py:forward_walk_operator does on one device.  Default None
    falls back to an independent stream (same expectation, far larger
    deep-window variance).
    """
    N_per   = walkers.config.shape[0]
    # op_batch_size divides the PER-DEVICE shard, not the global walker
    # count -- pmap has already split the walkers n_devices ways by the time
    # this runs, so `walkers` here is one shard.
    if op_batch_size is not None and N_per % op_batch_size:
        raise ValueError(
            f"op_batch_size={op_batch_size} must divide the PER-DEVICE shard "
            f"size {N_per} (= n_walkers // n_devices), not the global walker "
            f"count")
    dtype   = walkers.log_weight.dtype

    def apply_op_one(config, key):
        # the single-device operator application (same guard for a
        # configuration with no valid transition)
        return apply_operator_one(config, key, op, guide, cache_op)

    # Per-device, per-walker apply keys.
    # The key is fold_in(fold_in(rng_key, device_id), LOCAL index) and the
    # keys are materialised as data before the map, so chunking within a
    # shard keeps every walker's key -- exactly as in the single-device path.
    # Do not re-derive keys per chunk.
    device_id = jax.lax.axis_index(axis_name)
    apply_root = jax.random.fold_in(rng_key, device_id)
    apply_keys = jax.vmap(
        lambda i: jax.random.fold_in(apply_root, i)
    )(jnp.arange(N_per))
    if op_batch_size is None or op_batch_size >= N_per:
        new_configs, vals = jax.vmap(apply_op_one)(walkers.config, apply_keys)
    else:
        # Chunk the operator application WITHIN each shard.  This COMPOSES
        # with pmap's sharding rather than replacing it: per-device peak goes
        # as (N_walkers/n_devices) * (op_batch_size/N_per), so sharding buys
        # the first factor in parallel and batching the second sequentially.
        # Reach for sharding first; it is free.
        new_configs, vals = jax.lax.map(
            lambda ck: apply_op_one(ck[0], ck[1]),
            (walkers.config, apply_keys),
            batch_size=op_batch_size,
        )

    op_walkers = Walker(
        config=new_configs,
        log_weight=jnp.zeros(N_per, dtype=dtype),
        e_loc=jnp.zeros(N_per, dtype=dtype),
    )

    sorted_m = sorted(int(m) for m in m_array)
    m_max    = sorted_m[-1]

    def step_fn(carry, _):
        w, v, log_wm, key, (r, st) = carry
        key, rng = jax.random.split(key)
        # MUST match the main chain: fold_in 0 = evolve, 1 = reconfigure, and
        # the device index folded into the evolve key only (the
        # reconfiguration key stays replicated so every device builds the
        # same global CDF).
        k_evo_global = jax.random.fold_in(rng, 0)
        k_recon      = jax.random.fold_in(rng, 1)
        k_evo = jax.random.fold_in(k_evo_global, jax.lax.axis_index(axis_name))
        w, (r, st) = evolve_carried(w, (r, st), h_main, step_size, guide, method,
                                    k_evo, avg_log_weight, lam=lam, cache=cache_h)
        w, log_G_t, _, (v, r, st) = reconfiguration_global(
            w, k_recon, axis_name=axis_name, aux=(v, r, st))
        log_wm = log_wm + log_G_t
        v_mean = jax.lax.pmean(jnp.mean(v), axis_name=axis_name)
        # Split layout, mirroring evolution.py:forward_walk_operator: a single
        # `jnp.exp(log_wm) * v_mean` would underflow float32 at deep windows.
        # v_mean stays linear (it is signed); only log_wm needs the exponent
        # range.
        return (w, v, log_wm, key, (r, st)), (v_mean, log_wm)

    # The main chain's NEXT key; absent it, a disjoint stream replicated
    # across devices (do NOT fold_in device_id here -- the reconfiguration key
    # must be identical on every device or they build different global CDFs).
    start_key = (jax.random.fold_in(rng_key, 2**31 - 1)
                 if shadow_key is None else shadow_key)
    log_weight_multi = jnp.zeros((), dtype=dtype)
    # The shadow chain carries the rates of its walkers across the steps as
    # the main chain does; the first step's evaluation is done here.
    # The collective's shape is static, so no scalar communication survives
    # compilation. Account for the actual pmap axis, not all visible devices.
    n_devices = jax.lax.all_gather(jnp.int32(0), axis_name).shape[0]
    carried0 = initial_carried(op_walkers, h_main, guide, cache_h,
                               with_rates=_PMAP_CARRY_RATES and rates_fit_budget(
                                   n_devices * op_walkers.config.shape[0], h_main,
                                   op_walkers.log_weight.dtype, _carry_rates_bytes()))
    _, (per_step_val, per_step_lognorm) = jax.lax.scan(
        step_fn, (op_walkers, vals, log_weight_multi, start_key, carried0),
        None, length=m_max)

    # Row m holds the value after m shadow steps; row 0 is the un-propagated
    # ensemble (the global mean of the operator values, no accumulated
    # weight), as in forward_walk_operator.
    v0_mean = jax.lax.pmean(jnp.mean(vals), axis_name=axis_name)
    per_step_val = jnp.concatenate([v0_mean[None], per_step_val])
    per_step_lognorm = jnp.concatenate([log_weight_multi[None], per_step_lognorm])
    pick = jnp.array(sorted_m, dtype=jnp.int32)
    return per_step_val[pick], per_step_lognorm[pick]


# ── pmapped step factory ────────────────────────────────────────────────────

def _make_step_chunk(hamiltonian, lam, guide, method, step_size,
                     off_diag_operators=None, m_array=None,
                     cache_h=None, cache_off=None, op_batch_size=None,
                     gate_off_diag=False):
    """Build a pmapped function that runs a chunk of GFMC steps via lax.scan.

    The returned function takes ``(walkers_shard, carried_shard, rng_key,
    measured, steps, avg_log_weight)``, where ``carried_shard`` is the shard's
    carried pair ``(rates, state)`` (see
    :func:`pygfmc.engine.driver._initial_carried`; either entry may be None
    throughout), ``steps`` the replicated absolute chain step of every step
    of the chunk (the refresh schedule of an incremental guide) and
    ``measured`` is a replicated ``(n_chunk,)``
    boolean array: with ``gate_off_diag`` step t launches the off-diagonal
    shadow chains only where ``measured[t]``; without it every step measures
    and ``measured`` only sets the chunk length.

    Output tuple: (walkers, carried, key_out, log_G, E [, off_diag_vals, off_diag_lognorms])
      - off_diag_vals: (n_steps, n_off, n_p)  forward-walked values per p;
        NaN filler rows at unmeasured steps when gating.
    """
    off_diag_ops = list(off_diag_operators.values()) if off_diag_operators else []
    n_off = len(off_diag_ops)
    if n_off > 0 and m_array is None:
        raise ValueError("m_array required when off_diag_operators provided")
    sorted_m = sorted(int(m) for m in (m_array or []))
    n_p = len(sorted_m)


    def chunk_fn(walkers_shard, carried_shard, rng_key, measured, steps, avg_log_weight):
        """Run a chunk of steps; the key advances in the carry.

        The key IS the chain position -- key_{t+1}, sub_t = split(key_t), no
        step index anywhere -- so the chunk takes a starting key and hands
        back where it got to.  The shadow chain simply advances the same key
        from the next step, and chunk and call boundaries are invisible to
        the stream.
        """
        def step(carry, xs):
            measure_flag, step_abs = xs
            walkers, carried, key = carry
            carried = refresh_carried_at(step_abs, carried, walkers, hamiltonian,
                                         guide, cache_h)
            # identical recurrence to run_gfmc: the key carries the position
            key, rng = jax.random.split(key)
            k1 = jax.random.fold_in(rng, 0)     # evolve
            k2 = jax.random.fold_in(rng, 1)     # reconfigure (replicated)
            k3 = jax.random.fold_in(rng, 2)     # off-diagonal operators
            device_id = jax.lax.axis_index('devices')
            k_evo = jax.random.fold_in(k1, device_id)
            walkers, carried = evolve_carried(walkers, carried, hamiltonian, step_size,
                                              guide, method, k_evo, avg_log_weight,
                                              lam=lam, cache=cache_h)
            walkers, log_G_n, E_n, carried = reconfiguration_global(
                walkers, k2, aux=carried)
            outputs = (log_G_n, E_n)
            if off_diag_ops:
                # The main chain's own future step keys -- the shadow replays
                # them from the operator-applied ensemble (correlated
                # sampling).  Keys are state-independent, so expectations are
                # unchanged; only the variance improves.
                # Each op: (n_p,) values via global forward-walking
                def _measure(_):
                    _fw = [
                        forward_walk_operator_global(
                            walkers, op, hamiltonian, step_size, guide, method,
                            jax.random.fold_in(k3, i), avg_log_weight, sorted_m,
                            lam=lam,
                            cache_h=cache_h,
                            cache_op=(cache_off or {}).get(
                                list(off_diag_operators.keys())[i]),
                            op_batch_size=op_batch_size,
                            shadow_key=key,
                        )
                        for i, op in enumerate(off_diag_ops)
                    ]
                    return (jnp.stack([a for a, _ in _fw]),   # (n_off, n_p)
                            jnp.stack([b for _, b in _fw]))   # (n_off, n_p)
                if not gate_off_diag:
                    off_vals, off_norms = _measure(None)
                else:
                    # As in run_gfmc: lax.cond with a scalar predicate runs
                    # only the taken branch, so unmeasured steps skip the
                    # shadow chain.  The predicate is REPLICATED (identical
                    # on every device), so all devices take the same branch
                    # and the collectives inside it (the shadow ensemble's
                    # global reconfiguration) stay matched.  k3 and key are
                    # advanced every step regardless, so a measured step's
                    # value is bit-identical to the stride-1 run's.
                    _dt = walkers.log_weight.dtype
                    off_vals, off_norms = jax.lax.cond(
                        measure_flag, _measure,
                        lambda _: (jnp.full((n_off, n_p), jnp.nan, dtype=_dt),
                                   jnp.full((n_off, n_p), jnp.nan, dtype=_dt)),
                        None)
                outputs = outputs + (off_vals, off_norms)
            return (walkers, carried, key), outputs

        (walkers_shard, carried_shard, key_out), outputs = jax.lax.scan(
            step, (walkers_shard, carried_shard, rng_key), (measured, steps))
        return (walkers_shard, carried_shard, key_out) + tuple(outputs)

    # rng_key, the measurement mask and the steps are replicated; the walkers
    # and their carried pair shard (a None entry has no leaves to map).
    return jax.pmap(chunk_fn, axis_name='devices',
                    in_axes=(0, 0, None, None, None, None))


# ── main loop ───────────────────────────────────────────────────────────────

def _production_pmap(*, walkers_sharded, carried_sharded, step_offset, rng_key,
                     chunk_fn, avg_log_weight, n_steps, print_every, plan,
                     n_off, n_devices, verbosity):
    """The production loop of :func:`run_gfmc_pmap`: ``n_steps`` steps in
    chunks of ``print_every`` through the pmapped ``chunk_fn``, collecting
    the per-step series.  The caller's ``rng_key`` IS the production stream:
    it advances one split per step inside the chunks and the final value is
    what a continuation passes back, so the correlated-shadow guarantee spans
    chunk AND call boundaries with no bookkeeping.  ``step_offset`` is the
    number of steps the chain already holds, as in
    :func:`pygfmc.engine.driver._production`.

    Returns a namespace with the final ``walkers_sharded`` and
    ``rng_key_final``, the series ``weights_np``, ``E_np``, the dense
    ``off_diag_np`` / ``off_norm_np`` (measured rows only) with their
    ``off_steps_np``, and the ``runtime``.
    """
    p_array = plan.p_array
    progress_window = plan.p_max + (plan.m_max if n_off else 0)
    weights_chunks  = []
    E_chunks        = []
    off_diag_chunks = []   # list of (n_chunk, n_off, n_p) arrays
    off_norm_chunks = []   # matching shadow-chain LOG weight products
    off_step_chunks = []   # production step of every kept row

    production_start = time.time()
    chunk_start = production_start

    def _dispatch(step0, state):
        """Launch the chunk starting at step0 (asynchronously) from
        ``state = (walkers_sharded, carried_sharded, rng_key)``; returns the
        state the next chunk continues from and the chunk's outputs."""
        walkers_sharded, carried_sharded, rng_key = state
        n_chunk = min(print_every, n_steps - step0)
        _t, measured_np = _measured_steps(step0, n_chunk, n_steps, plan)
        outputs = chunk_fn(walkers_sharded, carried_sharded, rng_key,
                           jnp.asarray(measured_np),
                           jnp.asarray(_t + step_offset, dtype=jnp.int32), avg_log_weight)
        # the key advanced through the chunk; replicated, so take device 0.
        # This is what the NEXT chunk (and the next call) continues from.
        return ((outputs[0], outputs[1], outputs[2][0]),
                (step0, n_chunk, _t, measured_np, outputs))

    # Pipelined as in driver._production: chunk k+1 is dispatched BEFORE
    # chunk k's outputs are pulled to the host, so the devices are not idle
    # while the host works.  Dispatch is asynchronous and the state chain is
    # strictly sequential, so every result is that of the unpipelined loop.
    state = (walkers_sharded, carried_sharded, rng_key)
    state, pending = _dispatch(0, state)
    while pending is not None:
        step, n_chunk, _t, measured_np, outputs = pending
        walkers_sharded, carried_sharded, rng_key = state   # after this chunk
        if step + n_chunk < n_steps:
            state, pending = _dispatch(step + n_chunk, state)
        else:
            pending = None
        log_G_chunk = outputs[3]
        E_chunk = outputs[4]
        out_idx = 5

        # G_chunk, E_chunk replicated across devices; take device 0
        weights_chunks.append(np.asarray(log_G_chunk[0]))
        E_chunks.append(np.asarray(E_chunk[0]))
        _check_population(weights_chunks[-1], step, E_chunks[-1])
        if n_off > 0:
            # one slot per step (lax.cond branches share a shape); keep only
            # the measured rows, so raw['off_diag'] is compact as in run_gfmc
            off_chunk = outputs[out_idx]  # (n_dev, n_chunk, n_off, n_p)
            out_idx += 1
            off_diag_chunks.append(np.asarray(off_chunk[0])[measured_np])
            nrm_chunk = outputs[out_idx]
            out_idx += 1
            off_norm_chunks.append(np.asarray(nrm_chunk[0])[measured_np])
            off_step_chunks.append(_t[measured_np].astype(np.int64))
        del outputs
        step += n_chunk

        if verbosity >= 1:
            elapsed = time.time() - chunk_start
            it_per_sec = n_chunk / max(elapsed, 1e-9)
            line = (f"[gfmc-pmap] - step: {step}/{n_steps}, "
                    f"{it_per_sec:.1f} it/s, n_devices={n_devices}")
            if verbosity >= 3 and step - 1 > progress_window:
                weights_np = np.concatenate(weights_chunks)
                E_np       = np.concatenate(E_chunks)
                results, _ = _estimate_observables(p_array, weights_np, E_np,
                                                  cutoff=step - 1)
                line += f", Energy[p]: {results}"
            emit(line, verbosity)
            chunk_start = time.time()

    return SimpleNamespace(
        walkers_sharded=walkers_sharded, rng_key_final=rng_key,
        weights_np=np.concatenate(weights_chunks),
        E_np=np.concatenate(E_chunks),
        off_diag_np=(np.concatenate(off_diag_chunks, axis=0)
                     if off_diag_chunks else None),      # (n_meas, n_off, n_p)
        off_norm_np=(np.concatenate(off_norm_chunks, axis=0)
                     if off_norm_chunks else None),
        off_steps_np=(np.concatenate(off_step_chunks)
                      if off_step_chunks else None),
        runtime=time.time() - production_start)


def run_gfmc_pmap(walkers: Walker, n_steps: int, hamiltonian: Operator,
                  step_size: Union[int, float], guide: Optional[Guide],
                  p_array: Union[Sequence[int], np.ndarray], *, method: str = "cont_time",
                  rng_key: Optional[jax.Array] = None, lam: Optional[float] = None,
                  diag_operators: Optional[Mapping[str, DiagonalOperator]] = None,
                  off_diag_operators: Optional[Mapping[str, Operator]] = None,
                  avg_log_weight: Optional[float] = None, warmup_steps: Optional[int] = None,
                  verbosity: int = 7, number_of_outputs: int = 100,
                  n_devices: Optional[int] = None, return_raw: bool = True,
                  off_diag_stride: int = 1, off_diag_batch_size: Optional[int] = None,
                  _step_offset: int = 0) -> tuple:
    """Multi-device GFMC main loop.  Energy + off-diagonal operators.

    Args mirror :func:`pygfmc.engine.driver.run_gfmc`, with the same defaults,
    so that :func:`run` behaves the same whatever the device count.  In
    addition:

    Args:
        n_devices: number of devices to shard walkers across (default: all
                   visible devices).  The walker count must be divisible by it.

    Differences from ``run_gfmc``:
      * ``diag_operators`` raise NotImplementedError (ancestry tracing is
        single-device only).
      * Off-diagonal measurement gating (``off_diag_stride > 1`` and the
        equal windows implied by ``return_raw=False``) works as in
        ``run_gfmc``: the same steps are measured, unmeasured steps skip the
        shadow chain, and ``raw['off_diag_steps']`` gives each row's step.
      * Progress output: a progress line is printed from the first chunk on
        (``run_gfmc`` starts once a full progress window has been walked),
        and the per-operator lines of ``verbosity >= 8`` are not printed.

    The warm-up follows ``run_gfmc``: the ramp runs whenever
    ``warmup_steps > 0``; a supplied ``avg_log_weight`` is kept fixed (the
    ramp then only equilibrates the walkers).  The default
    ``warmup_steps=None`` runs 1000 warm-up steps without an
    ``avg_log_weight`` and none with one; an explicit ``warmup_steps=0``
    without ``avg_log_weight`` raises.
    ``stats`` carries the same ``parameters/*`` entries as ``run_gfmc``,
    including ``parameters/rng_key_next``.

    Off-diagonal operators use the forward-walking pure estimator with global
    reconfiguration of the shadow ensemble.

    ``rng_key`` is the production stream's starting position, as in
    ``run_gfmc``: keys advance ``key_{t+1} = split(key_t)`` in the scan carry
    and the warm-up, positivity check and statistics draw from side-streams,
    so the shadow chain replays the main chain's own future draws (correlated
    sampling) and passing ``stats['parameters/rng_key_next']`` back as
    ``rng_key`` (with the returned walkers, ``warmup_steps=0`` and the
    returned ``avg_log_weight``) continues the chain bit-identically on the
    same device count.  The
    chain is NOT the single-device chain at equal seed: the evolve key carries a ``fold_in(device_id)`` and reconfiguration
    is global over all shards, so sharding changes the trajectory — as a
    different seed would, not as a bias.

    ``stats`` carries the same :func:`pygfmc.engine.estimators.statistics` fields as
    ``run_gfmc`` — sampler health, the offspring distribution and ancestor
    survival / ``effective_sample_size`` — so a consumer of :func:`run` sees
    one contract whatever the device count.  That pass runs on the
    UNSHARDED ensemble and therefore on a single device: a population that
    only fits when sharded can OOM there, after production has finished.

    Returns the same tuple shape as :func:`pygfmc.engine.driver.run_gfmc`: four values
    when ``return_raw=False`` -- ``stats`` always comes back -- and five
    (adding ``raw``) when ``return_raw=True``.
    """
    wall_start = time.time()
    rng_key = _default_rng_key(rng_key, verbosity, "run_gfmc_pmap")
    if isinstance(rng_key, np.ndarray):
        # a key read back from stats or HDF5; as a JAX array it matches the
        # signature of the scans compiled by the run it continues
        rng_key = jnp.asarray(rng_key)
    key_data(rng_key)
    require_operator_protocol(hamiltonian, role="hamiltonian")
    for _name, _op in (off_diag_operators or {}).items():
        require_operator_protocol(_op, role=f"off-diagonal operator {_name!r}")
    require_guide_protocol(guide)
    require_incremental_support(guide, hamiltonian)
    if n_devices is None:
        n_devices = jax.device_count()
    n_devices = _validated_int("n_devices", n_devices, 1)
    n_steps = _validated_int("n_steps", n_steps, 1)
    require_step_size(method, step_size)
    lam_given = lam                   # recorded in stats as the caller gave it
    lam = require_lam(method, lam)
    warmup_steps = _validated_int(
        "warmup_steps", _default_warmup(avg_log_weight, warmup_steps), 0)
    number_of_outputs = _validated_int(
        "number_of_outputs", number_of_outputs, 1)
    if n_devices > jax.local_device_count():
        raise ValueError(
            f"n_devices={n_devices} exceeds the {jax.local_device_count()} "
            "locally addressable JAX devices.")
    off_diag_stride = _validated_int(
        "off_diag_stride", off_diag_stride, 1)
    N_walker = walkers.config.shape[0]
    if off_diag_batch_size is not None:
        # op_batch_size divides the PER-DEVICE shard, not the global walker
        # count -- pmap splits the walkers n_devices ways first.
        off_diag_batch_size = _validated_int(
            "off_diag_batch_size", off_diag_batch_size, 1)
        if N_walker % n_devices == 0 and (N_walker // n_devices) % off_diag_batch_size:
            raise ValueError(
                f"off_diag_batch_size={off_diag_batch_size} must divide the "
                f"PER-DEVICE shard size {N_walker // n_devices} "
                f"(= n_walkers // n_devices), not the global walker count")
    n_p_given = len(p_array) if p_array is not None else 0
    p_array = _validated_p_array(p_array)
    if len(p_array) != n_p_given:
        emit(f"[gfmc.run_gfmc_pmap] dedup'd p_array: {n_p_given - len(p_array)} "
             f"duplicate value(s) dropped, using {p_array}", verbosity)
    off_diag_names = list(off_diag_operators.keys()) if off_diag_operators else []
    n_off = len(off_diag_names)
    if diag_operators:
        raise NotImplementedError(
            "run_gfmc_pmap does not support diag_operators: the ancestry-corrected "
            "mixed estimator used by run_gfmc would require global parent-index "
            "tracking across devices, which is not implemented.  Use run_gfmc on a "
            "single device, or express the observable as an off_diag operator."
        )
    # The window grid and the measurement gating, planned exactly as in
    # run_gfmc, so the two backends measure the same steps.
    plan = _measurement_plan(n_steps, p_array, n_off, off_diag_stride, return_raw)
    _require_warmup_or_alw(avg_log_weight, warmup_steps)
    m_array = plan.m_array

    walkers_sharded = shard_walkers(walkers, n_devices)
    if n_off > 0:
        # Same rule as run_gfmc: an operator with an off-diagonal transition
        # (M_total > 1) needs even p; a purely diagonal one accepts odd p.
        has_off_diag = any(int(op.M_total) > 1 for op in off_diag_operators.values())
        bad = [p for p in p_array if p % 2 != 0] if has_off_diag else []
        if bad:
            raise ValueError(
                f"All p values must be even when off_diag_operators have "
                f"off-diagonal elements (M_total > 1); got {bad}"
            )

    cache_h = guide_cache(guide, hamiltonian)
    cache_off = {name: guide_cache(guide, op)
                 for name, op in (off_diag_operators or {}).items()}
    step_f = float(step_size)

    # Same key discipline as run_gfmc: the caller's rng_key is the production
    # stream and is NOT split here; the warm-up, the positivity probe and the
    # statistics draw from tagged side-streams off aux_root.
    rng_key_in = rng_key
    aux_root = jax.random.fold_in(rng_key, _AUX_ROOT)
    carried_sharded = _initial_carried_pmap(walkers_sharded, hamiltonian, guide, cache_h)
    walkers_sharded, carried_sharded, avg_log_weight = _warmup_pmap(
        walkers_sharded, carried_sharded, hamiltonian, guide, method, step_f, lam,
        cache_h, avg_log_weight, warmup_steps, aux_root, verbosity)
    # One pmap signature for the probe and the production chunks whether the
    # shift was pinned (a Python float) or measured (an array);
    # stats['avg_log_weight'] records the value as given.
    avg_log_weight_given = avg_log_weight
    avg_log_weight = _traced_alw(avg_log_weight, walkers.log_weight.dtype)

    if off_diag_operators:
        # As in run_gfmc: probe the operators on an ensemble evolved along a
        # side-stream, so configurations typical of the walk are tested
        # rather than the (possibly artificial) starting ones.
        probe_fn = _make_step_chunk(hamiltonian, lam, guide, method, step_f,
                                    cache_h=cache_h)
        probe_key = jax.random.fold_in(aux_root, _AUX_POSITIVITY)
        probe_sharded = probe_fn(
            walkers_sharded, carried_sharded, probe_key,
            jnp.zeros(_POSITIVITY_PROBE_STEPS, dtype=bool),
            jnp.arange(_POSITIVITY_PROBE_STEPS, dtype=jnp.int32), avg_log_weight)[0]
        check_off_diag_operator_positivity(
            unshard_walkers(probe_sharded), off_diag_operators, guide,
            caches=cache_off, verbose=verbosity >= 1)
        del probe_fn, probe_sharded

    # ── production ──────────────────────────────────────────────────────────
    chunk_fn = _make_step_chunk(hamiltonian, lam, guide, method, step_f,
                                off_diag_operators=off_diag_operators,
                                m_array=m_array, cache_h=cache_h,
                                cache_off=cache_off,
                                op_batch_size=off_diag_batch_size,
                                gate_off_diag=plan.gate_off_diag)
    print_every = max(1, n_steps // max(1, number_of_outputs))

    prod = _production_pmap(
        walkers_sharded=walkers_sharded, carried_sharded=carried_sharded,
        step_offset=int(_step_offset), rng_key=rng_key, chunk_fn=chunk_fn,
        avg_log_weight=avg_log_weight,
        n_steps=n_steps,
        print_every=print_every, plan=plan, n_off=n_off, n_devices=n_devices,
        verbosity=verbosity)
    walkers_sharded, rng_key = prod.walkers_sharded, prod.rng_key_final
    weights_np, E_np = prod.weights_np, prod.E_np
    off_diag_np, off_norm_np, off_steps_np = (prod.off_diag_np, prod.off_norm_np,
                                              prod.off_steps_np)
    production_runtime = prod.runtime

    final_results, obs_results = _estimate_observables(
        p_array, weights_np, E_np,
        off_diag_np=off_diag_np, op_names=off_diag_names or None,
        m_array=m_array, off_diag_norm_np=off_norm_np,
        off_diag_steps=off_steps_np,
    )

    walkers_final = unshard_walkers(walkers_sharded)
    # Same split as run_gfmc: diagnostics/timing in ``stats``, GFMC values in
    # ``raw``, assembled by the shared helpers so a consumer of ``run`` sees
    # one contract whatever the device count.  The end-of-run statistics()
    # pass is single-device: it drives its own evolution loop over the whole
    # UNSHARDED population, so its transient buffers are sized for N_walkers
    # on ONE device, not N_walkers/n_devices.  A population that only fits
    # when sharded can therefore OOM here, at the END of a long production
    # run; call statistics() on a slice of the ensemble in that case (n_eff
    # then describes the slice).
    _lineage_warning(m_array, n_steps, verbosity,
                     "[multi_gpu] - [run_gfmc_pmap] -",
                     ", and they run on ONE device over the full population.")
    # keyed by operator name, exactly as run_gfmc returns them: the two entry
    # points must hand back the same shape of object, or a caller that swaps
    # `run` for `run_gfmc` silently gets a different raw layout.
    off_diag_by_name = _off_by_name(off_diag_np, off_diag_names)
    stats = _run_stats(
        walkers=walkers_final, hamiltonian=hamiltonian, step_f=step_f,
        guide=guide, method=method, aux_root=aux_root,
        avg_log_weight=avg_log_weight, lam=lam, lam_given=lam_given,
        cache_h=cache_h, plan=plan, n_steps=n_steps,
        warmup_steps=warmup_steps, verbosity=verbosity,
        number_of_outputs=number_of_outputs, diag_operators=None,
        off_diag_operators=off_diag_operators, rng_key_in=rng_key_in,
        rng_key_next=rng_key, E_np=E_np, weights_np=weights_np, diag_np=None,
        off_diag_np=off_diag_by_name, production_runtime=production_runtime,
        wall_start=wall_start, number_of_devices=n_devices,
        avg_log_weight_given=avg_log_weight_given)
    # stats always comes back -- it is the run's diagnostic record and does not
    # depend on keeping the per-step series; return_raw gates ``raw`` alone.
    if return_raw:
        raw = _raw_series(
            weights_np=weights_np, E_np=E_np, diag_np=None,
            off_diag_np=off_diag_by_name,
            off_diag_norm_np=_off_by_name(off_norm_np, off_diag_names),
            off_diag_steps_np=off_steps_np, additional_diag_np=None,
            plan=plan, diag_names=[], off_diag_names=off_diag_names)
        return walkers_final, final_results, obs_results, stats, raw
    return walkers_final, final_results, obs_results, stats


def _initial_carried_pmap(walkers_sharded, hamiltonian, guide, cache_h):
    """Per-shard counterpart of :func:`pygfmc.engine.driver._initial_carried`
    (the size guard applies to the whole ensemble)."""
    n_dev, n_per = walkers_sharded.config.shape[:2]
    # Across devices the carried rates travel through the all_gather of the
    # global reconfiguration, M_total floats per walker on top of the
    # configuration's N_sites, so on a multi-GPU system the saved rate
    # evaluation competes with the extra collective traffic.  Which side
    # wins depends on the hardware; PYGFMC_PMAP_CARRY_RATES=0 keeps the rates
    # local to a step (the guide state, N_sites per walker, is still carried).
    with_rates = (_PMAP_CARRY_RATES and rates_fit_budget(
        n_dev * n_per, hamiltonian, walkers_sharded.log_weight.dtype,
        _carry_rates_bytes()))
    return jax.pmap(lambda w: initial_carried(w, hamiltonian, guide, cache_h,
                                              with_rates=with_rates))(walkers_sharded)


def _warmup_pmap(walkers_sharded, carried_sharded, hamiltonian, guide, method,
                 step_f, lam, cache_h, avg_log_weight, warmup_steps, aux_root,
                 verbosity):
    """The warm-up of :func:`run_gfmc_pmap`: estimate ``avg_log_weight``
    unless it is given, then run the ramp stages of
    :func:`pygfmc.engine.driver._warmup_schedule` on the sharded ensemble.

    Same semantics as the single-device warm-up: the ramp runs whenever
    ``warmup_steps > 0``; a pinned ``avg_log_weight`` only stops it from
    being UPDATED, the walkers are still equilibrated.  Each stage draws
    from its own side-stream, derived exactly as in ``run_gfmc``.  Returns
    ``(walkers_sharded, carried_sharded, avg_log_weight)``.
    """
    alw_pinned = avg_log_weight is not None
    if not alw_pinned:
        def estimate_shard(walkers_shard):
            return estimate_average_log_branch_weight(
                walkers_shard, hamiltonian, guide,
                method=method, lam=lam, cache=cache_h)

        shard_estimates = jax.pmap(estimate_shard)(walkers_sharded)
        avg_log_weight = jnp.mean(shard_estimates)
        avg_log_weight.block_until_ready()
    ramp = _warmup_schedule(method, step_f, warmup_steps)
    for stage_idx, (step_stage, n_warmup) in enumerate(ramp):
        chunk_fn = _make_step_chunk(
            hamiltonian, lam, guide, method, step_stage, cache_h=cache_h)
        warm_key = _warmup_stage_key(aux_root, n_warmup, step_stage)
        t0 = time.time()
        walkers_sharded, carried_sharded, _wk, log_G_chunk, _ = chunk_fn(
            walkers_sharded, carried_sharded, warm_key,
            jnp.zeros(int(n_warmup), dtype=bool),
            jnp.arange(int(n_warmup), dtype=jnp.int32),
            _traced_alw(avg_log_weight, walkers_sharded.log_weight.dtype))
        log_G_chunk[0].block_until_ready()
        elapsed = time.time() - t0
        measured_alw = avg_log_weight + (
            jax.scipy.special.logsumexp(log_G_chunk[0])
            - jnp.log(jnp.asarray(n_warmup, dtype=log_G_chunk.dtype))) / step_stage
        if not alw_pinned:
            avg_log_weight = measured_alw
        emit(f"[gfmc-pmap] - [warmup {stage_idx + 1}/{len(ramp)}] - "
             f"step={step_stage}, {n_warmup} steps @ "
             f"{n_warmup / elapsed:.1f} it/s — "
             f"avg_log_weight = {float(avg_log_weight):.6f}"
             + (f" (pinned; this stage measured "
                f"{float(measured_alw):.6f})" if alw_pinned else ""),
             verbosity, 8)
    if not ramp:
        emit(f"[gfmc-pmap] - no warmup (warmup_steps=0), "
             f"avg_log_weight={float(avg_log_weight):.6f}", verbosity, 8)
    return walkers_sharded, carried_sharded, avg_log_weight


# ── auto-dispatch ───────────────────────────────────────────────────────────

def run(walkers: Walker, n_steps: int, hamiltonian: Operator,
        step_size: Union[int, float], guide: Optional[Guide], p_array: Union[Sequence[int], np.ndarray],
        *, method: str = "cont_time", rng_key: Optional[jax.Array] = None,
        lam: Optional[float] = None, **kwargs) -> tuple:
    """Auto-dispatch GFMC: run_gfmc_pmap on >1 devices, run_gfmc otherwise.

    The recommended entry point.  Drop-in replacement for
    :func:`pygfmc.engine.driver.run_gfmc` — same positional arguments and keyword
    arguments (plus ``n_devices``, default ``jax.device_count()``), same
    return tuple.  ``lam`` is needed only by the discrete methods.  When more
    than one device is used it dispatches to :func:`run_gfmc_pmap`, which
    shards walkers across devices; otherwise it calls ``run_gfmc`` unchanged.

    Runs with ``diag_operators`` fall back to the single-device backend.
    Note that the two backends differ in a few details (see
    :func:`run_gfmc_pmap`).
    """
    n_devices = kwargs.pop('n_devices', None)
    if n_devices is None:
        n_devices = jax.device_count()
    else:
        n_devices = _validated_int("n_devices", n_devices, 1)

    # Off-diagonal measurement shards: each device evolves its own shadow
    # shard, the shadow ensemble branches globally (reconfiguration_global
    # with aux=vals), and the shadow replays the main chain's own future step
    # keys -- the same correlated-sampling estimator run_gfmc uses, with the
    # same measurement gating.  Only diagonal operators (ancestry tracing)
    # send a run back to one device.
    unsupported_pmap = bool(kwargs.get('diag_operators'))
    # NOTE: off_diag_batch_size is supported on BOTH paths, so it is
    # deliberately absent from this list.  It COMPOSES with sharding rather
    # than competing with it -- pmap splits the walkers n_devices ways and
    # the batch size then chunks within each shard, so per-device peak goes
    # as (n_walkers/n_devices) * (op_batch_size/N_per).  In the pmap path
    # it must divide the PER-DEVICE shard, not the global walker count.
    # Both paths advance the same key recurrence.

    if n_devices > 1 and not unsupported_pmap:
        return run_gfmc_pmap(
            walkers, n_steps, hamiltonian, step_size, guide,
            p_array, method=method, rng_key=rng_key, lam=lam, n_devices=n_devices, **kwargs,
        )
    return run_gfmc(
        walkers, n_steps, hamiltonian, step_size, guide,
        p_array, method=method, rng_key=rng_key, lam=lam, **kwargs,
    )
