"""Estimators and diagnostics.

Windowed mixed/pure estimators (:func:`_estimate_observables`), sampler and
lineage diagnostics (:func:`statistics`), autocorrelation times, and small helpers used by the run drivers.
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Optional, Sequence, Union
from collections.abc import Mapping
from functools import partial
from types import SimpleNamespace

from .._output import emit
from .evolution import (transition_rates, guide_cache, require_lam,
                        initial_carried, evolve_carried, gather_carried, refresh_carried,
                        refresh_carried_at, _STATE_REFRESH_STEPS, _carry_rates_bytes,
                        rates_fit_budget)
from ..walkers import Walker, reconfiguration
from ..protocols import Operator, Guide
from .._precision import float_dtype

#: Configurations the off-diagonal positivity check samples.
_N_POSITIVITY_CHECK = 100
#: Offspring bins of ``statistics``: exact counts 0 .. _OFFSPRING_BINS-1 plus
#: one overflow bin.
_OFFSPRING_BINS = 4


def _validated_p_array(p_array):
    """Return sorted unique positive integer estimator windows."""
    if p_array is None or len(p_array) == 0:
        raise ValueError("p_array must contain at least one positive window size.")
    checked = []
    for p in p_array:
        p_int = int(p)
        if float(p_int) != float(p):
            raise ValueError(f"p_array values must be integers, got {p!r}.")
        if p_int <= 0:
            raise ValueError(f"p_array values must be > 0, got {p_int}.")
        checked.append(p_int)
    return sorted(set(checked))


def _validated_steps(steps, n_rows, n_steps):
    """``off_diag_steps`` as an int64 array: one-dimensional, one entry per
    stored row, integer-valued, strictly increasing and within
    ``[0, n_steps)``.  Anything else raises; nothing is cast or clipped
    silently, since a wrong step silently misplaces a measurement."""
    arr = np.asarray(steps)
    if arr.ndim != 1 or arr.shape[0] != n_rows:
        raise ValueError(
            f"off_diag_steps must be a 1-D array with one entry per stored "
            f"row ({n_rows}), got shape {arr.shape}.")
    if arr.size == 0:
        return arr.astype(np.int64)
    if arr.dtype.kind not in "iu":
        if arr.dtype.kind != "f" or not np.all(np.isfinite(arr)) \
                or not np.all(arr == np.floor(arr)):
            raise ValueError("off_diag_steps must hold integer step indices.")
    arr = arr.astype(np.int64)
    if np.any(np.diff(arr) <= 0):
        raise ValueError("off_diag_steps must be strictly increasing.")
    if arr[0] < 0 or arr[-1] >= n_steps:
        raise ValueError(
            f"off_diag_steps must lie in [0, {n_steps}), got "
            f"[{int(arr[0])}, {int(arr[-1])}].")
    return arr


def autocorrelation(series: np.ndarray, c: float = 5.0) -> tuple[float, int]:
    """Integrated autocorrelation time of a per-step series.

    Returns ``(tau_int, window)``.  ``tau_int`` converts steps into independent
    samples -- ``N_eff = N / (2 tau_int)`` -- and inflates the error on a mean
    by ``sqrt(2 tau_int)``.

    rho(t) is estimated by FFT in O(N log N). The self-consistent window is
    the smallest M with ``M >= c * (0.5 + sum(abs(rho[1:M+1])))``. The absolute
    correlation envelope prevents premature termination for oscillating
    correlations; the returned time uses the SIGNED sum. Anticorrelated
    series can legitimately have ``0 < tau_int < 0.5``.

    ``tau_int`` is NaN for insufficient/constant/nonfinite data, when no
    window is found, or when the truncated signed sum is nonpositive. These
    cases do not support a reliable effective-sample-count estimate.

    NOTE this is the tau for statistics and budgeting.  The initial transient
    decays with the SLOWEST MODE instead, which is longer, so it is the wrong
    number for choosing a warm-up length.
    """
    if not np.isfinite(c) or c <= 0:
        raise ValueError("c must be finite and positive.")
    x = np.asarray(series, dtype=float).ravel()
    n = x.size
    if n < 16 or not np.all(np.isfinite(x)):
        return float("nan"), 0
    d = x - x.mean()
    acov0 = float(d @ d) / n
    if acov0 <= 0.0:                      # constant series: tau is undefined
        return float("nan"), 0
    m = 1 << (2 * n - 1).bit_length()     # next power of two >= 2n
    f = np.fft.rfft(d, m)
    acov = np.fft.irfft(f * np.conj(f), m)[:n].real / np.arange(n, 0, -1)
    rho = acov / acov[0]
    run = 0.5 + np.cumsum(rho[1:])
    M = np.arange(1, run.size + 1)
    # Cancellation between alternating correlations must not terminate the
    # window before their slowly decaying envelope has been sampled.
    envelope = 0.5 + np.cumsum(np.abs(rho[1:]))
    hit = np.flatnonzero(M >= c * envelope)
    if hit.size == 0:
        return float("nan"), 0
    i = int(hit[0])
    if run[i] <= 0 or not np.isfinite(run[i]):
        return float("nan"), int(M[i])
    return float(run[i]), int(M[i])


def dense_and_names(series, names=None, what="off_diag_np"):
    """``{name: (n_rows, ...)}`` -> ``(dense (n_rows, n_ops, ...), names)``.

    The off-diagonal series are keyed by operator name like the diagonal ones,
    but unlike them they always share a shape -- one scalar per operator per
    window -- so a dense array with a positional operator axis exists and the
    estimator uses it internally.  This converts at the boundary and leaves a
    dense array (plus its separate names) untouched.

    Args:
        series: mapping name -> array, a dense array, or None.
        names:  optional names; a mapping must hold exactly these operators
                and is converted in this order.
        what:   label used in error messages.

    Returns:
        ``(dense_or_None, names_or_None)``.
    """
    if isinstance(series, Mapping):
        keys = list(series)
        if not keys:
            return None, None
        if names is not None:
            names = list(names)
            if sorted(map(str, names)) != sorted(map(str, keys)):
                raise ValueError(
                    f"{what} is keyed by {keys} but the operator names are "
                    f"{names}; the two must hold the same operators.")
            keys = names                       # the names' order, not the mapping's
        return np.stack([np.asarray(series[k]) for k in keys], axis=1), keys
    return (None if series is None else np.asarray(series)), names


def _series_tau(arr, scale=1):
    """tau_int of every series in ``arr``, scaled to step units.

    ``arr`` is ``(n_steps, *trailing)`` and the result carries the trailing
    shape: ``(n_p,)`` for a scalar operator, ``(n_vals, n_p)`` for a
    vector-valued one -- one tau per value per window.  Components are not
    averaged, since they can decorrelate very differently (e.g. S(k=0) is
    conserved in a fixed-Sz sector while S(k=pi) is not).
    """
    a = np.asarray(arr)
    if a.ndim == 1:
        a = a[:, None]
    trailing = a.shape[1:]
    flat = a.reshape(a.shape[0], -1)
    taus = np.array([autocorrelation(flat[:, j])[0] * scale
                     for j in range(flat.shape[1])], dtype=float)
    return taus.reshape(trailing)


def autocorrelation_stats(E_np, log_weights_np, diag_np=None, diag_names=None,
                          off_diag_np=None, off_diag_names=None,
                          off_diag_stride=1):
    """``stats`` entries under ``autocorrelation/`` for one run.

    Returns a dict with keys ``autocorrelation/energy_estimator``,
    ``autocorrelation/log_weights``, ``autocorrelation/diag/<name>`` and
    ``autocorrelation/off_diag/<name>`` (integrated autocorrelation times in
    main-chain branching steps, see :func:`autocorrelation`).

    ``energy`` and ``log_weights`` are clean per-step series, so their tau is
    the physical decorrelation time of the walker ensemble.

    The per-observable entries are one tau per projection window p -- and for
    a vector-valued diagonal operator one tau per value per window, shape
    ``(n_vals, n_p)``, since its components need not decorrelate alike.  They
    are NOT purely physical: ``raw['diag'][name][..., j]`` holds a value already carried
    through m = p/2 ancestry steps, so consecutive entries share m-1 of them and
    tau grows with p by construction.  That makes them the right input for
    choosing ``off_diag_stride`` -- redundancy is redundancy whatever its
    cause -- and the WRONG input for an error bar, which should come from
    blocking the estimator instead (``evaluate_observables(..., n_blocks=...)``).

    tau is observable-specific and can differ by an order of magnitude between
    observables of the same run, so the energy's tau is not a usable proxy.

    Off-diagonal rows are already strided, so their tau is multiplied by
    ``off_diag_stride`` to come out in main-chain steps.  It is the tau of the
    stored forward-walk values, i.e. of the estimator's numerator.

    THESE TAUS ARE IN BRANCHING STEPS, NOT IMAGINARY TIME.  Decorrelation is
    a process in imaginary time, so a tau_int quoted in steps is only
    meaningful alongside the step_size it was measured at; convert through

        tau_between = off_diag_stride * step_size

    before carrying a stride between runs with different ``step_size``.
    """
    out = {}
    out["autocorrelation/energy_estimator"] = autocorrelation(E_np)[0]
    out["autocorrelation/log_weights"] = autocorrelation(log_weights_np)[0]
    if diag_np is not None:
        items = (diag_np.items() if isinstance(diag_np, Mapping)
                 else ((name, diag_np[:, k])
                       for k, name in enumerate(diag_names or ())))
        for name, arr in items:
            out[f"autocorrelation/diag/{name}"] = _series_tau(arr)
    if off_diag_np is not None:
        items = (off_diag_np.items() if isinstance(off_diag_np, Mapping)
                 else ((name, off_diag_np[:, k])
                       for k, name in enumerate(off_diag_names or ())))
        for name, arr in items:
            out[f"autocorrelation/off_diag/{name}"] = _series_tau(
                arr, scale=int(off_diag_stride))
    return out


def _diag_series_by_name(diag_op_np, diag_op_names):
    """``(diag_by_name, diag_op_names)``: the diagonal series as a mapping
    name -> array.

    ``diag_op_np`` is either the per-name mapping run_gfmc returns in
    ``raw['diag']`` -- {name: (n_steps, [n_vals], n_p)} -- or a dense array
    with a positional operator axis (a hand-built raw), which needs its
    names alongside.  The estimator works on the mapping; the dense form
    is split into one here.
    """
    if isinstance(diag_op_np, Mapping):
        if diag_op_names is None:
            diag_op_names = list(diag_op_np)
        elif list(diag_op_names) != list(diag_op_np):
            raise ValueError(
                f"diag_op_names {list(diag_op_names)} does not match the keys "
                f"of diag_op_np {list(diag_op_np)}; pass one or the other.")
        diag_by_name = {k: np.asarray(v) for k, v in diag_op_np.items()}
    elif diag_op_np is None:
        diag_by_name = None
    else:
        if diag_op_names is None:
            raise ValueError(
                "diag_op_np and diag_op_names must be supplied together "
                "(a dense diag_op_np carries no names; pass a {name: array} "
                "mapping instead to skip them).")
        _dense = np.asarray(diag_op_np)
        if _dense.ndim not in (3, 4):
            raise ValueError(
                f"diag_op_np must have 3 or 4 dimensions, got {_dense.shape}.")
        if _dense.shape[1] != len(diag_op_names):
            raise ValueError(
                "diag_op_np axes must be (n_samples, n_operators, [...,] n_p); "
                f"got {_dense.shape} against {len(diag_op_names)} names.")
        diag_by_name = {name: _dense[:, i]
                        for i, name in enumerate(diag_op_names)}
    return diag_by_name, diag_op_names


def _energy_and_diagonal_estimates(sorted_p, log_w_cumsum, E, diag_by_name,
                                   m_max_common):
    """The windowed energy and the ancestry-traced diagonal estimates at
    every p (formulas in :func:`_estimate_observables`).  Diagonal windows
    start at ``i >= m_max_common`` so that, with off-diagonal operators in
    the same call, every p uses one set of windows for either kind."""
    energy_results = {}
    obs_results = {}
    if diag_by_name:
        for name in diag_by_name:
            obs_results[name] = {}
    for j, p in enumerate(sorted_p):
        log_G_p = log_w_cumsum[p:] - log_w_cumsum[:-p]
        if len(log_G_p) == 0:  # guarded by validation above
            raise RuntimeError("validated energy window became empty")
        shift   = np.max(log_G_p)
        G_p     = np.exp(log_G_p - shift)
        G_p_sum = float(np.sum(G_p))
        energy_results[p] = float(np.sum(G_p * E[p:])) / G_p_sum
        if diag_by_name:
            log_G_p_trim = log_G_p[m_max_common:]
            if len(log_G_p_trim) == 0:  # guarded by validation above
                raise RuntimeError("validated diagonal window became empty")
            shift_trim   = np.max(log_G_p_trim)
            G_p_trim     = np.exp(log_G_p_trim - shift_trim)
            G_p_trim_sum = float(np.sum(G_p_trim))
            for name, arr in diag_by_name.items():
                # arr: (n_steps, n_p) for a scalar operator,
                #      (n_steps, n_vals, n_p) for a vector-valued one
                if arr.ndim == 2:
                    D = arr[:, j][p:][m_max_common:]                   # (n_steps,)
                    obs_results[name][p] = float(np.sum(G_p_trim * D)) / G_p_trim_sum
                else:
                    D = arr[:, :, j][p:][m_max_common:]                # (n_steps, n_vals)
                    obs_results[name][p] = (np.sum(G_p_trim[:, None] * D, axis=0)
                                            / G_p_trim_sum)  # (n_vals,)

    return energy_results, obs_results


def _off_diagonal_estimates(sorted_p, m_array, log_w_cumsum, off_diag_np,
                            off_diag_norm_np, off_diag_steps, op_names, n,
                            m_max_common):
    """The forward-walking pure estimates ``{name: {p: value}}`` of the
    off-diagonal operators (formulas in :func:`_estimate_observables`)."""
    obs_results = {name: {} for name in op_names}
    L = np.concatenate([[0.0], log_w_cumsum])
    # Which main-chain step each stored row was launched at: the rows say
    # so themselves, which is what lets the measurement grid be per-call
    # and still survive concatenation across runs.
    k_all    = np.asarray(off_diag_steps, dtype=np.int64)
    for j, (p, m) in enumerate(zip(sorted_p, m_array)):
        count = n - m - p
        if count <= 0:  # guarded by validation above
            raise RuntimeError("validated off-diagonal window became empty")
        # Select the measured window starts once per p, shared by all
        # operators.  Restrict numerator and denominator to the same rows.
        i_sel_lo = max(0, m_max_common - m)
        # Validated step indices are sorted. Locate the contiguous interval
        # without allocating a full-chain mask and integer gather per window.
        lo, hi = np.searchsorted(k_all, [p + i_sel_lo, p + count])
        rows = slice(lo, hi)
        i_sel = k_all[rows] - p
        if len(i_sel) == 0:
            raise ValueError(
                f"No stored off-diagonal measurement is usable for p={p}, "
                f"m={m}.")
        log_num = L[p + 1 + i_sel] - L[p - m + 1 + i_sel]
        log_den = L[p + m + 1 + i_sel] - L[p - m + 1 + i_sel]
        for idx, name in enumerate(op_names):
            fw      = off_diag_np[rows, idx, j]
            # Forward-walk layout (see forward_walk_operator): fw =
            # mean(v), linear and SIGNED; the shadow chain's accumulated
            # LOG weight product lives in off_diag_norm_np.
            #
            # That norm drifts to large negative values at deep windows
            # while the main chain's log_num carries the compensating
            # +O(m), so their SUM is O(1).  Exponentiating each factor
            # separately would underflow float32; adding in the exponent
            # and exponentiating ONCE keeps every depth representable.
            log_nrm   = off_diag_norm_np[rows, idx, j]
            log_terms = log_num + log_nrm
            finite    = np.isfinite(log_terms)
            shift = max(np.max(log_terms[finite]), np.max(log_den)) if finite.any() else np.max(log_den)
            num = float(np.sum(np.exp(log_terms - shift) * fw))
            den = float(np.sum(np.exp(log_den - shift)))
            obs_results[name][p] = num / den
    return obs_results


def _estimate_observables(p_array, log_weights, E, cutoff=None,
                         off_diag_np=None, op_names=None, m_array=None,
                         diag_op_np=None, diag_op_names=None,
                         off_diag_steps=None,
                         off_diag_norm_np=None):
    """Windowed importance-weighted estimator for energy and operators.

    PROJECTION DEPTH.  Throughout this package tau is the depth applied to EACH
    side of the bra-ket, i.e.

        tau = m * step_size ,      m = p // 2

    not the full p-step window.  Every estimator below is centred: the operator
    sits at the midpoint of the p-step window and each side is projected by m
    steps, so the bias decays as exp(-dE * tau) and an exponential fit against
    this tau returns the gap dE.  (Against p*step_size it would return dE/2.)
    It is also the same m the lineage is traced back for n_eff, so the two are
    directly comparable -- see `statistics`.

    The ENERGY is centred too, even though it is evaluated at the END of the
    window below.  [H, G] = 0, so <psi_T|G^m H G^m|psi_T> is identically
    <psi_T|H G^2m|psi_T> and H may be commuted to the edge; that is an
    implementation shortcut, not a deeper projection.  The energy converges
    twice as fast in the exponent -- exp(-2 dE tau) -- because H is diagonal in
    its own eigenbasis, so <psi_0|H|psi_1> = 0 kills the first-order term and
    the leading bias is second order in the guide error.  That is why the
    energy can read flat across a whole p grid on which an observable is still
    drifting; it is physics, not a different window.

    Energy (mixed estimator):

        <E>_p = Σ_i G_p(i) · E(i+p) / Σ_i G_p(i)

    where G_p(i) = weights(i+1) · … · weights(i+p).

    Diagonal operators (ancestry-traced, i.e. forward-walking, pure estimator):

        <D>_p = Σ_i G_p(i) · D(i+p) / Σ_i G_p(i)

    where

        D(i+p) ≡ diag_op_np[i+p, op_idx, p_idx]
               = (1/N_walker) Σ_α O_diag( x_{anc_α(m)}^{(i+m)} )

    is the diagonal-operator value at the step-(i+m) ancestor
    configuration of each walker α at step i+p, traced back m = p/2
    reconfigurations through the parent map.  This evaluates the
    operator at the midpoint of the p-step projection (m steps in
    from each end), so combined with the weighting Σ_i G_p(i)·D(i+p)
    it realises the pure estimator at projection length m on both
    bra and ket.  The naive per-step walker mean (no ancestry
    correction) is the biased single-sided variant.

    Off-diagonal operators (pure estimator, forward-walking).  Per
    measurement step k with m = p/2:

        numerator term    = G_main^{-m}(k) · G_aux^{+m}(k) · val(k)
        denominator term  = G_main^{-m}(k) · G_main^{+m}(k)

        <O>_p = Σ_k numerator term / Σ_k denominator term

    where G_main^{-m}(k) = weights(k-m+1) · … · weights(k) is the bra half of
    the p-step window centred on k, G_main^{+m}(k) = weights(k+1) · … ·
    weights(k+m) its ket half, and ``val(k)`` (``off_diag_np``) and
    ``log G_aux^{+m}(k)`` (``off_diag_norm_np``) come from
    :func:`pygfmc.engine.evolution.forward_walk_operator`: the operator is applied at
    step k and the ket half is supplied by a shadow chain propagated m steps
    from the operator-applied ensemble.  G_aux does NOT cancel; its ratio to
    G_main^{+m} is what converts mixed → pure.  k runs over the measured
    steps with p ≤ k < n_steps − m (plus the common trim described below),
    and both sums use the same set of k.

    When diagonal AND off-diagonal operators are estimated in one call, both
    are restricted to the common window starts i ≥ max(m_array) so that every
    p uses the same set of windows for either kind.

    Args:
        p_array:        list of window sizes
        log_weights:    1-D array of the per-step log branch weights log G_n
                        of the main walk (``raw['log_weights']``).
        E:              1-D array of per-step energy values
        cutoff:         truncate arrays to this length before computing
        off_diag_np:    forward-walk values, or None: a mapping name ->
                        (n_meas, n_p) as in ``raw['off_diag']``, or a dense
                        (n_meas, n_ops, n_p) array with ``op_names``.  One row
                        per MEASURED step; the row -> step mapping comes from
                        ``off_diag_steps``.  Window starts whose measurement
                        step was skipped are excluded from the windowed sums.
        op_names:       list of off-diagonal operator names, or None
        m_array:        list of m = p//2 values paired with sorted p_array, or None
        diag_op_np:     diagonal operator values, or None: a mapping name ->
                        (n_steps, n_p) / (n_steps, n_vals, n_p) as in
                        ``raw['diag']``, or a dense (n_steps, n_ops, [n_vals,]
                        n_p) array with ``diag_op_names``.
        diag_op_names:  list of diagonal operator names, or None
        off_diag_steps: (n_rows,) main-chain step index per stored row
                        (``raw['off_diag_steps']``), increasing.  REQUIRED
                        whenever off_diag_np is given.  The rows carry their
                        own positions, so the grid need not be uniform and
                        rows from several runs can be concatenated.
        off_diag_norm_np: the shadow chains' accumulated LOG weight products
                        (``raw['off_diag_norm']``), same shape/row mapping as
                        off_diag_np.  REQUIRED whenever off_diag_np is given.
                        It enters as a NUMERATOR FACTOR (log G_aux^{+m}
                        above), not as a divisor.

    Returns:
        energy_results : dict  p → float
        obs_results    : dict  name → dict  p → value, for all operators
                         (diag + off-diag); the value is a float, or an
                         ``(n_vals,)`` array for a vector-valued diagonal
                         operator.
    """
    weights = np.asarray(log_weights)
    E = np.asarray(E)
    if weights.ndim != 1 or E.ndim != 1:
        raise ValueError(
            f"log_weights and E must be one-dimensional; got {weights.shape} and {E.shape}.")
    if len(weights) != len(E):
        raise ValueError(
            f"log_weights and E must have equal lengths; got {len(weights)} and {len(E)}.")
    if len(weights) == 0:
        raise ValueError("log_weights and E must contain at least one sample.")

    sorted_p = _validated_p_array(p_array)
    if m_array is None:
        _m_array = [p // 2 for p in sorted_p]
    else:
        if len(m_array) != len(sorted_p):
            raise ValueError(
                f"m_array must have one entry per distinct p value; got "
                f"{len(m_array)} entries for {len(sorted_p)} windows.")
        _m_array = []
        for m in m_array:
            m_int = int(m)
            if float(m_int) != float(m) or m_int < 0:
                raise ValueError(
                    f"m_array values must be nonnegative integers, got {m!r}.")
            _m_array.append(m_int)

    off_diag_np, op_names = dense_and_names(off_diag_np, op_names, "off_diag_np")
    # the normalisation series are keyed like the values and must line up
    # with them operator by operator, so they are converted in the values'
    # name order (a mapping with other keys is rejected, not reordered)
    off_diag_norm_np, _ = dense_and_names(off_diag_norm_np, op_names,
                                          "off_diag_norm_np")

    diag_by_name, diag_op_names = _diag_series_by_name(diag_op_np, diag_op_names)
    if (diag_by_name is None) != (not diag_op_names):
        raise ValueError("diag_op_np and diag_op_names must be supplied together.")
    if (off_diag_np is None) != (op_names is None):
        raise ValueError("off_diag_np and op_names must be supplied together.")
    if off_diag_np is not None and off_diag_norm_np is None:
        raise ValueError(
            "off_diag_norm_np is required whenever off_diag_np is supplied.")
    if off_diag_np is not None and off_diag_steps is None:
        raise ValueError(
            "off_diag_steps is required whenever off_diag_np is supplied: it "
            "records the main-chain step each row was measured at.")

    off_diag_np = None if off_diag_np is None else np.asarray(off_diag_np)
    off_diag_norm_np = (None if off_diag_norm_np is None
                        else np.asarray(off_diag_norm_np))
    if off_diag_np is not None:
        off_diag_steps = _validated_steps(off_diag_steps, off_diag_np.shape[0],
                                          len(log_weights))

    if cutoff is not None:
        cutoff_int = int(cutoff)
        if float(cutoff_int) != float(cutoff) or cutoff_int <= 0:
            raise ValueError(f"cutoff must be a positive integer, got {cutoff!r}.")
        cutoff = min(cutoff_int, len(weights))
        weights = weights[:cutoff]
        E = E[:cutoff]
        if off_diag_np is not None:
            _keep = int(np.searchsorted(np.asarray(off_diag_steps),
                                        cutoff - 1, side="right"))
            off_diag_steps = np.asarray(off_diag_steps)[:_keep]
            off_diag_np = off_diag_np[:_keep]
            if off_diag_norm_np is not None:
                off_diag_norm_np = off_diag_norm_np[:_keep]
        if diag_by_name is not None:
            diag_by_name = {k: v[:cutoff] for k, v in diag_by_name.items()}

    n = len(weights)
    if not np.all(np.isfinite(E)):
        raise ValueError("Energy samples contain nonfinite values.")
    too_large = [p for p in sorted_p if p >= n]
    if too_large:
        raise ValueError(
            f"Each p value must be smaller than the number of available "
            f"samples ({n}); invalid values: {too_large}.")

    if diag_by_name is not None:
        for name, arr in diag_by_name.items():
            if arr.ndim not in (2, 3):
                raise ValueError(
                    f"diag_op_np['{name}'] must be (n_samples, n_p) for a "
                    f"scalar operator or (n_samples, n_vals, n_p) for a "
                    f"vector-valued one; got {arr.shape}.")
            if arr.shape[0] != n or arr.shape[-1] != len(sorted_p):
                raise ValueError(
                    f"diag_op_np['{name}'] has shape {arr.shape}; its first "
                    f"axis must be the {n} samples and its last the "
                    f"{len(sorted_p)} p values.")
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"diag_op_np['{name}'] contains nonfinite values.")

    if off_diag_np is not None:
        if off_diag_np.ndim != 3:
            raise ValueError(
                f"off_diag_np must have shape (n_measurements, n_operators, n_p); "
                f"got {off_diag_np.shape}.")
        if (off_diag_np.shape[1] != len(op_names)
                or off_diag_np.shape[2] != len(sorted_p)):
            raise ValueError(
                f"off_diag_np has operator/window axes {off_diag_np.shape[1:]}, "
                f"expected {(len(op_names), len(sorted_p))}.")
        if off_diag_norm_np.shape != off_diag_np.shape:
            raise ValueError(
                f"off_diag_norm_np must match off_diag_np exactly; got "
                f"{off_diag_norm_np.shape} and {off_diag_np.shape}.")
        if (not np.all(np.isfinite(off_diag_np))
                or not np.all(np.isfinite(off_diag_norm_np))):
            raise ValueError(
                "off_diag_np and off_diag_norm_np must contain only finite values.")
        insufficient = [p for p, m in zip(sorted_p, _m_array)
                        if n - m - p <= 0]
        if insufficient:
            raise ValueError(
                f"Insufficient samples for off-diagonal windows {insufficient}; "
                "each requires n_samples > p + m.")

    log_weights = weights
    if not np.all(np.isfinite(log_weights)):
        raise ValueError("Branch weights contain nonfinite values.")
    log_w_cumsum = np.cumsum(log_weights)

    # The off-diagonal pure estimator's shadow launch inherently needs m=p/2
    # steps of prior main-chain history, so its window-starts are restricted
    # to i >= m, while the diagonal/energy mixed estimator's ancestor-traced
    # windows have no such restriction (i >= 0). When BOTH a diag_operator
    # and an off_diag_operator are measured in the same call, those extra
    # leading windows the diagonal sum includes are real, finite-sample
    # terms (not floating-point noise) that make the two estimators of the
    # same underlying quantity disagree by more than expected.
    # Restricting every diag_operator AND off_diag_operator sum to the
    # common window-start i >= m_max = max(m_array) makes every reported p
    # use an identical set of windows. Energy and diag/off-diag-only calls
    # (m_max_common=0) are unaffected.
    m_max_common = (max(_m_array)
                     if (diag_by_name and off_diag_np is not None and op_names)
                     else 0)
    if m_max_common:
        insufficient_common = [p for p in sorted_p
                               if n - p - m_max_common <= 0]
        if insufficient_common:
            raise ValueError(
                f"Insufficient samples for the common diagonal/off-diagonal "
                f"window trim at p={insufficient_common}; each requires "
                f"n_samples > p + max(m_array)={m_max_common}.")

    energy_results, obs_results = _energy_and_diagonal_estimates(
        sorted_p, log_w_cumsum, E, diag_by_name, m_max_common)
    if off_diag_np is not None and op_names:
        obs_results.update(_off_diagonal_estimates(
            sorted_p, _m_array, log_w_cumsum, off_diag_np, off_diag_norm_np,
            off_diag_steps, op_names, n, m_max_common))

    return energy_results, obs_results


def check_off_diag_operator_positivity(walkers: Walker,
                                       off_diag_operators: Mapping[str, Operator],
                                       guide: Optional[Guide],
                                       caches: Optional[Mapping[str, Any]] = None,
                                       verbose: bool = True) -> None:
    """Check off-diagonal operator weights are non-negative.

    M_total == 1 (diagonal only): always passes, any sign allowed.
    M_total  > 1: all guide-transformed rates must be >= 0 (the estimator
    samples the successor by inverse CDF over the rates).

    Args:
        walkers:            Walker batch; its first 100 configurations are
                            tested.
        off_diag_operators: dict name -> operator.
        guide:              guiding wavefunction or None.
        caches:             optional dict name -> ``guide.build_cache(op)``.
        verbose:            print a one-line report when the check passes
                            (the drivers pass ``verbosity >= 1``).

    Raises:
        ValueError: if any rate is negative
    """
    n_check = _N_POSITIVITY_CHECK
    configs = walkers.config[:n_check]

    for name, op in off_diag_operators.items():
        if op.M_total == 1:
            continue

        def check_one(config):
            rates, _ = transition_rates(
                config, op, guide,
                cache=(caches or {}).get(name),
            )
            ok = jnp.all(jnp.isfinite(rates) & (rates >= 0))
            min_val = jnp.min(rates)
            return ok, min_val

        all_ok, min_vals = jax.vmap(check_one)(configs)

        if not jnp.all(all_ok):
            worst_min = float(jnp.min(min_vals))
            raise ValueError(
                f"Operator '{name}': negative rate detected "
                f"(min = {worst_min:.6f}). All rates must be >= 0."
            )

    emit("[estimators] - [check_off_diag_operator_positivity] - all operators "
         f"passed positivity check ({min(n_check, walkers.config.shape[0])} "
         f"configs, {len(off_diag_operators)} operators)", int(bool(verbose)))


def _row_keys(c):
    """Each row of a 2-D host array as ONE opaque void scalar, or None.

    Two rows have equal keys iff their bytes are equal, which coincides with
    elementwise ``==`` except for -0.0 (normalised to +0.0 here) and NaN
    (NaN != NaN, so such arrays -- and zero-width rows -- return None and the
    caller falls back to the elementwise path).  Sorting/uniquing these keys
    replaces a Python set of tuples or an N_sites-key lexsort, and is exact.
    """
    c = np.asarray(c)
    if c.ndim != 2 or c.shape[1] == 0:
        return None
    if c.dtype.kind in "fc":
        if np.isnan(c).any():
            return None
        c = c + c.dtype.type(0)          # -0.0 + 0.0 == +0.0 bytewise
    c = np.ascontiguousarray(c)
    return c.view(np.dtype((np.void, c.dtype.itemsize * c.shape[1]))).ravel()


def _count_unique_rows(c):
    """Number of distinct rows of a 2-D host array (elementwise equality)."""
    c = np.asarray(c)
    if len(c) == 0:
        return 0
    keys = _row_keys(c)
    if keys is not None:
        return int(np.unique(keys).size)
    idx = np.lexsort(c.T)
    changed = np.any(c[idx][1:] != c[idx][:-1], axis=1)
    return int(1 + np.sum(changed))


def count_unique_configs(walkers: Walker) -> int:
    """Number of distinct configurations in the walker population (on the host)."""
    return _count_unique_rows(walkers.config)


def estimate_average_log_branch_weight(walkers: Walker, hamiltonian: Operator,
                                       guide: Optional[Guide],
                                       method: str = "cont_time",
                                       lam: Optional[float] = None,
                                       cache: Any = None) -> float:
    """Returns a scalar estimate of the average log branch weight across walkers.

    Used as the starting value of ``avg_log_weight`` before the warm-up ramp.

    For the discrete methods (``single_step``, ``multi_step``; ``lam`` required):
        returns  mean( log(lam - sum(rates)) ), the average log branch weight per step.
    For ``cont_time`` (the default; ``lam`` unused):
        returns  mean( -sum(rates) ), the average negative local energy per unit
        imaginary time, which is the correct per-unit-time normalisation for the
        continuous-time propagator (``avg_log_weight * beta`` is subtracted from each
        walker's accumulated log-weight).  Using the discrete formula here would give
        a value ~log(lam - E_diag) ≪ -E_L, causing float32 overflow in the display
        and a bad initial estimate that the warmup ramp would have to correct.
    """
    lam = require_lam(method, lam)
    if method == "cont_time":
        def _val(config):
            rates, _ = transition_rates(config, hamiltonian, guide, cache=cache)
            return -jnp.sum(rates)
    else:
        def _val(config):
            rates, _ = transition_rates(config, hamiltonian, guide, cache=cache)
            return jnp.log(lam - jnp.sum(rates))

    return jnp.mean(jax.vmap(_val)(walkers.config))


def _walker_step_stats(config, hamiltonian, lam, guide, rng_key, cache=None):
    """Compute diag, p_d and k_raw for a single walker config (designed for vmap)."""
    rates, _ = transition_rates(config, hamiltonian, guide, cache=cache)
    b_x  = lam - jnp.sum(rates)
    p_d  = (lam - rates[0]) / b_x
    diag = rates[0]
    psi  = jax.random.uniform(rng_key)
    # p_d == 1 is possible on an absorbing row even when the Hamiltonian has
    # off-diagonal entries elsewhere.  Report an infinite hold instead of
    # performing an inf-to-int conversion.  p_d <= 0 is already an invalid
    # discrete propagator row and is reported separately by statistics().
    valid_geometric = (p_d > 0) & (p_d < 1)
    safe_log_p_d = jnp.where(valid_geometric, jnp.log(p_d), -1.0)
    sampled = jnp.floor(jnp.log1p(-psi) / safe_log_p_d) + 1.0
    k_raw = jnp.where(p_d >= 1, jnp.inf,
                      jnp.where(p_d <= 0, 1.0, sampled))
    return diag, p_d, k_raw


def _walker_cont_time_stats(config, hamiltonian, guide, rng_key, cache=None):
    """Return diagonal element, escape rate, and sampled continuous hold time."""
    rates, _ = transition_rates(config, hamiltonian, guide, cache=cache)
    R_x = -jnp.sum(rates[1:])
    psi = jax.random.uniform(rng_key)
    tau = jnp.where(R_x > 0, -jnp.log1p(-psi) / R_x, jnp.inf)
    return rates[0], R_x, tau


@partial(jax.jit, static_argnames=("k_max",))
def _lineage_step(anc, indices, k_max, e0, e_loc):
    """One branching step of ancestry + offspring statistics (see statistics).

    anc[i] = step-0 ancestor label of walker i.  Resampling replaces
    walker i by a copy of walker ``indices[i]``, so the new labels are
    just ``anc[indices]`` -- one gather.  c_j = #descendants of ancestor
    j, whence the two survival measures:
        fraction = #{c_j > 0} / N          (optimistic: distinct lineages)
        n_eff    = N^2 / sum_j c_j^2       (Kish's ESS, a COUNT in [1,N])
    The second is the one that bounds the pure estimator's variance.

    ``e0[j]`` is the local energy of step-0 walker j and ``e_loc[i]`` that
    of walker i after this step.  Pairing every walker with its own
    ancestor gives the lineage correlation of the local energy,
        corr = Pearson correlation of (e0[anc_new[i]], e_loc[i]) over i,
    1 at depth 0 and decaying to 0 as the population forgets its start.
    It is NaN when all walkers (or all ancestors) share one local energy.

    Module level (not a closure inside statistics()) so that its compiled
    form is reused across calls; N is read off the shapes.
    """
    N_walkers = indices.shape[0]
    _k_bins = jnp.arange(k_max, dtype=jnp.int32)
    counts = jnp.bincount(indices, length=N_walkers)          # offspring per parent
    row = jnp.concatenate([
        jnp.mean(counts[None, :] == _k_bins[:, None], axis=1),
        jnp.mean(counts >= k_max)[None],
    ])
    anc_new = anc[indices]
    c = jnp.bincount(anc_new, length=N_walkers).astype(float_dtype())
    frac = jnp.count_nonzero(c) / N_walkers
    ess = jnp.square(jnp.asarray(N_walkers, c.dtype)) / jnp.sum(jnp.square(c))
    x = e0[anc_new].astype(c.dtype)
    y = e_loc.astype(c.dtype)
    xc = x - jnp.mean(x)
    yc = y - jnp.mean(y)
    corr = jnp.sum(xc * yc) / jnp.sqrt(jnp.sum(xc * xc) * jnp.sum(yc * yc))
    return anc_new, row, frac, ess, corr


def _validated_m_array(m_array):
    """``m_array`` as a sorted list of distinct non-negative ints (empty for None)."""
    m_sorted = []
    for m in ([] if m_array is None else list(m_array)):
        m_int = int(m)
        if float(m_int) != float(m):
            raise ValueError(f"m_array values must be integers, got {m!r}.")
        m_sorted.append(m_int)
    m_sorted = sorted(set(m_sorted))
    if any(m < 0 for m in m_sorted):
        raise ValueError(f"m_array values must be nonnegative, got {m_sorted}.")
    return m_sorted


def _statistics_report(*, quiet, M, n_total, N_walkers, step_size, lam, method,
                       avg_log_weight, n_unique_pre, n_unique_post, diag_mins,
                       diag_maxs, p_d_mins, p_d_maxs, p_d_means, k_means,
                       k_maxs, changed_rates, overlap_rates, offspring, k_max,
                       m_sorted, anc_frac, anc_eff, anc_corr):
    """Print the :func:`statistics` report (nothing when ``quiet``)."""
    v = int(not quiet)
    tag = "[estimators] - [statistics] -"
    emit(f"{tag} {M} branching steps"
         + (f" (+{n_total - M} lineage-only)" if n_total > M else ""), v)
    emit(f"{tag} N_walkers: {N_walkers}", v)
    emit(f"{tag} step_size: {step_size}", v)
    # lam is None for cont_time, which does not use it
    emit(f"{tag} lam: " + ("not used" if lam is None else f"{lam:.6f}"), v)
    if method == "cont_time":
        emit(f"{tag} avg_log_weight: {float(avg_log_weight):.6f}", v)
    else:
        emit(f"{tag} average_b_x: {float(jnp.exp(avg_log_weight)):.6f}  "
             f"(log = {float(avg_log_weight):.6f})", v)
    emit(f"{tag} unique (pre): {np.mean(n_unique_pre):.1f} ± "
         f"{np.std(n_unique_pre):.1f}  ({np.mean(n_unique_pre)/N_walkers:.1%})", v)
    emit(f"{tag} unique (post): {np.mean(n_unique_post):.1f} ± "
         f"{np.std(n_unique_post):.1f}  ({np.mean(n_unique_post)/N_walkers:.1%})", v)
    emit(f"{tag} H_diag: min={np.min(diag_mins):.4f}, max={np.max(diag_maxs):.4f}"
         f"  (observed max over the {M} health steps)", v)
    if method == "cont_time":
        emit(f"{tag} escape rate: min={np.min(p_d_mins):.4f}, "
             f"max={np.max(p_d_maxs):.4f}", v)
        emit(f"{tag} hops per branching step (Gamma*step_size): "
             f"mean={np.mean(p_d_means)*float(step_size):.3f}, "
             f"max={np.max(p_d_maxs)*float(step_size):.3f}", v)
        emit(f"{tag} escape rate Gamma: mean={np.mean(p_d_means):.3f}, "
             f"min={np.min(p_d_mins):.3f}, max={np.max(p_d_maxs):.3f}", v)
        emit(f"{tag} hold time (a draw of 1/Gamma, not hops per step): "
             f"mean={np.mean(k_means):.4f}, max={np.max(k_maxs):.4f}", v)
    else:
        emit(f"{tag} p_d: min={np.min(p_d_mins):.4f}, max={np.max(p_d_maxs):.4f}", v)
        emit(f"{tag} hold length (sub-steps): mean={np.mean(k_means):.2f}, "
             f"max={np.max(k_maxs)}", v)
    emit(f"{tag} changed (per-index): {np.mean(changed_rates):.1%} avg over {M} steps", v)
    emit(f"{tag} overlap (pop):       {np.mean(overlap_rates):.1%} avg over {M} steps", v)
    off_mean = offspring.mean(axis=0)
    labels = [f"k={k}" for k in range(k_max)] + [f"k>={k_max}"]
    emit(f"{tag} offspring: "
         + "  ".join(f"{label}: {val:.4f}" for label, val in zip(labels, off_mean))
         + f"  (death=duplicate rate {off_mean[0]:.4f},"
         + f" survival_rate {1.0 - off_mean[0]:.4f})", v)
    if m_sorted:
        emit(f"{tag} {'m':>6} {'ancestors':>12} {'fraction':>10} {'n_eff':>14} "
             f"{'corr(e_L)':>10}", v)
        for m, f_, ef, cr in zip(m_sorted, anc_frac, anc_eff, anc_corr):
            emit(f"{tag} {m:>6} {int(round(f_ * N_walkers)):>12} "
                 f"{f_:>10.4f} {ef:>14.2f} {cr:>10.4f}", v)


def _statistics_dict(*, method, step_size, N_walkers, n_unique_pre,
                     n_unique_post, diag_mins, diag_maxs, p_d_mins, p_d_maxs,
                     p_d_means, k_means, k_maxs, changed_rates, overlap_rates,
                     offspring, k_max, m_sorted, anc_frac, anc_eff, anc_corr):
    """The ``stats`` dict of :func:`statistics` (keys documented there)."""
    stats = dict(
        n_unique_pre  = np.asarray(n_unique_pre),
        n_unique_post = np.asarray(n_unique_post),
        # Walker diversity rho = distinct configs / N_walkers (length M).
        unique_fraction = np.asarray(n_unique_post) / float(N_walkers),
        diag_min      = np.asarray(diag_mins),
        diag_max      = np.asarray(diag_maxs),
        changed_rate  = changed_rates,
        overlap_rate  = overlap_rates,
        H_diag_max_observed = np.asarray([np.max(diag_maxs)]),
        H_diag_min_observed = np.asarray([np.min(diag_mins)]),
    )
    if method == "cont_time":
        # hops_per_step under cont_time = the AVERAGE NUMBER OF CONFIGURATION
        # UPDATES PER BRANCHING STEP.  A branching step advances imaginary time by
        # step_size while the walker hops at the guide-transformed escape rate
        # Gamma, so the expected updates are Gamma*step_size; averaging over
        # the equilibrium ensemble makes <Gamma>*step_size exact by
        # stationarity (Gamma varies along the path, but its ensemble mean does
        # not).  This is the quantity that sets runtime -- per-step work is
        # hops, not m_max.
        #
        # It is NOT recoverable from hold_time_mean, and the two errors you
        # might make point in OPPOSITE directions:
        #   * 1/hold_time_mean is the HARMONIC mean of Gamma (hold_time is a
        #     draw of 1/Gamma), so by Jensen it UNDERSTATES <Gamma>, badly so
        #     when the Gamma distribution is broad.
        #   * escape_rate_max*step_size OVERSTATES typical work, often by
        #     several times.
        # Hence escape_rate_mean is reported explicitly rather than left to be
        # reconstructed.
        stats.update(
            escape_rate_min=np.asarray(p_d_mins),
            escape_rate_max=np.asarray(p_d_maxs),
            escape_rate_mean=np.asarray(p_d_means),
            hops_per_step_mean=np.asarray(p_d_means) * float(step_size),
            hops_per_step_max=np.asarray(p_d_maxs) * float(step_size),
            hold_time_mean=np.asarray(k_means),
            hold_time_max=np.asarray(k_maxs),
        )
    else:
        # Discrete methods: the hold length IS the sampled number of diagonal
        # sub-steps before an off-diagonal move (a geometric draw), which is
        # already "per branching step" in its own units.
        stats.update(
            p_d_min=np.asarray(p_d_mins),
            p_d_max=np.asarray(p_d_maxs),
            p_d_mean=np.asarray(p_d_means),
            hold_steps_mean=np.asarray(k_means),
            hold_steps_max=np.asarray(k_maxs),
        )
    for k in range(k_max):
        stats[f"offspring_k{k}"] = offspring[:, k]
    stats[f"offspring_k{k_max}plus"] = offspring[:, k_max]
    # Fraction of walkers with >= 1 offspring (length n_total).  Exact
    # complement of the k=0 bin; distinct from unique_fraction above.
    stats["survival_rate"] = 1.0 - offspring[:, 0]
    if m_sorted:
        stats["ancestor_m"]            = np.asarray(m_sorted, dtype=float)
        stats["ancestor_fraction"]     = np.asarray(anc_frac)
        stats["effective_sample_size"] = np.asarray(anc_eff)
        stats["ancestor_correlation"]   = np.asarray(anc_corr)
    return stats


def _health_step(walkers, carried, step_idx, run):
    """One branching step with the sampler-health probes (see statistics):
    per-walker stay/escape statistics of the current configurations, then
    evolve + reconfigure.  ``carried`` is the walkers' carried pair
    ``(rates, state)``.  Returns ``(walkers, carried, indices, probes)``."""
    k_stat     = jax.random.fold_in(run.rng_key, 3 * step_idx)
    k_evolve   = jax.random.fold_in(run.rng_key, 3 * step_idx + 1)
    k_resample = jax.random.fold_in(run.rng_key, 3 * step_idx + 2)

    stat_keys = jax.vmap(lambda i: jax.random.fold_in(k_stat, i))(jnp.arange(run.N_walkers))
    if run.method == "cont_time":
        diags, second_stats, third_stats = jax.vmap(
            lambda config, key: _walker_cont_time_stats(
                config, run.hamiltonian, run.guide, key, run.cache_h)
        )(walkers.config, stat_keys)
    else:
        diags, second_stats, third_stats = jax.vmap(
            lambda config, key: _walker_step_stats(
                config, run.hamiltonian, run.lam, run.guide, key, run.cache_h)
        )(walkers.config, stat_keys)

    walkers, carried = evolve_carried(walkers, carried, run.hamiltonian, run.step_size,
                                      run.guide, run.method, k_evolve,
                                      run.avg_log_weight, lam=run.lam, cache=run.cache_h)
    walkers, _, _, indices = reconfiguration(walkers, k_resample)
    carried = gather_carried(carried, indices)

    return walkers, carried, indices, (jnp.min(diags), jnp.max(diags),
                              jnp.min(second_stats), jnp.max(second_stats),
                              jnp.mean(second_stats.astype(jnp.float32)),
                              jnp.mean(third_stats.astype(jnp.float32)),
                              jnp.max(third_stats))


def _config_set(config):
    # elementwise-equality fallback of the overlap statistic (NaN rows)
    c = np.asarray(config)
    return set(map(tuple, c.tolist()))


def _health_steps(walkers, carried, n_steps, run):
    """The first ``n_steps`` branching steps with the host-side health
    diagnostics (unique counts, population overlap) and the lineage
    bookkeeping.  Returns ``(walkers, carried, health, lineage)``: the
    carried pair and two namespaces whose fields are the per-step lists."""
    health = SimpleNamespace(
        n_unique_pre=[], n_unique_post=[], diag_mins=[], diag_maxs=[],
        p_d_mins=[], p_d_maxs=[], p_d_means=[], k_means=[], k_maxs=[],
        changed_rates=[],   # fraction of walkers at index i that differ from step-0
        overlap_rates=[])   # fraction of step-t configs present in step-0 population
    # Everything the ancestry needs is a gather and two bincounts, so it runs
    # in JAX rather than on the host, avoiding a device->host sync per
    # branching step.  Results stay on device and are pulled once, after the
    # loop.
    lineage = SimpleNamespace(
        off_rows=[],                        # one (k_max+1,) device row per step
        frac=[], ess=[], corr=[],           # per-step lineage, sliced to m later
        anc=jnp.arange(run.N_walkers, dtype=jnp.int32))   # step-0 ancestor label

    config0  = np.asarray(walkers.config)
    keys0    = _row_keys(config0)
    uniq0    = np.unique(keys0) if keys0 is not None else None
    set0     = None                     # built only on the NaN fallback

    for i in range(n_steps):
        if i and i % _STATE_REFRESH_STEPS == 0:
            carried = refresh_carried(carried, walkers, run.hamiltonian, run.guide,
                                      run.cache_h)
        health.n_unique_pre.append(_count_unique_rows(walkers.config))
        walkers, carried, indices, (dmin, dmax, pmin, pmax, pmean,
                                    kmean, kmax) = _health_step(walkers, carried, i, run)
        health.diag_mins.append(dmin)
        health.diag_maxs.append(dmax)
        health.p_d_mins.append(pmin)
        health.p_d_maxs.append(pmax)
        health.p_d_means.append(pmean)
        health.k_means.append(kmean)
        health.k_maxs.append(kmax)

        config_t = np.asarray(walkers.config)
        keys_t = _row_keys(config_t)
        uniq_t = np.unique(keys_t) if keys_t is not None else None
        health.n_unique_post.append(int(uniq_t.size) if uniq_t is not None
                                    else _count_unique_rows(config_t))
        # metric 1: per-index change rate
        health.changed_rates.append(float(np.mean(np.any(config_t != config0, axis=1))))
        # metric 2: population overlap (fraction of step-t configs in step-0 set)
        if uniq0 is not None and uniq_t is not None:
            n_common = np.intersect1d(uniq0, uniq_t, assume_unique=True).size
            overlap = int(n_common) / max(int(uniq_t.size), 1)
        else:
            if set0 is None:
                set0 = _config_set(config0)
            set_t = _config_set(config_t)
            overlap = len(set0 & set_t) / max(len(set_t), 1)
        health.overlap_rates.append(overlap)

        lineage.anc, row, frac, ess, corr = _lineage_step(
            lineage.anc, indices, run.k_max, run.e0, walkers.e_loc)
        lineage.off_rows.append(row)
        lineage.frac.append(frac)
        lineage.ess.append(ess)
        lineage.corr.append(corr)
    return walkers, carried, health, lineage


def _lineage_scan(walkers, carried, anc, first, last, run):
    """Steps [first, last) as ONE lax.scan: evolve + reconfigure + lineage.

    Beyond M there are no host-side diagnostics left (no config sets, no
    unique counts), so the whole tail compiles into a single scan instead
    of a Python loop that re-dispatches every step.  This is where a deep
    m_array spends its time -- max(m_array) branching steps -- so the
    Python-level dispatch and the per-step sync were pure overhead.
    The key derivation is UNCHANGED (fold_in(rng_key, 3*step_idx+{1,2})
    with the same absolute step_idx), so the trajectory is identical to
    the unrolled version.  Returns ``(walkers, anc, (rows, frac, ess, corr))``.
    """
    k_max = run.k_max

    def body(carry, step_idx):
        walkers_, carried_, anc_ = carry
        carried_ = refresh_carried_at(step_idx, carried_, walkers_, run.hamiltonian,
                                      run.guide, run.cache_h)
        k_evolve   = jax.random.fold_in(run.rng_key, 3 * step_idx + 1)
        k_resample = jax.random.fold_in(run.rng_key, 3 * step_idx + 2)
        walkers_, carried_ = evolve_carried(
            walkers_, carried_, run.hamiltonian, run.step_size, run.guide,
            run.method, k_evolve, run.avg_log_weight, lam=run.lam, cache=run.cache_h)
        walkers_, _, _, indices = reconfiguration(walkers_, k_resample)
        anc_, row, frac, ess, corr = _lineage_step(anc_, indices, k_max,
                                                   run.e0, walkers_.e_loc)
        return (walkers_, gather_carried(carried_, indices), anc_), \
            (row, frac, ess, corr)

    if last <= first:
        empty = (jnp.zeros((0, k_max + 1)), jnp.zeros((0,)), jnp.zeros((0,)),
                 jnp.zeros((0,)))
        return walkers, anc, empty
    (walkers, _, anc), out = jax.lax.scan(
        body, (walkers, carried, anc), jnp.arange(first, last, dtype=jnp.int32))
    return walkers, anc, out


def _lineage_curves(lineage, m_sorted, N_walkers, k_max):
    """Pull the per-step lineage results to the host in one transfer and
    slice them at the depths in ``m_sorted``.  Returns
    ``(offspring, anc_frac, anc_eff, anc_corr)``."""
    offspring = np.asarray(jnp.concatenate(
        [r if r.ndim == 2 else r[None] for r in lineage.off_rows], axis=0)) \
        if lineage.off_rows else np.zeros((0, k_max + 1))
    frac_steps = np.asarray(jnp.concatenate(
        [f if f.ndim == 1 else f[None] for f in lineage.frac])) \
        if lineage.frac else np.zeros((0,))
    ess_steps = np.asarray(jnp.concatenate(
        [e if e.ndim == 1 else e[None] for e in lineage.ess])) \
        if lineage.ess else np.zeros((0,))
    corr_steps = np.asarray(jnp.concatenate(
        [c if c.ndim == 1 else c[None] for c in lineage.corr])) \
        if lineage.corr else np.zeros((0,))

    # frac_steps[j] / ess_steps[j] are the values AFTER branching step j, i.e.
    # at depth m = j+1.  m = 0 is the un-branched population: every walker is
    # its own ancestor, c_i = 1, so the Kish ESS is N^2/N = N -- NOT 1.
    anc_frac, anc_eff, anc_corr = [], [], []
    for m in m_sorted:
        if m == 0:
            anc_frac.append(1.0)
            anc_eff.append(float(N_walkers))
            anc_corr.append(1.0)
        else:
            anc_frac.append(float(frac_steps[m - 1]))
            anc_eff.append(float(ess_steps[m - 1]))
            anc_corr.append(float(corr_steps[m - 1]))
    return offspring, anc_frac, anc_eff, anc_corr


def statistics(walkers: Walker, hamiltonian: Operator, step_size: Union[int, float],
               guide: Optional[Guide], method: str, rng_key: jax.Array,
               avg_log_weight: Optional[float] = None, lam: Optional[float] = None,
               M: int = 10, quiet: bool = False, m_array: Optional[Sequence[int]] = None,
               cache: Any = None) -> tuple[Walker, dict[str, Any]]:
    """Run branching steps and collect diagnostic statistics.

    One evolution pass provides three families of diagnostics:

    1. **Sampler health** (first M steps): per-step min/max of the
       diagonal weight, survival probability p_d, diagonal sub-step count k,
       unique-configuration counts before/after each reconfiguration, and the
       changed/overlap mixing rates.
    2. **Offspring distribution** (every step): from each reconfiguration's
       parent map, the fraction of walkers with exactly k children for
       k = 0..3 plus an overflow bin (>= 4); fractions are in
       [0, 1] and sum to 1.  fraction[0] is the per-step lineage-death rate
       (= the duplicate rate sum_{k>=2}, since resampling conserves N) and
       scales ~linearly with step_size.
    3. **Ancestor survival** (optional, when ``m_array`` is given): each
       walker's step-0 ancestor label is carried through the resampling
       indices; at every m in m_array the fraction of distinct surviving
       ancestors and the Kish effective sample size N^2 / sum_i c_i^2 (c_i =
       number of descendants of ancestor i) are recorded, the quantity that
       bounds the pure estimator's effective sample size at projection depth
       m = p/2.  The same ancestry gives the lineage correlation of the
       local energy: the correlation between a walker's e_L at depth m and
       its own step-0 ancestor's, which measures in branching steps how
       long the population remembers its state.
       The run is extended to max(M, max(m_array)) steps; steps beyond M
       skip the (expensive) sampler-health diagnostics.

    Start from a stored, equilibrated population for meaningful ancestor
    curves (with near-uniform weights systematic resampling is the identity
    map, so the first steps of a fresh population show fraction 1.0).

    Args:
        walkers:        Walker batch to use as the starting point.
        hamiltonian:    Hamiltonian object.
        step_size:      Per-Markov-step size; forwarded to evolve_walkers
                        (integer n_bra for the discrete methods, float beta
                        for ``cont_time``).
        guide:          Guiding wavefunction or None.
        method:         "single_step", "multi_step", or "cont_time".
        rng_key:        JAX PRNGKey.
        avg_log_weight: Normalisation shift per sub-step (discrete methods) or
                        per unit imaginary time (``cont_time``); evolve_walkers
                        subtracts it * step_size.  ``None`` (default) uses
                        :func:`estimate_average_log_branch_weight` on the
                        given walkers.  Reconfiguration normalises the
                        weights, so every diagnostic here is invariant to
                        this shift up to float rounding; it only keeps the
                        log weights near zero.  Pass ``stats['avg_log_weight']``
                        of a run to reproduce that run's arithmetic exactly.
        lam:            GFMC shift λ; required by the discrete methods, unused
                        by ``cont_time``.
        M:              Number of full-diagnostic branching steps (default 10).
        quiet:          If True, suppress printed output.
        m_array:        Optional list of ancestor-survival depths (in steps;
                        tau = m * step_size).  None (default) disables the
                        lineage tracking and keeps the run at M steps.
        cache:          Optional guide.build_cache(hamiltonian) cache; computed
                        internally when None.

    The diagnostics run on the full population on one device.  A population
    that only fits when sharded over several devices can exhaust memory here;
    pass a slice of it (``jax.tree_util.tree_map(lambda x: x[:n], walkers)``)
    in that case, noting that ``effective_sample_size`` then describes the
    slice.

    Returns:
        ``(walkers, stats)``:

        * ``walkers``: Walker batch after max(M, max(m_array or [0])) steps.
        * ``stats``: dict with the keys below.

        * ``n_unique_pre`` (M,): unique configs before each reconfiguration.
        * ``n_unique_post`` (M,): unique configs after each reconfiguration.
        * ``unique_fraction`` (M,): n_unique_post / N_walkers -- the walker
          diversity rho plotted against the step size in the paper's
          convergence figure.  Around 0.95 is a healthy value.
        * ``survival_rate`` (n_total,): 1 - offspring_k0, i.e. the fraction of
          walkers that produce at least one offspring.  NOTE this is *not* the
          same as unique_fraction: it counts surviving parent slots, not
          distinct configurations, so two survivors carrying the same config
          count twice.  The two agree only when the configuration space is
          large compared to N_walkers (they differ by a factor of ~3 for an
          L=8 chain with 256 walkers).
        * ``diag_min`` (M,): per-step minimum diagonal weight.
        * ``diag_max`` (M,): per-step maximum diagonal weight.
        * ``p_d_min`` / ``p_d_max`` / ``p_d_mean`` / ``hold_steps_mean`` /
          ``hold_steps_max`` (M,): discrete methods only -- the stay
          probability and the sampled geometric hold length in sub-steps.
        * ``escape_rate_min`` / ``escape_rate_max`` / ``escape_rate_mean`` /
          ``hold_time_mean`` / ``hold_time_max`` / ``hops_per_step_mean`` /
          ``hops_per_step_max`` (M,): ``cont_time`` only -- the
          guide-transformed escape rate Gamma, the sampled exponential hold
          time, and Gamma*step_size = the average number of configuration
          updates per branching step (see the block that sets them).
        * ``changed_rate`` (M,): per-step fraction of walkers whose config
          changed from step 0.
        * ``overlap_rate`` (M,): per-step fraction of step-t configs present
          in the step-0 population.
        * ``H_diag_max_observed`` / ``H_diag_min_observed`` (1,): scalar
          extremes of the diagonal matrix element H_diag(x) over all walkers
          and all M health steps (diag_min/max are the raw rates[0] extremes;
          the max is the number lambda tuning needs: choose
          lam >= it + margin).
        * ``offspring_k<j>`` (n_total,): fraction of walkers with exactly j
          children per step, j = 0..3.
        * ``offspring_k4plus`` (n_total,): fraction with >= 4 children.
        * ``ancestor_m`` / ``ancestor_fraction`` / ``effective_sample_size``
          (len(m_array),): only when m_array is given.  ancestor_fraction is
          a FRACTION in (0,1]: #{c_i > 0} / N.  effective_sample_size is a
          COUNT in [1, N]: Kish's ESS, n_eff = N^2 / sum_i c_i^2
          (equivalently (sum_i c_i)^2 / sum_i c_i^2, since sum_i c_i = N),
          = 1 / P(two random walkers share an ancestor).  At m = 0 it is N.
        * ``ancestor_correlation`` (len(m_array),): only when m_array is
          given.  Pearson correlation between the local energy of each
          walker at depth m and that of its step-0 ancestor (every walker
          paired with its own ancestor, so lineages with many descendants
          weigh more).  1 at m = 0, decaying towards 0 over the population's
          decorrelation time; the per-walker counterpart of the run's
          ``autocorrelation/energy_estimator``.  NaN when every walker has
          the same local energy.
    """
    N_walkers = walkers.config.shape[0]
    if N_walkers <= 0:
        raise ValueError("statistics requires at least one walker.")
    if M < 1:
        raise ValueError(f"M must be >= 1, got {M}.")
    lam = require_lam(method, lam)
    k_max = _OFFSPRING_BINS

    # Precompute (guide, hamiltonian) external-partner tables once, same as
    # run_gfmc does internally (reuse the caller's cache when provided).
    if cache is None:
        cache = guide_cache(guide, hamiltonian)
    cache_h = cache
    if avg_log_weight is None:
        avg_log_weight = estimate_average_log_branch_weight(
            walkers, hamiltonian, guide, method=method, lam=lam, cache=cache_h)

    m_sorted = _validated_m_array(m_array)
    n_total  = max(M, m_sorted[-1]) if m_sorted else M

    # Local energy of the step-0 population, the reference of the lineage
    # correlation.  Recomputed rather than read from ``walkers.e_loc``, which
    # a freshly prepared population leaves at zero.
    rates0, state0 = initial_carried(walkers, hamiltonian, guide, cache_h)
    e0 = jnp.sum(rates0, axis=1)
    # The same byte budget as the drivers (evolution.rates_fit_budget).
    carried0 = (rates0 if rates_fit_budget(N_walkers, hamiltonian, walkers.log_weight.dtype,
                                           _carry_rates_bytes()) else None, state0)

    run = SimpleNamespace(rng_key=rng_key, hamiltonian=hamiltonian, guide=guide,
                          lam=lam, method=method, step_size=step_size,
                          avg_log_weight=avg_log_weight, cache_h=cache_h,
                          N_walkers=N_walkers, k_max=k_max, e0=e0)

    # The first M steps also measure sampler health, which needs the walker
    # configurations on the host (unique counts, population overlap), so they
    # stay a Python loop.  Everything after M is evolve + reconfigure +
    # lineage only and runs as a single scan.
    # The rates are carried across the steps as in the drivers (one rate
    # evaluation per step saved); rates0 doubles as the carried start.
    walkers, carried, health, lineage = _health_steps(walkers, carried0,
                                                      min(M, n_total), run)
    if n_total > M:
        walkers, lineage.anc, tail = _lineage_scan(walkers, carried, lineage.anc,
                                                   M, n_total, run)
        for name, arr in zip(("off_rows", "frac", "ess", "corr"), tail):
            getattr(lineage, name).append(arr)
    offspring, anc_frac, anc_eff, anc_corr = _lineage_curves(lineage, m_sorted,
                                                             N_walkers, k_max)

    n_unique_pre  = np.array(health.n_unique_pre)
    n_unique_post = np.array(health.n_unique_post)
    diag_mins = np.array(health.diag_mins)
    diag_maxs = np.array(health.diag_maxs)
    p_d_mins = np.array(health.p_d_mins)
    p_d_maxs = np.array(health.p_d_maxs)
    p_d_means = np.array(health.p_d_means)
    k_means = np.array(health.k_means)
    k_maxs = np.array(health.k_maxs)
    changed_rates = health.changed_rates
    overlap_rates = health.overlap_rates

    if method != "cont_time" and np.min(p_d_mins) < 0:
        n_bad     = int(np.sum(np.array(p_d_mins) < 0))
        worst_idx = int(np.argmin(p_d_mins))
        worst_pd  = float(np.min(p_d_mins))
        max_diag  = float(np.max(diag_maxs))
        raise ValueError(
            f"[gfmc.statistics] sampler positivity broken: "
            f"p_d_min < 0 in {n_bad}/{M} branching steps "
            f"(worst = {worst_pd:.4f} at step {worst_idx}). "
            f"This means at least one walker visited a configuration with "
            f"H_diag > lam. Current lam = {lam:.4f}, observed max H_diag = "
            f"{max_diag:.4f}. Increase lam to at least max(H_diag) + a small "
            f"margin (e.g. lam = Lx·Ly for the toric code)."
        )

    _statistics_report(
        quiet=quiet, M=M, n_total=n_total, N_walkers=N_walkers,
        step_size=step_size, lam=lam, method=method,
        avg_log_weight=avg_log_weight, n_unique_pre=n_unique_pre,
        n_unique_post=n_unique_post, diag_mins=diag_mins, diag_maxs=diag_maxs,
        p_d_mins=p_d_mins, p_d_maxs=p_d_maxs, p_d_means=p_d_means,
        k_means=k_means, k_maxs=k_maxs, changed_rates=changed_rates,
        overlap_rates=overlap_rates, offspring=offspring, k_max=k_max,
        m_sorted=m_sorted, anc_frac=anc_frac, anc_eff=anc_eff,
        anc_corr=anc_corr)

    stats = _statistics_dict(
        method=method, step_size=step_size, N_walkers=N_walkers,
        n_unique_pre=n_unique_pre, n_unique_post=n_unique_post,
        diag_mins=diag_mins, diag_maxs=diag_maxs, p_d_mins=p_d_mins,
        p_d_maxs=p_d_maxs, p_d_means=p_d_means, k_means=k_means,
        k_maxs=k_maxs, changed_rates=np.array(changed_rates),
        overlap_rates=np.array(overlap_rates), offspring=offspring,
        k_max=k_max, m_sorted=m_sorted, anc_frac=anc_frac, anc_eff=anc_eff,
        anc_corr=anc_corr)
    return walkers, stats
