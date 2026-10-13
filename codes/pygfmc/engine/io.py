"""HDF5 storage for GFMC runs.

:func:`write_to_HDF5` and :func:`run_from_HDF5` are the two entry points --
one stores a run, the other continues a stored chain;
:func:`load_raw_from_HDF5` and :func:`evaluate_observables` re-estimate from the
stored per-step series.  A file holds **one continuous Markov chain**: each
call appends a group ``/run_<n>`` (``n`` 0-based) carrying that call's
windowed energies, observable estimates, statistics and per-run parameters,
and — when ``raw`` is passed — appends the per-step series to the shared
top-level ``/raw`` group.  Because the raw series are stored, the energy can
be re-windowed for *any* projection length after the fact
(``evaluate_observables(load_raw_from_HDF5(path), p_array=...)``); the windowed values
in the group are a convenience, not the only record.

Continuing a chain needs two things that live outside the estimator output:
the **walker ensemble** and the **RNG stream**.  Both are stored.  The
ensemble is a single top-level ``/current_walkers`` group, overwritten by each
call — the name says which ensemble it is, since it tracks the newest run and
not any particular one,
because only the newest one can be continued from and a production ensemble
is large enough (~100 MB and up) that keeping one per run would dominate the
file.  The stream is recorded as ``parameters/rng_key`` (where the run
started) and ``parameters/rng_key_next`` (where it ended); a resuming call
feeds the latter back as ``rng_key`` so step keys
*advance* instead of being replayed.  :func:`run_from_HDF5` performs the whole
continuation, and it never warms up: the stored ensemble is
already equilibrated, so re-warming it would throw the equilibration away and
add a discontinuity in the middle of the chain.

Layout::

    /                     attrs: format, format_version, n_runs
    /parameters/<key>     the MUST_MATCH parameters, fixed for the whole file
    /current_walkers      config, log_weight, e_loc, rng_key
                          attrs: run, avg_log_weight;  rng_key = the
                          production-final key, so the walker file alone
                          continues the chain
    /raw/                 one continuous series, appended to by every run
        energy            (n_total,)                    per-step energy E_n
        log_weights       (n_total,)                    per-step log G_n
        diag              (n_total, n_ops, [n_vals,] n_p)  when widths agree
        diag_by_name/<nm> (n_total, [n_vals,] n_p)      when widths differ
        off_diag          (n_meas, n_ops, n_p)
        off_diag_norm     (n_meas, n_ops, n_p)
        off_diag_steps    (n_meas,)   absolute step index of each row
    /run_<n>/             attrs: n_steps, it_per_sec, runtime, total_wall_time,
                                 walker_hash, raw_step_begin/end,
                                 raw_meas_begin/end, walker_filename
        energy            (n_p,)      windowed energy, ordered by p
        observables/<nm>  (n_p, ...)  windowed operator estimates
        parameters/<key>  this run's MAY_DIFFER parameters
        statistics/<key>  everything else from stats, autocorrelation included
        additional_diag   diagonal lead-in for a continuation (see run_gfmc)
"""
import hashlib
import os
import re
import warnings

import numpy as np
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, Optional, Sequence, Union
import h5py

from .._output import emit
from ..protocols import Operator, DiagonalOperator, Guide
from ..walkers import Walker


FORMAT = "pygfmc"
FORMAT_VERSION = 2

#: Parameters that must be identical across every run in one file.  A file is
#: one continuous chain through one physical system measured one way, so a
#: change in any of these means the runs cannot be concatenated, re-windowed
#: together, or averaged — which is a mistake worth stopping the job for
#: rather than discovering in the analysis.  Grouped by why they matter:
#:
#: * the propagated distribution (``n_walkers`` sets the branching statistics,
#:   ``lam``/``step_size``/``method`` set the propagator itself),
#: * the system and the guide (a different guide file is a different
#:   importance function, hence a different weight sequence),
#: * the estimator layout (windows and forward-walk rows have to line up
#:   before values from two runs can enter the same average),
#: * the numerics (weight width follows ``jax_enable_x64``, so mixing widths
#:   mixes two different roundings of the same chain).
MUST_MATCH = (
    "n_walkers", "n_sites",
    "lam", "step_size", "method",
    "hamiltonian", "hamiltonian_hash",
    "has_guide", "guide", "guide_hash",
    "p_array", "m_array",
    "n_diag_operators", "n_off_diag_operators",
    "diag_operators", "off_diag_operators",
    "diag_operator_hashes", "off_diag_operator_hashes",
    "off_diag_stride", "off_diag_equal_windows",
    "config_dtype", "weight_dtype", "jax_enable_x64",
)

#: Parameters recorded per run but deliberately *not* validated, with the
#: reason each one legitimately varies.
MAY_DIFFER = {
    "n_steps": "a resumed campaign's last run is usually shorter",
    "warmup_steps": "0 when resuming from a stored ensemble",
    "rng_key": ("where this run's production stream started; each run starts "
                "where the previous one ended, so it differs per run"),
    "rng_key_next": "where this run's stream ended; checked for continuity",
    "verbosity": "printing only",
    "number_of_outputs": "printing only",
    "off_diag_offset": ("the local step of a call's first measurement; each "
                        "call anchors its own measurement grid (rows carry "
                        "their step in off_diag_steps), and equal-window "
                        "gating (return_raw=False) delays it past the "
                        "lead-in"),
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def pytree_fingerprint(obj):
    """Content hash of a Hamiltonian, operator or guide.

    Class names and operator names identify these objects by *label*, and
    every label can stay fixed while the thing behind it changes — a guide
    retrained into the same file, a coupling edited between two runs of a
    campaign, an operator rebuilt on a different bond under the same name.
    All would pass a name check and produce a chain that is not
    the chain the earlier runs walked.

    Everything involved is a registered JAX pytree, so hashing the flattened
    leaves together with the treedef (which carries the static/auxiliary data,
    e.g. SpinHalf's term flags) gives identity by content for a few hundred
    bytes, with no serialisation format to maintain.  Reconstructing these
    objects from a file is deliberately NOT attempted: no Hamiltonian class
    has a ``load``, ``generate_state`` is code rather than data, and
    user-defined operators may hold anything at all — a reconstruction path
    that worked only for the built-in classes would be worse than none.

    Two limitations worth knowing.  The hash is only as complete as the pytree
    registration, so state a class keeps outside its ``tree_flatten`` is
    invisible to it.  And a mismatch says *that* something changed, not what.

    Args:
        obj: Any registered pytree, or None.

    Returns:
        A 24-character hex digest; the empty string for None.
    """
    import jax

    if obj is None:
        return ""
    leaves, treedef = jax.tree_util.tree_flatten(obj)
    h = hashlib.blake2b(digest_size=12)
    h.update(type(obj).__name__.encode())
    # Default object reprs embed a memory address, which differs per process
    # and would make every resume look like a mismatch.  Scrub them: the worst
    # case is two distinct aux values colliding, after which the leaf hashes
    # still separate anything numerically different.
    h.update(re.sub(r"0x[0-9a-fA-F]+", "0x", repr(treedef)).encode())
    for leaf in leaves:
        arr = np.ascontiguousarray(np.asarray(leaf))
        h.update(f"{arr.dtype}{arr.shape}".encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def _operator_fingerprints(ops):
    """(names, hashes) for a dict/sequence of operators, in iteration order."""
    if ops is None:
        return [], []
    names = list(ops.keys()) if isinstance(ops, dict) else [
        getattr(o, "__name__", str(i)) for i, o in enumerate(ops)]
    values = list(ops.values()) if isinstance(ops, dict) else list(ops)
    return ([str(n) for n in names],
            [pytree_fingerprint(o) for o in values])


def _decode(val):
    """Normalise a stored value for comparison and for reporting."""
    if isinstance(val, bytes):
        return val.decode()
    arr = np.asarray(val)
    if arr.dtype.kind == "S":
        return (arr.item().decode() if arr.size == 1
                else [v.decode() for v in arr.ravel()])
    if arr.ndim == 0:
        return arr.item()
    return arr


def _match(stored, current):
    a, b = _decode(stored), _decode(current)
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.ndim:
        return bool(np.array_equal(a, b))
    if a.dtype.kind == "f" or b.dtype.kind == "f":
        # NaN is how an unset lam is stored; two unset values agree.
        if np.isnan(float(a)) and np.isnan(float(b)):
            return True
        return bool(np.isclose(float(a), float(b), rtol=1e-9, atol=0.0))
    return bool(a == b)


def _run_indices(f):
    return sorted(int(k.split("_", 1)[1]) for k in f
                  if k.startswith("run_") and k.split("_", 1)[1].isdigit())


def _split_stats(stats):
    """Route ``stats`` into (parameters, statistics).

    ``run_gfmc`` puts its own inputs under the ``parameters/`` prefix inside
    ``stats``; everything else — the sampler-health series, the offspring and
    ancestor diagnostics, ``autocorrelation/*``, ``it_per_sec``, ``runtime``,
    ``total_wall_time``, ``avg_log_weight`` — is a measured quantity.  They go
    to sibling groups so validation reads one of them and never has to filter
    the other.  Nothing is dropped.
    """
    params, other = {}, {}
    for key, val in (stats or {}).items():
        if key.startswith("parameters/"):
            params[key[len("parameters/"):]] = val
        else:
            other[key] = val
    return params, other


def _split_parameters(params):
    """Route parameters into (fixed, per_run) by :data:`MUST_MATCH`.

    A parameter that must agree across every run in the file has exactly one
    value for the file, so it belongs to the file and not to each run.  The
    layout then states which is which: whatever sits in ``/parameters`` is
    fixed for the whole chain, whatever sits in ``/run_n/parameters`` is what
    that call did differently.
    """
    fixed = {k: v for k, v in params.items() if k in MUST_MATCH}
    per_run = {k: v for k, v in params.items() if k not in MUST_MATCH}
    return fixed, per_run


def _append(f, key, block):
    """Append ``block`` along axis 0, creating the dataset on first use.

    The raw series are one continuous chain, so they are stored as one
    continuous set of arrays that grows with every appended run rather than as
    a per-run copy.  Axis 0 is the step axis in ``raw``'s own in-memory layout,
    so no transpose is involved and ``_estimate_observables`` consumes the
    stored arrays directly.
    """
    block = np.asarray(block)
    if key not in f:
        # Chunks span whole rows (every value and every p of one step), ~1 MiB
        # each, so appends touch few chunks and reading one operator never
        # crosses the p axis.  shuffle + gzip level 1 is lossless, readable by
        # any HDF5 tool, and on GFMC float32 series both faster and smaller
        # than gzip level 4 alone (measured: 1.6x faster writes, 76% vs 85%
        # of the uncompressed size).
        row_bytes = block.dtype.itemsize * int(np.prod(block.shape[1:], dtype=np.int64))
        # not capped by this block's length: the chunk shape is fixed at creation
        # and later, longer runs append to the same dataset
        rows = int(max(1, (1 << 20) // max(1, row_bytes)))
        f.create_dataset(key, data=block,
                         maxshape=(None,) + block.shape[1:],
                         chunks=(rows,) + block.shape[1:],
                         shuffle=True, compression="gzip", compression_opts=1)
        return 0, block.shape[0]
    n = f[key].shape[0]
    f[key].resize(n + block.shape[0], axis=0)
    f[key][n:] = block
    return n, n + block.shape[0]


def diag_mapping_to_dense(diag):
    """``{name: (n, [n_vals], n_p)}`` -> dense ``(n, n_ops, [n_vals], n_p)``.

    Returns ``None`` when the operators do not share a width, which is exactly
    when no dense array exists: the dense ``raw/diag`` layout has one
    positional operator axis and therefore one width for all of them.  Callers that need
    the dense form (this module's uniform-width fast path) treat ``None`` as "store per name instead".
    """
    if diag is None or not isinstance(diag, Mapping):
        return None if diag is None else np.asarray(diag)
    arrs = {k: np.asarray(v) for k, v in diag.items()}
    if not arrs:
        return None
    widths = {(1 if a.ndim == 2 else a.shape[1]) for a in arrs.values()}
    if len(widths) != 1:
        return None
    return np.stack(list(arrs.values()), axis=1)


def _write_group(g, mapping):
    for key, val in mapping.items():
        arr = np.asarray(val)
        if arr.dtype.kind == "U":            # h5py has no unicode conversion
            arr = arr.astype("S")
        g.create_dataset(key, data=arr)


def _check_parameters(f, params, source):
    """Validate the fixed parameters and the RNG stream against the file."""
    idx = _run_indices(f)
    ref = f["parameters"] if "parameters" in f else {}

    mismatches = [
        f"{k}: stored={_decode(ref[k][()])!r}, current={_decode(v)!r}"
        for k, v in params.items()
        if k in MUST_MATCH and k in ref and not _match(ref[k][()], v)
    ]
    if mismatches:
        raise ValueError(
            f"[{source}] parameter mismatch — a file holds one continuous "
            f"chain through one system measured one way, so these runs cannot "
            f"share it:\n  " + "\n  ".join(mismatches) +
            "\nWrite a new file, or see pygfmc.engine.io.MAY_DIFFER for what is allowed "
            "to vary." + ("\nA *_hash mismatch means the object's contents "
                          "differ even though its name does not -- a retrained "
                          "guide, an edited coupling, an operator on a "
                          "different site. See pygfmc.engine.io.pytree_fingerprint."
                          if any("_hash" in m for m in mismatches) else ""))

    # RNG stream CONTINUITY.  Keys advance key_{t+1} = split(key_t) with no
    # step index, so a run's stream is pinned by where it STARTED (rng_key)
    # and where it ENDED (rng_key_next).  A continuation starts on the key the
    # last run ended on -- an equality test.
    #
    # TWO SEVERITIES.  Reusing a
    # key another run already STARTED from is a replay: the two draw literally
    # the same random numbers, which silently correlates samples that are being
    # treated as independent -- that is an error.  Merely not continuing (a
    # fresh, unrelated key) is a gap: the runs are independent streams, which
    # is statistically fine but breaks the one-chain layout, so it warns.
    if not idx or "rng_key" not in params:
        return
    key_now = np.asarray(_decode(params["rng_key"]))
    for i in idx:
        p_i = f[f"run_{i}/parameters"]
        if "rng_key" not in p_i:
            continue
        if np.array_equal(np.asarray(p_i["rng_key"][()]), key_now):
            raise ValueError(
                f"[{source}] RNG stream replay: this run starts from the same "
                f"production key as run_{i}, so the two would draw identical "
                f"random numbers. Pass rng_key=<the key that came back with "
                f"the walkers> (stats['parameters/rng_key_next'], which "
                f"Walker.save(rng_key=...) checkpoints and Walker.load "
                f"returns), or write to a new file.")
    last = f[f"run_{idx[-1]}/parameters"]
    if "rng_key_next" in last and not np.array_equal(
            np.asarray(last["rng_key_next"][()]), key_now):
        warnings.warn(
            f"[{source}] run_{idx[-1]} ended on a different key than this run "
            f"starts from, so the file holds two independent streams rather "
            f"than one continuous chain. Pass "
            f"stats['parameters/rng_key_next'] as rng_key to continue it.",
            stacklevel=3)


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------
def write_to_HDF5(filename: str, stats: Mapping[str, Any],
                  walkers: Optional[Walker] = None,
                  energy: Optional[Mapping[int, float]] = None,
                  obs: Optional[Mapping[str, Mapping[int, Any]]] = None,
                  raw: Optional[Mapping[str, Any]] = None,
                  walker_filename: Optional[str] = None,
                  verbose: bool = True) -> int:
    """Append one GFMC run to an HDF5 file as a self-contained group.

    Creates the file if it does not exist, otherwise appends the next
    ``/run_<n>`` (0-based).  ``filename`` and ``stats`` are required; every
    other argument is optional.  ``MUST_MATCH`` parameters are validated
    against the file's ``/parameters`` group (and the RNG stream against the
    stored runs: a replayed key raises, a non-continuing key warns) before
    anything is written, and the write is skipped entirely if validation
    fails.

    Args:
        filename:        Path to the HDF5 file.
        stats:           ``stats`` dict from ``run_gfmc``.  **Required.**  Its
                         ``parameters/`` entries carry everything that
                         identifies the run, so without them a file has
                         nothing to validate an append against and nothing to
                         rebuild a continuation from — the two things the
                         format exists for.  The fixed ones go to
                         ``/parameters``, the per-run ones to
                         ``run_<n>/parameters``, and everything measured
                         (``autocorrelation/*``, ``it_per_sec``, ``runtime``,
                         ``total_wall_time``, the sampler-health and lineage
                         series, ``avg_log_weight``) to
                         ``run_<n>/statistics``.
        walkers:         Final :class:`Walker` batch.  Stored as the top-level
                         ``/current_walkers`` checkpoint, overwriting the
                         previous one — only the newest ensemble can continue
                         the chain.
        energy:          ``dict`` p → energy, as returned by ``run_gfmc``.
        obs:             ``dict`` name → ``dict`` p → value.
        raw:             ``raw`` dict from ``run_gfmc``.  Appended to the
                         top-level ``/raw`` series.  Storing it is what makes
                         the run re-estimable afterwards (via
                         :func:`load_raw_from_HDF5` and :func:`evaluate_observables`);
                         without it only the windowed values for this run's
                         ``p_array`` survive.
        walker_filename: Write the ensemble to this separate file instead of
                         embedding it, and record the path.  Useful when the
                         ensemble dwarfs the measurements or lives on faster
                         storage.
        verbose:         Print one line naming the group written (default
                         True); ``False`` keeps the call silent.

    Returns:
        The index ``n`` of the group just written.
    """
    if stats is None:
        raise ValueError(
            "write_to_HDF5 requires stats: its parameters/ entries are what "
            "identify the run, so without them the file cannot validate a "
            "later append and cannot rebuild a continuation. Pass the stats "
            "dict returned by run/run_gfmc.")
    params, other_stats = _split_stats(stats)
    from ..walkers.walker import key_data
    for key_name in ("rng_key", "rng_key_next"):
        if key_name in params:
            # Validate before opening the output file for mutation.
            params[key_name] = key_data(params[key_name])
    if not params:
        raise ValueError(
            "write_to_HDF5 got a stats dict with no 'parameters/' entries. "
            "Pass the dict returned by run/run_gfmc unmodified — run_gfmc "
            "records its own inputs there.")
    p_array = None
    if "p_array" in params:
        p_array = [int(v) for v in np.asarray(_decode(params["p_array"])).ravel()]
    elif energy:
        p_array = sorted(energy)

    new = not os.path.exists(filename)
    with h5py.File(filename, "w" if new else "a", locking=False) as f:
        if new:
            f.attrs["format"] = FORMAT
            f.attrs["format_version"] = FORMAT_VERSION
        else:
            _check_parameters(f, params, "write_to_HDF5")

        idx = _run_indices(f)
        n = (idx[-1] + 1) if idx else 0

        # The fixed parameters describe the file, so they are written once and
        # only validated afterwards; a per-run copy would be 26 identical
        # datasets per run and would leave open which of them is authoritative.
        fixed, per_run = _split_parameters(params)
        if fixed and "parameters" not in f:
            _write_group(f.create_group("parameters"), fixed)

        g = f.create_group(f"run_{n}")
        if per_run:
            _write_group(g.create_group("parameters"), per_run)
        if other_stats:
            _write_group(g.create_group("statistics"), other_stats)

        if energy:
            order = p_array or sorted(energy)
            g.create_dataset("energy", data=np.asarray(
                [energy[p] for p in order], dtype=np.float64))
            g["energy"].attrs["p_array"] = np.asarray(order, dtype=np.int64)
        if obs:
            og = g.create_group("observables")
            for name, per_p in obs.items():
                order = [p for p in (p_array or sorted(per_p)) if p in per_p]
                og.create_dataset(name, data=np.stack(
                    [np.asarray(per_p[p]) for p in order]))
                og[name].attrs["p_array"] = np.asarray(order, dtype=np.int64)

        if raw is not None:
            rg = f.require_group("raw")
            logw = raw.get("log_weights")
            span = {}
            # ``raw['diag']`` is a mapping name -> series.  Operators that
            # share a width go to the dense ``raw/diag`` (positional operator
            # axis, names in /parameters); a run whose operators return
            # DIFFERENT numbers of values has no dense form and is stored per
            # name under ``raw/diag_by_name``.
            # the off-diagonal series are keyed by name in memory but always
            # share a shape, so the dense dataset this file has always used
            # still exists -- rebuild it here rather than fork the layout.
            from .estimators import dense_and_names
            _od_dense, _od_names = dense_and_names(raw.get("off_diag"), None)
            _odn_dense, _ = dense_and_names(raw.get("off_diag_norm"), _od_names,
                                            "off_diag_norm")
            _diag_dense = diag_mapping_to_dense(raw.get("diag"))
            _diag_named = (raw.get("diag")
                           if (_diag_dense is None
                               and isinstance(raw.get("diag"), Mapping))
                           else None)
            for key, val in (("energy", raw.get("energy")),
                             ("log_weights", logw),
                             ("diag", _diag_dense),
                             ("off_diag", _od_dense),
                             # local -> absolute: the writer is the component
                             # that knows the file's history, so it supplies
                             # the chain numbering the call could not.
                             ("off_diag_steps",
                              (np.asarray(raw["off_diag_steps"], dtype=np.int64)
                               + int(rg["energy"].shape[0] if "energy" in rg
                                     else 0))
                              if raw.get("off_diag_steps") is not None
                              else None),
                             ("off_diag_norm", _odn_dense)):
                if val is not None:
                    span[key] = _append(rg, key, val)
            if _diag_named:
                dg = rg.require_group("diag_by_name")
                for _name, _arr in _diag_named.items():
                    _append(dg, str(_name), np.asarray(_arr))
            # Which rows this run contributed, so a run's own slice of the
            # continuous series is recoverable without summing n_steps.
            # (The rows carry their absolute step in off_diag_steps, so the
            # measurement grid needs no further bookkeeping here.)
            if "energy" in span:
                g.attrs["raw_step_begin"], g.attrs["raw_step_end"] = span["energy"]
            if "off_diag" in span:
                (g.attrs["raw_meas_begin"],
                 g.attrs["raw_meas_end"]) = span["off_diag"]
            # additional_diag is NOT a row of this chain: it holds the
            # diagonal values of the m_max steps that FOLLOW this run, traced
            # with this run's ancestry history, so a continuation can fill in
            # the rows its own fresh parent map cannot reach.  It therefore
            # lives with the run that produced it, not in the growing series.
            if raw.get("additional_diag") is not None:
                _extra = raw["additional_diag"]
                _extra_dense = diag_mapping_to_dense(_extra)
                if _extra_dense is not None:
                    g.create_dataset("additional_diag", data=_extra_dense)
                else:
                    eg = g.create_group("additional_diag_by_name")
                    for _name, _arr in _extra.items():
                        eg.create_dataset(str(_name), data=np.asarray(_arr))

        if "n_steps" in params:
            g.attrs["n_steps"] = int(_decode(params["n_steps"]))
        for key in ("it_per_sec", "runtime", "total_wall_time"):
            if key in other_stats:
                g.attrs[key] = float(np.asarray(other_stats[key]).ravel()[0])

        # ---- the walker checkpoint: the ensemble half of "resumable" ----
        if walkers is not None:
            # A fingerprint of the ensemble this run ended on, recorded per
            # run.  /current_walkers only ever holds the newest ensemble, so
            # without this there is nothing to check a continuation actually
            # started from where the previous run stopped.
            g.attrs["walker_hash"] = pytree_fingerprint(walkers)
            # The production-final key, so the walker checkpoint ALONE is
            # enough to continue the chain: Walker.load returns it as the
            # rng_key and run_gfmc resumes the stream from it.
            rng_key = None
            if "rng_key_next" in params:
                rng_key = np.asarray(_decode(params["rng_key_next"]),
                                     dtype=np.uint32)
            if walker_filename:
                walkers.save(walker_filename, rng_key=rng_key,
                             avg_log_weight=float(np.asarray(
                                 other_stats.get("avg_log_weight",
                                                 np.nan)).ravel()[0]))
                g.attrs["walker_filename"] = os.path.abspath(walker_filename)
                f.attrs["walker_filename"] = os.path.abspath(walker_filename)
            else:
                from ..walkers.walker import _compact_config
                # /current_walkers, not /walkers: the group holds the state
                # after the NEWEST run and is replaced on every call, so a
                # name suggesting a per-run record would be misleading.
                if "current_walkers" in f:
                    del f["current_walkers"]
                wg = f.create_group("current_walkers")
                wg.attrs["config_dtype"] = str(walkers.config.dtype)
                wg.create_dataset("config", data=_compact_config(walkers.config))
                wg.create_dataset("log_weight", data=np.asarray(walkers.log_weight))
                wg.create_dataset("e_loc", data=np.asarray(walkers.e_loc))
                if rng_key is not None:
                    # under "rng_key", the name Walker.load already reads
                    wg.create_dataset("rng_key", data=rng_key)
                wg.attrs["run"] = n
                if "avg_log_weight" in other_stats:
                    wg.attrs["avg_log_weight"] = float(np.asarray(
                        other_stats["avg_log_weight"]).ravel()[0])

        f.attrs["n_runs"] = n + 1

    emit(f"[write_to_HDF5] wrote {filename}:/run_{n}", int(bool(verbose)))
    return n


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
def _read_group(g):
    out = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out[name] = obj[()]
    g.visititems(visit)
    return out


class _StoredChain:
    """Read-only view of a file written by :func:`write_to_HDF5`.

    Opens the file once and answers what every reader needs from it: the run
    indices, the fixed parameters, one run's own parameters and statistics,
    the diagonal lead-in a run recorded, and where the walker ensemble is.
    Stored values come back decoded (bytes as ``str``, 0-d arrays as
    scalars).  Use as a context manager.
    """

    def __init__(self, filename):
        self.filename = filename
        self.file = h5py.File(filename, "r", locking=False)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.file.close()

    @property
    def run_indices(self):
        return _run_indices(self.file)

    def fixed_parameters(self):
        """``/parameters``: the MUST_MATCH set, fixed for the whole file."""
        if "parameters" not in self.file:
            return {}
        return {k: _decode(v) for k, v
                in _read_group(self.file["parameters"]).items()}

    def run_parameters(self, i):
        """``/run_<i>/parameters``: what that call did differently."""
        g = self.file[f"run_{i}"]
        if "parameters" not in g:
            return {}
        return {k: _decode(v) for k, v in _read_group(g["parameters"]).items()}

    def run_statistics(self, i):
        g = self.file[f"run_{i}"]
        if "statistics" not in g:
            return {}
        return {k: _decode(v) for k, v in _read_group(g["statistics"]).items()}

    def run(self, i):
        """One run as a dict: ``index``, ``energy``, ``p_array``,
        ``observables``, ``parameters`` (the fixed set merged with the run's
        own, also separately as ``fixed_parameters``/``run_parameters``) and
        ``statistics``."""
        g = self.file[f"run_{i}"]
        d = {"index": i, "energy": None, "p_array": None, "observables": {}}
        if "energy" in g:
            d["energy"] = g["energy"][()]
            d["p_array"] = g["energy"].attrs["p_array"]
        if "observables" in g:
            d["observables"] = {k: g[f"observables/{k}"][()]
                                for k in g["observables"]}
        # ``parameters`` merges the file's fixed set with this run's own, so
        # a caller sees one dict and does not have to know which half a given
        # name lives in.
        d["fixed_parameters"] = self.fixed_parameters()
        d["run_parameters"] = self.run_parameters(i)
        d["parameters"] = {**d["fixed_parameters"], **d["run_parameters"]}
        d["statistics"] = self.run_statistics(i)
        if d["p_array"] is None and "p_array" in d["parameters"]:
            d["p_array"] = np.asarray(d["parameters"]["p_array"])
        return d

    def newest(self):
        """The newest run (see :meth:`run`); raises when the file holds none."""
        idx = self.run_indices
        if not idx:
            raise ValueError(f"{self.filename} holds no run_<n> groups.")
        return self.run(idx[-1])

    def lead_in(self, i):
        """The diagonal lead-in block run ``i`` recorded for its continuation:
        a mapping name -> block when the operators differ in width, a dense
        block otherwise, ``None`` when there is none."""
        named = f"run_{i}/additional_diag_by_name"
        dense = f"run_{i}/additional_diag"
        if named in self.file:
            return {k: self.file[named][k][()] for k in self.file[named]}
        if dense in self.file:
            return self.file[dense][()]
        return None

    def walker_source(self, walker_filename=None):
        """Resolve the newest run's checkpoint, never a stale file-level hint."""
        if walker_filename is not None:
            return walker_filename, False
        newest = self.run_indices[-1]
        external = self.file[f"run_{newest}"].attrs.get("walker_filename")
        if external:
            return _decode(external), False
        embedded = ("current_walkers" in self.file
                    and self.file["current_walkers"].attrs.get("run") == newest)
        return None, embedded


def _read_from_HDF5(filename, run=-1):
    """Read one run back.

    Args:
        filename: Path to a file written by :func:`write_to_HDF5`.
        run:      0-based run index; negative counts from the newest
                  (default ``-1``).  Pass ``None`` for a list of every run.

    Returns:
        A dict with keys ``index``, ``energy``, ``p_array``, ``observables``,
        ``parameters`` (the file's fixed set merged with this run's own, also
        available separately as ``fixed_parameters``/``run_parameters``) and
        ``statistics`` — or a list of such dicts when ``run`` is ``None``.
        The raw series are not per run; read them with
        :func:`load_raw_from_HDF5`.
    """
    with _StoredChain(filename) as chain:
        idx = chain.run_indices
        if not idx:
            raise ValueError(f"{filename} holds no run_<n> groups.")
        wanted = idx if run is None else [idx[run]]
        out = [chain.run(i) for i in wanted]
    return out if run is None else out[0]


# --------------------------------------------------------------------------
# re-estimating from the raw per-step series
# --------------------------------------------------------------------------
def load_raw_from_HDF5(filename: str) -> dict[str, Any]:
    """Read a file's raw series back in the layout ``run`` returns them in.

    A file written by :func:`write_to_HDF5` holds ONE continuous chain: every
    run appends to the same ``/raw`` group, so what comes back here is the
    whole chain and not any single run's slice.  That is what makes
    re-estimating over it legitimate -- the run boundaries are joins in one
    chain rather than seams between independent ones.  ``run_<n>``'s
    ``raw_step_begin``/``raw_step_end`` attributes record which rows each run
    contributed, for a caller that does want one run's slice.

    The keys are the ones ``run`` puts in ``raw`` -- ``log_weights``,
    ``energy``, ``diag`` and ``off_diag``/``off_diag_norm`` (mappings keyed
    by operator name when the names are recorded) and ``off_diag_steps``, so
    the result goes straight into :func:`evaluate_observables`.  Keys the file does not hold are
    absent rather than None.

    The metadata ``p_array``, ``m_array``, ``diag_operators`` and
    ``off_diag_operators`` are read from the file's ``/parameters`` group, as
    in the ``raw`` that ``run`` returns.  They matter because the operator
    arrays are indexed by POSITION in the p grid -- ``off_diag[name][:, j]``
    is the value at window ``p_array[j]``, and nothing in the array itself
    says so.

    Args:
        filename: File written by :func:`write_to_HDF5`.

    Returns:
        ``dict`` of the raw series, plus the metadata named above.

    Raises:
        ValueError: when the file stores no raw series, i.e. it was written
            without ``raw=...`` and there is nothing to re-estimate from.
    """
    with _StoredChain(filename) as chain:
        f = chain.file
        if "raw" not in f or "energy" not in f["raw"] \
                or "log_weights" not in f["raw"]:
            raise ValueError(
                f"{filename} stores no raw series, so nothing can be "
                "re-estimated. Pass raw=... to write_to_HDF5 to keep the "
                "per-step energy and log weights.")
        rg = f["raw"]
        raw = {}
        for key in ("log_weights", "energy", "off_diag",
                    "off_diag_steps", "off_diag_norm"):
            if key in rg:
                raw[key] = rg[key][()]
        # the diagonal series come back keyed by operator name whichever way
        # they were stored: per name (widths differ) or dense with a
        # positional operator axis (widths agree; names in
        # /parameters/diag_operators, read below).
        _diag_named = ({k: rg["diag_by_name"][k][()] for k in rg["diag_by_name"]}
                       if "diag_by_name" in rg else None)
        _diag_dense = rg["diag"][()] if "diag" in rg else None
        par = chain.fixed_parameters()

    p_array = [int(v) for v in np.asarray(par.get("p_array", [])).ravel()]
    if p_array:
        raw["p_array"] = p_array
    m_array = [int(v) for v in np.asarray(par.get("m_array", [])).ravel()]
    if m_array:
        raw["m_array"] = m_array
    for key in ("diag_operators", "off_diag_operators"):
        names = [str(v) for v in np.atleast_1d(
            _decode(par.get(key, np.asarray([], dtype="S1"))))]
        if names:
            raw[key] = names

    od_names = raw.get("off_diag_operators") or []
    for key in ("off_diag", "off_diag_norm"):
        arr = raw.get(key)
        if arr is not None and len(od_names) == np.shape(arr)[1]:
            raw[key] = {n: arr[:, i] for i, n in enumerate(od_names)}

    diag_names = raw.get("diag_operators") or []
    if _diag_named is not None:
        # keep the run's own operator order when the file records it
        order = [n for n in diag_names if n in _diag_named] or list(_diag_named)
        raw["diag"] = {n: _diag_named[n] for n in order}
    elif _diag_dense is not None:
        raw["diag"] = ({n: _diag_dense[:, i] for i, n in enumerate(diag_names)}
                       if len(diag_names) == _diag_dense.shape[1]
                       else _diag_dense)   # no names stored: leave it dense
    return raw


def _normalized_raw(raw, p_array=None):
    """Validate a raw-series dict and return its parts as numpy arrays.

    A ``raw`` from ``run()`` or :func:`load_raw_from_HDF5` carries the
    operator names, ``m_array`` and ``p_array`` along; one assembled by hand
    (an analysis script slicing the series) need not, so missing names fall
    back to positional ones and the p grid may come from ``p_array``.
    Returns a namespace with ``energy``, ``log_weights``, ``off_diag``,
    ``off_diag_norm`` (dense ``(n_meas, n_off, n_p)`` or None),
    ``off_diag_steps``, ``diag`` (a name -> series mapping, a dense array or
    None), ``diag_names``, ``off_diag_names``, ``m_array`` and ``stored_p``,
    the sorted grid the operator arrays are measured on.

    Raises:
        ValueError: when the energy or log-weight series is missing, when no
            p grid is available, or when an operator array does not have
            ``len(stored_p)`` windows on its last axis.
    """
    from .estimators import dense_and_names, _validated_p_array
    energy_raw = raw.get("energy")
    if energy_raw is None:
        raise ValueError(
            "raw holds no 'energy' series, so nothing can be re-estimated. "
            "Pass the raw dict run() returns with return_raw=True, or "
            "load_raw_from_HDF5(file).")
    logw = raw.get("log_weights")
    if logw is None:
        raise ValueError(
            "raw holds no 'log_weights' series; the windowed estimator "
            "cannot be rebuilt without the per-step branch weights.")
    # numpy from here on: the replicas index these with boolean masks, and a
    # raw straight off the device would otherwise decide the dtype of the
    # estimate
    energy_raw = np.asarray(energy_raw)
    logw = np.asarray(logw)

    off_np, _od_names = dense_and_names(raw.get("off_diag"), None)
    # the normalisation series follow the operator order of the values
    off_norm, _ = dense_and_names(raw.get("off_diag_norm"), _od_names,
                                  "off_diag_norm")
    diag_raw = raw.get("diag")
    if isinstance(diag_raw, Mapping):
        diag_np = {k: np.asarray(v) for k, v in diag_raw.items()} or None
    else:
        diag_np = None if diag_raw is None else np.asarray(diag_raw)
    off_steps = raw.get("off_diag_steps")       # validated by the estimator
    off_steps = None if off_steps is None else np.asarray(off_steps)
    diag_names = raw.get("diag_operators")
    off_diag_names = raw.get("off_diag_operators") or _od_names
    if isinstance(diag_np, Mapping):
        diag_names = list(diag_np)          # a mapping carries its own names
    elif diag_np is not None and not diag_names:
        diag_names = [f"diag_{k}" for k in range(diag_np.shape[1])]
    if off_np is not None and not off_diag_names:
        off_diag_names = [f"off_diag_{k}" for k in range(off_np.shape[1])]

    # The grid the operator arrays live on.  It is not recoverable from the
    # arrays -- only their LENGTH along the p axis is -- so it has to come
    # from the file's parameters or from the caller.
    stored_p_array = raw.get("p_array", p_array)
    if stored_p_array is None:
        raise ValueError(
            "no p grid available: raw carries no 'p_array' (a raw dict from "
            "run() or load_raw_from_HDF5 does; one assembled by hand may "
            "not), so pass p_array=... -- the grid the run used.")
    stored_p = _validated_p_array(np.asarray(stored_p_array).ravel())

    # _estimate_observables rejects a length mismatch on its own, but its
    # message names its own argument (diag_op_np/off_diag_np) and not the grid
    # the caller got wrong, which is the only thing they can act on here.
    # A grid of the right LENGTH but the wrong values is silent, here and
    # there -- nothing in the arrays records which windows they were measured
    # at, which is why run() and load_raw_from_HDF5 carry p_array along.
    _grid_check = [("off_diag", off_np), ("off_diag_norm", off_norm)]
    if isinstance(diag_np, Mapping):
        _grid_check += [(f"diag['{k}']", v) for k, v in diag_np.items()]
    else:
        _grid_check.append(("diag", diag_np))
    for name, arr in _grid_check:
        if arr is not None and arr.shape[-1] != len(stored_p):
            raise ValueError(
                f"raw['{name}'] has {arr.shape[-1]} window(s) on "
                f"its last axis but the grid it is said to be on, "
                f"{stored_p}, has {len(stored_p)}. Set raw['p_array'] to "
                "the grid the run measured the operators at.")
    return SimpleNamespace(
        energy=energy_raw, log_weights=logw, off_diag=off_np,
        off_diag_norm=off_norm, off_diag_steps=off_steps, diag=diag_np,
        diag_names=diag_names, off_diag_names=off_diag_names,
        m_array=raw.get("m_array"), stored_p=stored_p)


def evaluate_observables(raw: Mapping[str, Any], p_array: Optional[Union[Sequence[int], np.ndarray]] = None,
                         n_blocks: Optional[int] = None) -> tuple:
    """Re-evaluate the energy and the observables from raw per-step series.

    Takes the ``raw`` dict ``run``/``run_gfmc`` return, or the one
    :func:`load_raw_from_HDF5` reads back from a file; both carry the p grid
    (``raw['p_array']``, ``raw['m_array']``) and the operator names alongside
    the series, so the call needs nothing else.  A ``raw`` assembled by hand
    -- an analysis script slicing the series -- may carry neither, and then
    ``p_array`` is required: the operator arrays are indexed by POSITION in
    the grid and nothing in them says what it was.  Operators of such a
    ``raw`` are named by their mapping keys, or positionally (``diag_0``,
    ``off_diag_0``, ...) for dense arrays.

    The windowed values a run returns are fixed at the grid that run used.
    The raw series are not, and because a file holds one continuous chain they
    can be re-estimated over the whole of it -- across run boundaries, which
    is legitimate here precisely because the boundaries are joins in one chain
    rather than seams between independent ones.

    The energy takes any ``p``.  The observables do not: their ancestry depth
    and forward-walking shadow chains are fixed while the chain is walking, so
    they are only ever available at the grid they were measured on, and asking
    for other values returns the energy there and the observables at that
    grid.

    Passing ``n_blocks`` additionally returns statistical errors from a
    blocked jackknife: the chain is cut into ``n_blocks`` contiguous blocks,
    each is deleted in turn and the *whole* estimator recomputed on the rest,
    and the error is the inflated scatter of those replicas,
    ``sqrt((nb-1)/nb * sum (x_b - mean)^2)``.  Recomputing rather than
    averaging per-block values is what makes it correct for a ratio estimator
    like ``sum G*E / sum G``, and it reuses ``_estimate_observables`` so the
    replicas inherit its window bookkeeping unchanged. If the sample count
    is not divisible by ``n_blocks``, trailing samples are discarded with a
    warning, from BOTH the point estimates and the error calculation.

    The block length is what matters, and it must exceed both the
    autocorrelation time and the window length: deleting one step from a
    20000-step chain with tau_int = 9 underestimated the error 4-fold, and the
    estimate only settled once blocks were longer than ~4p.  Choose
    ``n_blocks`` so that the block length ``n_steps / n_blocks`` clears
    ``max(2*tau_int, p)`` comfortably, and keep at least ~20 blocks or the
    error on the error takes over.  Two effects push the same way here: a
    replica is the surviving steps concatenated, so windows spanning the cut
    combine weights from either side of it, which understates the error by
    ~p/blen (measured 19 % at blen = 2p, under 5 % at blen = 10p).  Clearing
    the block-length condition suppresses that along with the
    autocorrelation, and at the plateau it is below the ~1/sqrt(2*n_blocks)
    uncertainty of the error estimate itself.  The reliable way to choose it is to scan
    ``n_blocks`` and take the value where the error stops growing: on the
    chain above it rose 8.4e-3, 8.6e-3, 8.9e-3 for block lengths 40, 80, 160
    and settled there.  tau_int for each series is already in the file under
    ``run_<n>/statistics/autocorrelation/``.

    Note that this is the statistical error only.  It says nothing about the
    systematic bias of a finite ``p``, which is the larger danger at deep
    projection: an over-projected estimate can be far off while carrying a
    small error bar, because the surviving lineages agree with each other.
    Independent runs across several ``p`` remain the only check for that.

    Args:
        raw:      ``raw`` dict from ``run``/``run_gfmc``, or from
                  :func:`load_raw_from_HDF5`.  ``energy`` and ``log_weights``
                  are required; ``diag``/``off_diag`` and their companions are
                  used when present.
        p_array:  Window sizes for the energy.  Defaults to the grid the
                  run used (``raw['p_array']``), in which case energy and
                  observables share one grid; any other grid re-windows the
                  energy alone.  For a hand-built ``raw`` without
                  ``p_array`` it is also the grid the operator arrays are
                  taken to sit on.
        n_blocks: Number of jackknife blocks.  ``None`` skips the error
                  estimate and returns the point estimates alone.

    Returns:
        ``(energy, obs)`` in ``run_gfmc``'s own form -- ``dict`` p -> energy
        and ``dict`` name -> ``dict`` p -> value -- or
        ``(energy, obs, energy_err, obs_err)`` with the same key structure
        when ``n_blocks`` is given.  ``obs`` is empty when ``raw`` holds no
        operator series.

    Raises:
        ValueError: when ``raw`` holds no energy or log-weight series, when no
            p grid is available from either ``p_array`` or ``raw``, when the
            operator arrays do not match the grid they are said to be on, or
            when ``n_blocks`` leaves fewer than 2 blocks.
    """
    from .estimators import (_estimate_observables,
                             _validated_p_array, _validated_steps)
    from .._validation import _validated_int

    r = _normalized_raw(raw, p_array)
    energy_raw, logw, stored_p = r.energy, r.log_weights, r.stored_p
    off_np, off_norm, off_steps = r.off_diag, r.off_diag_norm, r.off_diag_steps
    diag_np, diag_names, off_diag_names, m_array = (r.diag, r.diag_names,
                                                    r.off_diag_names, r.m_array)

    def _mask_diag(diag, mask):
        """Apply a step mask to every diagonal series (mapping or dense)."""
        if diag is None:
            return None
        if isinstance(diag, Mapping):
            return {k: v[mask] for k, v in diag.items()}
        return diag[mask]

    if n_blocks is not None:
        n_blocks = _validated_int("n_blocks", n_blocks, 1)
        n = energy_raw.shape[0]
        blen = n // n_blocks
        if n_blocks < 2 or blen < 1:
            raise ValueError(
                f"n_blocks={n_blocks} over {n} steps leaves {blen} steps per "
                "block; a jackknife needs at least 2 usable blocks.")
        if off_np is not None:
            # Validate against the original chain before dropping rows:
            # truncation must not hide malformed measurement indices.
            if off_steps is None:
                raise ValueError("off_diag_steps is required whenever off_diag is supplied.")
            off_steps = _validated_steps(off_steps, len(off_np), n)
        usable = blen * n_blocks
        if usable != n:
            warnings.warn(
                f"Discarding {n - usable} trailing samples for equal jackknife "
                "blocks; point estimates and errors both use the retained samples.",
                stacklevel=2)
            if off_np is not None:
                keep = off_steps < usable
                off_np = off_np[keep]
                off_norm = None if off_norm is None else off_norm[keep]
                off_steps = off_steps[keep]
            energy_raw, logw = energy_raw[:usable], logw[:usable]
            diag_np = _mask_diag(diag_np, slice(0, usable))

    # The stored grid first: the operator arrays are indexed by position in
    # it, so it is the only grid at which they can be read.
    E_stored, obs = _estimate_observables(
        stored_p, logw, energy_raw,
        off_diag_np=off_np, op_names=off_diag_names, m_array=m_array,
        diag_op_np=diag_np, diag_op_names=diag_names,
        off_diag_steps=off_steps,
        off_diag_norm_np=off_norm)

    same_grid = p_array is None or _validated_p_array(p_array) == stored_p
    if same_grid:
        E_out, obs_out = E_stored, obs
        grid = stored_p
    else:
        # A different grid: energy only, with no operator arrays to misalign.
        grid = _validated_p_array(p_array)
        E_out, _ = _estimate_observables(grid, logw, energy_raw)
        obs_out = obs

    if n_blocks is None:
        return E_out, obs_out

    n = energy_raw.shape[0]
    if n_blocks < 20:
        warnings.warn(
            f"n_blocks={n_blocks} is few: the error on the error is then "
            "comparable to the error itself. ~20 or more is advisable.",
            stacklevel=2)

    def _replicas(keep_fn):
        """keep_fn(mask) -> the estimator value on the retained steps."""
        out = []
        for b in range(n_blocks):
            mask = np.ones(n, dtype=bool)
            mask[b * blen:(b + 1) * blen] = False
            out.append(keep_fn(mask))
        return out

    def _jk_err(values):
        """Inflated scatter of the leave-one-out replicas.

        Each replica keeps all but one block, so the replicas are far more
        alike than independent estimates would be; the (nb-1) factor scales
        that artificially small spread back up.
        """
        arr = np.asarray(values, dtype=float)
        centred = arr - arr.mean(axis=0, keepdims=True)
        return np.sqrt((n_blocks - 1) / n_blocks
                       * np.sum(centred ** 2, axis=0))

    # Which main-chain step each stored off-diagonal row was launched at.
    if off_np is not None:
        row_step = off_steps

    # Observables: only at the stored grid, and only when there are series to
    # re-estimate.  The operator arrays are masked along the step axis with
    # the same mask, so a replica stays internally consistent.  Deleting a
    # block CLOSES THE GAP -- the replica is the survivors concatenated -- so
    # the off-diagonal rows have to be both dropped and RENUMBERED into the
    # shortened chain, or their steps point past the end of it.  One
    # estimator call per block gives the energy of every window and every
    # operator at once; the energy is re-estimated separately only when it
    # was asked for on another grid than the stored one.
    obs_err = {}
    need_obs = bool(obs_out) and (diag_np is not None or off_np is not None)
    e_reps = None
    if need_obs:
        names = list(obs_out)

        def _full_rep(m):
            rows = rep_steps = None
            if off_np is not None:
                new_pos = np.cumsum(m) - 1      # kept step -> its new index
                rows = m[row_step]
                rep_steps = new_pos[row_step[rows]]
            return _estimate_observables(
                stored_p, logw[m], energy_raw[m],
                off_diag_np=(off_np[rows] if off_np is not None else None),
                op_names=off_diag_names, m_array=m_array,
                diag_op_np=_mask_diag(diag_np, m),
                diag_op_names=diag_names,
                off_diag_steps=rep_steps,
                off_diag_norm_np=(off_norm[rows] if off_norm is not None
                                  else None))
        reps = _replicas(_full_rep)
        # One jackknife per (operator, window) rather than one over a flattened
        # vector of all of them: the operators need not share a width any more,
        # so the per-replica values are ragged and there is nothing to flatten.
        for nm in names:
            obs_err[nm] = {p: _jk_err([np.asarray(o[nm][p], dtype=float)
                                       for _, o in reps])
                           for p in stored_p}
        if same_grid:
            e_reps = [[e[p] for p in grid] for e, _ in reps]
    if e_reps is None:
        # Energy: available at any grid, so use whichever was asked for.
        def _energy_rep(m):
            est = _estimate_observables(grid, logw[m], energy_raw[m])[0]
            return [est[p] for p in grid]
        e_reps = _replicas(_energy_rep)
    energy_err = dict(zip(grid, _jk_err(e_reps)))

    return E_out, obs_out, energy_err, obs_err


def resume_from_HDF5(filename: str, hamiltonian: Optional[Operator] = None,
                     guide: Optional[Guide] = None,
                     diag_operators: Optional[Mapping[str, DiagonalOperator]] = None,
                     off_diag_operators: Optional[Mapping[str, Operator]] = None,
                     walker_filename: Optional[str] = None, dtype: Any = None,
                     config_dtype: Any = None) -> dict[str, Any]:
    """Assemble the ``run_gfmc`` arguments that continue a stored chain.

    :func:`run_from_HDF5` does the whole continuation in one call; use this
    lower-level function when you want the pieces, e.g. to pass the stored
    ensemble and key to a run with different operators.

    Reads the newest run's embedded or external checkpoint, validates its
    walker hash when recorded, and returns it with that run's parameters.
    An explicit ``walker_filename`` overrides the stored source but must
    still match the newest ensemble. Returns keyword arguments::

        kw = resume_from_HDF5("out.h5")
        walkers, E, obs, stats, raw = run_gfmc(
            kw.pop("walkers"), n_steps, H, step_size, guide, p_array,
            method=method, lam=lam, **kw)      # the stored run's settings
        write_to_HDF5("out.h5", stats, walkers, E, obs, raw)

    The returned ``rng_key`` is the stored production-final key, so the step
    keys continue past everything already walked, and ``warmup_steps=0`` because the
    stored ensemble is already equilibrated — re-warming it would discard that
    equilibration and put a discontinuity in the middle of the chain, which is
    exactly what a single-chain-per-file layout is meant to prevent.  The
    stored ``avg_log_weight`` is passed along for the same reason.

    Pass the objects the continuation will use and they are verified against
    the stored run before anything is returned, so a mismatch costs nothing::

        kw = resume_from_HDF5("out.h5", hamiltonian=H, guide=guide,
                             diag_operators=diag_ops)

    This is the natural place for that check.  ``write_to_HDF5`` performs it as
    well, but only once the run has finished.  Omitting the objects leaves the
    contents unverified and warns as much — the file records what the earlier
    runs used, and only the caller holds what this one is about to use.

    Args:
        filename:        File written by :func:`write_to_HDF5`.
        hamiltonian:     Hamiltonian the continuation will use.  Verified by
                         class name and content fingerprint against the stored
                         run when given.
        guide:           Guiding wavefunction, verified the same way.
        diag_operators:  Diagonal operator dict, verified by name, order and
                         fingerprint.
        off_diag_operators: As ``diag_operators``.
        walker_filename: Read the ensemble from this file instead of the
                         embedded checkpoint (defaults to the path recorded at
                         write time, if any).
        dtype:           JAX dtype for weights/local energies.  Default: the
                         ``weight_dtype`` recorded in the file.
        config_dtype:    JAX dtype for configurations.  Default: the recorded
                         ``config_dtype``.

    Returns:
        ``dict`` of ``run_gfmc`` keyword arguments: ``walkers``,
        ``avg_log_weight``, ``rng_key`` (the stored production-final key, so
        the continuation resumes the stream) and ``warmup_steps``.
    """
    import jax.numpy as jnp
    from ..walkers import Walker

    # Verify BEFORE handing back the ensemble.  A mismatch caught here costs
    # nothing; the same mismatch caught by write_to_HDF5 costs the whole run.
    supplied = {k: v for k, v in (("hamiltonian", hamiltonian),
                                  ("guide", guide),
                                  ("diag_operators", diag_operators),
                                  ("off_diag_operators", off_diag_operators))
                if v is not None}
    if supplied:
        check_run_compatible(filename, **supplied)
    else:
        warnings.warn(
            "resume_from_HDF5 could not verify what this continuation will run "
            "against what the file already holds: the stored fingerprints "
            "describe the earlier runs, and only the caller holds the objects "
            "this one is about to use. Pass hamiltonian=, guide=, "
            "diag_operators= and off_diag_operators= to have them checked here "
            "rather than by write_to_HDF5 after the run.", stacklevel=2)

    with _StoredChain(filename) as chain:
        d = chain.newest()
        external, embedded = chain.walker_source(walker_filename)
        expected_hash = chain.file[f"run_{d['index']}"].attrs.get("walker_hash")
    p = d["parameters"]

    def _jdt(name, override):
        if override is not None:
            return override
        raw_name = p.get(name)
        return jnp.dtype(str(raw_name)) if raw_name else jnp.float32

    dtype = _jdt("weight_dtype", dtype)
    config_dtype = _jdt("config_dtype", config_dtype)

    if external and not embedded:
        walkers, rng_key, alw = Walker.load(external, dtype=dtype,
                                            config_dtype=config_dtype)
    elif embedded:
        walkers, rng_key, alw = Walker.load(filename, group="current_walkers",
                                            dtype=dtype,
                                            config_dtype=config_dtype)
    else:
        raise ValueError(
            f"{filename} stores no walker ensemble, so the chain cannot be "
            "continued from it. Pass walkers=... to write_to_HDF5 (or "
            "walker_filename=...) on the run you intend to resume from.")

    if expected_hash is not None:
        # Validate before any requested compute-dtype conversion. Config
        # storage may be compact integers; restore its original dtype.
        with h5py.File(external if external else filename, "r") as checkpoint:
            group = checkpoint["walkers" if external else "current_walkers"]
            original = Walker(
                np.asarray(group["config"], dtype=group.attrs.get(
                    "config_dtype", p.get("config_dtype", "float32"))),
                np.asarray(group["log_weight"]), np.asarray(group["e_loc"]))
        if pytree_fingerprint(original) != _decode(expected_hash):
            raise ValueError("Walker checkpoint does not match the newest run's walker_hash.")
    if alw is None and "avg_log_weight" in d["statistics"]:
        alw = float(np.asarray(d["statistics"]["avg_log_weight"]).ravel()[0])
    # Continue the SAME key stream rather than starting a new one: the stored
    # production-final key IS the resume point, so passing it as rng_key makes
    # the continuation bit-identical to an uninterrupted run.  No step counter
    # and no chain object -- the key carries the position.
    if "rng_key_next" in p:
        rng_key = jnp.asarray(np.asarray(p["rng_key_next"], dtype=np.uint32))

    kw = dict(
        walkers=walkers,
        avg_log_weight=(float(alw) if alw is not None else None),
        rng_key=rng_key,

        # The ensemble is equilibrated; warming it again would undo that.
        warmup_steps=0,
    )
    return kw


def run_from_HDF5(filename: str, hamiltonian: Operator, guide: Optional[Guide] = None,
                  diag_operators: Optional[Mapping[str, DiagonalOperator]] = None,
                  off_diag_operators: Optional[Mapping[str, Operator]] = None,
                  n_steps: Optional[int] = None,
                  walker_filename: Optional[str] = None, **run_kwargs) -> tuple:
    """Continue a stored chain in one call, and return what ``run_gfmc`` returns.

    Everything the continuation needs already sits in the file — the fixed
    parameters in ``/parameters``, the ensemble in ``/current_walkers``, the
    stream in the newest run's ``rng_key_next``.  This
    reads all of it, verifies the objects handed in against the stored
    fingerprints, and runs::

        walkers, energy, obs, stats, raw = run_from_HDF5(
            "out.h5", H, guide, diag_ops, off_diag_ops)
        write_to_HDF5("out.h5", stats, walkers, energy, obs, raw)

    No warmup is performed.  The stored ensemble is already equilibrated and
    its ``avg_log_weight`` is passed through, so re-warming it would discard
    that equilibration and leave a discontinuity mid-chain.

    The diagonal observables come out bit-identical to an uninterrupted run.
    A fresh ancestry map cannot trace back past its own start, so the first
    ``p/2`` rows would otherwise be zeros; the previous run recorded exactly
    those values by walking ``m_max`` steps further while it still had the
    history, and they are substituted here.  The off-diagonal observables need
    no such repair — their shadow chains only look forward.

    ``lam``, ``step_size``, ``p_array``, ``method`` and ``off_diag_stride``
    come from the file rather than from arguments, which is what makes the
    continuation *the same calculation* instead of a similar one.  Anything in :data:`MAY_DIFFER` may still be
    passed through ``run_kwargs`` — ``verbosity`` and ``number_of_outputs``
    typically are.

    Args:
        filename:           File written by :func:`write_to_HDF5`.
        hamiltonian:        The Hamiltonian.  Compared against the stored class
                            name and fingerprint.
        guide:              Guiding wavefunction, compared the same way.
        diag_operators:     Diagonal operator dict, compared by name, order and
                            fingerprint.
        off_diag_operators: As ``diag_operators``.
        n_steps:            Steps to run.  Defaults to the newest stored run's
                            ``n_steps``, so a self-looping campaign repeats
                            equal blocks without restating the length.
        walker_filename:    Read the ensemble from this file instead of the
                            embedded checkpoint.
        **run_kwargs:       Passed to ``run_gfmc``, overriding what was read
                            from the file.  Raises on any :data:`MUST_MATCH`
                            name, since overriding one of those is precisely
                            what the file is there to prevent.

    Returns:
        ``(walkers, energy, obs, stats, raw)`` — exactly ``run``'s tuple
        (a 4-tuple without ``raw`` if ``return_raw=False`` is passed).

    Raises:
        ValueError: on a parameter or fingerprint mismatch, an RNG stream
            replay, a missing ensemble, or a ``MUST_MATCH`` override.
    """
    # `run` rather than `run_gfmc`: it is the recommended entry point and
    # auto-dispatches to the multi-GPU backend, falling back on its own for
    # features that backend lacks.  The keyword and return contract is
    # identical either way.
    from .multi_gpu import run

    blocked = sorted(set(run_kwargs) & set(MUST_MATCH) |
                     ({"lam"} if "lam" in run_kwargs else set()))
    if blocked:
        raise ValueError(
            f"[run_from_HDF5] {blocked} cannot be overridden — they are read "
            "from the file so that the continuation is the same calculation. "
            "Start a new file to change any of them.")

    kw = resume_from_HDF5(filename, hamiltonian=hamiltonian, guide=guide,
                         diag_operators=diag_operators,
                         off_diag_operators=off_diag_operators,
                         walker_filename=walker_filename)
    with _StoredChain(filename) as chain:
        d = chain.newest()
        # The lead-in block the previous run recorded: diagonal values for
        # exactly the steps this call is about to re-walk, traced with the
        # history this call does not have.
        lead_in = chain.lead_in(d["index"])
        # The steps the chain already holds, so the fixed refresh schedule
        # of a guide in the incremental mode continues where it was.
        steps_done = sum(int(chain.run_parameters(i).get("n_steps", 0))
                         for i in chain.run_indices)
    p = d["parameters"]

    if n_steps is None:
        if "n_steps" not in p:
            raise ValueError(
                f"{filename} records no n_steps for its newest run, so the "
                "block length cannot be inherited. Pass n_steps=... .")
        n_steps = int(p["n_steps"])

    p_array = [int(v) for v in np.asarray(p["p_array"]).ravel()]
    step_size = float(p["step_size"])
    # step_size is an integer count of sub-steps for the discrete methods and a
    # float imaginary time for cont_time; hand back the integer where it is one
    # so run_gfmc's own integer check sees what it expects.
    if float(step_size).is_integer():
        step_size = int(step_size)

    call = dict(
        method=str(p["method"]),
        off_diag_stride=int(p["off_diag_stride"]),
    )
    # The measurement gating is a MUST_MATCH parameter, so the continuation
    # reproduces the stored choice explicitly rather than re-deriving it from
    # its own return_raw.
    if "off_diag_equal_windows" in p:
        call["off_diag_equal_windows"] = bool(int(p["off_diag_equal_windows"]))
    call.update(kw)
    call.setdefault("_step_offset", steps_done)
    call.update(run_kwargs)
    # lam is stored as NaN for cont_time, which takes none
    lam = float(p["lam"])
    lam = None if np.isnan(lam) else lam
    out = run(call.pop("walkers"), n_steps, hamiltonian,
              step_size, guide, p_array, lam=lam,
              diag_operators=diag_operators,
              off_diag_operators=off_diag_operators, **call)

    if lead_in is not None and len(out) > 4 and out[4] is not None:
        raw = out[4]
        diag = raw.get("diag")
        if diag is not None:
            # This call re-walked the same steps from the same ensemble with the
            # same rng_key, so its walkers match the uninterrupted run exactly
            # and only the trace is short.  Substituting the recorded values for
            # the first rows therefore reproduces the uninterrupted result --
            # not an approximation to it.  Rows beyond p/2 already agree with
            # the block, so overwriting the whole overlap is harmless.
            # ``diag`` is a mapping name -> series; the stored lead-in is
            # either the same mapping (widths differ) or a dense block
            # (widths agree), whose operator axis is positional in this run's
            # operator order.
            names = list(diag)
            spliced = {}
            for i, name in enumerate(names):
                arr = np.asarray(diag[name])
                if isinstance(lead_in, Mapping):
                    src = lead_in.get(name)
                else:
                    # dense block: (m_max, n_ops, [n_vals,] n_p), operator
                    # axis positional in this run's operator order
                    src = lead_in[:, i] if np.ndim(lead_in) >= 3 else None
                if src is None:
                    raise ValueError(
                        f"[run_from_HDF5] the stored diagonal lead-in has no "
                        f"series for operator '{name}'; the operator set "
                        "changed, which the parameter check should already "
                        "have caught.")
                src = np.asarray(src)
                if src.shape[1:] != arr.shape[1:]:
                    raise ValueError(
                        f"[run_from_HDF5] stored diagonal lead-in for "
                        f"'{name}' has shape {src.shape[1:]} but this run's "
                        f"rows are {arr.shape[1:]}. The operator set or p "
                        "grid changed, which the parameter check should "
                        "already have caught.")
                k = min(src.shape[0], arr.shape[0])
                out_arr = np.array(arr, copy=True)
                out_arr[:k] = src[:k]
                spliced[name] = out_arr
            raw["diag"] = spliced
    return out


def check_run_compatible(filename: str, stats: Optional[Mapping[str, Any]] = None,
                         **parameters) -> None:
    """Validate a planned run against a stored file *before* walking.

    ``write_to_HDF5`` performs the same check, but only once the run has
    finished — by which point a mismatch has already cost the wall-clock.
    Call this first with the parameters you are about to use and the job stops
    immediately instead::

        check_run_compatible(path, n_walkers=N, lam=lam, step_size=8,
                             method="multi_step", p_array=[4, 8, 16],
                             hamiltonian=H, guide=guide,
                             diag_operators=diag_ops,
                             off_diag_operators=off_ops)

    ``hamiltonian``, ``guide`` and the two operator dicts may be passed as the
    objects themselves, in which case their class names, operator names and
    content fingerprints (see :func:`pytree_fingerprint`) are all derived and
    compared — so a retrained guide or an edited coupling is caught even
    though every name still agrees.  Names are
    checked, not only counts — two runs measuring the same *number* of
    different operators produce datasets of identical shape whose rows mean
    different things, which no later shape check can catch.

    Accepted either as keyword arguments (bare names, no ``parameters/``
    prefix) or as a ``stats`` dict from a previous ``run_gfmc``.  Unknown and
    :data:`MAY_DIFFER` names are ignored, so passing a whole parameter set is
    fine.  A missing file is compatible with anything.

    Raises:
        ValueError: on a :data:`MUST_MATCH` mismatch or an RNG stream replay.
    """
    if not os.path.exists(filename):
        return
    params, _ = _split_stats(stats)
    params.update(parameters)
    # Accept the objects themselves, so the pre-run call can pass what it is
    # about to hand to run_gfmc rather than restating names and hashes.
    for key, count in (("diag_operators", "n_diag_operators"),
                       ("off_diag_operators", "n_off_diag_operators")):
        ops = params.get(key)
        if ops is None or isinstance(ops, np.ndarray):
            continue
        names, hashes = _operator_fingerprints(ops)
        params[key] = np.asarray(names, dtype="S64")
        params.setdefault(count, len(names))
        # Only fingerprint real operators; a caller listing bare names (or
        # None placeholders) gets the name check and nothing more.
        if all(h for h in hashes):
            params.setdefault(f"{key[:-1]}_hashes",
                              np.asarray(hashes, dtype="S32"))
    for key, target in (("hamiltonian", "hamiltonian_hash"),
                        ("guide", "guide_hash")):
        obj = params.get(key)
        if obj is None or isinstance(obj, (str, bytes, np.ndarray)):
            continue
        params[target] = np.asarray(pytree_fingerprint(obj), dtype="S32")
        params[key] = np.asarray(type(obj).__name__, dtype="S32")
        if key == "guide":
            params.setdefault("has_guide", 1)
    with _StoredChain(filename) as chain:
        _check_parameters(chain.file, params, "check_run_compatible")


# Underscore-prefixed aliases of the two functions above.
_check_run_compatible = check_run_compatible
_resume_from_HDF5 = resume_from_HDF5
