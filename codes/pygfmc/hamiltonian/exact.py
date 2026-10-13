"""Generic exact diagonalization over the Hamiltonian protocol.

:func:`exact_diagonalization` builds the full matrix of any object that
implements ``get_rates``/``generate_state`` and solves for the lowest few
eigenpairs, then evaluates diagonal and off-diagonal observables on each of
them.  It needs nothing from the Hamiltonian beyond the protocol, so it
covers every built-in model and user-defined operator classes alike.

Two implementation choices worth stating, because the obvious alternatives are
much worse.

**Indexing is arithmetic, not a lookup table.**  A configuration is a digit
string in base ``d = len(local_hilbert_space)`` with site 0 most significant,
so its index is ``Σ_i digit_i · d^(N-1-i)``.  Mapping configurations through a
``dict`` keyed by tuples would need ``dim`` Python objects (hundreds of MB by
N = 20 at d = 2) and would convert a float array to a hashable tuple on every
lookup, which is precisely the inner loop of the matrix build.  The arithmetic
form costs nothing and vectorises over a whole successor block at once.

**The basis is enumerated with a vectorised odometer**, not a Python loop or
``itertools.product``::

    digits  = (np.arange(dim)[:, None] // d ** np.arange(N-1, -1, -1)) % d
    configs = values[digits]

Rates for the whole block then come from one batched ``jax.vmap(get_rates)``
call, reusing the same JIT path production runs use.  Configurations are
processed in chunks so that the ``(chunk, M_total, N)`` successor tensor stays
bounded regardless of ``dim``.
"""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from .._validation import _validated_int

__all__ = ["exact_diagonalization"]

#: Refuse beyond this many basis states rather than exhaust memory silently.
_MAX_DIM = 1 << 24

#: Configurations per batched get_rates call.
_CHUNK = 4096

#: At or below this basis dimension the eigensolve is dense.
_DENSE_DIM = 16

#: Refuse an allocation estimated above this many bytes (the sparse matrix
#: of the operator, or the dense matrix of a full-spectrum solve).
_MAX_BYTES = 8 << 30

#: Relative asymmetry above which an operator matrix is not Hermitian.
_HERMITIAN_RTOL = 1e-5


@contextlib.contextmanager
def _float64():
    """Run the whole routine in double precision, whatever the caller set.

    Exact diagonalization is the reference the Monte Carlo is checked against,
    so it should not inherit the sampler's working precision.  Everything this
    module does -- index arithmetic, matrix assembly, the eigensolve, the
    observable contractions -- is float64, and JAX is switched to x64 for the
    duration so a double-precision Hamiltonian is not silently truncated on
    the way in.

    The flag is restored on the way out, including on an exception.  That is
    not optional: ``jax_enable_x64`` is global, the walker initializers take
    their width from it, and a leaked True hands float64 walkers to a float32
    Hamiltonian, which fails deep inside the production scan with a carry-type
    mismatch that points nowhere near here.

    One residual this cannot remove.  Rates come from the Hamiltonian, so a
    model built with ``dtype=jnp.float32`` stores float32 couplings and the
    spectrum carries that ~1e-7 relative error.  No downstream precision
    recovers it; build the model with ``dtype=jnp.float64`` when it matters.
    """
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _basis_chunk(lo, hi, values, powers, d):
    """Configurations with indices ``[lo, hi)`` as a ``(hi - lo, N)`` array."""
    idx = np.arange(lo, hi, dtype=np.int64)
    return values[(idx[:, None] // powers) % d]


def _index_of(configs, values, powers, tol=1e-6):
    """Basis index of each configuration in ``configs`` (shape ``(..., N)``).

    Returns ``(index, valid)``.  ``valid`` is False wherever a site value is
    not in the local Hilbert space -- which is not an error: ``Spin`` and
    ``BoseHubbard`` return successors outside the physical basis for forbidden
    moves (``S+`` at ``m = +S``, a boson past ``n_max``) and rely on those
    slots carrying zero rate.  Indexing them anyway would fabricate matrix
    elements at wrong positions, so they are reported and dropped by the
    caller.

    Written in ``jax.numpy`` so it runs inside the jitted rate evaluation:
    only the integer indices and the mask leave the device, never the
    ``(chunk, M, N)`` float successor tensor.  Must be traced with x64
    enabled (see :func:`_float64`) so ``powers`` stays int64.
    """
    diff = jnp.abs(configs[..., None] - values)
    digit = jnp.argmin(diff, axis=-1)
    valid = jnp.take_along_axis(diff, digit[..., None], axis=-1)[..., 0] <= tol
    return jnp.sum(digit * powers, axis=-1), valid.all(axis=-1)


def _operator_matrix(op, dim, values, powers, d, N, dtype=np.float64):
    """Full sparse matrix of ``op``, built from ``get_rates`` alone.

    Column ``x`` holds the transitions out of configuration ``x``, following
    the same reconstruction the rate protocol defines -- ``H[x, x] =
    rates[0]`` and ``H[y, x] += rates[k]`` for the successor ``y`` of
    transition ``k >= 1``.
    """
    values_j = jnp.asarray(values, dtype=jnp.float64)
    powers_j = jnp.asarray(powers, dtype=jnp.int64)

    def one(c):
        rates, new_cfg = op.get_rates(c, return_configs=True)
        if new_cfg is None or rates.shape[0] <= 1:
            return rates, None, None
        # Successor -> basis index on the device; only (M-1,) ints + mask
        # come back to the host.
        tgt, valid = _index_of(new_cfg[1:, :], values_j, powers_j)
        return rates, tgt, valid

    rates_fn = jax.jit(jax.vmap(one))
    rows, cols, data = [], [], []
    for lo in range(0, dim, _CHUNK):
        hi = min(lo + _CHUNK, dim)
        cfg = _basis_chunk(lo, hi, values, powers, d)
        rates, tgt, valid = rates_fn(jnp.asarray(cfg, dtype=jnp.float64))
        rates = np.asarray(rates, dtype=dtype)              # (n, M)
        src = np.arange(lo, hi, dtype=np.int64)

        rows.append(src)                                    # diagonal
        cols.append(src)
        data.append(rates[:, 0])

        if tgt is not None:
            tgt = np.asarray(tgt)                           # (n, M-1) int64
            valid = np.asarray(valid)                       # (n, M-1) bool
            off = rates[:, 1:]
            # Drop zero-rate slots as well as unphysical successors: a zero
            # contributes nothing, and skipping it keeps the triplet lists
            # small for sparse operators.
            leaving = ~valid & (off != 0.0)
            if np.any(leaving):
                bad = np.argwhere(leaving)[0]
                raise ValueError(
                    f"{type(op).__name__}: transition {int(bad[1]) + 1} of basis "
                    f"state {int(lo + bad[0])} has amplitude {float(off[tuple(bad)])!r} "
                    "but leads outside the supplied local_hilbert_space; only "
                    "transitions of zero amplitude may leave the basis.")
            keep = valid & (off != 0.0)
            src_b = np.broadcast_to(src[:, None], off.shape)
            rows.append(tgt[keep])
            cols.append(src_b[keep])
            data.append(off[keep])
            del src_b

    return sparse.coo_matrix(
        (np.concatenate(data),
         (np.concatenate(rows), np.concatenate(cols))),
        shape=(dim, dim), dtype=dtype).tocsr()


def _diagonal_values(op, dim, values, powers, d, dtype=np.float64):
    """``op.diagonal_weight`` over the whole basis, shape ``(dim,)`` or
    ``(dim, n_vals)`` for a vector-valued operator (e.g. per-bond ZZ)."""
    fn = jax.jit(jax.vmap(op.diagonal_weight))
    out = []
    for lo in range(0, dim, _CHUNK):
        hi = min(lo + _CHUNK, dim)
        cfg = _basis_chunk(lo, hi, values, powers, d)
        out.append(np.asarray(fn(jnp.asarray(cfg, dtype=jnp.float64)),
                              dtype=dtype))
    return np.concatenate(out, axis=0)


def _require_hermitian(H, name):
    """Raise unless the sparse matrix ``H`` has finite entries and is
    symmetric to ``_HERMITIAN_RTOL`` of its largest entry: the symmetric
    eigensolvers assume it, and a malformed custom operator should fail here
    rather than return a plausible number."""
    if not np.all(np.isfinite(H.data)):
        raise ValueError(f"{name}: the operator matrix has non-finite entries.")
    scale = float(abs(H).max()) if H.nnz else 0.0
    asym = float(abs(H - H.T).max()) if H.nnz else 0.0
    if asym > _HERMITIAN_RTOL * max(scale, 1.0):
        raise ValueError(
            f"{name}: the operator matrix is not Hermitian (largest "
            f"|H - H^T| = {asym:.3g} against entries up to {scale:.3g}); "
            "get_rates must give H[y, x] == H[x, y] for every pair of basis "
            "states.")


def exact_diagonalization(hamiltonian, local_hilbert_space=None, N=None,
                          diag_operators=None, off_diag_operators=None,
                          number_of_states=1, return_states=False):
    """Lowest eigenpairs of ``hamiltonian``, with observables on each.

    Args:
        hamiltonian:        Any object implementing the Hamiltonian protocol
                            (``M_total``, ``get_rates``, ``generate_state``).
                            Its matrix is rebuilt from ``get_rates`` alone.
        local_hilbert_space: The per-site values, in basis-digit order,
                            e.g. ``(1, -1)`` for spin-1/2, ``(-1, 0, 1)``
                            for S = 1, ``(0, 1, 2)`` for bosons with
                            ``n_max = 2``.  The order fixes the basis
                            indexing of the returned states (energies do not
                            depend on it).  Defaults to
                            ``hamiltonian.local_hilbert_space``, which the
                            pre-implemented classes provide.
        N:                  Number of sites; the basis has
                            ``len(local_hilbert_space) ** N`` states.
                            Defaults to ``hamiltonian.N``, so it only has to
                            be given for a class that exposes no ``N`` -- a
                            user-defined Hamiltonian implementing just the
                            protocol.
        diag_operators:     ``dict`` name → operator, measured as
                            ``Σ_x |ψ_x|² O(x)`` via ``diagonal_weight``.
        off_diag_operators: ``dict`` name → operator, measured as
                            ``ψᵀ O ψ`` with ``O`` built from its own
                            ``get_rates``.
        number_of_states:   How many lowest eigenpairs to solve for.
        return_states:      Also return the eigenvectors, shape
                            ``(dim, number_of_states)``.  Needed by ``ExactGroundState``, which turns them
                            into guide amplitudes.

    Returns:
        ``(energies, diag_obs, off_diag_obs)``, or that tuple with ``states``
        appended when ``return_states`` is True.  ``energies`` has shape
        ``(number_of_states,)``; the two dicts mirror the input dicts key for
        key, each value carrying one entry per state along the leading axis.

    Note:
        The routine works in float64 throughout, whatever precision the caller
        is running in.  The one residual it cannot remove is the Hamiltonian's
        own dtype: a ``dtype=jnp.float32`` model stores float32 couplings, and
        the spectrum inherits that ~1e-7 relative error; build the model with
        ``dtype=jnp.float64`` when it matters.

    Raises:
        ValueError: on a basis larger than 2**24 states, an ``N`` or
            ``local_hilbert_space`` contradicting what the Hamiltonian reports
            (``N``, ``d`` or the local values), a ``local_hilbert_space`` with
            repeated or non-finite values, ``number_of_states`` outside
            ``[1, dim]``, a transition of nonzero amplitude that leaves the
            basis, a non-Hermitian or non-finite operator matrix, or an
            estimated allocation above ``_MAX_BYTES``.
    """
    if local_hilbert_space is None:
        local_hilbert_space = getattr(hamiltonian, "local_hilbert_space", None)
        if local_hilbert_space is None:
            raise ValueError(
                f"local_hilbert_space was not given and "
                f"{type(hamiltonian).__name__} exposes no "
                "local_hilbert_space attribute.")
    values = np.asarray(local_hilbert_space, dtype=float).ravel()
    d = values.size
    if d < 2:
        raise ValueError(
            f"local_hilbert_space needs at least two values, got {values.tolist()}.")
    if not np.all(np.isfinite(values)) or np.unique(values).size != d:
        raise ValueError(
            f"local_hilbert_space must hold distinct finite values, got "
            f"{values.tolist()}.")
    own = getattr(hamiltonian, "local_hilbert_space", None)
    if own is not None:
        own = np.asarray(own, dtype=float).ravel()
        if own.size != d:
            raise ValueError(
                f"local_hilbert_space has {d} values {values.tolist()} but the "
                f"Hamiltonian acts on {own.size} values {own.tolist()}: a "
                "different local dimension.")
        if not np.array_equal(np.sort(own), np.sort(values)):
            raise ValueError(
                f"local_hilbert_space {values.tolist()} differs from the values "
                f"{own.tolist()} the Hamiltonian acts on; a different order is "
                "fine, different values are a different basis.")
    if N is None:
        N = getattr(hamiltonian, "N", None)
        if N is None:
            raise ValueError(
                f"N was not given and {type(hamiltonian).__name__} exposes no "
                "N attribute, so the size of the basis is unknown.")
    N = int(N)
    dim = d ** N
    if dim > _MAX_DIM:
        raise ValueError(
            f"basis of {d}**{N} = {dim} states exceeds the {_MAX_DIM} limit; "
            "exact diagonalization is not the right tool at this size.")
    number_of_states = _validated_int("number_of_states", number_of_states, 1)
    if number_of_states > dim:
        raise ValueError(
            f"number_of_states must satisfy 1 <= k <= dim = {dim}, "
            f"got {number_of_states}.")
    # Estimated allocations: the COO triplets of the operator (one float64
    # and two int64 per nonzero, at most M_total per basis state) and, for a
    # full-spectrum solve, the dense matrix with its eigenvectors.
    m_total = int(getattr(hamiltonian, "M_total", 1))
    sparse_bytes = dim * m_total * (8 + 8 + 8)
    dense = dim <= _DENSE_DIM or number_of_states >= dim
    dense_bytes = 2 * dim * dim * 8 if dense else 0
    if max(sparse_bytes, dense_bytes) > _MAX_BYTES:
        raise ValueError(
            f"exact diagonalization of {dim} states would allocate about "
            f"{max(sparse_bytes, dense_bytes) / 2**30:.1f} GiB "
            f"({'dense full-spectrum solve' if dense_bytes > sparse_bytes else 'sparse operator'}), "
            f"above the {_MAX_BYTES / 2**30:.0f} GiB limit; it is not the right "
            "tool at this size.")

    powers = (d ** np.arange(N - 1, -1, -1)).astype(np.int64)

    # A wrong local_hilbert_space produces a plausible-looking spectrum on the
    # wrong Hilbert space, which is the worst kind of failure here.  It cannot
    # be detected from the rate protocol alone -- a spin-1/2 model evaluates
    # happily on a three-state basis -- so cross-check whatever the Hamiltonian
    # chooses to expose.  Absent attributes are skipped rather than assumed.
    h_N = getattr(hamiltonian, "N", None)
    if h_N is not None and int(h_N) != N:
        raise ValueError(
            f"N={N} but the Hamiltonian reports N={int(h_N)}; the basis would "
            "not be the one it acts on.")
    h_d = getattr(hamiltonian, "d", None)
    if h_d is not None and int(h_d) != d:
        raise ValueError(
            f"local_hilbert_space has {d} values {values.tolist()} but the "
            f"Hamiltonian reports a local dimension of {int(h_d)}.")

    with _float64():
        H = _operator_matrix(hamiltonian, dim, values, powers, d, N)
        _require_hermitian(H, type(hamiltonian).__name__)
        # eigsh needs k < dim and is unreliable on tiny matrices, so small
        # bases (or a request for the whole spectrum) are solved densely.
        if dim <= _DENSE_DIM or number_of_states >= dim:
            energies, states = np.linalg.eigh(H.toarray())
            energies = energies[:number_of_states]
            states = states[:, :number_of_states]
        else:
            # A fixed start vector: ARPACK otherwise advances its own seed
            # between calls, and the energies of two diagonalisations in
            # one process then differ in the last bits.
            v0 = np.random.default_rng(0).standard_normal(dim)
            energies, states = eigsh(H, k=number_of_states, which="SA", v0=v0)
            order = np.argsort(energies)
            energies, states = energies[order], states[:, order]

        diag_obs, off_diag_obs = {}, {}
        for name, op in (diag_operators or {}).items():
            # <O> = sum_x |psi_x|^2 O(x): diagonal operators need no matrix
            vals = _diagonal_values(op, dim, values, powers, d)
            diag_obs[name] = (states ** 2).T @ vals   # (k,) or (k, n_vals)
        for name, op in (off_diag_operators or {}).items():
            op_matrix = _operator_matrix(op, dim, values, powers, d, N)
            off_diag_obs[name] = np.einsum("ik,ik->k", states, op_matrix @ states)

    if return_states:
        return energies, diag_obs, off_diag_obs, states
    return energies, diag_obs, off_diag_obs
