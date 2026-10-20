"""StructureFactor vs an INDEPENDENT Kronecker construction.

A factor of 2 or a swapped J_pm/J_pmpm still gives a well-formed operator, so
the reference builds ``sum_pairs w_ij (A_iB_j + h.c.)`` from 2x2 single-site
matrices, sharing no code with the rate path, and compares the full 2^N matrix.
Basis: site 0 most significant, index 0 = up (+1).
"""
import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from core.reference import reference_basis_index
from pygfmc.hamiltonian import SpinHalf, StructureFactor, SzSzStructureFactor, is_valid_k

SP = np.array([[0.0, 1.0], [0.0, 0.0]])          # |up><down|
SM = SP.T.copy()
MAT = {"X": (0.5 * (SP + SM)).astype(complex), "Y": (SP - SM) / (2.0j),
       "Z": np.diag([0.5, -0.5]).astype(complex),
       "Sp": SP.astype(complex), "Sm": SM.astype(complex)}
CH = {("Sp", "Sm"): "SpSm", ("Sp", "Sp"): "SpSp", ("Z", "Z"): "SzSz"}


@pytest.fixture(autouse=True)
def _float64(x64):
    """Float64 matrix identities, inside the conftest x64 scope."""
    with x64():
        yield


def embed(op2, site, N):
    """Kron op2 onto `site`, site 0 most significant."""
    out = np.array([[1.0 + 0j]])
    for s in range(N):
        out = np.kron(out, op2 if s == site else np.eye(2))
    return out


def geom(positions, N):
    """The full i<j pair list and its raw displacements (onsite excluded)."""
    iu, ju = np.triu_indices(N, k=1)
    return np.stack([iu, ju], axis=1).astype(np.int32), positions[iu] - positions[ju]


def random_geom(N, seed, d=3):
    positions = np.random.default_rng(seed).normal(size=(N, d))
    return positions, *geom(positions, N)


def reference(pairs, dr, k, A, B, N, shift=1.0):
    """sum_pairs (shift + cos(k.dr)) (A_i B_j + h.c.), built from scratch."""
    op_matrix = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for (i, j), d in zip(pairs, dr):
        T = embed(MAT[A], int(i), N) @ embed(MAT[B], int(j), N)
        op_matrix += (shift + np.cos(np.dot(d, k))) * (T + T.conj().T)
    return op_matrix


def build_sf(*args, **kwargs):
    """StructureFactor, expecting the diagonal-channel warning exactly for SzSz."""
    if str(kwargs.get("channel", "SpSm")).lower().replace(" ", "") == "szsz":
        with pytest.warns(UserWarning, match="DIAGONAL"):
            return StructureFactor(*args, **kwargs)
    return StructureFactor(*args, **kwargs)


def from_get_rates(op, N):
    """Rebuild the matrix using ONLY get_rates (row 0 of new_configs is the diagonal)."""
    M = np.zeros((2 ** N, 2 ** N))
    for bits in itertools.product([1.0, -1.0], repeat=N):
        cfg = jnp.asarray(bits, dtype=jnp.float32)
        col = int(reference_basis_index(op, cfg))
        rates, new_cfgs = op.get_rates(cfg, return_configs=True)
        rates = np.asarray(rates, dtype=float)
        M[col, col] += rates[0]
        for t in range(1, rates.shape[0]):
            M[int(reference_basis_index(op, new_cfgs[t])), col] += rates[t]
    return M


def zz_diag_values(op, N):
    """(diagonal_weight, reference index) over the full basis."""
    bits = [jnp.asarray(b, dtype=jnp.float64)
            for b in itertools.product([1.0, -1.0], repeat=N)]
    return (np.array([float(op.diagonal_weight(b)) for b in bits]),
            [int(reference_basis_index(op, b)) for b in bits])


# ── the off-diagonal constructor ────────────────────────────────────────────

@pytest.mark.parametrize("A,B", [("Sp", "Sm"), ("Sp", "Sp")])
def test_matches_kronecker_reference(A, B):
    """Both channels equal the Kronecker reference over the full 2^N matrix."""
    N = 6
    positions, pairs, dr = random_geom(N, 7)
    k = np.array([0.7, -1.3, 0.4])
    op = StructureFactor(positions, k, channel=CH[(A, B)], dtype=jnp.float64)
    ref = reference(pairs, dr, k, A, B, N)
    assert np.max(np.abs(ref.imag)) < 1e-12, "reference must be real"
    err = np.max(np.abs(from_get_rates(op, N) - ref.real))
    assert err / max(np.max(np.abs(ref.real)), 1.0) < 1e-10, f"{A}{B}: {err:.3e}"


def test_non_vacuous():
    """Perturbing one weight by 1e-4 relative must be caught."""
    N = 5
    positions, pairs, dr = random_geom(N, 3)
    k = np.array([1.1, 0.2, -0.5])
    op = StructureFactor(positions, k, channel="SpSm", dtype=jnp.float64)
    W = 1.0 + np.cos(dr @ k)          # recomputed: reproducing it is part of the check
    assert np.all(W > 0), "this k must leave every pair with nonzero weight"
    bad = W.copy()
    bad[0] *= 1.0001
    op_bad = SpinHalf.create(pairs, J_x=jnp.asarray(2 * bad), J_y=jnp.asarray(2 * bad),
                             J_z=0.0, N=N, dtype=jnp.float64, sign_check=False)
    assert np.max(np.abs(from_get_rates(op, N) - from_get_rates(op_bad, N))) > 1e-6


@pytest.mark.parametrize("ch", ["SpSm", "SpSp", "SpSm+SpSp"])
def test_rates_non_negative(ch):
    """The off-diagonal estimator requires every rate >= 0 (inverse-CDF sampling)."""
    N = 6
    positions, pairs, dr = random_geom(N, 11)
    op = StructureFactor(positions, np.array([0.9, 0.3, -0.2]), channel=ch,
                         dtype=jnp.float64)
    worst = min(float(np.min(np.asarray(op.get_rates(jnp.asarray(b))[0])[1:]))
                for b in itertools.product([1.0, -1.0], repeat=N))
    assert worst >= -1e-12, f"{ch}: most negative off-diagonal rate {worst:.3e}"


def test_k_zero_reduces_to_the_plain_pair_sum():
    """At k=0 the weight is 2 on every pair, so J_pm = 2 (J_x = J_y = 4)."""
    N = 5
    positions, pairs, dr = random_geom(N, 5)
    op = StructureFactor(positions, np.zeros(3), channel="SpSm", dtype=jnp.float64)
    plain = SpinHalf.create(pairs, J_x=4.0, J_y=4.0, J_z=0.0, N=N, dtype=jnp.float64,
                            sign_check=False)
    assert np.max(np.abs(from_get_rates(op, N) - from_get_rates(plain, N))) < 1e-10


def test_blocked_and_invalid_combinations():
    """YY, S.S, misspelled ZZ and mixed species raise with the route named."""
    N = 4
    positions, pairs, dr = random_geom(N, 0)
    k = np.zeros(3)
    for ch in ("ZZ", "Sz"):
        with pytest.raises(ValueError, match="spelled 'SzSz'"):
            StructureFactor(positions, k, channel=ch)
    with pytest.raises(ValueError, match="NEGATIVE"):
        StructureFactor(positions, k, channel="YY")
    with pytest.raises(ValueError, match="then add"):
        StructureFactor(positions, k, channel="S.S")
    for ch in ["XY", "XZ", "SpZ", "Q", "SpSm-SpSp"]:
        with pytest.raises(ValueError, match="not reachable"):
            StructureFactor(positions, k, channel=ch)


def test_input_validation():
    """The shape contract on positions and k is enforced."""
    N = 4
    positions, pairs, dr = random_geom(N, 0)
    k = np.array([0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match=r"positions must be \(N, d\)"):
        StructureFactor(positions[:, 0], k)
    with pytest.raises(ValueError, match="they must match"):
        StructureFactor(positions, k[:2])
    with pytest.raises(ValueError, match="ONE wavevector"):
        StructureFactor(positions, np.stack([k, k]))
    with pytest.raises(ValueError, match="at least two sites"):
        StructureFactor(positions[:1], k)


def test_periodic_image_does_not_matter_for_valid_k():
    """Shifting sites by periodic vectors leaves the operator unchanged for an
    allowed k, and changes it for a k off the grid."""
    L = np.array([[3.0, 0.0], [1.0, 2.0]])           # periodic vectors (rows)
    rng = np.random.default_rng(31)
    N = 5
    positions = rng.normal(size=(N, 2))
    shifted = positions + rng.integers(-2, 3, size=(N, 2)) @ L
    good = 2 * np.pi * np.array([1, -2]) @ np.linalg.inv(L).T
    assert is_valid_k(good, L) and not is_valid_k(1.3 * good, L)
    for k, same in ((good, True), (1.3 * good, False)):
        a = from_get_rates(StructureFactor(positions, k, dtype=jnp.float64), N)
        b = from_get_rates(StructureFactor(shifted, k, dtype=jnp.float64), N)
        assert (np.max(np.abs(a - b)) < 1e-9) == same


def test_is_valid_k():
    """k.L_j / 2 pi integer for every row; single k gives a bool, many an array."""
    L = np.array([[4.0, 0.0], [2.0, 2.0 * np.sqrt(3.0)]])
    ms = np.array([[0, 0], [1, 0], [0, 1], [3, -2]])
    ks = 2 * np.pi * ms @ np.linalg.inv(L).T
    assert np.array_equal(is_valid_k(ks, L), np.ones(4, dtype=bool))
    assert is_valid_k(ks[1], L) is True
    assert is_valid_k(ks[1] * 0.5, L) is False
    assert np.array_equal(is_valid_k(np.vstack([ks[2], 0.5 * ks[2]]), L), [True, False])
    with pytest.raises(ValueError, match=r"\(d, d\)"):
        is_valid_k(ks, L[:1])
    with pytest.raises(ValueError, match="k_vectors must be"):
        is_valid_k(np.zeros(3), L)


def test_incommensurate_k_is_accepted_without_complaint():
    """The constructor does NOT check k, by design; commensurability is the caller's job (is_valid_k)."""
    N = 4
    positions, pairs, dr = random_geom(N, 0)
    good = np.pi * np.array([1.0, 0.0, 0.0])
    for k in (good, good * 1.37):
        op = StructureFactor(positions, k, channel="SpSm", dtype=jnp.float64)
        W = 1.0 + np.cos(dr @ k)
        assert op.n_off == int(np.sum(W > 1e-12)) == N * (N - 1) // 2
        np.testing.assert_allclose(np.asarray(op.J_pm), W, rtol=1e-12)


def full_ordered(positions, q, A, B, N):
    """The textbook AB(q) = sum_{i != j} exp(-i q.r_ij) A_iB_j over ORDERED pairs, no h.c."""
    op_matrix = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for i in range(N):
        for j in range(N):
            if i != j:
                op_matrix += (np.exp(-1j * np.dot(q, positions[i] - positions[j]))
                      * (embed(MAT[A], i, N) @ embed(MAT[B], j, N)))
    return op_matrix


@pytest.mark.parametrize("A,B", [("Sp", "Sm"), ("Sp", "Sp"), ("Z", "Z")])
def test_which_AB0_plus_ABk_it_actually_is(A, B):
    """``O == herm(0) + herm(k)`` always; ``O == Re[AB(0)+AB(k)]`` (no factor 2) only for SpSm and SzSz."""
    N = 5
    positions, pairs, dr = random_geom(N, 4)
    k = np.array([0.8, -0.5, 1.2])
    op_matrix = from_get_rates(build_sf(positions, k, channel=CH[(A, B)],
                                        dtype=jnp.float64), N)
    scale = max(np.max(np.abs(op_matrix)), 1.0)

    H = (reference(pairs, dr, np.zeros(3), A, B, N, shift=0.0)
         + reference(pairs, dr, k, A, B, N, shift=0.0))
    assert np.max(np.abs(H.imag)) < 1e-12, "cos-weighted form must be real here"
    assert np.max(np.abs(op_matrix - H.real)) / scale < 1e-10, "herm identity"

    f0, fk = full_ordered(positions, np.zeros(3), A, B, N), full_ordered(positions, k, A, B, N)
    matches = np.max(np.abs(op_matrix - (f0 + fk).real)) / scale < 1e-10
    hermitian = np.max(np.abs(fk - fk.conj().T)) < 1e-12
    assert matches == (CH[(A, B)] in ("SpSm", "SzSz"))
    assert hermitian == (CH[(A, B)] in ("SpSm", "SzSz"))
    if CH[(A, B)] == "SpSp":          # no scalar multiple works
        for f in (0.5, 1.0, 2.0):
            assert np.max(np.abs(op_matrix - f * (f0 + fk).real)) / scale > 1e-3


@pytest.mark.parametrize("geometry", ["random", "lattice"])
def test_szsz_channel_is_szsz_structure_factor_at_0_plus_k(geometry):
    """StructureFactor(channel="SzSz") == SzSzStructureFactor at q = 0 plus q = k,
    configuration by configuration: both are the ordered pair sum, one with
    the 1 + cos weight, the other through |rho_q|^2."""
    if geometry == "random":
        N = 6
        positions = np.random.default_rng(42).normal(size=(N, 3))
        ks = [np.array([0.7, -1.3, 0.4]), np.array([2.1, 0.2, -0.9])]
    else:                                            # 3 x 2 triangular torus
        a = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]])
        L = np.array([3.0, 2.0])[:, None] * a
        positions = np.array([x * a[0] + y * a[1] for y in range(2) for x in range(3)])
        N = len(positions)
        ks = [2 * np.pi * np.array(m) @ np.linalg.inv(L).T for m in ([1, 0], [1, 1], [2, -1])]
        assert is_valid_k(np.array(ks), L).all()
    configs = jnp.asarray(list(itertools.product([1.0, -1.0], repeat=N)), dtype=jnp.float64)
    for k in ks:
        op = build_sf(positions, k, channel="SzSz", dtype=jnp.float64)
        assert op.M_total == 1, "the SzSz channel is diagonal"
        got = np.array([float(op.diagonal_weight(c)) for c in configs])
        sf = SzSzStructureFactor(positions, np.stack([np.zeros_like(k), k]), dtype=jnp.float64)
        want = np.asarray(sf.diagonal_weight(configs)).sum(axis=1)
        assert np.max(np.abs(got - want)) < 1e-10


def test_szsz_channel_matches_kronecker_reference():
    """The SzSz channel is sum_{i<j} (1 + cos)(Z_iZ_j + h.c.) = 2(1 + cos) Z_iZ_j."""
    N = 5
    positions, pairs, dr = random_geom(N, 6, d=2)
    k = np.array([1.4, -0.6])
    op = build_sf(positions, k, channel="szsz", dtype=jnp.float64)
    ref = reference(pairs, dr, k, "Z", "Z", N)
    assert np.max(np.abs(from_get_rates(op, N) - ref.real)) < 1e-10


def test_channels_are_additive():
    """O("SpSm+SpSp") == O("SpSm") + O("SpSp") exactly; names are case/space-insensitive."""
    N = 5
    positions, pairs, dr = random_geom(N, 21)
    k = np.array([0.9, -0.7, 0.2])

    def build(channel):
        return from_get_rates(StructureFactor(positions, k, dtype=jnp.float64,
                                              channel=channel), N)

    o_pm, o_pp, o_both = build("SpSm"), build("SpSp"), build("SpSm+SpSp")
    assert np.max(np.abs(o_both - (o_pm + o_pp))) < 1e-10
    assert np.array_equal(build("spsm + spsp"), o_both)
    assert np.array_equal(build("spsm"), o_pm)
    assert np.array_equal(build("SPSP"), o_pp)
    for removed_alias in ("XX", "PM", "SmSp", "PP", "SpSp+SpSm"):
        with pytest.raises(ValueError, match="not reachable"):
            StructureFactor(positions, k, channel=removed_alias)


# ── the diagonal SzSz(k) operator ───────────────────────────────────────────

def zz_operator(pairs, dr, k, N, dtype=jnp.float64):
    """The diagonal ZZ(k) operator built directly from SpinHalf (raw cos, no +h.c.)."""
    return SpinHalf.create(pairs, J_x=0.0, J_y=0.0,
                           J_z=jnp.asarray(np.cos(dr @ k), dtype=dtype), N=N, dtype=dtype)


def szz_reference(pairs, dr, q, N):
    """sum_pairs cos(q.dr) Z_i Z_j, built from scratch."""
    op_matrix = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for (i, j), d in zip(pairs, dr):
        op_matrix += np.cos(np.dot(q, d)) * (embed(MAT["Z"], int(i), N) @ embed(MAT["Z"], int(j), N))
    return op_matrix


def test_scalar_Jx_alongside_array_Jz_builds():
    """A scalar J_x next to a per-pair J_z array builds the diagonal operator and
    gives the explicit pair sum of J_z S^z S^z."""
    N = 4
    pairs = np.array([(i, j) for i in range(N) for j in range(i + 1, N)], dtype=np.int32)
    w = jnp.asarray(np.arange(1, len(pairs) + 1, dtype=float))
    op = SpinHalf.create(pairs, J_x=0.0, J_y=0.0, J_z=w, N=N, dtype=jnp.float64)
    assert op.M_total == 1
    cfg = jnp.asarray([1.0, -1.0, 1.0, 1.0], dtype=jnp.float64)
    ref = sum(float(w[e]) * float(cfg[a]) * float(cfg[b]) / 4.0
              for e, (a, b) in enumerate(pairs))
    assert abs(float(op.diagonal_weight(cfg)) - ref) < 1e-12


# ── arbitrary dimension ─────────────────────────────────────────────────────

@pytest.mark.parametrize("d", [1, 2, 3, 4])
def test_any_dimension(d):
    """Nothing is 3D-specific; the weight is a dot product."""
    N = 5
    rng = np.random.default_rng(100 + d)
    positions = rng.normal(size=(N, d))
    k = rng.normal(size=d)
    pairs, dr = geom(positions, N)
    op = StructureFactor(positions, k, channel="SpSm", dtype=jnp.float64)
    ref = reference(pairs, dr, k, "Sp", "Sm", N)
    assert np.max(np.abs(from_get_rates(op, N) - ref.real)) < 1e-10

    opd = zz_operator(pairs, dr, k, N)
    assert opd.M_total == 1
    got, idx = zz_diag_values(opd, N)
    assert np.max(np.abs(got - np.diag(szz_reference(pairs, dr, k, N).real)[idx])) < 1e-10


