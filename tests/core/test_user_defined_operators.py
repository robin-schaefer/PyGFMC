"""The paper's user-defined operator classes, verified against PyGFMC.

The appendix classes are reproduced VERBATIM so the published snippets stay
executable.  Checked: the Hamiltonian protocol against an independent
Kronecker construction, ``return_configs`` vs ``generate_state``, runs against
exact diagonalization, the custom guide, and the library's multi-k
``SzSzStructureFactor`` against the appendix pair sum.
"""
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.reference import pyrochlore_cubic_geometry, reference_basis_index
from pygfmc import run_gfmc
from pygfmc.walkers import prepare_walkers_random, prepare_walkers_from_configs
from pygfmc.hamiltonian import SpinHalf, chain, exact_diagonalization, pyrochlore
# aliased to keep it apart from the paper's single-k PairSzSz below
from pygfmc.hamiltonian import SzSzStructureFactor as LibraryStructureFactor
from pygfmc.wavefunction import ExactGroundState


# ---------------------------------------------------------------------------
# VERBATIM from the paper, appendix "An n-body Hamiltonian"
# ---------------------------------------------------------------------------
class NBody:
    def __init__(self, clusters, J_z, J_x, h=0.0):
        self.clusters = jnp.asarray(clusters)          # (n_terms, n)
        self.J_z, self.J_x, self.h = J_z, J_x, h
        self.M_total  = 1 + self.clusters.shape[0]

    def get_rates(self, config, return_configs=False):
        diag  = self.diagonal_weight(config)
        off   = jnp.full(self.clusters.shape[0], -self.J_x)
        rates = jnp.concatenate([jnp.array([diag]), off])
        if not return_configs:
            return rates, None
        flips = jax.vmap(lambda c: config.at[c].multiply(-1))(self.clusters)
        return rates, jnp.concatenate([config[None, :], flips])

    def generate_state(self, config, idx):
        flipped = config.at[self.clusters[idx - 1]].multiply(-1)
        return jnp.where(idx == 0, config, flipped)

    def diagonal_weight(self, config):
        return (-self.J_z * jnp.sum(jnp.prod(config[self.clusters], axis=1))
                - self.h * jnp.sum(config))

jax.tree_util.register_pytree_node(
    NBody,
    lambda h: ((h.clusters,), (h.J_z, h.J_x, h.h)),
    lambda aux, children: NBody(children[0], *aux))


# ---------------------------------------------------------------------------
# VERBATIM from the paper, appendix "S^z S^z structure factor"
# ---------------------------------------------------------------------------
class PairSzSz:
    def __init__(self, pairs, dr, k):
        self.pairs = jnp.asarray(pairs)                # (n_pairs, 2)
        self.w = 2.0 * jnp.cos(jnp.asarray(dr) @ jnp.asarray(k))

    def diagonal_weight(self, config):
        s = config[self.pairs]
        return 0.25 * jnp.sum(self.w * s[:, 0] * s[:, 1])


# ---------------------------------------------------------------------------
# VERBATIM from the paper, appendix "User-defined guiding wavefunctions"
# ---------------------------------------------------------------------------
class MagnetizationGuide:
    def __init__(self, beta):
        self.beta = jnp.asarray(beta)

    def log_psi(self, config, cache=None):
        # config is (N,) or (n, N)
        return self.beta * jnp.sum(config, axis=-1)

    def log_diff_from_configs(self, config, new_configs, cache=None):
        return self.log_psi(new_configs) - self.log_psi(config)

    def log_diff_from_operator(self, config, op, cache=None):
        # flipping the sites of cluster k changes the magnetization
        # by -2 sum_{i in cluster k} sigma_i; entry 0 is the diagonal
        dm = -2.0 * jnp.sum(config[op.clusters], axis=1)
        return self.beta * jnp.concatenate([jnp.zeros(1, dtype=dm.dtype), dm])

jax.tree_util.register_pytree_node(
    MagnetizationGuide,
    lambda g: ((g.beta,), None),
    lambda aux, children: MagnetizationGuide(*children),
)


# ---------------------------------------------------------------------------

_L = 6
_CLUSTERS = [[i, (i + 1) % _L] for i in range(_L)]
_J_Z, _J_X = 1.0, 0.7
_LAM = 10.0
_COORDS = np.array([[i, 0.0] for i in range(_L)])
_PAIRS = [(i, j) for i in range(_L) for j in range(_L) if i < j]
_DR = np.array([_COORDS[i] - _COORDS[j] for i, j in _PAIRS])


def _nbody(J_x=0.4):
    return NBody(_CLUSTERS, J_z=_J_Z, J_x=J_x)


def _pauli_matrix(J_x=0.4):
    """H from Kronecker products of PAULI matrices -- no shared code."""
    sx, sz = np.array([[0.0, 1.0], [1.0, 0.0]]), np.diag([1.0, -1.0])

    def term(op, sites):
        M = np.array([[1.0]])
        for s in range(_L):
            M = np.kron(M, op if s in sites else np.eye(2))
        return M

    return sum(-_J_Z * term(sz, c) - J_x * term(sx, c) for c in _CLUSTERS)


def _basis():
    """Site 0 most significant, +1 before -1."""
    return list(itertools.product([1, -1], repeat=_L))


# ── the n-body Hamiltonian ──────────────────────────────────────────────────

def test_nbody_get_rates_rebuilds_the_matrix():
    """H[y,x] from get_rates must equal the Kronecker construction."""
    h = _nbody(_J_X)
    H_ref = _pauli_matrix(_J_X)
    idx = {b: i for i, b in enumerate(_basis())}
    H = np.zeros_like(H_ref)
    for b, col in idx.items():
        cfg = jnp.array(b, dtype=jnp.float32)
        r = np.asarray(h.get_rates(cfg)[0])
        H[col, col] += r[0]
        for k in range(1, h.M_total):
            nb = tuple(np.asarray(h.generate_state(cfg, k)).astype(int).tolist())
            H[idx[nb], col] += r[k]
    assert np.abs(H - H_ref).max() < 1e-5, np.abs(H - H_ref).max()
    off = H_ref[~np.eye(len(H_ref), dtype=bool)]
    assert off.max() <= 1e-12, f"positive off-diagonal element {off.max()}"


def test_nbody_return_configs_agrees_with_generate_state():
    """The materialized tensor and the on-the-fly path must not disagree."""
    h = _nbody(_J_X)
    cfg = jnp.array([1, -1, 1, 1, -1, -1], dtype=jnp.float32)
    r_bare, c_bare = h.get_rates(cfg)
    r_full, c_full = h.get_rates(cfg, return_configs=True)
    assert c_bare is None, "return_configs=False must not materialize configs"
    assert c_full.shape == (h.M_total, _L)
    assert bool(jnp.all(r_bare == r_full))
    assert bool(jnp.all(c_full[0] == cfg)), "row 0 must repeat the input"
    for k in range(h.M_total):          # row 0 included: term 0 is the diagonal
        assert bool(jnp.all(c_full[k] == h.generate_state(cfg, k))), f"row {k}"


def test_nbody_gfmc_matches_exact_diagonalization():
    """A real run with the user-defined Hamiltonian must reach the ED energy."""
    h = _nbody(_J_X)
    E0 = float(np.linalg.eigvalsh(_pauli_matrix(_J_X))[0])
    walkers = prepare_walkers_from_configs(jnp.ones((1024, _L), dtype=jnp.float32))
    _, energy, _, _, _ = run_gfmc(
        walkers, 600, h, 1, None, [8, 16, 32], lam=_LAM,
        rng_key=jax.random.PRNGKey(0), warmup_steps=200, verbosity=0)
    best = energy[max(energy)]
    assert abs(best - E0) / _L < 0.02, f"gfmc={best:.5f} vs ED={E0:.5f}"


def test_structure_factor_diagonal_weight_is_exact():
    """The appendix structure factor equals an explicit pair sum."""
    cfg = jnp.array([1, -1, 1, 1, -1, -1], dtype=jnp.float32)
    for k in ([np.pi, 0.0], [0.0, 0.0], [np.pi / 3, 0.0]):
        sf_op = PairSzSz(_PAIRS, _DR, k)
        ref = sum(2 * np.cos(np.dot(k, _COORDS[i] - _COORDS[j]))
                  * (float(cfg[i]) / 2) * (float(cfg[j]) / 2) for i, j in _PAIRS)
        assert np.isclose(float(sf_op.diagonal_weight(cfg)), ref, atol=1e-5), k


# ── a protocol-only Hamiltonian through exact diagonalization ───────────────

def test_protocol_hamiltonian_exact_diagonalization():
    """exact_diagonalization needs nothing beyond the protocol."""
    h = _nbody()
    assert not hasattr(h, "N") and not hasattr(h, "local_hilbert_space")
    reference = np.sort(np.linalg.eigvalsh(_pauli_matrix()))[:3]
    energies, diag, off = exact_diagonalization(h, (-1, 1), _L, number_of_states=3)
    np.testing.assert_allclose(energies, reference, atol=1e-6)
    assert diag == {} and off == {}
    with pytest.raises(ValueError, match="exposes no"):     # N must be demanded
        exact_diagonalization(h, (-1, 1))


def test_protocol_hamiltonian_observables():
    """Diagonal and off-diagonal observables on a protocol-only Hamiltonian."""
    h = _nbody()
    zz = PairSzSz([[0, 1], [1, 2]], dr=[[1.0], [1.0]], k=[0.0])
    energies, diag, off = exact_diagonalization(
        h, (-1, 1), _L, diag_operators={"sf": zz},
        off_diag_operators={"flip": _nbody()}, number_of_states=2)
    assert set(diag) == {"sf"} and set(off) == {"flip"}
    assert np.shape(diag["sf"]) == (2,) and np.shape(off["flip"]) == (2,)
    np.testing.assert_allclose(off["flip"], energies, atol=1e-6)   # <H> = E

    _, _, _, states = exact_diagonalization(h, (-1, 1), _L, return_states=True)
    configs = np.array(list(itertools.product([1.0, -1.0], repeat=_L)))
    direct = float((states[:, 0] ** 2) @ jax.vmap(zz.diagonal_weight)(
        jnp.asarray(configs, dtype=jnp.float32)))
    assert diag["sf"][0] == pytest.approx(direct, rel=0, abs=1e-6)


def test_protocol_hamiltonian_exact_guide():
    """ExactGroundState on a protocol-only Hamiltonian needs local_hilbert_space, then is exact."""
    h = _nbody()
    with pytest.raises(ValueError, match="local_hilbert_space"):
        ExactGroundState(h)
    guide = ExactGroundState(h, local_hilbert_space=(-1, 1), N=_L)
    reference = float(np.linalg.eigvalsh(_pauli_matrix())[0])
    assert guide.E0 == pytest.approx(reference, rel=0, abs=1e-6)
    assert guide.log_amp.shape == (2 ** _L,)
    assert guide.N == _L and guide.local_hilbert_space == (-1.0, 1.0)

    walkers = prepare_walkers_from_configs(jnp.ones((256, _L)))
    _, energy, _, _, _ = run_gfmc(walkers, 200, h, 1, guide, [4, 8], lam=8.0,
                                  rng_key=jax.random.PRNGKey(0), warmup_steps=40,
                                  verbosity=0)
    for p, value in energy.items():
        assert float(value) == pytest.approx(reference, rel=0, abs=1e-3), p


def test_protocol_hamiltonian_exact_guide_round_trips(tmp_path):
    """The generic index map survives HDF5 and pytree round-trips."""
    h = _nbody()
    guide = ExactGroundState(h, local_hilbert_space=(-1, 1), N=_L)
    path = str(tmp_path / "guide.h5")
    guide.save(path)
    loaded = ExactGroundState.load(path, h)
    assert loaded.local_hilbert_space == guide.local_hilbert_space
    assert loaded.N == guide.N
    assert bool(jnp.array_equal(loaded.log_amp, guide.log_amp))
    assert loaded.E0 == pytest.approx(guide.E0, rel=0, abs=0)

    rebuilt = jax.tree_util.tree_unflatten(*reversed(jax.tree_util.tree_flatten(guide)))
    assert rebuilt.local_hilbert_space == guide.local_hilbert_space
    assert rebuilt._powers == guide._powers and rebuilt.N == guide.N
    cfg = jnp.ones(_L)
    assert int(rebuilt.basis_index(cfg)) == int(guide.basis_index(cfg))


def test_generic_index_agrees_with_the_per_class_map():
    """Default and reversed local orderings give the same amplitude per configuration."""
    h = SpinHalf.create(chain(4).edges, -1.0, -1.0, 1.0, h_x=0.3)
    per_class = ExactGroundState(h)                       # (1, -1) by default
    generic = ExactGroundState(h, local_hilbert_space=(-1, 1))
    assert per_class.local_hilbert_space == (1.0, -1.0)
    assert per_class.E0 == pytest.approx(generic.E0, rel=0, abs=1e-6)
    configs = [jnp.array(c) for c in itertools.product([1.0, -1.0], repeat=4)]
    for c in configs:        # the default ordering IS the standard basis index
        assert int(per_class.basis_index(c)) == int(reference_basis_index(h, c))
    np.testing.assert_allclose(np.sort(np.asarray(per_class.log_amp)),
                               np.sort(np.asarray(generic.log_amp)), atol=1e-6)
    for c in configs:
        a = float(per_class.log_amp[per_class.basis_index(c)])
        b = float(generic.log_amp[generic.basis_index(c)])
        assert a == pytest.approx(b, rel=0, abs=1e-6), c


# ── the custom guide ────────────────────────────────────────────────────────

def test_custom_guide_routines_agree():
    """The three guide routines must return the same ratios."""
    h = _nbody()
    guide = MagnetizationGuide(0.25)
    rng = np.random.default_rng(0)
    for _ in range(5):
        cfg = jnp.asarray(rng.choice([-1.0, 1.0], size=_L), dtype=jnp.float32)
        _, successors = h.get_rates(cfg, return_configs=True)
        from_op = np.asarray(guide.log_diff_from_operator(cfg, h))
        from_cfg = np.asarray(guide.log_diff_from_configs(cfg, successors))
        from_psi = np.asarray(jax.vmap(guide.log_psi)(successors) - guide.log_psi(cfg))
        np.testing.assert_allclose(from_op, from_cfg, atol=1e-6)
        np.testing.assert_allclose(from_op, from_psi, atol=1e-6)
        assert from_op[0] == pytest.approx(0.0, abs=1e-7)


def test_custom_guide_at_beta_zero_reduces_to_no_guide():
    """beta = 0 makes every ratio exp(0) = 1, so the walk draws the same moves
    as the unguided one; e_L agrees to float32 rounding only, because the
    forbidden-transition mask in transition_rates changes how XLA fuses the
    sum of the rates.

    Pinned to multi_step: under cont_time the while_loop amplifies the
    last-bit differences between the guided and unguided code paths.
    """
    h = _nbody()
    common = dict(rng_key=jax.random.PRNGKey(3), warmup_steps=0,
                  avg_log_weight=2.0, verbosity=0, method="multi_step")

    def walk(guide):
        return run_gfmc(prepare_walkers_from_configs(jnp.ones((256, _L))), 300, h,
                        1, guide, [4, 8], lam=8.0, **common)

    w_free, E_free, _, _, raw_free = walk(None)
    w_zero, E_zero, _, _, raw_zero = walk(MagnetizationGuide(0.0))
    for key in ("energy", "log_weights"):
        np.testing.assert_allclose(np.asarray(raw_free[key]), np.asarray(raw_zero[key]),
                                   rtol=0, atol=1e-5, err_msg=key)
    assert bool(jnp.all(w_free.config == w_zero.config))
    for p in E_free:
        assert float(E_free[p]) == pytest.approx(float(E_zero[p]), rel=0, abs=1e-5)

    _, _, _, _, raw_beta = walk(MagnetizationGuide(0.3))
    assert not np.array_equal(np.asarray(raw_free["energy"]),
                              np.asarray(raw_beta["energy"]))


def test_custom_guide_gfmc_reaches_the_exact_ground_state():
    """GFMC is unbiased whatever the guide; the tolerance covers the MC error."""
    h = _nbody()
    exact = float(exact_diagonalization(h, (-1, 1), _L)[0][0])
    _, energy, _, _, _ = run_gfmc(
        prepare_walkers_from_configs(jnp.ones((1024, _L))), 1000, h, 1,
        MagnetizationGuide(0.1), [8, 32], lam=8.0, rng_key=jax.random.PRNGKey(1),
        warmup_steps=200, verbosity=0)
    for p, value in energy.items():
        assert float(value) == pytest.approx(exact, rel=0, abs=0.05), p


# ── the library's multi-k SzSzStructureFactor vs the appendix pair sum ──────

_SF_L = 6
_SF_POS = np.arange(_SF_L, dtype=float)[:, None]
_SF_KS = (2 * np.pi * np.arange(_SF_L) / _SF_L)[:, None]      # the allowed grid
_SF_CFG = jnp.array([1, -1, 1, 1, -1, -1], dtype=jnp.float32)


def _appendix_pair_sum(k, minimum_image):
    """The appendix class, over i<j, with or without the minimum image."""
    pairs = [(i, j) for i in range(_SF_L) for j in range(_SF_L) if i < j]
    dr = np.array([_SF_POS[i] - _SF_POS[j] for i, j in pairs])
    if minimum_image:
        dr = dr - _SF_L * np.round(dr / _SF_L)
    return PairSzSz(pairs, dr, k)


@pytest.mark.parametrize("minimum_image", [False, True])
def test_structure_factor_multi_k_matches_the_appendix_pair_sum(minimum_image):
    """Same numbers as the pair sum at every allowed k, in both image conventions."""
    sf = LibraryStructureFactor(_SF_POS, _SF_KS)
    assert sf.k_vectors.shape[0] == _SF_L
    rng = np.random.default_rng(0)
    cfgs = [_SF_CFG] + [jnp.asarray(rng.choice([1.0, -1.0], size=_SF_L), dtype=jnp.float32)
                        for _ in range(10)]
    for cfg in cfgs:
        ref = np.array([float(_appendix_pair_sum(k, minimum_image).diagonal_weight(cfg))
                        for k in _SF_KS])
        np.testing.assert_allclose(np.asarray(sf.diagonal_weight(cfg)), ref,
                                   rtol=0, atol=1e-5)


def test_structure_factor_takes_the_wavevectors_as_given():
    """k is not validated (positions cannot decide it); only shapes are checked."""
    sf = LibraryStructureFactor(_SF_POS, [[2 * np.pi * 1.5 / _SF_L]])
    assert np.isfinite(float(sf.diagonal_weight(_SF_CFG)[0]))
    with pytest.raises(ValueError, match="one component per spatial"):
        LibraryStructureFactor(_SF_POS, [[0.0, 0.0]])


def test_structure_factor_zero_k_is_the_magnetization():
    """S(0) = ((sum_i s_i)^2 - N)/4 exactly: the i==j terms are dropped."""
    zero = LibraryStructureFactor(_SF_POS, [[0.0]])
    m = float(np.sum(np.asarray(_SF_CFG)))
    s0 = float(zero.diagonal_weight(_SF_CFG)[0])
    assert s0 == pytest.approx((m ** 2 - _SF_L) / 4, abs=1e-5)
    assert s0 + _SF_L / 4 == pytest.approx(m ** 2 / 4, abs=1e-5)

    batch = jnp.stack([_SF_CFG, -_SF_CFG])
    sf = LibraryStructureFactor(_SF_POS, _SF_KS)
    assert np.asarray(sf.diagonal_weight(batch)).shape == (2, _SF_L)
    np.testing.assert_allclose(np.asarray(sf.diagonal_weight(batch))[0],
                               np.asarray(sf.diagonal_weight(_SF_CFG)), rtol=0, atol=1e-6)


def test_structure_factor_is_a_pytree_and_jits():
    """Registered as a pytree with the phase tables as children; jits."""
    sf = LibraryStructureFactor(_SF_POS, _SF_KS)
    leaves, treedef = jax.tree_util.tree_flatten(sf)
    assert len(leaves) == 4, [np.shape(x) for x in leaves]
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    np.testing.assert_array_equal(np.asarray(rebuilt.diagonal_weight(_SF_CFG)),
                                  np.asarray(sf.diagonal_weight(_SF_CFG)))
    np.testing.assert_allclose(
        np.asarray(jax.jit(lambda o, c: o.diagonal_weight(c))(sf, _SF_CFG)),
        np.asarray(sf.diagonal_weight(_SF_CFG)), rtol=0, atol=1e-6)


def test_structure_factor_multi_k_in_a_run_equals_separate_operators():
    """One vector-valued operator == n_k scalar appendix operators, through run_gfmc.

    Both runs share rng_key and diagonal operators consume no randomness, so
    the main chains coincide.  An XXZ ring, since two NBody instances in one
    jit cache raise (its snippet keeps an array in the pytree aux).
    """
    ks = _SF_KS[[0, 1, 3]]
    h = SpinHalf.create(chain(_SF_L).edges, J_x=-1.0, J_y=-1.0, J_z=1.0)
    walkers = prepare_walkers_random(512, _SF_L, jax.random.PRNGKey(7))
    common = dict(rng_key=jax.random.PRNGKey(3), warmup_steps=50, verbosity=0)

    _, _, obs_vec, stats_vec, raw = run_gfmc(
        walkers, 400, h, 1, None, [8, 16], lam=2.5,
        diag_operators={"sf": LibraryStructureFactor(_SF_POS, ks)}, **common)
    _, _, obs_sep, stats_sep, _ = run_gfmc(
        walkers, 400, h, 1, None, [8, 16], lam=2.5,
        diag_operators={f"sf_{i}": _appendix_pair_sum(k, True)
                        for i, k in enumerate(ks)}, **common)

    assert set(raw["diag"]) == {"sf"}, "raw['diag'] is keyed by operator name"
    assert raw["diag"]["sf"].shape == (400, len(ks), 2), "(n_steps, n_vals, n_p)"
    for p in obs_vec["sf"]:
        merged = np.asarray(obs_vec["sf"][p])
        assert merged.shape == (len(ks),), merged.shape
        separate = np.array([float(obs_sep[f"sf_{i}"][p]) for i in range(len(ks))])
        np.testing.assert_allclose(merged, separate, rtol=2e-4, atol=1e-5)

    tau_merged = np.asarray(stats_vec["autocorrelation/diag/sf"])
    assert tau_merged.shape == (len(ks), 2), tau_merged.shape
    tau_separate = np.array([np.asarray(stats_sep[f"autocorrelation/diag/sf_{i}"])
                             for i in range(len(ks))])
    assert tau_separate.shape == tau_merged.shape
    np.testing.assert_allclose(tau_merged, tau_separate, rtol=1e-3)
    assert tau_merged.max() > 1.5 * tau_merged.min(), tau_merged   # per-value tau


# ── the shipped class vs the appendix pair sum on 1D/2D/3D lattices ─────────
#
# Algebraically equal but summed in different orders: float64 agrees to 1e-9,
# float32 to 32*eps*N, and k = 0 (integer arithmetic) bitwise in both.

def _tiling(dims, primitive, basis=None):
    """Positions of a complete ``dims`` tiling of ``primitive``, and its cell (rows)."""
    cells = np.array(list(np.ndindex(*dims)), dtype=float) @ primitive
    if basis is not None:
        cells = (cells[:, None, :]
                 + np.asarray(basis, dtype=float)[None]).reshape(-1, cells.shape[1])
    return cells, np.asarray(dims, dtype=float)[:, None] * primitive


def _sf_cases():
    h = np.sqrt(3.0) / 2.0
    fcc = pyrochlore(2, 2, 2)
    return {
        "1d_chain_L12":            _tiling((12,), np.eye(1)),
        "2d_square_4x4":           _tiling((4, 4), np.eye(2)),
        "2d_rect_3x5":             _tiling((3, 5), np.eye(2)),
        "2d_triangular_4x4":       _tiling((4, 4), np.array([[1.0, 0.0], [0.5, h]])),
        "2d_oblique_2site_basis":  _tiling((3, 3), np.array([[1.0, 0.0], [0.3, 0.9]]),
                                           basis=[[0.0, 0.0], [0.5, 0.25]]),
        "3d_cubic_3x3x3":          _tiling((3, 3, 3), np.eye(3)),
        "3d_orthorhombic_2x3x4":   _tiling((2, 3, 4), np.diag([1.0, 1.3, 0.7])),
        "3d_pyrochlore_fcc_N32":   (fcc.extras["positions"],
                                    fcc.extras["supercell_dims"][:, None]
                                    * fcc.extras["primitive_vectors"]),
        "3d_pyrochlore_cubic_N128": pyrochlore_cubic_geometry(2, 2, 2),
    }


_SF_CASES = _sf_cases()


def _allowed_k(cell, n_k=4, seed=0):
    """Reciprocal-lattice wavevectors k = 2 pi m cell^-T, m = 0 first."""
    d = cell.shape[0]
    recip = 2 * np.pi * np.linalg.inv(cell).T
    rng = np.random.default_rng(seed)
    ms = [np.zeros(d, dtype=int), np.ones(d, dtype=int), np.arange(1, d + 1, dtype=int)]
    while len(ms) < n_k:
        ms.append(rng.integers(-3, 4, size=d))
    return np.array([m @ recip for m in ms[:n_k]]), ms[:n_k]


def _appendix_ops(pos, cell, ks):
    """One appendix operator per k, over i<j with MINIMUM-IMAGE dr."""
    N = pos.shape[0]
    pairs = np.array([(i, j) for i in range(N) for j in range(N) if i < j])
    frac = (pos[pairs[:, 0]] - pos[pairs[:, 1]]) @ np.linalg.inv(cell)
    dr = (frac - np.round(frac)) @ cell
    return [PairSzSz(pairs, dr, k) for k in ks]


@pytest.mark.parametrize("case", sorted(_SF_CASES))
def test_structure_factor_equals_the_appendix_pair_sum_on_any_lattice(case, x64):
    """|sum_i e^{ik r_i} s_i|^2 == the O(N^2) pair sum, on symmetric and asymmetric cells."""
    pos, cell = _SF_CASES[case]
    N = pos.shape[0]
    ks, ms = _allowed_k(cell)
    assert np.allclose(ms[0], 0), "the first k must be 0, the exact anchor"
    configs = np.random.default_rng(12345).choice([1.0, -1.0], size=(5, N))

    sf32 = LibraryStructureFactor(pos, ks)
    assert sf32.k_vectors.shape[0] == len(ks)
    ref32 = _appendix_ops(pos, cell, ks)
    atol32 = 32 * np.finfo(np.float32).eps * N
    for cfg in configs:
        c = jnp.asarray(cfg, dtype=jnp.float32)
        got = np.asarray(sf32.diagonal_weight(c))
        assert got.shape == (len(ks),)
        ref = np.array([float(o.diagonal_weight(c)) for o in ref32])
        np.testing.assert_allclose(got, ref, rtol=0, atol=atol32)
        assert got[0] == ref[0], (case, got[0], ref[0])

    with x64():
        sf64 = LibraryStructureFactor(pos, ks, dtype=jnp.float64)
        ref64 = _appendix_ops(pos, cell, ks)
        for cfg in configs:
            c = jnp.asarray(cfg, dtype=jnp.float64)
            got = np.asarray(sf64.diagonal_weight(c))
            ref = np.array([float(o.diagonal_weight(c)) for o in ref64])
            np.testing.assert_allclose(got, ref, rtol=0, atol=1e-9)
            assert got[0] == ref[0], (case, got[0], ref[0])
