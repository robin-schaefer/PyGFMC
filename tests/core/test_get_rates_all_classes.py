"""Every Hamiltonian class: ``get_rates`` must reproduce the full matrix.

For each class one model is built with every parameter active (per-edge /
per-site arrays, so a broadcast-instead-of-indexed coupling cannot pass), H is
rebuilt column by column from ``get_rates`` over the ENTIRE Hilbert space and
compared with the independent Kronecker construction in ``reference.py``.
``generate_state(config, k)`` is checked against the successor tensor at every
configuration and slot, since production samples through it.
"""
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pygfmc.hamiltonian import BoseHubbard, PlaquetteModel, Spin, SpinHalf, ToricCode
from core.rates import all_configs, rebuild_matrix
from core.reference import reference_dense

_DTYPE = jnp.float32
_ATOL = 1e-5                      # float32 rate tensor vs float64 reference
_EDGES_6 = [(i, j) for i in range(6) for j in range(i + 1, 6)]   # all-to-all


_all_configs = all_configs


def _rebuild_and_compare(h, configs, name, H_ref=None):
    """Assemble H from get_rates over all configs; compare to the reference."""
    H_rebuilt = rebuild_matrix(h, configs, dtype=_DTYPE, name=name)
    if H_ref is None:
        H_ref = reference_dense(h)
    assert H_rebuilt.shape == H_ref.shape
    if not np.allclose(H_rebuilt, H_ref, atol=_ATOL):
        d = np.abs(H_rebuilt - H_ref)
        i, j = np.unravel_index(np.argmax(d), d.shape)
        raise AssertionError(
            f"{name}: get_rates matrix differs from the reference in "
            f"{int((d > _ATOL).sum())}/{d.size} entries; worst H[{i},{j}]: "
            f"{H_rebuilt[i, j]!r} vs {H_ref[i, j]!r}")
    return H_rebuilt


# ── the "all parameters on" models ──────────────────────────────────────────

def _model_spinhalf():
    """Non-uniform J_x, J_y, J_z per edge and h_z, h_x per site, sign-problem free."""
    rng = np.random.default_rng(0)
    n = len(_EDGES_6)
    J_x = -rng.uniform(0.6, 1.2, n)
    J_y = J_x + rng.uniform(0.05, 0.4, n)
    h = SpinHalf.create(_EDGES_6, J_x=J_x, J_y=J_y, J_z=rng.uniform(-1, 1, n),
                        h_z=-rng.uniform(-1, 1, 6), h_x=rng.uniform(0.1, 0.6, 6),
                        dtype=_DTYPE)
    assert np.all(np.asarray(h.J_pm) <= 0) and np.all(np.asarray(h.J_pmpm) <= 0)
    assert h.n_hx > 0 and h.n_jpm > 0
    return h, (1.0, -1.0)


def _model_spin_s1():
    """S=1 with J_z, J_pm, J_pmpm, h_z and the shared transverse h_pm active."""
    rng = np.random.default_rng(1)
    n = len(_EDGES_6)
    h = Spin.create(_EDGES_6, S=1.0, J_z=rng.uniform(-1, 1, n),
                    J_pm=-rng.uniform(0.2, 1.0, n), J_pmpm=-rng.uniform(0.1, 0.5, n),
                    h_z=-rng.uniform(-1, 1, 6), h_pm=rng.uniform(0.1, 0.5, 6),
                    dtype=_DTYPE)
    return h, (-1.0, 0.0, 1.0)


def _model_bose_hubbard():
    """Hopping t, pair term eta (breaks number conservation), U and mu."""
    rng = np.random.default_rng(2)
    n = len(_EDGES_6)
    h = BoseHubbard.create(_EDGES_6, t=rng.uniform(0.2, 1.0, n),
                           eta=rng.uniform(0.2, 1.0, n), U=1.3, mu=-0.4, n_max=2,
                           dtype=_DTYPE)
    return h, (0.0, 1.0, 2.0)


def _model_toric_code():
    """Stars, plaquettes and both fields on the 2x2 torus (N=8)."""
    h = ToricCode.create(2, 2, A_s=0.7, B_p=1.1, h_z=0.4, h_x=0.3, dtype=_DTYPE)
    assert h.M_total == 1 + h.n_stars + h.N
    return h, (1.0, -1.0)


def _model_plaquette():
    """Ring exchange g and RK term mu on all 15 four-site plaquettes of 6 sites."""
    rng = np.random.default_rng(3)
    plaqs = np.array(list(itertools.combinations(range(6), 4)))
    h = PlaquetteModel.create(plaqs, g=rng.uniform(0.3, 1.0, len(plaqs)),
                              mu=rng.uniform(-1, 1, len(plaqs)), dtype=_DTYPE)
    return h, (1.0, -1.0)


_MODELS = {"SpinHalf": _model_spinhalf, "Spin": _model_spin_s1,
           "BoseHubbard": _model_bose_hubbard, "ToricCode": _model_toric_code,
           "PlaquetteModel": _model_plaquette}


@pytest.mark.parametrize("name", list(_MODELS))
def test_get_rates(name):
    """The matrix rebuilt from get_rates equals the reference, over the full basis."""
    h, values = _MODELS[name]()
    _rebuild_and_compare(h, _all_configs(values, h.N), name)


@pytest.mark.parametrize("name", list(_MODELS))
def test_diagonal_weight_matches_get_rates(name):
    """The fast ``diagonal_weight(c)`` must equal ``get_rates(c)[0][0]``."""
    h, values = _MODELS[name]()
    rng = np.random.default_rng(1234)
    configs = jnp.asarray(rng.choice(np.asarray(values), size=(512, h.N)), dtype=_DTYPE)
    diag_fast = np.asarray(jax.vmap(h.diagonal_weight)(configs), dtype=np.float64)
    diag_rates = np.asarray(jax.vmap(lambda c: h.get_rates(c)[0][0])(configs),
                            dtype=np.float64)
    assert diag_fast.shape == diag_rates.shape == (configs.shape[0],)
    assert np.ptp(diag_rates) > 0, f"{name}: diagonal is constant, test is vacuous"
    bad = np.flatnonzero(np.abs(diag_fast - diag_rates) > _ATOL)
    assert bad.size == 0, (
        f"{name}: diagonal_weight disagrees with get_rates for {bad.size}/512 "
        f"configurations, e.g. {np.asarray(configs[bad[0]])}")


def test_toric_per_object_couplings():
    """Per-star/plaquette/site couplings: scalar == uniform array, non-uniform rebuilds."""
    Lx = Ly = 2
    n_v, N = Lx * Ly, 2 * Lx * Ly

    uni = ToricCode.create(Lx, Ly, A_s=0.7, B_p=1.1, h_z=0.4, h_x=0.3, dtype=_DTYPE)
    arr = ToricCode.create(Lx, Ly, A_s=np.full(n_v, 0.7), B_p=np.full(n_v, 1.1),
                           h_z=np.full(N, 0.4), h_x=np.full(N, 0.3), dtype=_DTYPE)
    assert uni.uniform_As and not arr.uniform_As
    assert uni.M_total == arr.M_total
    assert np.abs(reference_dense(uni) - reference_dense(arr)).max() < 1e-6

    rng = np.random.default_rng(11)
    h = ToricCode.create(Lx, Ly, A_s=np.array([1.0, 0.2, 1.0, 1.0]),
                         B_p=rng.uniform(0.5, 1.5, n_v), h_z=rng.uniform(-0.3, 0.3, N),
                         h_x=rng.uniform(0.0, 0.4, N), dtype=_DTYPE)
    M = _rebuild_and_compare(h, _all_configs((1.0, -1.0), h.N), "ToricCode/non-uniform")
    assert np.abs(M - M.T).max() < 1e-6

    with pytest.raises(ValueError, match="A_s must be"):
        ToricCode.create(Lx, Ly, A_s=np.array([1.0, -0.1, 1.0, 1.0]), dtype=_DTYPE)
    with pytest.raises(ValueError, match="h_x must be"):
        ToricCode.create(Lx, Ly, h_x=np.array([0.1] * (N - 1) + [-0.2]), dtype=_DTYPE)
    with pytest.raises(ValueError, match="shape"):
        ToricCode.create(Lx, Ly, B_p=np.ones(n_v + 1), dtype=_DTYPE)


_MOVE_MODELS = {
    "spin": lambda: Spin.create(np.asarray([[i, (i + 1) % 6] for i in range(6)]), S=1.0,
                                J_z=0.3, J_pm=-0.5, J_pmpm=-0.2, h_z=-0.1, h_pm=0.3, N=6),
    "bose_hubbard": lambda: BoseHubbard.create(np.asarray([[i, (i + 1) % 6] for i in range(6)]),
                                               t=1.0, eta=0.3, U=2.0, mu=0.5, h_pm=0.2,
                                               N=6, n_max=2),
}


@pytest.mark.parametrize("name", sorted(_MOVE_MODELS))
def test_move_reproduces_generate_state(name):
    """``move`` names the (at most two) sites of every transition and their
    increments, and adding them gives exactly ``generate_state``."""
    H = _MOVE_MODELS[name]()
    rng = np.random.default_rng(0)
    values = np.asarray(H.local_hilbert_space, dtype=np.float32)
    for _ in range(3):
        x = jnp.asarray(rng.choice(values, size=H.N), dtype=jnp.float32)
        for idx in range(H.M_total):
            sites, deltas = H.move(x, jnp.int32(idx))
            assert sites.shape == (2,) and deltas.shape == (2,)
            y = x.at[sites[0]].add(deltas[0]).at[sites[1]].add(deltas[1])
            np.testing.assert_array_equal(np.asarray(y), np.asarray(H.generate_state(x, jnp.int32(idx))))
