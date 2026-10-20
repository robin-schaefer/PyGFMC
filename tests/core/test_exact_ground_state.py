"""Tests for the ExactGroundState guiding wavefunction.

Covers the guide's basis-index bijection, ``log_diff_from_configs`` against a
brute-force reference built from the independent reference matrix, the
zero-variance / zero-bias properties under ``run_gfmc``, pytree and HDF5
round-trips, and PlaquetteModel.
"""
import itertools

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.reference import reference_basis_index, reference_ground_state, reference_matrix
from pygfmc import run_gfmc
from pygfmc.engine.evolution import transition_rates
from pygfmc.walkers import prepare_walkers_sz
from pygfmc.hamiltonian import (BoseHubbard, PlaquetteModel, Spin, SpinHalf, ToricCode, chain,
                                exact_diagonalization)
from pygfmc.wavefunction import ExactGroundState

_MODELS = {
    "spinhalf": (lambda: SpinHalf.create(chain(4).edges, -1.0, -1.0, 1.0, h_x=0.3),
                 (-1.0, 1.0)),
    "spin_S1": (lambda: Spin.create(chain(3).edges, S=1.0, J_pm=-1.0, J_z=1.0,
                                    h_pm=-0.2),
                (-1.0, 0.0, 1.0)),
    "bose_hubbard": (lambda: BoseHubbard.create(chain(3).edges, t=1.0, U=2.0, mu=0.5,
                                                h_pm=-0.1, eta=0.2, n_max=2),
                     (0.0, 1.0, 2.0)),
    "toric": (lambda: ToricCode.create(2, 2, A_s=1.0, B_p=0.5), (-1.0, 1.0)),
}


@pytest.fixture
def spinhalf4():
    return SpinHalf.create(chain(4).edges, -1.0, -1.0, 1.0)


@pytest.mark.parametrize("name", list(_MODELS))
def test_bijection(name):
    """The guide's basis index maps the full configuration space onto 0..dim-1."""
    build, values = _MODELS[name]
    h = build()
    guide = ExactGroundState(h)
    cfgs = jnp.asarray(list(itertools.product(values, repeat=h.N)), dtype=jnp.float32)
    idxs = jax.vmap(guide.basis_index)(cfgs)
    assert sorted(np.asarray(idxs).tolist()) == list(range(len(values) ** h.N))


def _brute_log_diff(h, config):
    """log|psi0[new]| - log|psi0[old]| per transition, from the reference matrix.

    Out-of-basis successors (zero-rate forbidden moves) are clipped; the caller
    compares only transitions that carry rate.
    """
    _, psi = reference_ground_state(h)
    log_amp = np.log(np.maximum(np.abs(psi), 1e-30))
    dim = log_amp.shape[0]
    idx_old = int(reference_basis_index(h, np.asarray(config)))
    out = np.zeros(h.M_total)
    for k in range(1, h.M_total):
        idx_new = int(reference_basis_index(h, np.asarray(h.generate_state(config, k))))
        out[k] = log_amp[min(max(idx_new, 0), dim - 1)] - log_amp[idx_old]
    return out


@pytest.mark.parametrize("name,seed", [("spinhalf", 0), ("spin_S1", 1), ("bose_hubbard", 2)])
def test_log_diff(name, seed):
    """log_diff_from_configs matches the brute-force ratio on rate-carrying moves."""
    build, values = _MODELS[name]
    h = build()
    guide = ExactGroundState(h)
    rng = np.random.default_rng(seed)
    for _ in range(5):
        cfg = jnp.asarray(rng.choice(values, size=h.N), dtype=jnp.float32)
        keep = np.asarray(h.get_rates(cfg)[0]) != 0.0
        keep[0] = False
        np.testing.assert_allclose(np.asarray(guide.log_diff_from_configs(cfg))[keep],
                                   _brute_log_diff(h, cfg)[keep], atol=1e-5)


@pytest.mark.parametrize("name", ["spinhalf", "spinhalf_sector", "spin_S1", "bose_hubbard"])
def test_exact_guide_local_energy_on_every_supported_configuration(name):
    """The actual guided rates give E0 on the entire nonzero support."""
    build, values = ((lambda: SpinHalf.create(chain(4).edges, -1.0, -1.0, 1.0),
                      (-1.0, 1.0)) if name == "spinhalf_sector" else _MODELS[name])
    h = build()
    guide = ExactGroundState(h)
    e0, psi = reference_ground_state(h)
    configs = np.asarray(list(itertools.product(values, repeat=h.N)))
    indices = [reference_basis_index(h, c) for c in configs]
    support = np.abs(psi[indices]) > 1e-10
    assert support.any()
    if name == "spinhalf_sector":
        assert not support.all(), "exercise exact wavefunctions with nodes too"
    energies = jax.vmap(lambda c: transition_rates(c, h, guide)[0].sum())(
        jnp.asarray(configs[support], dtype=jnp.float32))
    np.testing.assert_allclose(energies, e0, rtol=0, atol=2e-5)
    assert float(np.std(energies)) < 1e-5


def test_zero_variance_integration():
    """With psi_T = psi_0 the GFMC energy collapses onto E_0 within float32 noise."""
    lat = chain(6, periodic=True)
    h = SpinHalf.create(lat.edges, -1.0, -1.0, 1.0)
    guide = ExactGroundState(h)
    walkers = prepare_walkers_sz(128, lat.N, jax.random.PRNGKey(0), sz=0)
    _, results, _, _, _ = run_gfmc(
        walkers, n_steps=300, hamiltonian=h, lam=2.5, step_size=8, guide=guide,
        p_array=[8, 16], method="single_step", rng_key=jax.random.PRNGKey(1),
        verbosity=0, warmup_steps=100)
    assert set(results) == {8, 16}
    for p, energy in results.items():
        err_per_site = abs(energy - guide.E0) / lat.N
        assert err_per_site < 1e-4, f"p={p}: |dE|/N = {err_per_site:.2e}"


def test_off_diag_observable_is_exact_at_every_p():
    """With psi_T = psi_0 the forward-walked estimate is unbiased and flat in p.

    A wrong window offset or mis-normalized shadow chain would drift with p.
    """
    lat = chain(6, periodic=True)
    h = SpinHalf.create(lat.edges, -1.0, -1.0, 1.0)
    guide = ExactGroundState(h)

    n_e = len(lat.edges)
    one, zero = np.zeros(n_e), np.zeros(n_e)
    one[0] = 1.0                      # <S+S- + h.c.> on bond 0 only
    spm = SpinHalf.create(list(lat.edges), J_x=2 * one, J_y=2 * one, J_z=zero,
                          sign_check=False)
    _, psi0 = reference_ground_state(h)
    exact = float(psi0 @ (reference_matrix(spm) @ psi0))

    walkers = prepare_walkers_sz(2048, lat.N, jax.random.PRNGKey(0), sz=0)
    _, _, obs, _, _ = run_gfmc(
        walkers, n_steps=1500, hamiltonian=h, lam=float(n_e) / 4 + 1.0, step_size=1,
        guide=guide, p_array=[2, 4, 8, 16], method="single_step",
        rng_key=jax.random.PRNGKey(1), off_diag_operators={"spm_bond": spm},
        verbosity=0, warmup_steps=100)

    got = {p: float(np.ravel(v)[0]) for p, v in obs["spm_bond"].items()}
    for p, val in got.items():
        assert abs(val - exact) < 1e-2, f"p={p}: {val:.6f} vs exact {exact:.6f}"
    assert max(got.values()) - min(got.values()) < 5e-3, f"drifts with p: {got}"


def test_hdf5_roundtrip(spinhalf4, tmp_path):
    """save/load reproduces the amplitudes and writes the metadata attributes."""
    g = ExactGroundState(spinhalf4)
    path = str(tmp_path / "exact_gs.h5")
    g.save(path, note="test_run")
    g2 = ExactGroundState.load(path, spinhalf4)
    np.testing.assert_array_equal(np.asarray(g.log_amp), np.asarray(g2.log_amp))
    cfg = jnp.array([1., -1., 1., -1.], dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(g.log_diff_from_configs(cfg)),
                               np.asarray(g2.log_diff_from_configs(cfg)), atol=0)
    with h5py.File(path, "r") as f:
        attrs = dict(f["exact_ground_state"].attrs)
    assert attrs["family"] == "SpinHalf" and attrs["N"] == 4
    assert attrs["note"] == "test_run"


def test_hdf5_load_rejects_wrong_family(spinhalf4, tmp_path):
    """Loading into a different Hamiltonian family raises."""
    path = str(tmp_path / "exact_gs.h5")
    ExactGroundState(spinhalf4).save(path)
    h_other = Spin.create(chain(2).edges, S=1.0, J_pm=-1.0, J_z=1.0)
    with pytest.raises(ValueError, match="family"):
        ExactGroundState.load(path, h_other)


def test_plaquette_works_without_config_to_basis_index():
    """The index comes from local_hilbert_space, so PlaquetteModel is supported."""
    h = PlaquetteModel.create(jnp.array([[0, 1, 2, 3]], dtype=jnp.int32), g=1.0, mu=0.0)
    assert not hasattr(h, "config_to_basis_index")
    guide = ExactGroundState(h)
    assert guide.local_hilbert_space == h.local_hilbert_space
    assert guide.log_amp.shape == (2 ** 4,)
    assert guide.E0 == pytest.approx(reference_ground_state(h)[0], rel=0, abs=1e-6)


# ── input validation of the reference solver ─────────────────────────────

class _TwoSiteOperator:
    """Site 0 of a two-site spin-1/2 chain is flipped by one term; the
    amplitudes are chosen per test (a successor outside the basis, or an
    asymmetric pair of matrix elements)."""
    M_total = 2
    N = 2
    local_hilbert_space = (-1.0, 1.0)

    def __init__(self, amplitude_up, amplitude_down, successor_value=None):
        self.amplitude_up, self.amplitude_down = amplitude_up, amplitude_down
        self.successor_value = successor_value

    def get_rates(self, config, return_configs=False):
        amp = jnp.where(config[0] > 0, self.amplitude_down, self.amplitude_up)
        rates = jnp.stack([jnp.zeros(()), jnp.asarray(amp, jnp.float32)])
        new = jnp.stack([config, self.generate_state(config, 1)]) if return_configs else None
        return rates, new

    def generate_state(self, config, idx):
        target = -config[0] if self.successor_value is None else self.successor_value
        return jnp.where(idx == 0, config, config.at[0].set(target))


def test_basis_values_must_be_the_hamiltonians_own():
    """A reordered basis is the same basis; different values are not."""
    from pygfmc.hamiltonian import SpinHalf
    H = SpinHalf.create(np.zeros((0, 2), dtype=np.int64), h_z=1.0, N=2)
    e_ref = exact_diagonalization(H)[0][0]
    assert exact_diagonalization(H, local_hilbert_space=(1.0, -1.0))[0][0] == e_ref
    with pytest.raises(ValueError, match="differs from the values"):
        exact_diagonalization(H, local_hilbert_space=(0.5, -0.5))
    with pytest.raises(ValueError, match="distinct finite"):
        exact_diagonalization(H, local_hilbert_space=(1.0, 1.0))
    with pytest.raises(ValueError, match="distinct finite"):
        exact_diagonalization(H, local_hilbert_space=(1.0, np.nan))


def test_nonzero_transition_leaving_the_basis_is_an_error():
    op = _TwoSiteOperator(-1.0, -1.0, successor_value=2.0)
    with pytest.raises(ValueError, match="leads outside the supplied local_hilbert_space"):
        exact_diagonalization(op)


def test_non_hermitian_operator_is_rejected():
    with pytest.raises(ValueError, match="not Hermitian"):
        exact_diagonalization(_TwoSiteOperator(-1.0, -2.0))
    e = exact_diagonalization(_TwoSiteOperator(-1.0, -1.0))[0]
    np.testing.assert_allclose(e[0], -1.0, atol=1e-6)


def test_allocation_guard_refuses_a_huge_dense_solve():
    """A full-spectrum request on 2**20 states would need a 16 TiB dense
    matrix; the estimate stops it before anything is built."""
    from pygfmc.hamiltonian import SpinHalf, chain
    H = SpinHalf.create(chain(20).edges, J_z=1.0)
    with pytest.raises(ValueError, match="would allocate about"):
        exact_diagonalization(H, number_of_states=2 ** 20)
    with pytest.raises(ValueError, match="number_of_states"):
        exact_diagonalization(H, number_of_states=0)
