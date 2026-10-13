"""Input checks and small cases of the ``pygfmc.hamiltonian`` subpackage.

1. ``SpinHalf``, ``Spin`` and ``BoseHubbard`` reject malformed edges (wrong
   shape, negative, non-integer, non-finite or overflowing indices) and accept
   valid edge lists, including empty ones, unchanged.
2. ``SpinHalf.create`` raises for ``J_pm > 0`` (a propagator sign problem) and
   stays silent for sign-problem-free models and measurement operators.
3. Exact diagonalization works on bases with at most four states.
4. ``PlaquetteModel.create`` rejects odd-length plaquettes.
"""
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from core.reference import reference_dense
from pygfmc.hamiltonian import (PlaquetteModel, SpinHalf, StructureFactor, ToricCode,
                                Spin, BoseHubbard, chain, exact_diagonalization)


@pytest.mark.parametrize("model", [SpinHalf, Spin, BoseHubbard])
@pytest.mark.parametrize("edges", [[0, 1], [[0, 1, 2]], [[-1, 0]],
                                   [[0, 1.5]], [[0, np.nan]], [[0, np.inf]],
                                   np.empty((0, 3)), [[0, 2**63]]])
def test_models_reject_invalid_edges(model, edges):
    """Invalid connectivity must not silently change the Hamiltonian."""
    with pytest.raises(ValueError, match="edge"):
        model.create(edges, N=3)


@pytest.mark.parametrize("edges", [[], np.empty((0, 2)), [[0., 1.]],
                                   np.array([[0, 1]], dtype=np.uint64)])
def test_edge_validation_preserves_valid_inputs(edges):
    from pygfmc.hamiltonian._common import edges_array

    arr, count, n_min = edges_array(edges)
    assert arr.dtype == np.int64
    assert arr.shape == (len(edges), 2)
    assert count == len(edges)
    assert n_min == (2 if len(edges) else 0)


def test_spinhalf_positive_jpm_raises():
    """J_pm > 0 raises at construction, uniform and per-edge; sign_check=False builds it."""
    edges = chain(6).edges
    with pytest.raises(ValueError, match="sign_check=False"):
        SpinHalf.create(edges, J_x=1.0, J_y=1.0, J_z=0.5)
    J = -np.ones(len(edges))
    J[2] = 0.4
    with pytest.raises(ValueError, match="J_pm"):
        SpinHalf.create(edges, J_x=J, J_y=J, J_z=0.5)
    meas = SpinHalf.create(edges, J_x=J, J_y=J, J_z=0.5, sign_check=False)
    assert float(meas.J_pm[2]) == pytest.approx(0.2)


def test_spinhalf_sign_problem_free_models_stay_silent():
    """Sign-problem-free models and sign_check=False measurement operators do not warn."""
    edges = chain(6).edges
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SpinHalf.create(edges, J_x=-1.0, J_y=-0.5, J_z=0.5, h_x=0.2)
        SpinHalf.create(edges, J_x=2.0, J_y=2.0, J_z=0.0, sign_check=False)
        pos = np.random.default_rng(0).normal(size=(5, 3))
        for ch in ("SpSm", "SpSp", "SpSm+SpSp"):
            StructureFactor(pos, np.array([0.3, 0.1, 0.2]), channel=ch,
                            dtype=jnp.float32)


def test_exact_ground_state_energy_on_tiny_bases():
    """Exact diagonalization works on dim-2 and dim-4 bases."""
    h = SpinHalf.create([], h_x=1.0, h_z=-0.6, N=1, dtype=jnp.float64)
    H = reference_dense(h)
    e, _, _, psi = exact_diagonalization(h, return_states=True)
    assert e[0] == pytest.approx(np.linalg.eigvalsh(H)[0], abs=1e-12)
    assert np.allclose(H @ psi[:, 0], e[0] * psi[:, 0], atol=1e-10)
    for m in (PlaquetteModel.create([[0, 1]], g=1.0, mu=0.3, dtype=jnp.float64),
              ToricCode.create(1, 1, A_s=1.0, B_p=0.7, h_x=0.3, dtype=jnp.float64)):
        e, _, _ = exact_diagonalization(m)
        assert e[0] == pytest.approx(np.linalg.eigvalsh(reference_dense(m))[0], abs=1e-10)


def test_plaquette_rejects_odd_length():
    """Odd plaquette lengths raise; even ones are accepted."""
    with pytest.raises(ValueError, match="EVEN"):
        PlaquetteModel.create([[0, 1, 2]], g=1.0)
    with pytest.raises(ValueError, match="EVEN"):
        PlaquetteModel.create([[0, 1, 2, 3, 4]], g=0.0, mu=1.0)
    PlaquetteModel.create([[0, 1, 2, 3, 4, 5]], g=1.0)
