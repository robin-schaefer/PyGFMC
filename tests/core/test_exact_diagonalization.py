"""Generic ``exact_diagonalization`` against the independent reference matrices.

The routine needs nothing but the rate protocol, so it is checked against the
Kronecker constructions in ``reference.py``, which share no code with the rate
path.  PlaquetteModel is included because its basis convention differs from
the other classes.
"""
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.reference import reference_dense, reference_ground_state, reference_matrix
from pygfmc.hamiltonian import (BoseHubbard, PlaquetteModel, Spin, SpinHalf, ToricCode,
                                chain, exact_diagonalization)

_ATOL = 1e-6          # float32 couplings bound the agreement

# (id, builder, local_hilbert_space, N) -- the models test_exact_ground_state uses
_MODELS = [
    ("spinhalf",
     lambda: SpinHalf.create(chain(4).edges, -1.0, -1.0, 1.0, h_x=0.3),
     (-1, 1), 4),
    ("spin_S1",
     lambda: Spin.create(chain(3).edges, S=1.0, J_pm=-1.0, J_z=1.0, h_pm=-0.2),
     (-1, 0, 1), 3),
    ("bose_hubbard",
     lambda: BoseHubbard.create(chain(3).edges, t=1.0, U=2.0, mu=0.5,
                                h_pm=-0.1, eta=0.2, n_max=2),
     (0, 1, 2), 3),
    ("toric_code", lambda: ToricCode.create(2, 2, A_s=1.0, B_p=0.5), (-1, 1), 8),
    ("plaquette",
     lambda: PlaquetteModel.create(jnp.array([[0, 1, 2, 3]], dtype=jnp.int32),
                                   g=1.0, mu=0.0),
     (-1, 1), 4),
]
_IDS = [m[0] for m in _MODELS]


def _dense_spectrum(h, k):
    return np.sort(np.linalg.eigvalsh(reference_dense(h)))[:k]


@pytest.mark.parametrize("name,build,lhs,N", _MODELS, ids=_IDS)
def test_ground_state_matches_the_per_class_construction(name, build, lhs, N):
    """E0 from the rate protocol equals E0 from the reference matrix."""
    h = build()
    energies, diag, off = exact_diagonalization(h, lhs, N, number_of_states=1)
    assert energies.shape == (1,)
    assert diag == {} and off == {}
    assert energies[0] == pytest.approx(reference_ground_state(h)[0], rel=0, abs=_ATOL)


@pytest.mark.parametrize("name,build,lhs,N", _MODELS, ids=_IDS)
def test_low_lying_spectrum_is_sorted_and_matches_dense(name, build, lhs, N):
    """The requested states come back sorted and match the dense reference."""
    h = build()
    energies, _, _ = exact_diagonalization(h, lhs, N, number_of_states=3)
    assert energies.shape == (3,)
    assert np.all(np.diff(energies) >= -1e-9), "eigenvalues must be sorted"
    np.testing.assert_allclose(energies, _dense_spectrum(h, 3), atol=_ATOL)


def test_observables_match_a_direct_evaluation():
    """Diagonal and off-diagonal observables against a dense computation.

    ``J_z=1`` builds ``J_z S^z S^z``, i.e. ``sigma_i sigma_j / 4``.
    """
    edges = chain(4).edges
    h = SpinHalf.create(edges, -1.0, -1.0, 1.0, h_x=0.3)
    zz = SpinHalf.create([edges[0]], J_x=0.0, J_y=0.0, J_z=1.0, sign_check=False)
    sx = SpinHalf.create(edges, J_x=0.0, J_y=0.0, J_z=0.0, h_x=-2.0, sign_check=False)

    energies, diag, off = exact_diagonalization(
        h, (-1, 1), 4, diag_operators={"zz01": zz},
        off_diag_operators={"sx_tot": sx}, number_of_states=2)
    assert set(diag) == {"zz01"} and set(off) == {"sx_tot"}
    assert np.shape(diag["zz01"]) == (2,) and np.shape(off["sx_tot"]) == (2,)

    E0, psi = reference_ground_state(h)
    assert energies[0] == pytest.approx(E0, rel=0, abs=_ATOL)
    # reference basis: site 0 most significant, digit 0 -> sigma = +1
    configs = np.array(list(itertools.product([1.0, -1.0], repeat=4)))
    i, j = int(edges[0][0]), int(edges[0][1])
    zz_direct = float((psi ** 2) @ (configs[:, i] * configs[:, j] / 4.0))
    sx_direct = float(psi @ (reference_matrix(sx) @ psi))
    assert diag["zz01"][0] == pytest.approx(zz_direct, rel=0, abs=_ATOL)
    assert off["sx_tot"][0] == pytest.approx(sx_direct, rel=0, abs=1e-5)


def test_runs_in_float64_and_restores_the_caller_precision():
    """Double precision inside, and ``jax_enable_x64`` restored even on a raise."""
    h = SpinHalf.create(chain(4).edges, -1.0, -1.0, 1.0, h_x=0.3)
    before = jax.config.jax_enable_x64
    energies, _, _ = exact_diagonalization(h, (-1, 1), 4)
    assert jax.config.jax_enable_x64 == before
    assert energies.dtype == np.float64
    with pytest.raises(ValueError):
        exact_diagonalization(h, (-1, 1), 4, number_of_states=999)
    assert jax.config.jax_enable_x64 == before


def test_input_validation():
    """Bad local spaces, sizes, state counts and contradicting N/d raise."""
    h = SpinHalf.create(chain(4).edges, -1.0, -1.0, 1.0)
    with pytest.raises(ValueError, match="at least two values"):
        exact_diagonalization(h, (1,), 4)
    with pytest.raises(ValueError, match="exceeds the"):
        exact_diagonalization(h, (-1, 1), 40)
    with pytest.raises(ValueError, match="number_of_states"):
        exact_diagonalization(h, (-1, 1), 4, number_of_states=0)
    with pytest.raises(ValueError, match="number_of_states"):
        exact_diagonalization(h, (-1, 1), 4, number_of_states=17)   # dim = 16
    with pytest.raises(ValueError, match="Hamiltonian reports N"):
        exact_diagonalization(h, (-1, 1), 5)
    spin1 = Spin.create(chain(3).edges, S=1.0, J_pm=-1.0, J_z=1.0)
    with pytest.raises(ValueError, match="local dimension"):
        exact_diagonalization(spin1, (-1, 1), 3)
