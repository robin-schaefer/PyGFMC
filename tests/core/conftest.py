"""Fixtures shared by ``test_guides.py`` and ``test_guided_methods.py``."""
import pytest

from core.guides import xxz_ring
from pygfmc.hamiltonian import exact_diagonalization


@pytest.fixture(scope="module")
def h():
    """Sign-problem-free ferromagnetic XXZ ring with a transverse field."""
    return xxz_ring()


@pytest.fixture(scope="module")
def e0(h):
    """Exact ground-state energy from exact diagonalization."""
    return float(exact_diagonalization(h)[0][0])
