"""The spin-1/2 guides and the model ``test_guides.py`` and ``test_guided_methods.py`` share.

A sign-problem-free ferromagnetic XXZ ring of ``L = 6`` sites with a
transverse field (64 states, so exact diagonalization is instant) and one
deterministic instance of each of the four documented spin-1/2 guides; plus
``log_ratios_from_log_psi``, the reference for a guide's fast ratios used by
the Jastrow tests, and ``require_netket``, the skip shared by the NetKet tests.
"""
import numpy as np
import pytest

from pygfmc.hamiltonian import SpinHalf, chain
from pygfmc.wavefunction import ClusterCPS, ClusterMLP, ExactGroundState, JastrowSpinHalf

L = 6
EDGES = chain(L, periodic=True).edges
CLUSTERS = np.asarray([[i, (i + 1) % L] for i in range(L)], dtype=np.int32)
GUIDE_NAMES = ["Jastrow", "ClusterCPS", "ClusterMLP", "ExactGroundState"]


def log_ratios_from_log_psi(guide, H, config):
    """``log psi_T(x'_k) - log psi_T(x)`` for every term of ``H``, from
    ``log_psi`` on the materialized successors: the reference for a guide's
    fast ``log_diff_from_operator``."""
    _, new_configs = H.get_rates(config, return_configs=True)
    return guide.log_psi(new_configs) - guide.log_psi(config)


def xxz_ring():
    """``J_x = J_y = J_z = -1`` on the ring, with ``h_x = 0.4``."""
    return SpinHalf.create(EDGES, J_x=-1.0, J_y=-1.0, J_z=-1.0, h_x=0.4)


def build_guide(name, h):
    """A fresh, deterministic guide; ``ExactGroundState`` diagonalizes ``h``."""
    if name == "Jastrow":
        return JastrowSpinHalf(L, EDGES, beta=0.2)
    if name == "ClusterCPS":
        return ClusterCPS([[CLUSTERS, None, True]], init="gaussian", seed=0)
    if name == "ClusterMLP":
        return ClusterMLP([[CLUSTERS, [4]]], seed=0)
    if name == "ExactGroundState":
        return ExactGroundState(h)
    raise KeyError(name)


def require_netket():
    """Skip when NetKet is absent or fails on import.

    ``pytest.importorskip`` skips only on ``ImportError``; a NetKet release that
    raises something else on import (e.g. a ``TypeError`` on an unsupported
    Python version) would fail the test instead.  ``import_netket`` reports both
    cases as ``ImportError``.
    """
    from pygfmc.wavefunction._optional import import_netket
    try:
        import_netket()
    except ImportError as e:
        pytest.skip(str(e))
