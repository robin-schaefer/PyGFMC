"""Guiding wavefunctions (trial states) for PyGFMC.

Every guide implements at least one of ``log_diff_from_operator``,
``log_diff_from_configs`` or ``log_psi`` (see ``README.md``).  Training with
NetKet is optional (``pip install -e "./codes[netket]"``); importing this package
never imports NetKet, optax or flax.
"""

from .jastrow_spin_half import JastrowSpinHalf
from .exact_ground_state import ExactGroundState
from .cluster_mlp import ClusterMLP
from .cluster_cps import ClusterCPS
from .jastrow_bose_hubbard import JastrowBoseHubbard
from .jastrow_spin import JastrowSpin
from .generic_netket_training import train_with_netket, make_hilbert

__all__ = [
    "JastrowSpinHalf", "JastrowSpin", "JastrowBoseHubbard", "ClusterCPS", "ClusterMLP",
    "ExactGroundState", "train_with_netket", "make_hilbert",
]
