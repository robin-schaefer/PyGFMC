"""The SpinHalf transition layout that the spin-1/2 guides index by.

``JastrowSpinHalf``, ``ClusterCPS`` and ``ClusterMLP`` lay their log-amplitude
ratios out as ``[diagonal, n_hx_static single-site flips, n_jpm_static edge
flips on (edge_i, edge_j)]``, the layout of ``SpinHalf.get_rates``.  Any
operator exposing those four members is accepted (``SpinHalf`` itself, a
``StructureFactor`` operator, or a user class with the same layout); any other
operator is rejected with a ``TypeError`` naming the requirement.
"""

LAYOUT_ATTRS = ("n_hx_static", "n_jpm_static", "edge_i", "edge_j")


def require_spin_half_layout(op, guide_name):
    """Raise ``TypeError`` unless ``op`` exposes the SpinHalf transition layout."""
    if not all(hasattr(op, name) for name in LAYOUT_ATTRS):
        raise TypeError(
            f"{guide_name} needs an operator with the SpinHalf transition "
            f"layout ({', '.join(LAYOUT_ATTRS)}); got {type(op).__name__}")
