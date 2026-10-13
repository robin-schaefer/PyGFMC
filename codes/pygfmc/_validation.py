"""Argument validation shared by the engine and the Hamiltonian package."""
import numpy as np

__all__ = ["_validated_int"]


def _validated_int(name, value, minimum):
    """Validate an integer-valued public control parameter."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    try:
        result = int(value)
        is_integral = float(result) == float(value)
    except (TypeError, ValueError, OverflowError):
        is_integral = False
        result = minimum - 1
    if not is_integral or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    return result
