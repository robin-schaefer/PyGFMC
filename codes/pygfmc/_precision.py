"""The floating-point width the package computes in: float32, or float64 when
``jax_enable_x64`` is on (the walkers' weights follow the same switch)."""
import jax
import jax.numpy as jnp


def float_dtype():
    """``jnp.float64`` when ``jax_enable_x64`` is set, else ``jnp.float32``."""
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
