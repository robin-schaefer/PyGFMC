"""Systematic resampling (reconfiguration) of the walker population."""
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from .walker import Walker


def systematic_parents(w, psi):
    """Parent indices of systematic resampling with normalised weights ``w``
    (``sum(w) == 1`` up to rounding) and offset ``psi`` in ``[0, 1)``.

    Parent ``k`` is drawn for every ``u = (i + psi) / N`` with
    ``cdf[k-1] <= u < cdf[k]`` (``side="right"``), so a walker of zero weight,
    across which the cdf is flat, is never chosen, including at ``u == 0``
    for a leading zero weight.  Rounding can leave ``cdf[-1]`` below the
    largest ``u``, so the result is capped at the last walker of positive
    weight rather than at ``N - 1``, which may carry none.
    """
    N = w.shape[0]
    u = (jnp.arange(N, dtype=w.dtype) + psi) / N
    cdf = jnp.cumsum(w)
    last_positive = jnp.max(jnp.where(w > 0, jnp.arange(N), 0))
    return jnp.minimum(jnp.searchsorted(cdf, u, side="right"), last_positive)


def reconfiguration(walkers: Walker, rng_key: jax.Array
                    ) -> tuple[Walker, jax.Array, jax.Array, jax.Array]:
    """Systematic resampling of the walker population.

    Resamples walkers proportional to their weights using the systematic
    scheme (one shared random offset).  After resampling all weights are
    reset to one (log-weight = 0).

    Args:
        walkers:        Walker batch to resample.
        rng_key:        JAX PRNGKey.

    Returns:
        ``(walkers, log_G_n, E_n, indices)``.  ``log_G_n`` is the logarithm
        of the mean walker weight ``G_n = (1/N) Σ_i w_i`` (the step's branch
        weight, kept in log space because its product over a window spans
        hundreds of orders of magnitude), ``E_n`` the weight-averaged local
        energy ``Σ_i w_i e_L(x_i) / Σ_i w_i``, and ``indices[i]`` the parent
        of new walker ``i`` (the parent map is needed to gather the new
        configurations, so it is always available and always returned).
    """
    N_walkers = walkers.config.shape[0]
    log_w   = walkers.log_weight
    log_G_n = logsumexp(log_w)
    w   = jnp.exp(log_w - log_G_n)
    E_n = (walkers.e_loc * w).sum()
    psi = jax.random.uniform(rng_key)
    indices = systematic_parents(w, psi)
    new_walkers = Walker(
        config=walkers.config[indices],
        log_weight=jnp.zeros(N_walkers, dtype=walkers.log_weight.dtype),
        e_loc=walkers.e_loc[indices],
    )
    log_avg_weight = log_G_n - jnp.log(jnp.asarray(
        N_walkers, dtype=walkers.log_weight.dtype))
    return new_walkers, log_avg_weight, E_n, indices
