"""Inverse-CDF sampling of a move: only slots with positive weight can be drawn.

A zero-weight slot is a move with zero amplitude.  For Bose-Hubbard with a
pair term such a slot leads outside the physical basis (occupation -1), and
the run fails with NaN a few hundred steps in; for spin models the wrong move
is taken silently.
"""
import numpy as np
import jax
import jax.numpy as jnp

from pygfmc.engine.evolution import sample_index


def _draw(cdf, weights, n=4096):
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    return np.asarray(jax.vmap(lambda k: sample_index(cdf, weights, k))(keys))


def test_zero_weight_slot_is_never_drawn():
    """A cumsum evaluated as a parallel scan can rise by an ulp across a
    zero-weight slot; the noise is exaggerated here to a 0.5 sliver so that a
    plain searchsorted lands in it a quarter of the time."""
    weights = jnp.array([1.0, 0.0, 1.0], dtype=jnp.float32)
    noisy_cdf = jnp.array([1.0, 1.5, 2.0], dtype=jnp.float32)
    idx = _draw(noisy_cdf, weights)
    assert set(np.unique(idx)) <= {0, 2}
    # the sliver was really drawn (a plain searchsorted would return slot 1 there)
    assert np.mean(idx == 0) > 0.6


def test_index_stays_inside_the_array():
    """Trailing zero-weight slots must not be reachable either, including
    through the clamp of an out-of-range searchsorted result."""
    weights = jnp.array([0.0, 2.0, 0.0, 0.0], dtype=jnp.float32)
    cdf = jnp.array([0.0, 2.0, 2.0000002, 2.0000005], dtype=jnp.float32)
    assert set(np.unique(_draw(cdf, weights))) == {1}


def test_exact_cdf_draws_proportionally():
    weights = jnp.array([1.0, 0.0, 3.0], dtype=jnp.float32)
    idx = _draw(jnp.cumsum(weights), weights, n=20000)
    assert set(np.unique(idx)) <= {0, 2}
    assert abs(np.mean(idx == 2) - 0.75) < 0.02
