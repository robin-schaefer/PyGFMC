"""JastrowBoseHubbard: the fast ratio routine must agree with log_psi.

``log_diff_from_operator`` never builds the successor configurations, so it
is checked against ``log_psi`` evaluated on the successors that
``BoseHubbard.get_rates`` returns, for every block of the transition layout
(hopping, pair term, single-site creation/annihilation) and every parameter
form (tied, per site / per pair, absent).
"""
import itertools

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from pygfmc.hamiltonian import BoseHubbard, chain
from pygfmc.wavefunction import JastrowBoseHubbard
from core.guides import log_ratios_from_log_psi, require_netket

L = 9
EDGES = np.asarray(chain(L).edges)
# guide pairs differ from the operator edges, and (1, 2) appears twice
PAIRS = np.concatenate([EDGES, [[0, 4], [2, 7], [1, 2]]])
RNG = np.random.default_rng(3)

MODELS = {
    "hopping": dict(t=1.0, U=2.0, n_max=2),
    "hopping+pairs": dict(t=1.0, eta=0.25, U=4.0, mu=1.0, n_max=3),
    "pairs+h_pm": dict(eta=0.3, h_pm=-0.2, U=1.0, n_max=3, N=L),
    "all": dict(t=1.0, eta=0.2, h_pm=-0.1, U=3.0, mu=0.5, n_max=2, N=L),
}

FORMS = {
    "tied beta, h, d": dict(beta=0.21, h=0.13, d=0.37),
    "tied beta only": dict(beta=0.21),
    "per pair / per site": dict(beta=RNG.uniform(-0.2, 0.4, len(PAIRS)),
                                h=RNG.uniform(-0.3, 0.3, L), d=RNG.uniform(0.1, 0.6, L)),
    "tied beta, per site d, no h": dict(beta=0.15, d=RNG.uniform(0.1, 0.6, L)),
}


@pytest.mark.parametrize("name, form", list(itertools.product(MODELS, FORMS)))
def test_fast_ratio_matches_log_psi(name, form):
    H = BoseHubbard.create(EDGES, **MODELS[name])
    guide = JastrowBoseHubbard(L, PAIRS, **FORMS[form])
    configs = jnp.asarray(np.random.default_rng(0).integers(0, int(H.n_max) + 1, (100, L)),
                          dtype=jnp.float32)
    fast = jax.jit(jax.vmap(lambda c: guide.log_diff_from_operator(c, H)))(configs)
    ref = jax.vmap(lambda c: log_ratios_from_log_psi(guide, H, c))(configs)
    assert fast.shape == (100, H.M_total)
    np.testing.assert_allclose(fast, ref, atol=1e-5)


@pytest.mark.parametrize("name, form", list(itertools.product(MODELS, FORMS)))
def test_cache_tables_reproduce_the_inline_ratios(name, form):
    """``build_cache`` holds the walker-independent tables; the cached ratios equal
    the on-the-fly ones bit for bit, and the cache is memoised per operator."""
    H = BoseHubbard.create(EDGES, **MODELS[name])
    guide = JastrowBoseHubbard(L, PAIRS, **FORMS[form])
    cache = guide.build_cache(H)
    assert guide.build_cache(H) is cache
    configs = jnp.asarray(np.random.default_rng(2).integers(0, int(H.n_max) + 1, (40, L)),
                          dtype=jnp.float32)
    inline = jax.jit(jax.vmap(lambda c: guide.log_diff_from_operator(c, H)))(configs)
    cached = jax.jit(jax.vmap(lambda c: guide.log_diff_from_operator(c, H, cache)))(configs)
    np.testing.assert_array_equal(np.asarray(inline), np.asarray(cached))


@pytest.mark.parametrize("form", FORMS)
def test_incremental_state_follows_the_moves(form):
    """The incremental neighbor sums, updated through 200 random moves with the
    operator's ``move``, stay within float rounding of a rebuild, and the ratios
    computed from them match the plain ones."""
    H = BoseHubbard.create(EDGES, **MODELS["all"])
    guide = JastrowBoseHubbard(L, PAIRS, incremental=True, **FORMS[form])
    cache = guide.build_cache(H)
    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.integers(0, int(H.n_max) + 1, L), dtype=jnp.float32)
    state = guide.init_state(x, cache)
    for _ in range(200):
        idx = jnp.int32(rng.integers(0, H.M_total))
        sites, deltas = H.move(x, idx)
        assert np.array_equal(np.asarray(x.at[sites[0]].add(deltas[0]).at[sites[1]].add(deltas[1])),
                              np.asarray(H.generate_state(x, idx)))
        state = guide.update_state(state, x, sites, deltas, cache)
        x = H.generate_state(x, idx)
    np.testing.assert_allclose(state, guide.init_state(x, cache), atol=1e-5)
    np.testing.assert_allclose(guide.log_diff_from_operator(x, H, cache, state=state),
                               guide.log_diff_from_operator(x, H, cache), atol=1e-5)


def test_incremental_update_without_pairs_is_a_no_op():
    """A guide built with ``pairs=[]`` has no neighbor sums to update."""
    H = BoseHubbard.create(EDGES, **MODELS["all"])
    guide = JastrowBoseHubbard(L, [], beta=0.3, d=0.2, incremental=True)
    cache = guide.build_cache(H)
    x = jnp.asarray(np.random.default_rng(1).integers(0, int(H.n_max) + 1, L), dtype=jnp.float32)
    state = guide.init_state(x, cache)
    assert state.shape == (L,) and not np.any(np.asarray(state))
    sites, deltas = H.move(x, jnp.int32(1))
    np.testing.assert_array_equal(np.asarray(guide.update_state(state, x, sites, deltas, cache)),
                                  np.asarray(state))


def test_wrong_coupling_is_detected():
    """Non-vacuity: flipping the sign of beta must break the agreement."""
    H = BoseHubbard.create(EDGES, **MODELS["all"])
    good = JastrowBoseHubbard(L, PAIRS, beta=0.21, h=0.13, d=0.37)
    bad = JastrowBoseHubbard(L, PAIRS, beta=-0.21, h=0.13, d=0.37)
    config = jnp.asarray([2, 0, 1, 1, 2, 0, 1, 2, 1], dtype=jnp.float32)
    assert jnp.max(jnp.abs(bad.log_diff_from_operator(config, H)
                           - log_ratios_from_log_psi(good, H, config))) > 0.1


def test_parameters_keep_the_form_they_were_given():
    def float_leaves(g):
        return [leaf.shape for leaf in jax.tree_util.tree_leaves(g)
                if jnp.issubdtype(leaf.dtype, jnp.floating)]
    assert float_leaves(JastrowBoseHubbard(L, PAIRS, beta=0.5)) == [()]
    assert float_leaves(JastrowBoseHubbard(L, PAIRS, beta=0.1, h=0.2, d=0.5)) == [(), (), ()]
    assert float_leaves(JastrowBoseHubbard(L, PAIRS, **FORMS["per pair / per site"])) == \
        [(len(PAIRS),), (L,), (L,)]
    guide = JastrowBoseHubbard(L, PAIRS, beta=0.1, d=0.5)
    assert guide._h is None and np.all(np.asarray(guide.h) == 0.0)
    assert np.allclose(guide.beta, 0.1) and guide.beta.shape == (len(PAIRS),)


def test_bad_shapes_raise():
    with pytest.raises(ValueError):
        JastrowBoseHubbard(L, PAIRS, beta=0.1, d=np.ones(L + 1))
    with pytest.raises(ValueError):
        JastrowBoseHubbard(L, PAIRS, beta=np.ones(L))
    with pytest.raises(ValueError):
        JastrowBoseHubbard(L, PAIRS, beta=None)


@pytest.mark.parametrize("form", FORMS)
def test_save_load_and_pytree(tmp_path, form):
    guide = JastrowBoseHubbard(L, PAIRS, **FORMS[form])
    configs = jnp.asarray(np.random.default_rng(1).integers(0, 3, (20, L)), dtype=jnp.float32)
    path = tmp_path / "guide.h5"
    guide.save(path)
    loaded = JastrowBoseHubbard.load(path)
    np.testing.assert_array_equal(loaded.log_psi(configs), guide.log_psi(configs))
    assert jax.tree_util.tree_structure(loaded) == jax.tree_util.tree_structure(guide)
    leaves, tree = jax.tree_util.tree_flatten(guide)
    rebuilt = jax.tree_util.tree_unflatten(tree, leaves)
    np.testing.assert_array_equal(rebuilt.log_psi(configs), guide.log_psi(configs))


def test_train_with_netket_keeps_the_form():
    """Training lowers the variational energy and keeps the parameter form.

    A local space of three occupations needs an explicit ergodic sampler (site
    swaps keep the occupation histogram), so the test passes NetKet's
    single-site ``MetropolisLocal`` on the unconstrained occupation space.
    """
    require_netket()
    import netket as nk
    from pygfmc.wavefunction import make_hilbert, train_with_netket
    n = 6
    H = BoseHubbard.create(chain(n).edges, t=1.0, U=4.0, mu=1.0, n_max=2)
    sampler = nk.sampler.MetropolisLocal(make_hilbert(H.local_hilbert_space, n))
    guide = JastrowBoseHubbard(n, chain(n).edges, beta=0.0, d=0.0)
    trained, history = train_with_netket(H, guide, sampler=sampler, n_iter=30,
                                         n_samples=512, seed=1, return_history=True)
    assert np.all(np.isfinite(history))
    assert jax.tree_util.tree_structure(trained) == jax.tree_util.tree_structure(guide)
    assert trained._h is None and trained._beta.shape == () and trained._d.shape == ()
    assert float(trained._d) > 0.05                 # learns to suppress double occupancy
    assert np.mean(history[-5:]) < history[0]
