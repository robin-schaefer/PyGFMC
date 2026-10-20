"""JastrowSpin: the guide for spin-S models.

The fast ratio routine never builds the successor configurations, so it is
checked against ``log_psi`` evaluated on the successors that ``Spin.get_rates``
returns, for every block of the transition layout (two-site S+S-, S+S+ and the
single-site S+, S-), for spin 1 and spin 3/2 and for every parameter form.
"""
import itertools

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from pygfmc import run
from pygfmc.hamiltonian import Spin, chain, exact_diagonalization
from pygfmc.walkers import prepare_walkers_from_configs
from pygfmc.wavefunction import JastrowSpin
from core.guides import log_ratios_from_log_psi, require_netket

L = 7
EDGES = np.asarray(chain(L).edges)
# guide pairs differ from the operator edges, and (1, 2) appears twice
PAIRS = np.concatenate([EDGES, [[0, 3], [2, 5], [1, 2]]])
RNG = np.random.default_rng(0)

FORMS = {
    "tied, no h, no d": dict(beta=0.3),
    "tied, tied h, tied d": dict(beta=-0.25, h=0.15, d=0.4),
    "per pair, per site": dict(beta=RNG.uniform(-0.5, 0.5, len(PAIRS)),
                               h=RNG.uniform(-0.3, 0.3, L), d=RNG.uniform(-0.3, 0.6, L)),
}


def _build_spin_model(S, seed):
    rng = np.random.default_rng(seed)
    n_e = len(EDGES)
    return Spin.create(EDGES, S=S, J_z=rng.uniform(-1, 1, n_e), J_pm=-rng.uniform(0.2, 1, n_e),
                       J_pmpm=-rng.uniform(0.1, 0.5, n_e), h_z=-rng.uniform(-0.5, 0.5, L),
                       h_pm=rng.uniform(0.1, 0.6, L), N=L)


# built once, each from its own seed, so no test depends on which ran before it
SPIN_MODELS = {1.0: _build_spin_model(1.0, 10), 1.5: _build_spin_model(1.5, 15)}


def _spin_model(S):
    return SPIN_MODELS[S]


def _configs(S, n):
    values = np.arange(-S, S + 0.5, 1.0)
    return jnp.asarray(np.random.default_rng(1).choice(values, size=(n, L)), dtype=jnp.float32)


@pytest.mark.parametrize("S, form", list(itertools.product([1.0, 1.5], FORMS)))
def test_fast_ratio_matches_log_psi(S, form):
    H = _spin_model(S)
    guide = JastrowSpin(L, PAIRS, **FORMS[form])
    configs = _configs(S, 60)
    fast = jax.jit(jax.vmap(lambda c: guide.log_diff_from_operator(c, H)))(configs)
    ref = jax.vmap(lambda c: log_ratios_from_log_psi(guide, H, c))(configs)
    assert fast.shape == (60, H.M_total)
    # forbidden moves (beyond +-S) carry zero rate; their ratio still has to be finite
    np.testing.assert_allclose(fast, ref, atol=1e-4)


@pytest.mark.parametrize("form", FORMS)
def test_cache_tables_reproduce_the_inline_ratios(form):
    """``build_cache`` holds the walker-independent tables; the cached ratios equal
    the on-the-fly ones bit for bit, and the cache is memoised per operator."""
    H = _spin_model(1.0)
    guide = JastrowSpin(L, PAIRS, **FORMS[form])
    cache = guide.build_cache(H)
    assert guide.build_cache(H) is cache
    configs = _configs(1.0, 40)
    inline = jax.jit(jax.vmap(lambda c: guide.log_diff_from_operator(c, H)))(configs)
    cached = jax.jit(jax.vmap(lambda c: guide.log_diff_from_operator(c, H, cache)))(configs)
    np.testing.assert_array_equal(np.asarray(inline), np.asarray(cached))


@pytest.mark.parametrize("form", FORMS)
def test_incremental_state_follows_the_moves(form):
    """The incremental neighbor sums, updated through 200 random moves with the
    operator's ``move``, stay within float rounding of a rebuild, and the ratios
    computed from them match the plain ones."""
    H = _spin_model(1.0)
    guide = JastrowSpin(L, PAIRS, incremental=True, **FORMS[form])
    cache = guide.build_cache(H)
    rng = np.random.default_rng(5)
    x = _configs(1.0, 1)[0]
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


def _incremental_run(walkers, H, guide, n_steps, number_of_outputs, rng_key):
    return run(walkers, n_steps, H, 0.25, guide, [2, 4], rng_key=rng_key,
               warmup_steps=20, number_of_outputs=number_of_outputs, verbosity=0)


def test_incremental_mode_does_not_depend_on_the_output_chunking(tmp_path):
    """The refresh of the incremental state runs on a fixed schedule of chain
    steps, so the same seed gives bit-identical results whatever
    ``number_of_outputs`` is, and a run split over two HDF5 blocks reproduces
    the contiguous run (the refresh at step 100 falls into the second block)."""
    from pygfmc import write_to_HDF5, run_from_HDF5
    n = 6
    H = Spin.create(chain(n).edges, S=1.0, J_z=-1.0, h_pm=0.65)
    guide = JastrowSpin(n, chain(n).edges, beta=-0.3, d=0.1, incremental=True)
    w0 = prepare_walkers_from_configs(jnp.ones((128, n)))
    key = jax.random.PRNGKey(3)
    w_a, e_a, _, _, raw_a = _incremental_run(w0, H, guide, 120, 3, key)
    w_b, e_b, _, _, raw_b = _incremental_run(w0, H, guide, 120, 30, key)
    assert e_a == e_b
    np.testing.assert_array_equal(np.asarray(raw_a["energy"]), np.asarray(raw_b["energy"]))
    np.testing.assert_array_equal(np.asarray(w_a.config), np.asarray(w_b.config))
    # split: 60 steps, stored, then 60 more from the file
    w_1, e_1, o_1, st_1, raw_1 = _incremental_run(w0, H, guide, 60, 3, key)
    path = str(tmp_path / "split.h5")
    write_to_HDF5(path, st_1, walkers=w_1, energy=e_1, obs=o_1, raw=raw_1, verbose=False)
    w_2, _, _, _, raw_2 = run_from_HDF5(path, H, guide=guide, n_steps=60,
                                        number_of_outputs=3, verbosity=0)
    np.testing.assert_array_equal(
        np.concatenate([np.asarray(raw_1["energy"]), np.asarray(raw_2["energy"])]),
        np.asarray(raw_a["energy"]))
    np.testing.assert_array_equal(np.asarray(w_2.config), np.asarray(w_a.config))


def test_incremental_update_without_pairs_is_a_no_op():
    """A guide built with ``pairs=[]`` has no neighbor sums to update."""
    H = Spin.create(EDGES, S=1.0, J_pm=-0.5, N=L)
    guide = JastrowSpin(L, [], beta=0.3, h=0.1, incremental=True)
    cache = guide.build_cache(H)
    x = _configs(1.0, 1)[0]
    state = guide.init_state(x, cache)
    assert state.shape == (L,) and not np.any(np.asarray(state))
    sites, deltas = H.move(x, jnp.int32(1))
    np.testing.assert_array_equal(np.asarray(guide.update_state(state, x, sites, deltas, cache)),
                                  np.asarray(state))


def test_float64_parameters_round_trip_exactly(tmp_path, x64):
    """Under ``jax_enable_x64`` a float64 parameter is stored and loaded as such."""
    with x64():
        beta = np.full(len(PAIRS), 0.123456789123, dtype=np.float64)
        guide = JastrowSpin(L, PAIRS, beta=beta, h=0.987654321987, dtype=jnp.float64)
        guide.save(str(tmp_path / "g64.h5"))
        back = JastrowSpin.load(str(tmp_path / "g64.h5"), dtype=jnp.float64)
        np.testing.assert_array_equal(np.asarray(back.beta), beta)
        assert float(back._h) == 0.987654321987


def test_forbidden_transitions_keep_their_zero_rate_under_a_huge_field():
    """A forbidden move (S+ on a site already at +S) has a bare rate of 0; the
    guide ratio evaluated on the unphysical successor may overflow, and the
    product must stay an exact 0 rather than 0 * inf = NaN."""
    from pygfmc.engine.evolution import transition_rates
    H = Spin.create(np.zeros((0, 2), dtype=np.int64), S=1.0, h_pm=0.5, N=3)
    guide = JastrowSpin(3, [], beta=0.0, h=100.0)
    x = jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float32)        # every S+ forbidden
    rates = np.asarray(transition_rates(x, H, guide, cache=guide.build_cache(H))[0])
    assert np.all(np.isfinite(rates))
    assert np.all(rates[1:4] == 0.0)                              # the S+ block: 0, not 0 * inf
    # the opposite sign overflows the ALLOWED S- ratios, and that is not hidden
    guide = JastrowSpin(3, [], beta=0.0, h=-100.0)
    rates = np.asarray(transition_rates(x, H, guide, cache=guide.build_cache(H))[0])
    assert np.all(rates[1:4] == 0.0) and np.all(np.isneginf(rates[4:]))


def test_malformed_pair_lists_are_rejected():
    """Pair lists get the edge checks: shape (n, 2), integer-valued, in range."""
    for bad in ([[0, 1.5]], [[0, 1, 2]], [[0, L]], [[-1, 0]], [[0, np.nan]]):
        with pytest.raises(ValueError):
            JastrowSpin(L, bad, beta=0.3)
    assert JastrowSpin(L, np.empty((0, 2)), beta=0.3).pair_i.shape == (0,)


def test_wrong_coupling_is_detected():
    """Non-vacuity: flipping the sign of d must break the agreement."""
    H = _spin_model(1.0)
    good = JastrowSpin(L, PAIRS, **FORMS["tied, tied h, tied d"])
    bad = JastrowSpin(L, PAIRS, beta=-0.25, h=0.15, d=-0.4)
    config = _configs(1.0, 1)[0]
    assert jnp.max(jnp.abs(bad.log_diff_from_operator(config, H)
                           - log_ratios_from_log_psi(good, H, config))) > 0.1


def test_parameters_keep_the_form_they_were_given():
    def float_leaves(g):
        return [leaf.shape for leaf in jax.tree_util.tree_leaves(g)
                if jnp.issubdtype(leaf.dtype, jnp.floating)]
    assert float_leaves(JastrowSpin(L, PAIRS, beta=0.3)) == [()]
    assert float_leaves(JastrowSpin(L, PAIRS, beta=0.3, h=0.1, d=0.2)) == [(), (), ()]
    assert float_leaves(JastrowSpin(L, PAIRS, **FORMS["per pair, per site"])) == [(len(PAIRS),), (L,), (L,)]


@pytest.mark.parametrize("form", FORMS)
def test_save_load_and_pytree(tmp_path, form):
    guide = JastrowSpin(L, PAIRS, **FORMS[form])
    configs = _configs(1.0, 20)
    path = tmp_path / "guide.h5"
    guide.save(path)
    loaded = JastrowSpin.load(path)
    np.testing.assert_array_equal(loaded.log_psi(configs), guide.log_psi(configs))
    assert jax.tree_util.tree_structure(loaded) == jax.tree_util.tree_structure(guide)
    leaves, tree = jax.tree_util.tree_flatten(guide)
    rebuilt = jax.tree_util.tree_unflatten(tree, leaves)
    np.testing.assert_array_equal(rebuilt.log_psi(configs), guide.log_psi(configs))


@pytest.mark.parametrize("incremental", [False, True])
def test_gfmc_spin1_tfim_matches_ed(incremental):
    """A short guided GFMC run on the spin-1 transverse-field Ising chain, with
    the neighbor sums rebuilt at every evaluation and in the incremental mode."""
    n = 6
    H = Spin.create(chain(n).edges, S=1.0, J_z=-1.0, h_pm=0.65)   # h = 1.3, near h_c
    E0 = float(exact_diagonalization(H, number_of_states=1)[0][0])
    guide = JastrowSpin(n, chain(n).edges, beta=-0.3, d=0.1, incremental=incremental)
    walkers = prepare_walkers_from_configs(jnp.ones((256, n)))
    _, energy, _, _, _ = run(walkers, 1500, H, lam=None, step_size=0.25, guide=guide,
                             p_array=[2, 8], rng_key=jax.random.PRNGKey(0),
                             warmup_steps=300, verbosity=0)
    assert abs(energy[8] - E0) / n < 0.01


def test_train_with_netket_keeps_the_form():
    require_netket()
    from pygfmc.wavefunction import train_with_netket, make_hilbert
    import netket as nk
    n = 6
    H = Spin.create(chain(n).edges, S=1.0, J_z=-1.0, h_pm=0.65)
    sampler = nk.sampler.MetropolisLocal(make_hilbert(H.local_hilbert_space, n))
    assert sorted(np.asarray(sampler.hilbert.local_states).tolist()) == [-1, 0, 1]
    guide = JastrowSpin(n, chain(n).edges, beta=-0.1, d=0.0)
    trained, history = train_with_netket(H, guide, sampler=sampler, n_iter=5, n_samples=256,
                                         return_history=True)
    assert np.all(np.isfinite(history))
    assert jax.tree_util.tree_structure(trained) == jax.tree_util.tree_structure(guide)
    assert float(trained._beta) != -0.1 or float(trained._d) != 0.0
