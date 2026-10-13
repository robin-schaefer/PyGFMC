"""Pytree hygiene for every guide in ``pygfmc/wavefunction/``.

A guide is a traced argument of the jitted evolution, so (1) its aux data must
be hashable and self-equal (a raw array raises on the second jit-cache lookup;
a NaN makes identical instances miss the cache), and (2) ``tree_unflatten``
must rebuild the full public object.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pygfmc.wavefunction as wf
from pygfmc.hamiltonian import SpinHalf, chain

L = 6
EDGES = chain(L, periodic=True).edges
PAIRS = np.asarray([[i, (i + 1) % L] for i in range(L)], dtype=np.int32)


def _h():
    return SpinHalf.create(EDGES, J_x=-1.0, J_y=-1.0, J_z=-1.0, h_x=0.4)


# name -> zero-argument builder producing a fresh, deterministic instance
BUILDERS = {
    "Jastrow": lambda: wf.JastrowSpinHalf(L, EDGES, beta=0.2),
    "ClusterCPS": lambda: wf.ClusterCPS(
        [dict(clusters=PAIRS, symmetries=np.array([[0, 1], [1, 0]]), parity=True)],
        init="gaussian", seed=0),
    "ClusterMLP": lambda: wf.ClusterMLP([[PAIRS, [4]]], seed=0),
    "ExactGroundState": lambda: wf.ExactGroundState(_h()),
    "ExactGroundState_log_amp": lambda: wf.ExactGroundState(_h(), log_amp=np.zeros(2 ** L)),
}


def _aux_leaves(treedef):
    """Every aux payload in the tree (root and nested nodes)."""
    out, stack = [], [treedef]
    while stack:
        td = stack.pop()
        data = td.node_data()
        if data is not None:
            out.append(data[1])
        stack.extend(td.children())
    return out


def _bad_aux(x):
    """Describe the first array / NaN found in ``x``, else None."""
    if isinstance(x, (np.ndarray, jax.Array)):
        return f"array of shape {np.shape(x)}"
    if isinstance(x, float) and x != x:
        return "NaN float"
    items = x if isinstance(x, (tuple, list)) else x.values() if isinstance(x, dict) else ()
    for y in items:
        r = _bad_aux(y)
        if r:
            return r
    return None


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_aux_is_hashable_and_self_equal(name):
    """Two identical instances have equal, hashable treedefs and share one jit trace."""
    a, b = BUILDERS[name](), BUILDERS[name]()
    ta, tb = jax.tree_util.tree_structure(a), jax.tree_util.tree_structure(b)
    for aux in _aux_leaves(ta):
        assert _bad_aux(aux) is None, f"{name} aux holds {_bad_aux(aux)}"
    assert ta == tb, f"two identical {name} instances have unequal treedefs"
    hash(ta)

    calls = []

    @jax.jit
    def f(g):
        calls.append(1)                    # runs once per TRACE
        return sum(jnp.sum(jnp.asarray(x, jnp.float32))
                   for x in jax.tree_util.tree_leaves(g))

    f(a)
    f(b)
    assert len(calls) == 1, f"{name}: second identical instance re-traced"


def test_exact_ground_state_E0_survives_roundtrip():
    """E0 (NaN when log_amp is supplied, finite otherwise) survives tree_map."""
    g = wf.ExactGroundState(_h(), log_amp=np.zeros(2 ** L))
    assert np.isnan(jax.tree_util.tree_map(lambda x: x, g).E0)
    g = wf.ExactGroundState(_h())
    g2 = jax.tree_util.tree_map(lambda x: x, g)
    assert g2.E0 == g.E0 and np.isfinite(g2.E0)


def test_jastrow_beta_after_tree_roundtrip():
    """JastrowSpinHalf.beta survives tree_flatten / tree_unflatten."""
    beta = np.linspace(-0.3, 0.4, len(EDGES)).astype(np.float32)
    g = wf.JastrowSpinHalf(L, EDGES, beta=beta)
    g2 = jax.tree_util.tree_unflatten(*reversed(jax.tree_util.tree_flatten(g)))
    np.testing.assert_array_equal(np.asarray(g2.beta), np.asarray(g.beta))
    np.testing.assert_array_equal(np.asarray(g.beta), beta)


def test_jastrow_beta_after_jit():
    """JastrowSpinHalf.beta is rebuilt from a jitted tree_map and readable inside jit."""
    g = wf.JastrowSpinHalf(L, EDGES, beta=0.25)

    @jax.jit
    def scale(guide):
        return jax.tree_util.tree_map(
            lambda x: x * 2 if jnp.issubdtype(x.dtype, jnp.floating) else x, guide)

    np.testing.assert_allclose(np.asarray(scale(g).beta), 0.5)
    np.testing.assert_allclose(float(jax.jit(lambda gd: jnp.sum(gd.beta))(g)),
                               0.25 * len(EDGES), rtol=1e-6)


def test_jastrow_beta_equals_minus_J_everywhere():
    """beta is -J for the per-pair couplings, in the input pair order: every
    per-site table entry of a pair holds -beta of that pair (both endpoints,
    unsorted pairs included), and log_psi is -sum_pairs beta_ij x_i x_j."""
    edges = np.asarray([[3, 1], [0, 2], [4, 5], [1, 2]])
    beta = np.asarray([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
    g = wf.JastrowSpinHalf(6, edges, beta=beta)
    np.testing.assert_array_equal(np.asarray(g.beta), beta)
    site_J = np.asarray(g.site_J)
    partner = np.asarray(g.site_other)
    valid = np.asarray(g.site_valid_mask)
    for (a, b), beta_ab in zip(edges, beta):
        for i, j in ((a, b), (b, a)):
            slot = np.flatnonzero(valid[i] & (partner[i] == j))
            assert slot.size == 1, (i, j)
            assert site_J[i, slot[0]] == -beta_ab, (i, j, site_J[i, slot[0]], beta_ab)
    assert int(valid.sum()) == 2 * len(edges)
    x = jnp.asarray([1.0, -1.0, -1.0, 1.0, 1.0, -1.0])
    expected = -sum(float(bb) * float(x[i]) * float(x[j]) for (i, j), bb in zip(edges, beta))
    np.testing.assert_allclose(float(np.ravel(g.log_psi(x))[0]), expected, rtol=0, atol=1e-6)
