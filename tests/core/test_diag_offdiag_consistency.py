"""The same diagonal operator measured through BOTH estimator pipelines must agree.

The diagonal (ancestry-traced) and off-diagonal (forward-walking) pipelines
share almost no code, so agreement is a real cross-check; both are also
anchored to the exact ground-state value.  The machine-precision version is
``test_diag_offdiag_window_alignment.py``.  Also here: the m = 0 forward walk
(p = 1) of a diagonal operator, a protocol-only operator as an off-diagonal
observable, and the jit cache of a continuation that passes the measured shift.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pygfmc.hamiltonian import SpinHalf, chain, exact_diagonalization
from pygfmc import run_gfmc
from pygfmc.engine import driver
from pygfmc.walkers import prepare_walkers_sz
from core.walkers import balanced_walkers

_DTYPE = jnp.float32


def test_diagonal_op_diag_vs_offdiag_consistency():
    """Ising energy via the diagonal and off-diagonal estimators matches ED and each other."""
    lattice = chain(6, periodic=True)
    h       = SpinHalf.create(lattice.edges, -1.0, -1.0, 1.0, dtype=_DTYPE)
    h_ising = SpinHalf.create(lattice.edges,  0.0,  0.0, 1.0, dtype=_DTYPE)
    lam     = len(lattice.edges) / 4.0 + 1.0

    _, _, off = exact_diagonalization(h, off_diag_operators={"ising": h_ising})
    ising_ed = float(off["ising"][0])

    rng = jax.random.PRNGKey(55)
    rng, init_key = jax.random.split(rng)
    walkers = balanced_walkers(256, lattice.N, init_key, dtype=_DTYPE)

    _, results, obs, _, _ = run_gfmc(
        walkers, 1000, h, 10, lam=lam,
        guide=None, p_array=[2, 4, 8, 16], method="single_step", rng_key=rng,
        diag_operators={"ising_diag": h_ising},
        off_diag_operators={"ising_offdiag": h_ising},
        warmup_steps=200, verbosity=0,
    )
    best_p = min([2, 4, 8, 16], key=lambda p: results[p])
    ising_diag    = obs["ising_diag"][best_p]
    ising_offdiag = obs["ising_offdiag"][best_p]

    assert abs(ising_diag    - ising_ed) < 0.1
    assert abs(ising_offdiag - ising_ed) < 0.1
    assert abs(ising_diag - ising_offdiag) < 0.1


def test_forward_walk_at_m_zero_is_the_plain_mean():
    """p = 1 means m = 0: the forward walk reads the un-propagated ensemble.

    The raw p = 1 column is then the plain walker mean at the measured step,
    i.e. the diagonal pipeline's m = 0 value at that step, with an exactly zero
    log norm; and it differs from the p = 4 column, so the depth is honored.
    """
    lattice = chain(6, periodic=True)
    h = SpinHalf.create(lattice.edges, -1.0, -1.0, 1.0, dtype=_DTYPE)
    zz = SpinHalf.create(lattice.edges, 0.0, 0.0, 1.0, dtype=_DTYPE)
    walkers = balanced_walkers(64, lattice.N, jax.random.PRNGKey(0), dtype=_DTYPE)
    _, _, obs, _, raw = run_gfmc(
        walkers, 40, h, 0.2, None, [1, 2, 4], rng_key=jax.random.PRNGKey(1),
        diag_operators={"zz_diag": zz}, off_diag_operators={"zz_off": zz},
        warmup_steps=8, verbosity=0)
    assert raw["p_array"] == [1, 2, 4] and raw["m_array"] == [0, 1, 2]
    off = np.asarray(raw["off_diag"]["zz_off"])
    norm = np.asarray(raw["off_diag_norm"]["zz_off"])
    diag = np.asarray(raw["diag"]["zz_diag"])
    steps = np.asarray(raw["off_diag_steps"])
    assert np.all(norm[:, 0] == 0.0)
    np.testing.assert_array_equal(off[:, 0], diag[steps, 0])
    assert not np.array_equal(off[:, 0], off[:, 2])
    assert abs(float(obs["zz_off"][1]) - float(obs["zz_diag"][1])) < 0.01


class _ProtocolIdentity(NamedTuple):
    """Only the protocol: ``M_total`` and ``get_rates`` (no ``n_off``, no
    ``generate_state``, which ``M_total == 1`` does not need)."""
    M_total: int = 1

    def get_rates(self, config, return_configs=False):
        return jnp.ones(1, dtype=config.dtype), None


def test_protocol_only_operator_measures_off_diagonally():
    """A NamedTuple with ``M_total == 1`` and ``get_rates`` runs as an
    off-diagonal operator, and the identity's estimate is 1 at every window."""
    lattice = chain(6, periodic=True)
    h = SpinHalf.create(lattice.edges, -1.0, -1.0, 1.0, dtype=_DTYPE)
    walkers = balanced_walkers(32, lattice.N, jax.random.PRNGKey(2), dtype=_DTYPE)
    _, _, obs, _, raw = run_gfmc(
        walkers, 30, h, 0.2, None, [1, 2, 4], rng_key=jax.random.PRNGKey(3),
        off_diag_operators={"one": _ProtocolIdentity()}, warmup_steps=8, verbosity=0)
    assert np.all(np.asarray(raw["off_diag"]["one"]) == 1.0)
    for p, value in obs["one"].items():
        assert abs(float(value) - 1.0) < 1e-5, (p, value)


def test_continuing_with_the_measured_shift_compiles_nothing_new():
    """A continuation that passes ``stats['avg_log_weight']`` (a float) hits the
    scans its warm-up run compiled: the shift is a traced argument, not part of
    the jit cache key."""
    h = SpinHalf.create(chain(6).edges, -1.0, -1.0, 1.0, dtype=_DTYPE)
    zz = SpinHalf.create(chain(6).edges, 0.0, 0.0, 1.0, dtype=_DTYPE)
    kw = dict(diag_operators={"zz": zz}, number_of_outputs=2, verbosity=0)

    def cache_size():
        return (driver._scan_cached._cache_size()
                + driver._scan_cached_donated._cache_size())

    walkers = prepare_walkers_sz(16, 6, jax.random.PRNGKey(0), sz=0)
    walkers, _, _, stats, _ = run_gfmc(walkers, 20, h, 0.2, None, [2, 4],
                                       rng_key=jax.random.PRNGKey(1), warmup_steps=8, **kw)
    compiled = cache_size()
    assert compiled > 0
    # the key as a JAX array: the shift is what this test is about
    next_key = jnp.asarray(stats["parameters/rng_key_next"])
    run_gfmc(walkers, 20, h, 0.2, None, [2, 4], rng_key=next_key,
             avg_log_weight=stats["avg_log_weight"], warmup_steps=0, **kw)
    assert cache_size() == compiled
