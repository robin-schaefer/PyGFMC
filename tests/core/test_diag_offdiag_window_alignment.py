"""Diagonal and off-diagonal estimators of the same SzSz agree to float64 precision.

The ancestry-traced diagonal estimator and the M_total=1 forward-walking
estimator are identical per window; after the estimator's common-window trim
both pooled sums run over the same windows, so they must agree at every p.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pygfmc import run_gfmc
from pygfmc.walkers import prepare_walkers_from_configs
from pygfmc.engine.estimators import estimate_average_log_branch_weight
from pygfmc.hamiltonian import SpinHalf


@pytest.fixture(autouse=True)
def _float64(x64):
    """The claim is float64 agreement; the conftest scope keeps x64 from leaking."""
    with x64():
        yield


class _SzSzDiag:
    def __init__(self, i, j):
        self.i, self.j = i, j

    def diagonal_weight(self, config):
        return config[self.i] * config[self.j]


class _SzSzOffDiag:
    """Same SzSz(i,j), routed through the off-diagonal forward-walking estimator."""
    M_total = 1
    n_off = 0

    def __init__(self, i, j):
        self.i, self.j = i, j

    def get_rates(self, config, return_configs=False):
        return jnp.asarray([config[self.i] * config[self.j]], dtype=config.dtype), None


jax.tree_util.register_pytree_node(
    _SzSzDiag, lambda o: ((), (o.i, o.j)), lambda aux, children: _SzSzDiag(*aux))
jax.tree_util.register_pytree_node(
    _SzSzOffDiag, lambda o: ((), (o.i, o.j)), lambda aux, children: _SzSzOffDiag(*aux))


def test_diag_offdiag_pipelines_agree_tfim6():
    """SzSz(4,5) on the L=6 TFIM: diag and off-diag estimates agree to 1e-9 at every p."""
    L = 6
    i0, i1 = 4, 5
    edges = np.asarray([(i, (i + 1) % L) for i in range(L)], dtype=np.int32)
    h = SpinHalf.create(edges, J_x=0., J_y=0., J_z=-4., h_x=1.5, dtype=jnp.float64)
    lam = 15.0
    p_array = [2, 4, 8, 16]

    wk = prepare_walkers_from_configs(jnp.ones((64, L), dtype=jnp.float64))
    alw = estimate_average_log_branch_weight(
        wk, h, None, method="multi_step", lam=lam)

    _, _, obs, _, _ = run_gfmc(
        wk, 2000, h, 4, None, p_array, method="multi_step", lam=lam,
        rng_key=jax.random.PRNGKey(11),
        diag_operators={"SzSz_diag": _SzSzDiag(i0, i1)},
        off_diag_operators={"SzSz_offdiag": _SzSzOffDiag(i0, i1)},
        avg_log_weight=alw, verbosity=0, warmup_steps=200,
        off_diag_stride=1)

    for p in p_array:
        d = obs["SzSz_diag"][p]
        o = obs["SzSz_offdiag"][p]
        assert abs(d - o) < 1e-9, f"p={p}: diag={d!r} offdiag={o!r} diff={abs(d - o):.3e}"
