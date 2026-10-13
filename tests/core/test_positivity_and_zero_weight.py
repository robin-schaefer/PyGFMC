"""Off-diagonal measurement operators: the positivity check and the zero-weight guard.

The forward-walking estimator draws a successor by inverse CDF over the
operator's rates, so a negative rate is refused up front by
``check_off_diag_operator_positivity`` on both backends.  A configuration on
which every rate of the operator vanishes (a bond hop with both sites empty)
is legal, and must leave the shadow chain finite.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pygfmc import run_gfmc
from pygfmc.engine.estimators import check_off_diag_operator_positivity
from pygfmc.engine.evolution import forward_walk_operator
from pygfmc.engine.multi_gpu import run_gfmc_pmap
from pygfmc.hamiltonian import BoseHubbard, SpinHalf, chain
from pygfmc.walkers import prepare_walkers_from_configs

_L = 8
_EDGES = chain(_L).edges
_H = SpinHalf.create(_EDGES, J_z=-1.0, h_x=1.0)


def _bad_op():
    """-sum_i S^x_i: every off-diagonal rate is -1/2."""
    return SpinHalf.create([], N=_L, h_x=1.0)


def _good_op():
    return SpinHalf.create([], N=_L, h_x=-1.0, sign_check=False)


def test_positivity_check_passes_and_fails_as_it_should(capsys):
    w = prepare_walkers_from_configs(jnp.ones((16, _L)))
    check_off_diag_operator_positivity(w, {"sx": _good_op()}, None, verbose=False)
    assert capsys.readouterr().out == ""
    check_off_diag_operator_positivity(w, {"sx": _good_op()}, None)
    assert "passed positivity check" in capsys.readouterr().out
    with pytest.raises(ValueError, match="negative rate"):
        check_off_diag_operator_positivity(w, {"sx": _bad_op()}, None)
    # a diagonal-only operator may carry any sign
    zz = SpinHalf.create(_EDGES, J_z=-7.0)
    check_off_diag_operator_positivity(w, {"zz": zz}, None, verbose=False)


@pytest.mark.parametrize("backend", ["run_gfmc", "run_gfmc_pmap"])
def test_negative_rate_operator_is_refused_before_production(backend):
    """Both drivers raise during the positivity probe, before any production step."""
    w = prepare_walkers_from_configs(jnp.ones((32, _L)))
    kw = dict(rng_key=jax.random.PRNGKey(0), warmup_steps=4, verbosity=0,
              off_diag_operators={"sx": _bad_op()})
    if backend == "run_gfmc_pmap":
        kw["n_devices"] = 1
    fn = run_gfmc if backend == "run_gfmc" else run_gfmc_pmap
    with pytest.raises(ValueError, match="negative rate"):
        fn(w, 20, _H, 0.2, None, [2], **kw)
    # the same call with the correct sign runs
    kw["off_diag_operators"] = {"sx": _good_op()}
    out = fn(w, 20, _H, 0.2, None, [2], **kw)
    assert np.isfinite(float(out[2]["sx"][2]))


def test_operator_with_no_allowed_move_on_a_configuration_stays_finite():
    """A bond-hop measurement on walkers whose bond is empty gives finite values, not NaN.

    The operator's total weight is zero on such a configuration; the shadow
    chain must then carry a zero contribution instead of an out-of-basis
    successor.
    """
    n = 6
    lat = chain(n)
    H = BoseHubbard.create(lat.edges, t=1.0, U=2.0, n_max=2)
    hop01 = BoseHubbard.create(lat.edges, t=-np.eye(1, len(lat.edges), 0)[0],
                               n_max=2, N=n, sign_check=False)
    # every walker starts with sites 0 and 1 empty and two particles elsewhere
    base = jnp.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    w = prepare_walkers_from_configs(jnp.tile(base, (64, 1)))
    rates, _ = hop01.get_rates(base)
    assert float(jnp.sum(jnp.abs(rates))) == 0.0, "the fixture must have no allowed hop"

    # applied directly to the empty-bond ensemble, the operator contributes
    # exactly zero at every shadow depth (no NaN from a clamped successor)
    fw, fw_norm = forward_walk_operator(
        w, hop01, H, 0.1, None, "cont_time", jax.random.PRNGKey(0), 0.0, [1, 2, 4])
    assert np.all(np.asarray(fw) == 0.0) and np.all(np.isfinite(np.asarray(fw_norm)))

    # through the driver the walkers move first, so later rows are non-zero,
    # but every value and every estimate stays finite
    _, energy, obs, stats, raw = run_gfmc(
        w, 40, H, 0.1, None, [2, 4], rng_key=jax.random.PRNGKey(1), warmup_steps=0,
        avg_log_weight=0.0, off_diag_operators={"hop01": hop01}, verbosity=0)
    assert all(np.isfinite(float(v)) for v in energy.values())
    assert all(np.isfinite(float(v)) for v in obs["hop01"].values())
    od = np.asarray(raw["off_diag"]["hop01"])
    assert np.all(np.isfinite(od)) and np.all(np.isfinite(np.asarray(raw["off_diag_norm"]["hop01"])))
    assert np.any(od != 0.0), "the walkers do hop onto the bond later, so the test is not vacuous"


def test_population_extinction_is_an_error_not_nan():
    """When lam equals the diagonal element of every walker, every branch
    weight is zero, the ensemble cannot be renormalized and the driver names
    the step instead of returning NaN."""
    from pygfmc import run_gfmc
    from pygfmc.hamiltonian import SpinHalf, chain
    n = 4
    H = SpinHalf.create(chain(n).edges, J_z=1.0)          # diagonal only
    up = prepare_walkers_from_configs(jnp.ones((16, n)))
    lam = float(len(chain(n).edges)) * 0.25                # == H_xx of all-up
    with pytest.raises(ValueError, match="population extinct at step 0"):
        run_gfmc(up, 10, H, 2, None, [2], method="multi_step", lam=lam,
                 rng_key=jax.random.PRNGKey(0), verbosity=0)
