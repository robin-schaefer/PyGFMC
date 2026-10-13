"""The keys and invariants of ``statistics()`` and of the ``stats`` a run returns.

The diagnostics are the quantities a user reads to tune a run, so their names
and ranges are a contract: the per-method key sets, the offspring fractions
summing to one, the Kish effective sample size equal to N at depth 0 and
non-increasing with depth, and the run's own inputs under ``parameters/``.
"""
import functools

import jax
import numpy as np
import pytest

from pygfmc import run_gfmc, statistics
from pygfmc.hamiltonian import SpinHalf, chain
from pygfmc.walkers import prepare_walkers_sz
from pygfmc.wavefunction import JastrowSpinHalf

_H = SpinHalf.create(chain(8).edges, J_x=-1.0, J_y=-1.0, J_z=1.0)
_G = JastrowSpinHalf(_H.N, _H.edges, beta=0.3)
_COMMON = {"n_unique_pre", "n_unique_post", "unique_fraction", "diag_min", "diag_max",
           "changed_rate", "overlap_rate", "H_diag_max_observed", "H_diag_min_observed",
           "offspring_k0", "offspring_k1", "offspring_k2", "offspring_k3",
           "offspring_k4plus", "survival_rate"}
_DISCRETE = {"p_d_min", "p_d_max", "p_d_mean", "hold_steps_mean", "hold_steps_max"}
_CONT = {"escape_rate_min", "escape_rate_max", "escape_rate_mean", "hold_time_mean",
         "hold_time_max", "hops_per_step_mean", "hops_per_step_max"}
_LINEAGE = {"ancestor_m", "ancestor_fraction", "effective_sample_size",
            "ancestor_correlation"}


@functools.lru_cache(maxsize=None)
def _equilibrated():
    """Walkers and shift after a short run; computed once (JAX arrays are immutable)."""
    w = prepare_walkers_sz(256, 8, jax.random.PRNGKey(0), sz=0)
    w, _, _, st, _ = run_gfmc(w, 40, _H, 0.2, _G, [2], rng_key=jax.random.PRNGKey(1),
                              warmup_steps=40, verbosity=0)
    return w, st["avg_log_weight"]


@pytest.mark.parametrize("method,step_size,lam", [("cont_time", 0.2, None),
                                                   ("multi_step", 2, 4.0),
                                                   ("single_step", 2, 4.0)])
def test_statistics_keys_and_shapes(method, step_size, lam):
    """Each method exposes exactly its own diagnostic family, all of length M (or n_total)."""
    w, alw = _equilibrated()
    M, m_array = 4, [1, 2, 6]
    _, st = statistics(w, _H, step_size, _G, method, jax.random.PRNGKey(2), alw,
                       lam=lam, M=M, quiet=True, m_array=m_array)
    expected = _COMMON | _LINEAGE | (_CONT if method == "cont_time" else _DISCRETE)
    assert set(st) == expected, (set(st) ^ expected)
    n_total = max(M, max(m_array))
    for k in _COMMON - {"H_diag_max_observed", "H_diag_min_observed", "survival_rate"} \
            - {f"offspring_k{j}" for j in range(4)} - {"offspring_k4plus"}:
        assert np.shape(st[k]) == (M,), k
    for k in [f"offspring_k{j}" for j in range(4)] + ["offspring_k4plus", "survival_rate"]:
        assert np.shape(st[k]) == (n_total,), k
    for k in _LINEAGE:
        assert np.shape(st[k]) == (len(m_array),), k


def test_offspring_fractions_sum_to_one_and_define_survival():
    w, alw = _equilibrated()
    _, st = statistics(w, _H, 0.2, _G, "cont_time", jax.random.PRNGKey(3), alw,
                       M=6, quiet=True)
    total = sum(st[f"offspring_k{j}"] for j in range(4)) + st["offspring_k4plus"]
    np.testing.assert_allclose(total, 1.0, atol=1e-6)
    np.testing.assert_allclose(st["survival_rate"], 1.0 - st["offspring_k0"], atol=1e-7)
    assert np.all(st["unique_fraction"] <= 1.0) and np.all(st["unique_fraction"] > 0.0)


def test_effective_sample_size_starts_at_N_and_never_grows_with_depth():
    w, alw = _equilibrated()
    m_array = [0, 1, 2, 4, 8]
    _, st = statistics(w, _H, 0.2, _G, "cont_time", jax.random.PRNGKey(4), alw,
                       M=2, quiet=True, m_array=m_array)
    ess = np.asarray(st["effective_sample_size"])
    frac = np.asarray(st["ancestor_fraction"])
    assert list(st["ancestor_m"]) == m_array
    assert ess[0] == 256.0 and frac[0] == 1.0
    assert np.all(np.diff(ess) <= 1e-9), ess
    assert np.all(np.diff(frac) <= 1e-9), frac
    assert np.all(ess >= 1.0) and np.all((frac > 0) & (frac <= 1))
    # the ESS never exceeds the number of distinct surviving ancestors
    assert np.all(ess <= frac * 256 + 1e-6)


def test_ancestor_correlation_is_one_at_depth_zero_and_decays_with_the_step():
    """corr(e_L) is exactly 1 at m = 0 and for a frozen population, within [-1, 1]
    otherwise, and decays faster for a larger step size."""
    w, alw = _equilibrated()
    key = jax.random.PRNGKey(4)
    _, st = statistics(w, _H, 0.2, _G, "cont_time", key, alw,
                       M=2, quiet=True, m_array=[0, 1, 2, 4, 8])
    corr = np.asarray(st["ancestor_correlation"])
    assert corr.shape == (5,) and corr[0] == 1.0
    assert np.all(np.isfinite(corr)) and np.all(np.abs(corr) <= 1.0)
    assert corr[1] < 0.9 and corr[1] > corr[4], corr
    # a step too short for any hop: every walker keeps its ancestor's energy
    _, frozen = statistics(w, _H, 1e-9, _G, "cont_time", key, alw,
                           M=2, quiet=True, m_array=[1, 4])
    np.testing.assert_allclose(frozen["ancestor_correlation"], 1.0, atol=1e-5)
    # ten hops per step: the population has forgotten its start after one step
    _, big = statistics(w, _H, 2.0, _G, "cont_time", key, alw,
                        M=2, quiet=True, m_array=[1])
    assert abs(big["ancestor_correlation"][0]) < 0.3, big["ancestor_correlation"]


def test_cont_time_hops_per_step_is_escape_rate_times_step_size():
    w, alw = _equilibrated()
    _, st = statistics(w, _H, 0.3, _G, "cont_time", jax.random.PRNGKey(5), alw,
                       M=3, quiet=True)
    np.testing.assert_allclose(st["hops_per_step_mean"], st["escape_rate_mean"] * 0.3)
    np.testing.assert_allclose(st["hops_per_step_max"], st["escape_rate_max"] * 0.3)
    assert np.all(st["hold_time_mean"] > 0)


def test_avg_log_weight_is_optional_and_only_a_numerical_shift():
    """Omitting avg_log_weight uses the walkers' own estimate, bit-identically;
    a different shift leaves every diagnostic within float noise."""
    from pygfmc.engine.estimators import estimate_average_log_branch_weight
    w, alw = _equilibrated()
    kw = dict(M=4, quiet=True, m_array=[1, 2, 4])
    key = jax.random.PRNGKey(7)
    est = estimate_average_log_branch_weight(w, _H, _G, method="cont_time")
    _, default = statistics(w, _H, 0.2, _G, "cont_time", key, **kw)
    _, explicit = statistics(w, _H, 0.2, _G, "cont_time", key, est, **kw)
    for k in default:
        np.testing.assert_array_equal(np.asarray(default[k]), np.asarray(explicit[k]), k)
    # the production run's shift, several units away from the estimate
    _, shifted = statistics(w, _H, 0.2, _G, "cont_time", key, alw + 3.0, **kw)
    for k in ("n_unique_post", "offspring_k0", "effective_sample_size", "escape_rate_mean"):
        np.testing.assert_allclose(np.asarray(shifted[k]), np.asarray(default[k]),
                                   rtol=0.15, atol=0.02, err_msg=k)


def test_run_stats_carry_the_diagnostics_and_the_inputs():
    """A driver's stats hold the statistics() keys, the inputs and the autocorrelation times."""
    w = prepare_walkers_sz(64, 8, jax.random.PRNGKey(0), sz=0)
    zz = SpinHalf.create(chain(8).edges, J_z=1.0)
    _, _, _, st, _ = run_gfmc(w, 40, _H, 0.2, _G, [2, 4], rng_key=jax.random.PRNGKey(6),
                              warmup_steps=8, diag_operators={"zz": zz}, verbosity=0)
    for k in _COMMON | _CONT | _LINEAGE:
        assert k in st, k
    for k in ("avg_log_weight", "it_per_sec", "runtime", "total_wall_time",
              "number_of_devices", "autocorrelation/energy_estimator", "autocorrelation/log_weights",
              "autocorrelation/diag/zz", "parameters/rng_key", "parameters/rng_key_next",
              "parameters/n_steps", "parameters/step_size", "parameters/method",
              "parameters/off_diag_equal_windows", "parameters/hamiltonian_hash"):
        assert k in st, k
    assert list(st["ancestor_m"]) == [1, 2]
    assert not any(k.startswith("parameters/prod_key") or k == "production_runtime"
                   for k in st)
