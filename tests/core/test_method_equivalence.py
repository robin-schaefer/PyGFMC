"""single_step and multi_step sample the same process, so expectations must agree.

multi_step draws a geometric hold count instead of testing the diagonal stay
once per transition, so the PRNG streams (and raw series) differ, but
avg_log_weight -> log(lam - E_0), the energy and a diagonal observable must
agree between the two methods.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pygfmc import run_gfmc
from pygfmc.engine.estimators import estimate_average_log_branch_weight
from pygfmc.walkers import prepare_walkers_from_configs
from pygfmc.hamiltonian import SpinHalf, chain, exact_diagonalization

_L = 8
_EDGES = chain(_L).edges
_N_BONDS = len(_EDGES)
# max|H_diag| = |J_z| * L / 4 = 8, so lam = 11 clears it with margin.
_LAM = 11.0
_METHODS = ("single_step", "multi_step")
_STEP_SIZES = (1, 2, 4, 8)
_P = 16
_N_WALKERS = 1024
_N_STEPS = 800
_WARMUP = 300
_SEEDS = (200, 201)


def _model():
    return SpinHalf.create(_EDGES, J_x=0.0, J_y=0.0, J_z=-4.0, h_x=1.5)


def _zz_all_bonds():
    """Sum of S^z S^z over every bond (less seed-sensitive than one bond)."""
    return SpinHalf.create(_EDGES, J_x=0.0, J_y=0.0, J_z=1.0, sign_check=False)


def _run(method, step_size, seed):
    """One run; returns (avg_log_weight, E[p], <SzSz> averaged over bonds)."""
    _, energy, obs, stats, _ = run_gfmc(
        prepare_walkers_from_configs(jnp.ones((_N_WALKERS, _L))), _N_STEPS, _model(),
        step_size, None, [_P], method=method, lam=_LAM,
        diag_operators={"zz": _zz_all_bonds()}, verbosity=0,
        warmup_steps=_WARMUP, rng_key=jax.random.PRNGKey(seed))
    return (float(np.ravel(stats["avg_log_weight"])[0]),
            float(energy[_P]),
            float(np.ravel(obs["zz"][_P])[0]) / _N_BONDS)


@pytest.fixture(scope="module")
def measured():
    """Every (method, step_size, seed) combination, run once for the module."""
    return {(m, ss, s): _run(m, ss, s)
            for m in _METHODS for ss in _STEP_SIZES for s in _SEEDS}


@pytest.fixture(scope="module")
def exact():
    energies, diag, _ = exact_diagonalization(
        _model(), diag_operators={"zz": _zz_all_bonds()}, number_of_states=1)
    return float(energies[0]), float(diag["zz"][0]) / _N_BONDS


def test_methods_are_not_bit_identical():
    """The PRNG streams differ, so a shared seed gives different raw series (asserted on purpose)."""
    common = dict(verbosity=0, warmup_steps=0, avg_log_weight=2.8)
    raws = {}
    for method in _METHODS:
        _, _, _, _, raw = run_gfmc(
            prepare_walkers_from_configs(jnp.ones((256, _L))), 200, _model(), 1,
            None, [8], method=method, lam=_LAM, rng_key=jax.random.PRNGKey(5), **common)
        raws[method] = np.asarray(raw["energy"])
    assert not np.array_equal(raws["single_step"], raws["multi_step"]), (
        "single_step and multi_step produced identical raw series; if the "
        "implementations were changed to share a PRNG stream this test is the "
        "one to update, but agreement of EXPECTATION values is what the other "
        "tests here check")


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("step_size", _STEP_SIZES)
def test_avg_log_weight_matches_log_lam_minus_E0(method, step_size, measured,
                                                 exact):
    """avg_log_weight per sub-step -> log(lam - E_0) for every method and step size."""
    E0, _ = exact
    target = np.log(_LAM - E0)
    for seed in _SEEDS:
        alw = measured[(method, step_size, seed)][0]
        assert alw == pytest.approx(target, abs=0.02), (
            f"{method} ss={step_size} seed={seed}: avg_log_weight {alw:.6f} "
            f"vs log(lam - E0) = {target:.6f}; measured spread over all "
            f"configurations was <= 1.9e-3")


@pytest.mark.parametrize("step_size", _STEP_SIZES)
def test_methods_agree_on_the_energy(step_size, measured, exact):
    """Same expectation value, so the windowed energies must agree."""
    E0, _ = exact
    single = np.mean([measured[("single_step", step_size, s)][1]
                      for s in _SEEDS])
    multi = np.mean([measured[("multi_step", step_size, s)][1]
                     for s in _SEEDS])
    assert single == pytest.approx(multi, abs=0.05), (
        f"ss={step_size}: single_step E={single:.5f} vs multi_step "
        f"E={multi:.5f}")
    # both must also be near the true ground state -- agreeing with each other
    # while both being wrong would pass the comparison above
    for name, value in (("single_step", single), ("multi_step", multi)):
        assert value == pytest.approx(E0, abs=0.05), (
            f"{name} at ss={step_size}: E={value:.5f} vs exact {E0:.5f}")


@pytest.mark.parametrize("step_size", _STEP_SIZES)
def test_methods_agree_on_a_diagonal_observable(step_size, measured):
    """A bond-averaged SzSz agrees between methods (at fixed p both are equally unconverged)."""
    single = np.mean([measured[("single_step", step_size, s)][2]
                      for s in _SEEDS])
    multi = np.mean([measured[("multi_step", step_size, s)][2]
                     for s in _SEEDS])
    assert single == pytest.approx(multi, abs=5e-3), (
        f"ss={step_size}: <SzSz>_avg single_step={single:.6f} vs "
        f"multi_step={multi:.6f}")


def test_the_observable_converges_towards_exact_with_step_size(measured,
                                                               exact):
    """Both methods approach exact SzSz as step_size (projection depth) grows, so agreement is not vacuous."""
    _, exact_zz = exact
    for method in _METHODS:
        devs = [abs(np.mean([measured[(method, ss, s)][2] for s in _SEEDS])
                    - exact_zz) for ss in _STEP_SIZES]
        assert devs[0] > devs[-1], (
            f"{method}: deviation from exact did not shrink with step size, "
            f"{[f'{d:.4f}' for d in devs]}")
        assert devs[-1] < 0.01, (
            f"{method}: still {devs[-1]:.4f} from exact at the largest step "
            f"size")


def test_cont_time_lam_none_reports_statistics(capsys):
    """cont_time takes lam=None; the end-of-run statistics report prints it.

    The report runs after the whole chain and only at verbosity >= 1, so this
    checks that it formats an absent ``lam`` ("lam: not used") instead of
    raising at the end of a run.
    """
    w = prepare_walkers_from_configs(jnp.ones((64, _L)))
    run_gfmc(w, 20, _model(), 0.25, None, [2, 4], method="cont_time",
             rng_key=jax.random.PRNGKey(0), verbosity=1,
             avg_log_weight=estimate_average_log_branch_weight(
                 w, _model(), None, method="cont_time"))
    assert "lam: not used" in capsys.readouterr().out
