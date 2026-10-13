"""Accuracy, seed spread and convergence checks, separate from broad regressions.

Tiny TFIMs have unique ground states (both transverse and longitudinal fields).
Independent dense ED supplies energy and magnetization. Three independent runs
at each population distinguish sampling spread from projection/population bias.
Bounds are fixed physical accuracy targets, not fitted to observed seed errors.
"""
import os

import jax
import numpy as np
import pytest

from core.reference import reference_ground_state, reference_matrix
from pygfmc import run_gfmc
from pygfmc.hamiltonian import SpinHalf, chain
from pygfmc.walkers import prepare_walkers_random

pytestmark = [pytest.mark.slow, pytest.mark.skipif(
    os.environ.get("RUN_GFMC_BENCH") != "1", reason="set RUN_GFMC_BENCH=1")]


def _mean_sem(values):
    values = np.asarray(values)
    assert np.isfinite(values).all()
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(len(values))


@pytest.mark.parametrize("hx", [0.8, 1.6])
@pytest.mark.parametrize("method", ["single_step", "multi_step", "cont_time"])
def test_small_system_accuracy_and_convergence(hx, method):
    edges = chain(4, periodic=True).edges
    h = SpinHalf.create(edges, J_z=1.0, h_x=hx, h_z=0.2)
    z = SpinHalf.create(edges, h_z=-1.0)
    e0, psi = reference_ground_state(h)
    reference = np.asarray([e0, psi @ (reference_matrix(z) @ psi)]) / h.N
    populations = []
    for n_walkers in (256, 1024):
        samples = []
        for seed in (17, 29, 43):
            init, key = jax.random.split(jax.random.PRNGKey(seed))
            walkers = prepare_walkers_random(n_walkers, h.N, init)
            _, energies, obs, _, _ = run_gfmc(
                walkers, n_steps=2048, hamiltonian=h, guide=None, lam=3.0,
                step_size=4 / 3 if method == "cont_time" else 4,
                p_array=[8, 16, 32], method=method, rng_key=key,
                diag_operators={"z": z}, warmup_steps=512, verbosity=0)
            samples.append([[float(energies[p]) / h.N,
                             float(np.asarray(obs["z"][p]).item()) / h.N]
                            for p in (8, 16, 32)])
        samples = np.asarray(samples)
        mean, sem = _mean_sem(samples)
        # Fixed accuracy ceilings prevent a noisy run from passing merely
        # because its confidence interval is large. Four SEM is a conservative
        # diagnostic, not a calibrated Gaussian CI from only three seeds.
        accuracy = np.asarray([0.01, 0.02])
        floor = np.asarray([0.002, 0.005])
        assert np.all(sem < accuracy / 2), (method, hx, n_walkers, sem)
        assert np.all(np.abs(mean - reference) < accuracy), (mean, reference, sem)
        assert np.all(np.abs(mean - reference) < floor + 4 * sem), (mean, reference, sem)
        # Paired seeds/windows: check a plateau without demanding monotonic
        # improvement from noisy estimates.
        delta, delta_sem = _mean_sem(samples[:, -1] - samples[:, -2])
        assert np.all(np.abs(delta) < floor + 4 * delta_sem), (delta, delta_sem)
        populations.append(samples[:, -1])
    delta, delta_sem = _mean_sem(populations[1] - populations[0])
    assert np.all(np.abs(delta) < floor + 4 * delta_sem), (delta, delta_sem)
