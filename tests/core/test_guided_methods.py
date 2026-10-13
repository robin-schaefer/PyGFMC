"""Guided GFMC reaches the exact ground state under ``multi_step`` and ``cont_time``.

``test_guides.py`` runs the spin-1/2 guides under ``single_step`` only; here the
same guides run under ``multi_step`` and the default ``cont_time``, and the
boson guide ``JastrowBoseHubbard`` runs end to end, against exact
diagonalization of a Bose-Hubbard chain.
"""
import jax
import numpy as np
import pytest

from core.guides import GUIDE_NAMES, L, build_guide
from pygfmc import run_gfmc
from pygfmc.hamiltonian import BoseHubbard, chain, exact_diagonalization
from pygfmc.walkers import prepare_walkers_random, prepare_walkers_sz
from pygfmc.wavefunction import JastrowBoseHubbard

LAM = L / 4.0 + 3.0
# (method, step_size): matched projection depth, tau = (p/2) * step_size / lam
METHODS = {"multi_step": 4, "cont_time": 4.0 / LAM}


@pytest.mark.parametrize("method", sorted(METHODS))
@pytest.mark.parametrize("name", sorted(GUIDE_NAMES))
def test_spin_half_guides_under_every_method(name, method, h, e0):
    """E(p=8) within 0.02/site of E0 for every guide under multi_step and cont_time."""
    guide = build_guide(name, h)
    walkers = prepare_walkers_random(1024, L, jax.random.PRNGKey(0))
    _, energy, _, _, _ = run_gfmc(
        walkers, 400, h, METHODS[method], guide, [2, 8], method=method, lam=LAM,
        rng_key=jax.random.PRNGKey(1), warmup_steps=200, verbosity=0)
    e = np.array([float(energy[p]) for p in (2, 8)])
    assert np.isfinite(e).all()
    err = np.abs(e - e0) / L
    assert err[-1] < 0.02, f"{name}/{method}: E(p=8) = {e[-1]:.6f} vs E0 = {e0:.6f}"
    assert err[-1] <= err[0] + 0.01, f"{name}/{method}: deeper window got worse: {err}"


@pytest.mark.parametrize("method,step_size,lam", [("cont_time", 0.1, None),
                                                   ("multi_step", 2, 12.0)])
def test_jastrow_bose_hubbard_matches_ed(method, step_size, lam):
    """A guided Bose-Hubbard chain at unit filling reaches E0 within 0.02/site."""
    n = 5
    lat = chain(n)
    H = BoseHubbard.create(lat.edges, t=1.0, U=3.0, mu=1.0, n_max=2)
    guide = JastrowBoseHubbard(n, lat.edges, beta=0.05, h=0.0, d=0.25)
    E0 = float(exact_diagonalization(H, number_of_states=1)[0][0])
    walkers = prepare_walkers_sz(1024, n, jax.random.PRNGKey(0), sz=n,
                                 states=(0.0, 1.0, 2.0))
    _, energy, _, _, _ = run_gfmc(
        walkers, 600, H, step_size, guide, [2, 8], method=method, lam=lam,
        rng_key=jax.random.PRNGKey(1), warmup_steps=300, verbosity=0)
    err = abs(float(energy[8]) - E0) / n
    assert err < 0.02, f"{method}: E(p=8) = {float(energy[8]):.6f} vs E0 = {E0:.6f}"


def test_gutzwiller_guide_is_not_vacuous():
    """The guide changes the walk: a guided chain differs from the free one with the same key."""
    n = 5
    lat = chain(n)
    H = BoseHubbard.create(lat.edges, t=1.0, U=3.0, mu=1.0, n_max=2)
    walkers = prepare_walkers_sz(128, n, jax.random.PRNGKey(0), sz=n, states=(0.0, 1.0, 2.0))
    common = dict(rng_key=jax.random.PRNGKey(2), warmup_steps=0, avg_log_weight=1.0,
                  verbosity=0)
    free = run_gfmc(walkers, 40, H, 0.1, None, [2], **common)
    guided = run_gfmc(walkers, 40, H, 0.1, JastrowBoseHubbard(n, lat.edges, beta=0.0, d=0.5),
                      [2], **common)
    assert not np.array_equal(np.asarray(free[4]["energy"]), np.asarray(guided[4]["energy"]))
