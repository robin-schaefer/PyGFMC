"""run_gfmc vs TeNPy DMRG on the L=50 two-leg ladder (N=100).

Same scoring as ``test_gfmc_vs_quspin.py`` (``helpers/scoring.py``), with a
looser ``ATOL_OBS``.  Opt-in: set ``RUN_GFMC_BENCH=1``.
"""

from __future__ import annotations

import os

import h5py
import jax
import pytest

from benchmark.helpers import gfmc_bench as gb
from benchmark.helpers.generate_tenpy_reference import OUTPUT_DIR
from benchmark.helpers.models import TENPY_MODELS
from benchmark.helpers.scoring import check_energy, check_observables
from benchmark.helpers.reference_contract import validate_reference

# Looser than the small systems' 0.08: guide-free GFMC cannot build the
# ladder's long-range order from a uniform trial state, and 4x the steps and
# walkers barely move the resulting bias.
ATOL_OBS = 0.15

# Reference observables read from each DMRG file, per family; the scoring
# compares every observable the run measures and fails on a missing reference.
_OBS_KEYS = {
    "SpinHalf": ("z_tot", "pm_bond", "x_tot"),
    "Spin": ("z_tot", "pm_bond", "pm_tot"),
    "BoseHubbard": ("n_tot", "n2_tot", "hop_bond", "pair_bond"),
}

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_GFMC_BENCH") != "1",
    reason="GFMC benchmark is opt-in; set RUN_GFMC_BENCH=1 to run",
)

_CASES = [(m, meth) for m in TENPY_MODELS for meth in gb.methods_for(m)]
_IDS = [f"{m['id']}-{meth}" for m, meth in _CASES]


@pytest.mark.parametrize("cfg,method", _CASES, ids=_IDS)
def test_gfmc_matches_tenpy(cfg, method):
    path = os.path.join(OUTPUT_DIR, cfg["id"] + ".h5")
    if not os.path.exists(path):
        pytest.fail(f"missing reference {path}; run "
                    "python -m benchmark.helpers.generate_tenpy_reference")

    with h5py.File(path, "r", locking=False) as f:
        validate_reference(f, cfg)
        E0_ref = float(f.attrs["E0"])
        ref_obs = {k: float(f.attrs[k]) for k in _OBS_KEYS[cfg["family"]]}

    energy_results, obs_results = gb.run_benchmark(cfg, jax.random.PRNGKey(0),
                                                   method=method)
    e_gfmc, p_best = gb.best_p_value(energy_results)
    check_energy(cfg, method, e_gfmc, E0_ref, p_best)
    check_observables(cfg, method, ref_obs, obs_results, ATOL_OBS)
