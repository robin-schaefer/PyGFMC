"""run_gfmc vs QuSpin exact diagonalization on small graphs and tori.

Each ``models.QUSPIN_MODELS`` entry runs under its own GFMC settings; the
largest-p energy and every measured observable are compared against the
stored reference (scoring rules in ``helpers/scoring.py``).  Opt-in: set
``RUN_GFMC_BENCH=1``.
"""

from __future__ import annotations

import os

import h5py
import jax
import pytest

from benchmark.helpers import gfmc_bench as gb
from benchmark.helpers.generate_quspin_reference import OUTPUT_DIR
from benchmark.helpers.models import QUSPIN_MODELS
from benchmark.helpers.scoring import check_energy, check_observables
from benchmark.helpers.reference_contract import validate_reference

# set from the worst asserted deviation over the suite (0.035, ToricCode
# <B_p>) plus the measured seed-to-seed spread (<= 0.017)
ATOL_OBS = 0.08

_NON_OBS_ATTRS = {"E0", "N", "id", "family", "phase", "graph_seed"}

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_GFMC_BENCH") != "1",
    reason="GFMC benchmark is opt-in; set RUN_GFMC_BENCH=1 to run",
)

# one case per (model, propagation scheme); the SpinHalf points run all three
_CASES = [(m, meth) for m in QUSPIN_MODELS for meth in gb.methods_for(m)]
_IDS = [f"{m['id']}-{meth}" for m, meth in _CASES]


@pytest.mark.parametrize("cfg,method", _CASES, ids=_IDS)
def test_gfmc_matches_quspin(cfg, method):
    path = os.path.join(OUTPUT_DIR, cfg["id"] + ".h5")
    if not os.path.exists(path):
        pytest.fail(f"missing reference {path}; run "
                    "python -m benchmark.helpers.generate_quspin_reference")

    with h5py.File(path, "r", locking=False) as f:
        validate_reference(f, cfg)
        E0_ref = float(f.attrs["E0"])
        ref_obs = {k: float(f.attrs[k]) for k in f.attrs
                   if k not in _NON_OBS_ATTRS and not k.startswith("param_")}

    energy_results, obs_results = gb.run_benchmark(cfg, jax.random.PRNGKey(0),
                                                   method=method)
    e_gfmc, p_best = gb.best_p_value(energy_results)
    check_energy(cfg, method, e_gfmc, E0_ref, p_best)
    check_observables(cfg, method, ref_obs, obs_results, ATOL_OBS)
