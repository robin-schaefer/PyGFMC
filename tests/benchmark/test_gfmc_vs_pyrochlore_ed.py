"""run_gfmc on the N=128 pyrochlore vs constrained ED / RK analytics.

``test_pyrochlore_reference_data_is_ice`` is cheap and always runs: it
re-derives from the committed lattice file that the hexagons are 6-cycles of
the bond graph and that every stored starting configuration is 2-in-2-out
with zero flux.  A start in the wrong flux sector would converge smoothly to
another sector's energy, since hexagon flips conserve the flux.

``test_gfmc_matches_pyrochlore_reference`` is the opt-in benchmark, with
tight tolerances:

  * ``ring_exchange`` (mu=0) lands ~6e-5/site from the constrained-ED value
    on a Hilbert space of 8.1e10 states;
  * ``rk_point`` (mu=+1) is a zero-variance point: the local energy vanishes
    for every configuration, so E must be exactly 0.0 at every p.
"""

from __future__ import annotations

import os

import h5py
import jax
import numpy as np
import pytest

from benchmark.helpers import gfmc_bench as gb
from benchmark.helpers.models import PYROCHLORE_MODELS
from benchmark.helpers.scoring import check_observable_contract
from benchmark.helpers.reference_contract import validate_reference
from benchmark.helpers.pyrochlore_n128 import (
    LATTICE_H5, N_HEXAGONS, N_SITES, REFERENCE_DIR, SUBLATTICE_VECTORS, tetrahedra,
)

ATOL_E_SITE = 5e-3       # per site; measured deviation is ~6e-5
ATOL_E_RK = 1e-6         # the RK energy is exactly zero, not approximately
ATOL_OBS_REL = 0.01      # relative; measured deviation is ~2e-4

@pytest.mark.skipif(not os.path.exists(LATTICE_H5),
                    reason=f"missing {LATTICE_H5}; run "
                           "python -m benchmark.helpers.generate_pyrochlore_reference")
def test_pyrochlore_reference_data_is_ice():
    with h5py.File(LATTICE_H5, "r", locking=False) as f:
        hexagons = f["hexagons"][()].astype(np.int64)
        edges = f["edges"][()].astype(np.int64)
        states = f["ice_states"][()].astype(np.int64)
        types = f["sublattice_type"][()].astype(np.int64)
        f_tetra = f["tetrahedra"][()].astype(np.int64)

    assert hexagons.shape == (N_HEXAGONS, 6)
    assert states.shape[1] == N_SITES
    assert set(np.unique(states)) == {-1, 1}

    bonds = {(min(a, b), max(a, b)) for a, b in edges}
    for h in hexagons:
        for k in range(6):
            a, b = int(h[k]), int(h[(k + 1) % 6])
            assert (min(a, b), max(a, b)) in bonds, (
                f"hexagon {h} is not a 6-cycle of the bond graph -- "
                "PlaquetteModel's alternating-flippability test would then "
                "not be the ring-exchange move")

    # derived from the bond list, not trusted from the file
    tetra = tetrahedra(edges)
    assert tetra.shape == (64, 4)
    assert np.array_equal(tetra, f_tetra), "stored tetrahedra != bond-graph 4-cliques"
    assert np.all(states[:, tetra].sum(2) == 0), "stored states are not 2-in-2-out"

    flux = np.einsum("sn,nk->sk", states, SUBLATTICE_VECTORS[types])
    assert np.all(flux == 0), (
        "stored states are not in the zero-flux sector "
        f"(max |flux| component {np.abs(flux).max()})"
    )


@pytest.mark.skipif(os.environ.get("RUN_GFMC_BENCH") != "1",
                    reason="GFMC benchmark is opt-in; set RUN_GFMC_BENCH=1 to run")
@pytest.mark.parametrize("cfg", PYROCHLORE_MODELS,
                          ids=[m["id"] for m in PYROCHLORE_MODELS])
def test_gfmc_matches_pyrochlore_reference(cfg):
    path = os.path.join(REFERENCE_DIR, cfg["id"] + ".h5")
    if not os.path.exists(path):
        pytest.fail(f"missing reference {path}; run "
                    "python -m benchmark.helpers.generate_pyrochlore_reference")

    with h5py.File(path, "r", locking=False) as f:
        validate_reference(f, cfg)
        E0_ref = float(f.attrs["E0"])
        ref_obs = {k: float(f.attrs[k]) for k in ("flippable_frac", "ring_flip_tot")
                   if k in f.attrs}

    energy_results, obs_results = gb.run_benchmark(cfg, jax.random.PRNGKey(0))
    e_gfmc, p_best = gb.best_p_value(energy_results)

    print(f"\n  {cfg['id']:35s}  E: gfmc={float(e_gfmc):.6f}  ref={E0_ref:.6f}  (p={p_best})")
    if cfg["phase"] == "rk_point":
        assert set(energy_results) == set(cfg["gfmc"]["p_array"])
        for p, energy in energy_results.items():
            assert abs(float(energy)) < ATOL_E_RK, (
                f"{cfg['id']}: RK energy must vanish at p={p}; got {float(energy)!r}"
            )
    else:
        diff_site = abs(float(e_gfmc) - E0_ref) / cfg["N"]
        assert diff_site < ATOL_E_SITE, (
            f"{cfg['id']}: energy disagrees with the constrained-ED reference: "
            f"gfmc={float(e_gfmc):.6f}, ref={E0_ref:.6f} ({diff_site:.2e}/site)"
        )

    for name in sorted(check_observable_contract(cfg, ref_obs, obs_results)):
        ref_val = ref_obs[name]
        gfmc_val, _ = gb.best_p_value(obs_results[name])
        # The generator already converts ED ring flips to the positive,
        # stoquastic convention. Flippability is intrinsically nonnegative.
        gfmc_val = float(np.ravel(gfmc_val)[0])
        assert gfmc_val >= 0 and ref_val >= 0
        if name == "flippable_frac":
            assert gfmc_val <= 1 and ref_val <= 1
        rel = abs(gfmc_val - ref_val) / max(abs(ref_val), 1e-12)
        print(f"    {name:16s} gfmc={gfmc_val:.6f}  ref={ref_val:.6f}  reldiff={rel:.4f}")
        assert rel < ATOL_OBS_REL, (
            f"{cfg['id']}: observable {name!r} disagrees: "
            f"gfmc={gfmc_val!r}, ref={ref_val!r}"
        )
