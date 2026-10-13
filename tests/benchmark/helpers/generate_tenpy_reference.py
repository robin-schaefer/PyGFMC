#!/usr/bin/env python3
"""Generate the TeNPy DMRG reference data for every entry in ``models.TENPY_MODELS``.

For each phase point on the L=50 two-leg ladder (100 sites): build the
matching TeNPy model (``ladder.build_ladder_model``), run two-site DMRG, and
store the ground energy plus bulk observables (bond correlators on the middle
rung, away from the open ends) to ``reference_data/tenpy/<id>.h5``.  Needs
TeNPy.  Run from ``tests/``:

    python -m benchmark.helpers.generate_tenpy_reference [--only <id_substr>]
"""

from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np

from benchmark.helpers.ladder import build_ladder_model
from benchmark.helpers.models import LADDER_L, TENPY_MODELS

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "reference_data", "tenpy")

# The stored files record max_chi, chi_saturated and max_trunc_err; use
# --chi-max with --outdir to measure the shift at another cap.
CHI_MAX = 128
DMRG_PARAMS = {
    "trunc_params": {"chi_max": CHI_MAX, "svd_min": 1e-10},
    "max_E_err": 1e-10,
    "max_S_err": 1e-8,
    "max_sweeps": 30,
    "max_hours": 0.35,
    "mixer": True,
}


def _initial_product_state(family, n_sites):
    if family == "BoseHubbard":
        # start near unit filling (n=1 everywhere); DMRG relaxes from there.
        return ["1"] * n_sites
    # Neel-like alternating start -- generic for both FM- and AFM-like phases.
    return ["up", "down"] * (n_sites // 2) + (["up"] if n_sites % 2 else [])


def _bulk_bond_mps_indices(lat, L):
    """MPS indices of the middle rung bond (x=L//2, u=0) <-> (x=L//2, u=1)."""
    x_mid = L // 2
    i0 = lat.lat2mps_idx(np.array([x_mid, 0]))
    j0 = lat.lat2mps_idx(np.array([x_mid, 1]))
    return int(i0), int(j0)


def _observables(family, psi, lat, L):
    i0, j0 = _bulk_bond_mps_indices(lat, L)

    if family in ("SpinHalf", "Spin"):
        z_tot = float(np.real(np.sum(psi.expectation_value("Sz"))))
        pm_bond = float(np.real(
            psi.expectation_value_term([("Sp", i0), ("Sm", j0)])
            + psi.expectation_value_term([("Sm", i0), ("Sp", j0)])
        ))
        obs = dict(z_tot=z_tot, pm_bond=pm_bond)
        if family == "SpinHalf":
            # Sum X_i = 2 * Sum Sx_i (Pauli convention, matches quspin_ref.py's x_tot).
            obs["x_tot"] = 2.0 * float(np.real(np.sum(psi.expectation_value("Sx"))))
        else:
            # Sum (S+_i + S-_i) = 2 * Sum Sx_i, matches quspin_ref.py's pm_tot.
            obs["pm_tot"] = 2.0 * float(np.real(np.sum(psi.expectation_value("Sx"))))
        return obs

    if family == "BoseHubbard":
        n_tot = float(np.real(np.sum(psi.expectation_value("N"))))
        n2_tot = float(np.real(np.sum(psi.expectation_value("NN"))))
        hop_bond = float(np.real(
            psi.expectation_value_term([("Bd", i0), ("B", j0)])
            + psi.expectation_value_term([("B", i0), ("Bd", j0)])
        ))
        pair_bond = float(np.real(
            psi.expectation_value_term([("Bd", i0), ("Bd", j0)])
            + psi.expectation_value_term([("B", i0), ("B", j0)])
        ))
        return dict(n_tot=n_tot, n2_tot=n2_tot, hop_bond=hop_bond, pair_bond=pair_bond)

    raise ValueError(f"unknown family {family!r}")


def _write_h5(path, cfg, E0, obs, dmrg_info):
    assert dmrg_info["max_chi"] > 0, f"{cfg['id']}: DMRG produced max_chi=0 (failed run?)"

    with h5py.File(path, "w") as f:
        f.attrs["id"] = cfg["id"]
        f.attrs["family"] = cfg["family"]
        f.attrs["phase"] = cfg["phase"]
        f.attrs["N"] = int(cfg["N"])
        f.attrs["L"] = LADDER_L
        f.attrs["bc"] = "open_legs"
        for k, v in cfg["params"].items():
            f.attrs[f"param_{k}"] = v
        f.attrs["E0"] = float(E0)
        f.attrs["e_per_site"] = float(E0) / cfg["N"]
        for k, v in obs.items():
            f.attrs[k] = float(v)
        for k, v in dmrg_info.items():
            f.attrs[f"dmrg_{k}"] = v


def main():
    from tenpy.networks.mps import MPS
    from tenpy.algorithms import dmrg

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None,
                         help="only run models whose id contains this substring")
    parser.add_argument("--chi-max", type=int, default=CHI_MAX,
                         help="MPS bond-dimension cap (default %(default)s)")
    parser.add_argument("--outdir", default=OUTPUT_DIR,
                         help="write here instead of reference_data/tenpy "
                              "(use a separate directory for chi-convergence re-runs, "
                              "since filenames are id-keyed only and would overwrite)")
    args = parser.parse_args()

    dmrg_params = dict(DMRG_PARAMS)
    dmrg_params["trunc_params"] = dict(DMRG_PARAMS["trunc_params"], chi_max=args.chi_max)
    outdir = args.outdir

    os.makedirs(outdir, exist_ok=True)
    models = TENPY_MODELS
    if args.only:
        models = [m for m in models if args.only in m["id"]]

    print(f"Generating TeNPy DMRG reference data for {len(models)} models -> {outdir}/\n")
    for cfg in models:
        t0 = time.time()
        M = build_ladder_model(cfg["family"], LADDER_L, cfg["params"], bc_MPS="finite")
        n_sites = M.lat.N_sites
        product_state = _initial_product_state(cfg["family"], n_sites)
        psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc="finite")

        eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
        E0, psi = eng.run()

        obs = _observables(cfg["family"], psi, M.lat, LADDER_L)
        max_chi = int(max(psi.chi))
        dmrg_info = dict(
            max_chi=max_chi,
            chi_max_requested=int(args.chi_max),
            chi_saturated=bool(max_chi >= args.chi_max),
            max_trunc_err=(float(max(eng.trunc_err_list)) if eng.trunc_err_list else 0.0),
            n_sweeps=int(eng.sweeps),
        )

        path = os.path.join(outdir, cfg["id"] + ".h5")
        _write_h5(path, cfg, E0, obs, dmrg_info)

        dt = time.time() - t0
        print(f"  {cfg['id']:40s}  E0={E0:.6f}  e/N={E0/cfg['N']:.6f}  "
              f"chi_max={dmrg_info['max_chi']:<4d} ({dt:.1f}s)")

    print("\nDone.")


if __name__ == "__main__":
    main()
