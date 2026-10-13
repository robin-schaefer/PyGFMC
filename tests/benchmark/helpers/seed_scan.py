#!/usr/bin/env python3
"""Run the QuSpin and pyrochlore benchmark models at several seeds; write a CSV.

Each benchmark test runs ONE seed, so what a tolerance really has to cover is
the seed-to-seed spread plus any reproducible GFMC bias.  This script
measures both.  Run from ``tests/``:

    python -m benchmark.helpers.seed_scan --seeds 0 1 2 [--only toriccode]
    python -m benchmark.helpers.seed_scan --analyze seed_scan.csv

Columns: id, family, phase, N, seed, quantity, gfmc, ref, reldiff, asserted.
``quantity`` is ``energy`` (deviation per site) or an observable name (the
same deviation convention as the benchmark). Every compared row is asserted;
unavailable references are explicitly reported by the observable contract.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import h5py
import jax
import numpy as np

from benchmark.helpers import gfmc_bench as gb
from benchmark.helpers.generate_quspin_reference import OUTPUT_DIR as QUSPIN_DIR
from benchmark.helpers.models import PYROCHLORE_MODELS, QUSPIN_MODELS
from benchmark.helpers.pyrochlore_n128 import REFERENCE_DIR
from benchmark.helpers.scoring import check_observable_contract, deviation
from benchmark.helpers.reference_contract import validate_reference

_SKIP_ATTRS = {"E0", "N", "id", "family", "phase", "graph_seed"}


def _reference(cfg):
    """(E0, {obs: value}) for one model, from whichever reference dir owns it."""
    if cfg in PYROCHLORE_MODELS:
        path = os.path.join(REFERENCE_DIR, cfg["id"] + ".h5")
        with h5py.File(path, "r", locking=False) as f:
            validate_reference(f, cfg)
            return float(f.attrs["E0"]), {
                k: float(f.attrs[k]) for k in ("flippable_frac", "ring_flip_tot")
                if k in f.attrs
            }
    path = os.path.join(QUSPIN_DIR, cfg["id"] + ".h5")
    with h5py.File(path, "r", locking=False) as f:
        validate_reference(f, cfg)
        return float(f.attrs["E0"]), {
            k: float(f.attrs[k]) for k in f.attrs
            if k not in _SKIP_ATTRS and not k.startswith("param_")
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default="seed_scan.csv")
    args = ap.parse_args()

    models = QUSPIN_MODELS + PYROCHLORE_MODELS
    if args.only:
        models = [m for m in models if args.only in m["id"]]

    rows = []
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "family", "phase", "N", "seed", "quantity",
                    "gfmc", "ref", "reldiff", "asserted"])

        for cfg in models:
            try:
                E0_ref, ref_obs = _reference(cfg)
            except (OSError, KeyError) as exc:
                print(f"  !! {cfg['id']}: no reference ({exc})", flush=True)
                continue

            for seed in args.seeds:
                t0 = time.time()
                energy, obs = gb.run_benchmark(cfg, jax.random.PRNGKey(seed))
                e_gfmc, p_best = gb.best_p_value(energy)
                d = abs(float(e_gfmc) - E0_ref) / cfg["N"]
                w.writerow([cfg["id"], cfg["family"], cfg["phase"], cfg["N"],
                            seed, "energy", float(e_gfmc), E0_ref, d, 1])
                rows.append((cfg["id"], seed, "energy", d))
                print(f"  {cfg['id']:45s} seed={seed} E/site diff={d:.4f} "
                      f"(p={p_best}, {time.time() - t0:.0f}s)", flush=True)

                for name in sorted(check_observable_contract(cfg, ref_obs, obs)):
                    ref_val = ref_obs[name]
                    val, _ = gb.best_p_value(obs[name])
                    val = float(np.ravel(val)[0])
                    rel = (abs(val - ref_val) / max(abs(ref_val), 1e-12)
                           if cfg in PYROCHLORE_MODELS else
                           deviation(name, val, ref_val, cfg["N"]))
                    asserted = 1
                    w.writerow([cfg["id"], cfg["family"], cfg["phase"], cfg["N"],
                                seed, name, val, ref_val, rel, asserted])
                    rows.append((cfg["id"], seed, name, rel))
                fh.flush()

    print(f"\nwrote {args.out} ({len(rows)} rows)")
    analyze(args.out)


def analyze(path):
    """Summarise a scan: worst deviation |dev| (includes reproducible GFMC
    bias) and seed spread (max - min over seeds, the part that is luck).  A
    These diagnostics do not establish convergence or justify automatically
    enlarging tolerances."""
    from collections import defaultdict

    per_case = defaultdict(list)      # (id, quantity) -> [(seed, gfmc, ref, rel)]
    with open(path) as fh:
        for r in csv.DictReader(fh):
            per_case[(r["id"], r["quantity"])].append(
                (int(r["seed"]), float(r["gfmc"]), float(r["ref"]),
                 float(r["reldiff"]), int(r["asserted"]), int(r["N"]), r["family"]))

    print("\n%-45s %-14s %9s %9s %s" % ("model", "quantity", "max|dev|", "spread", "asserted"))
    worst_dev = worst_spread = 0.0
    rows = []
    for (mid, q), vals in sorted(per_case.items()):
        rel = [v[3] for v in vals]
        gf = [v[1] for v in vals]
        asserted = vals[0][4]
        ref, N, family = vals[0][2], vals[0][5], vals[0][6]
        if q == "energy":
            scale = N
        elif family == "PlaquetteModel":
            scale = max(abs(ref), 1e-12)
        elif q.endswith("_tot"):
            scale = N
        elif q.endswith("_bond"):
            scale = 1.0
        else:
            scale = max(abs(ref), 1.0)
        dev, spread = max(rel), (max(gf) - min(gf)) / scale
        rows.append((dev, spread, mid, q, asserted, len(vals)))
        if asserted and q != "energy":
            worst_dev = max(worst_dev, dev)
            worst_spread = max(worst_spread, spread)
    for dev, spread, mid, q, asserted, n in sorted(rows, reverse=True)[:25]:
        print("%-45s %-14s %9.4f %9.4f %s (n=%d)"
              % (mid[:45], q, dev, spread, "yes" if asserted else "no ", n))

    print(f"\nasserted observables: worst |dev| = {worst_dev:.4f}, "
          f"worst seed spread = {worst_spread:.4f}")
    print("  Check projection/population convergence and sampling uncertainty "
          "before changing tolerances; quantities use different normalizations.")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--analyze":
        analyze(sys.argv[2])
    else:
        main()
