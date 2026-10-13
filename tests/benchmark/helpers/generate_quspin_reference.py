#!/usr/bin/env python3
"""Generate the QuSpin reference data for every entry in ``models.QUSPIN_MODELS``.

For each model: build the PyGFMC Hamiltonian, compute the QuSpin ground
energy and observables (``quspin_ref.compute_reference``), cross-check the
energy against PyGFMC's own ``pygfmc.hamiltonian.exact_diagonalization``, and write
``reference_data/quspin/<id>.h5``.  Needs QuSpin.  Run from ``tests/``:

    python -m benchmark.helpers.generate_quspin_reference [--only <id_substr>]
"""

from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np

from pygfmc.hamiltonian import exact_diagonalization

from benchmark.helpers import quspin_ref as qr
from benchmark.helpers.models import QUSPIN_MODELS

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "reference_data", "quspin")

ATOL_OWN_ED = 1e-6


def _check_own_ed(cfg, h, E0_quspin):
    """PyGFMC's ED (matrix rebuilt from get_rates) must agree with QuSpin."""
    E0_own = float(exact_diagonalization(h)[0][0])
    if abs(E0_quspin - E0_own) >= ATOL_OWN_ED * (abs(E0_own) + 1.0):
        raise AssertionError(f"{cfg['id']}: own-ED E0={E0_own!r} disagrees with "
                             f"QuSpin E0={E0_quspin!r}")


def _write_h5(path, cfg, h, ref):
    with h5py.File(path, "w") as f:
        f.attrs["id"] = cfg["id"]
        f.attrs["family"] = cfg["family"]
        f.attrs["phase"] = cfg["phase"]
        f.attrs["N"] = int(cfg["N"])
        for k, v in cfg["params"].items():
            f.attrs[f"param_{k}"] = v
        if "graph_seed" in cfg:
            f.attrs["graph_seed"] = cfg["graph_seed"]
        edges = getattr(h, "edges", None)
        if edges is not None and len(np.asarray(edges)) > 0:
            f.create_dataset("edges", data=np.asarray(edges, dtype=np.int32))
        for k, v in ref.items():
            f.attrs[k] = v if isinstance(v, str) else float(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None,
                        help="only run models whose id contains this substring")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    models = [m for m in QUSPIN_MODELS if not args.only or args.only in m["id"]]
    print(f"Generating QuSpin reference data for {len(models)} models -> {OUTPUT_DIR}/\n")
    for cfg in models:
        t0 = time.time()
        h, edges = cfg["build"]()
        ref = qr.compute_reference(cfg["family"], h, edges)
        _check_own_ed(cfg, h, ref["E0"])
        _write_h5(os.path.join(OUTPUT_DIR, cfg["id"] + ".h5"), cfg, h, ref)
        print(f"  {cfg['id']:45s}  N={cfg['N']:<4d}  E0={ref['E0']:.8f}  "
              f"({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
