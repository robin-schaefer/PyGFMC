#!/usr/bin/env python3
"""Generate the N=128 pyrochlore ring-exchange reference data.

The reference energy is not computed here: it comes from an external Lanczos
diagonalization inside the spin-ice manifold (see ``pyrochlore_n128.py``),
passed as ``--source``.  The script converts it into the suite's format while
re-deriving, from the raw lattice, every property the comparison relies on:

  * the hexagons are listed in cyclic ring order (so ``PlaquetteModel``'s
    alternating-flippability test is the ring-exchange move);
  * the external H is 2 terms of coefficient +1 per hexagon on exactly those
    hexagons, i.e. the PyGFMC model up to the sign gauge;
  * the sector-0 state is 2-in-2-out on all 64 tetrahedra (the 4-cliques of
    the bond graph) with zero flux, and sector 0 is the unique zero-flux and
    the largest sector;
  * the per-hexagon ring-flip expectation values sum to E0.

It then builds the walker ensemble by ring-flip diversification of the
sector-0 representative and writes

  ``lattice_N128.h5``                   hexagons, edges, ice states, ...
  ``pyrochlore_N128_ring_exchange.h5``  g=+1, mu=0   -- E0 from the ED
  ``pyrochlore_N128_rk_point.h5``       g=+1, mu=+1  -- E0 = 0 analytically

At the RK point H = sum_p (|flippable_p><flippable_p| - flip_p) annihilates
the equal-amplitude superposition over the sector, so E0 = 0 exactly and the
observables are uniform sector averages, estimated with the same ring-flip
chain.  Run from ``tests/``:

    python -m benchmark.helpers.generate_pyrochlore_reference --source graph.h5
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np

from benchmark.helpers.pyrochlore_n128 import (
    LATTICE_H5, N_HEXAGONS, N_SITES, REFERENCE_DIR,
    SUBLATTICE_VECTORS, ZERO_FLUX_SECTOR_DIM,
    flippable_mask, ring_flip_diversify, tetrahedra,
)

N_ICE_STATES = 2048        # == the benchmark's n_walkers; ~256 KB as int8
N_EQUIL_SWEEPS = 2000      # ring-flip sweeps used to decorrelate the ensemble
N_RK_SWEEPS = 600          # sweeps for the RK classical average
N_RK_DISCARD = 100


def _check_source(f):
    """Re-derive everything the benchmark depends on; return the pieces."""
    lat, res = f["lattice_gr"], f["results_gr"]
    hexagons = lat["hexagons"][()].astype(np.int64)
    edges = lat["edges"][()].astype(np.int64)
    types = lat["type_per_site"][()].astype(np.int64)

    assert int(lat["N"][()][0]) == N_SITES
    assert hexagons.shape == (N_HEXAGONS, 6), hexagons.shape

    bonds = {(min(a, b), max(a, b)) for a, b in edges}
    assert len(bonds) == 384, len(bonds)
    for h in hexagons:                     # cyclic ring order
        for k in range(6):
            a, b = int(h[k]), int(h[(k + 1) % 6])
            assert (min(a, b), max(a, b)) in bonds, f"hexagon {h} is not a cycle"

    # the external Hamiltonian: 2 terms of coefficient +1 per hexagon
    H = f["graph_gr/H_gr"]
    n_terms = int(H["number_of_terms"][()][0])
    assert n_terms == 2 * N_HEXAGONS, n_terms
    hex_set = {tuple(h) for h in hexagons}
    for t in range(n_terms):
        g = H[f"term{t}_gr"]
        assert float(g["coef"][()][0]) == 1.0
        assert tuple(g["sites_to_act"][()].astype(np.int64)) in hex_set

    tetra = tetrahedra(edges, N_SITES)
    assert tetra.shape == (64, 4), tetra.shape

    states = np.where(res["state_per_sector"][()] == 1, 1, -1).astype(np.int8)
    dims = res["dim_per_sector"][()]
    flux = np.einsum("sn,nk->sk", states.astype(np.int64),
                      SUBLATTICE_VECTORS[types])
    zero = np.where(np.abs(flux).sum(1) == 0)[0]
    assert len(zero) == 1 and zero[0] == 0, f"zero-flux sectors: {zero}"
    assert int(np.argmax(dims)) == 0, "sector 0 is not the largest"
    assert int(dims[0]) == ZERO_FLUX_SECTOR_DIM, int(dims[0])
    assert np.all(states[:, tetra].sum(2) == 0), "sector representatives are not ice"

    E0 = float(res["sector0_gr/E0"][()][0])
    obs = res["sector0_gr/all_obs_per_state"][()][0]     # per-hexagon <flip>
    assert abs(2.0 * obs.sum() - E0) < 1e-9, (2.0 * obs.sum(), E0)

    return hexagons, edges, tetra, types, states[0], E0, obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="graph.h5 of the external constrained-ED run")
    ap.add_argument("--n-states", type=int, default=N_ICE_STATES)
    ap.add_argument("--seed", type=int, default=20260809)
    args = ap.parse_args()

    os.makedirs(REFERENCE_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with h5py.File(args.source, "r") as f:
        hexagons, edges, tetra, types, rep, E0, obs_per_hex = _check_source(f)
    print(f"source checks passed: {args.source}")
    print(f"  E0 = {E0:.8f}   (zero-flux sector, dim {ZERO_FLUX_SECTOR_DIM})")

    # ── GFMC starting ensemble ────────────────────────────────────────────
    configs = np.tile(rep, (args.n_states, 1))
    configs = ring_flip_diversify(configs, hexagons, N_EQUIL_SWEEPS, rng)
    assert np.all(configs[:, tetra].sum(2) == 0), "diversified states left the ice manifold"
    n_unique = len(np.unique(configs, axis=0))
    print(f"  ice ensemble: {n_unique}/{args.n_states} distinct configurations")

    with h5py.File(LATTICE_H5, "w") as f:
        f.create_dataset("hexagons", data=hexagons.astype(np.int32))
        f.create_dataset("edges", data=edges.astype(np.int32))
        f.create_dataset("ice_states", data=configs.astype(np.int8))
        # per-site pyrochlore sublattice label; the flux (and hence the
        # sector) is sum_i sigma_i * SUBLATTICE_VECTORS[type[i]], so this is
        # not derivable from the bond list alone and must be carried along.
        f.create_dataset("sublattice_type", data=types.astype(np.int8))
        f.create_dataset("tetrahedra", data=tetra.astype(np.int32))
        f.attrs["N"] = N_SITES
        f.attrs["n_hexagons"] = N_HEXAGONS
        f.attrs["sector"] = "zero_flux"
        f.attrs["sector_dim"] = ZERO_FLUX_SECTOR_DIM
        f.attrs["n_equil_sweeps"] = N_EQUIL_SWEEPS
        f.attrs["seed"] = args.seed
        f.attrs["source"] = args.source

    # ── mu = 0: the constrained-ED reference ──────────────────────────────
    # PyGFMC's H = -g sum_p flip_p with g = +1 gives a flip matrix element
    # of -1, where the external ED used +1; on the (bipartite) ice flip graph
    # that is a gauge, so E0 is unchanged while every <flip_p> changes sign --
    # hence the explicit minus sign below. Stored ring_flip_tot uses the
    # positive stoquastic convention, which the benchmark checks signed.
    ring_flip_tot = -float(obs_per_hex.sum() * 2.0)
    assert ring_flip_tot >= 0
    with h5py.File(os.path.join(REFERENCE_DIR,
                                 "pyrochlore_N128_ring_exchange.h5"), "w") as f:
        f.attrs["id"] = "pyrochlore_N128_ring_exchange"
        f.attrs["family"] = "PlaquetteModel"
        f.attrs["phase"] = "ring_exchange"
        f.attrs["N"] = N_SITES
        f.attrs["param_g"] = 1.0
        f.attrs["param_mu"] = 0.0
        f.attrs["E0"] = E0
        f.attrs["E0_source"] = "constrained_ED_lanczos_ice_manifold"
        f.attrs["ring_flip_tot"] = ring_flip_tot
        f.create_dataset("ring_flip_per_hexagon", data=-obs_per_hex * 2.0)

    # ── mu = +1: the analytic RK point ────────────────────────────────────
    # <n_flippable> under the uniform (== RK) distribution, blocked over
    # sweeps of the same ring-flip chain.
    cfg = configs.copy()
    per_sweep = []
    for _ in range(N_RK_SWEEPS):
        per_sweep.append(flippable_mask(cfg, hexagons).sum(1).mean())
        cfg = ring_flip_diversify(cfg, hexagons, 1, rng)
    m = np.array(per_sweep[N_RK_DISCARD:])
    blocks = m[: (len(m) // 10) * 10].reshape(10, -1).mean(1)
    n_flip, n_flip_err = blocks.mean(), blocks.std(ddof=1) / np.sqrt(10)
    print(f"  RK <n_flippable> = {n_flip:.5f} +- {n_flip_err:.5f} of {N_HEXAGONS}")

    with h5py.File(os.path.join(REFERENCE_DIR,
                                 "pyrochlore_N128_rk_point.h5"), "w") as f:
        f.attrs["id"] = "pyrochlore_N128_rk_point"
        f.attrs["family"] = "PlaquetteModel"
        f.attrs["phase"] = "rk_point"
        f.attrs["N"] = N_SITES
        f.attrs["param_g"] = 1.0
        f.attrs["param_mu"] = 1.0
        f.attrs["E0"] = 0.0
        f.attrs["E0_source"] = "analytic_rk_sum_of_projectors"
        f.attrs["flippable_frac"] = float(n_flip / N_HEXAGONS)
        f.attrs["flippable_frac_err"] = float(n_flip_err / N_HEXAGONS)
        f.attrs["ring_flip_tot"] = float(n_flip)          # == n_hex * flippable_frac
        f.attrs["ring_flip_tot_err"] = float(n_flip_err)
        f.attrs["obs_source"] = "uniform_sector_average_ring_flip_chain"

    print(f"\nwrote reference_data/pyrochlore/ ({REFERENCE_DIR})")


if __name__ == "__main__":
    main()
