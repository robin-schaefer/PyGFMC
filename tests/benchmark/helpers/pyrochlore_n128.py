"""The N=128 pyrochlore hexagon ring-exchange benchmark.

Ring exchange on the 128 hexagons of a 2x2x2 cubic-cell pyrochlore (N = 128
sites, 384 bonds, 64 tetrahedra), restricted to the spin-ice manifold
(2-in-2-out on every tetrahedron).  Hexagon flips conserve the ice rule and
the topological flux, so the manifold splits into 152 sectors; the reference
lives in the zero-flux sector, the largest, of dimension 81 026 675 568
(~8.1e10) -- far beyond dense ED.

Reference: an external, massively parallel Lanczos diagonalization inside
the ice manifold gives ``E0 = -24.63446378`` for
``H = +1 * sum_hex (ring flip + h.c.)``.  ``PlaquetteModel`` is
``H = -g sum_p flip_p`` with ``g >= 0``, i.e. flip matrix element -1.  The
ring-flip graph on the ice manifold is bipartite, so the gauge
``|c> -> (-1)^f(c) |c>`` maps one sign onto the other and the ground energy
is the same number.

``reference_data/pyrochlore/lattice_N128.h5`` holds the hexagons plus an
ensemble of zero-flux ice configurations used to seed the walkers.  The flux
is exactly conserved, so walkers started elsewhere could never reach the
reference sector and the run would converge smoothly to the wrong energy.
"""

from __future__ import annotations

import itertools
import os

import h5py
import numpy as np

REFERENCE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "reference_data", "pyrochlore",
)
LATTICE_H5 = os.path.join(REFERENCE_DIR, "lattice_N128.h5")

N_SITES = 128
N_HEXAGONS = 128
ZERO_FLUX_SECTOR_DIM = 81026675568

# Ring-exchange sublattice directions: site i of pyrochlore sublattice
# type_per_site[i] carries the diamond-bond vector below, so the total flux
# is B = sum_i sigma_i * e[type(i)] and the zero-flux sector is B == 0.
SUBLATTICE_VECTORS = np.array(
    [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.int64
)

_ALTERNATING = np.array([1, -1, 1, -1, 1, -1], dtype=np.int8)


def flippable_mask(configs, hexagons):
    """(n_cfg, n_hex) bool: hexagon p is flippable in config c.

    Flippable == the six spins strictly alternate around the ring, in the
    order the hexagon is listed -- the same condition ``PlaquetteModel``
    applies, which is why the stored hexagon order must be the cyclic ring
    order (the generator asserts consecutive sites are lattice neighbors).
    """
    configs = np.asarray(configs, dtype=np.int8)
    sub = configs[:, hexagons]                       # (n_cfg, n_hex, 6)
    return (np.all(sub == _ALTERNATING, axis=2)
            | np.all(sub == -_ALTERNATING, axis=2))


def ring_flip_diversify(configs, hexagons, n_sweeps, rng):
    """Decorrelate an ice ensemble by random hexagon flips, in place.

    Each sweep offers every hexagon to every walker and flips it with
    probability 1/2 when flippable.  The proposal is its own inverse and is
    accepted with a walker-independent constant probability, so the chain is
    reversible with respect to the *uniform* distribution over the reachable
    set -- which is precisely one flux sector of the ice manifold.  Two
    consequences the callers rely on:

      * the sector (and the ice constraint) is preserved exactly, by
        construction rather than by a check;
      * the equilibrium distribution is the RK ground state's probability
        distribution, so the same routine also produces the analytic
        reference value for the mu=+1 phase point.

    The per-walker coin is essential: a deterministic "flip if flippable"
    rule evolves every walker identically and returns one distinct
    configuration no matter how many walkers are carried.
    """
    configs = np.asarray(configs, dtype=np.int8).copy()
    n_walkers, n_hex = configs.shape[0], hexagons.shape[0]
    for _ in range(n_sweeps):
        for p in rng.permutation(n_hex):
            sub = configs[:, hexagons[p]]
            flip = ((np.all(sub == _ALTERNATING, axis=1)
                     | np.all(sub == -_ALTERNATING, axis=1))
                    & (rng.random(n_walkers) < 0.5))
            if flip.any():
                idx = np.where(flip)[0]
                configs[np.ix_(idx, hexagons[p])] *= -1
    return configs


def tetrahedra(edges, n_sites=N_SITES):
    """The tetrahedra, derived as the 4-cliques of the bond graph."""
    adj = [set() for _ in range(n_sites)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return np.array(sorted({
        tuple(sorted((i, j, k, m)))
        for i in range(n_sites)
        for j, k, m in itertools.combinations(sorted(adj[i]), 3)
        if k in adj[j] and m in adj[j] and m in adj[k]
    }), dtype=np.int64)


def load_lattice(path=LATTICE_H5):
    """Return (hexagons (128,6) int64, ice_states (n, 128) int8 spins +-1)."""
    with h5py.File(path, "r", locking=False) as f:
        return f["hexagons"][()].astype(np.int64), f["ice_states"][()].astype(np.int8)
