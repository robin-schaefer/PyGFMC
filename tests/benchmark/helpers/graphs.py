"""The fixed random connected graphs of the small-system QuSpin benchmark.

One graph per size N (R4, R9 and R13 of the paper), used to build the
Spin/SpinHalf/BoseHubbard test Hamiltonians.  The edge lists are written out
here.  They were drawn once as G(N, m) random graphs with m = max(round(1.5 N),
N - 1) edges (average degree about 3); ``seed`` records the draw and is stored
as provenance in the reference files, which also hold the edges, so the
reference contract checks both.
"""

from __future__ import annotations

from pygfmc.hamiltonian import Lattice

# N -> (seed, edges)
_GRAPHS = {
    4: (1004, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
    9: (1009, [(0, 2), (0, 3), (0, 6), (0, 8), (1, 3), (1, 4), (1, 7), (1, 8),
               (2, 5), (2, 7), (2, 8), (3, 7), (5, 7), (7, 8)]),
    13: (1013, [(0, 6), (0, 10), (1, 3), (1, 5), (1, 6), (1, 10), (1, 11),
                (2, 6), (2, 10), (2, 11), (3, 4), (3, 12), (4, 11), (5, 6),
                (5, 10), (7, 8), (7, 9), (7, 12), (10, 11), (11, 12)]),
}

# N -> (Lattice, seed)
RANDOM_GRAPHS = {N: (Lattice(name="random_graph", N=N, edges=edges), seed)
                 for N, (seed, edges) in _GRAPHS.items()}
