from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Lattice:
    """A finite lattice: a site count, an edge list and optional metadata.

    Attributes:
        name:   lattice family, e.g. ``"chain"``, ``"square"``, ``"pyrochlore"``.
        N:      number of sites.
        edges:  list of ``(i, j)`` site pairs.
        J_x, J_y, J_z: optional suggested couplings (scalar or per-edge list);
                only :func:`chain` sets them, and
                nothing in the package reads them automatically.
        extras: factory-specific data.  The pyrochlore factories store
                ``hexagons``, ``type_per_hexagon``, ``positions``, ``supercell_dims``,
                ``primitive_vectors`` and ``sublattice_offsets``; :func:`kagome`
                stores ``hexagons``, ``hexagon_centers``, ``positions``,
                ``supercell_dims``, ``primitive_vectors``, ``sublattice_offsets``
                and ``torus``.
    """

    name: str
    N: int
    edges: list[tuple[int, int]]
    J_x: float | list[float] | None = None
    J_y: float | list[float] | None = None
    J_z: float | list[float] | None = None
    extras: dict = field(default_factory=dict, compare=False, hash=False)


def chain(n_sites: int, periodic: bool = True) -> Lattice:
    """Nearest-neighbour chain of ``n_sites`` sites.

    The closing bond ``(n_sites-1, 0)`` is added when ``periodic`` and
    ``n_sites > 2``.  Sets the suggested couplings ``J_x = J_y = -1``,
    ``J_z = 1``.
    """
    edges = [(i, i + 1) for i in range(n_sites - 1)]
    if periodic and n_sites > 2:
        edges.append((n_sites - 1, 0))
    return Lattice(name="chain", N=n_sites, edges=edges, J_x=-1.0, J_y=-1.0, J_z=1.0)


def pyrochlore(nx: int, ny: int, nz: int, periodic: bool = True) -> Lattice:
    """Pyrochlore lattice built from explicit FCC unit-cell edge templates (O(N)).

    4 sites per FCC unit cell.  N = 4*nx*ny*nz,  n_edges = N*3 (periodic).
    Cell index: z + y*nz + x*ny*nz  (z changes fastest).  Periodic boundaries
    require nx, ny, nz >= 2.  Positions use a conventional cubic cell of side
    1 (NN distance 1/(2√2)); ``extras`` carries hexagons, positions and the
    supercell (see :class:`Lattice`).  With open boundaries only the hexagons
    lying completely inside the lattice are listed.

    12 edge templates per cell::

        Within A-tetrahedron (offset 0,0,0):
            (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        Inter-cell / B-tetrahedron connections:
            src=1 → tar=0, offset(+x, 0,  0)
            src=2 → tar=0, offset( 0,+y,  0)
            src=3 → tar=0, offset( 0, 0, +z)
            src=2 → tar=1, offset(-x,+y,  0)
            src=3 → tar=1, offset(-x, 0, +z)
            src=3 → tar=2, offset( 0,-y, +z)
    """
    # (src_local, tar_local, offset_x, offset_y, offset_z)
    templates = [
        (0, 1,  0,  0,  0), (0, 2,  0,  0,  0), (0, 3,  0,  0,  0),
        (1, 2,  0,  0,  0), (1, 3,  0,  0,  0), (2, 3,  0,  0,  0),
        (1, 0,  1,  0,  0), (2, 0,  0,  1,  0), (3, 0,  0,  0,  1),
        (2, 1, -1,  1,  0), (3, 1, -1,  0,  1), (3, 2,  0, -1,  1),
    ]

    if periodic and (nx < 2 or ny < 2 or nz < 2):
        raise ValueError(
            f"pyrochlore requires nx, ny, nz >= 2 for periodic boundary conditions "
            f"(got {nx}, {ny}, {nz}). Inter-cell bond templates wrap to the same cell "
            f"for dimensions of size 1, producing duplicate edges and wrong degree-6 connectivity."
        )

    edges: set[tuple[int, int]] = set()

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                ci = z + y * nz + x * ny * nz
                for (s, t, ox, oy, oz) in templates:
                    raw_tx, raw_ty, raw_tz = x + ox, y + oy, z + oz

                    if not periodic:
                        if not (0 <= raw_tx < nx and 0 <= raw_ty < ny and 0 <= raw_tz < nz):
                            continue

                    tx = raw_tx % nx
                    ty = raw_ty % ny
                    tz = raw_tz % nz
                    cj = tz + ty * nz + tx * ny * nz
                    src, tar = ci * 4 + s, cj * 4 + t
                    edges.add((min(src, tar), max(src, tar)))

    N = 4 * nx * ny * nz

    # ── hexagons ──────────────────────────────────────────────────────────
    # 4 hexagon types per unit cell → N hexagons total.
    # Each row lists the 6 site indices forming the hexagonal loop.
    # Templates: (local_site, offset_x, offset_y, offset_z) for each vertex.
    hex_templates = [
        # type 0: local [0,3,1,0,3,1]
        [(0, 0, 0, 0), (3, 0, 0, 0), (1,-1, 0, 1), (0,-1, 0, 1), (3,-1, 0, 0), (1,-1, 0, 0)],
        # type 1: local [0,2,3,0,2,3]
        [(0, 0, 0, 0), (2, 0, 0, 0), (3, 0, 1,-1), (0, 0, 1,-1), (2, 0, 0,-1), (3, 0, 0,-1)],
        # type 2: local [0,2,1,0,2,1]
        [(0, 0, 0, 0), (2, 0, 0, 0), (1,-1, 1, 0), (0,-1, 1, 0), (2,-1, 0, 0), (1,-1, 0, 0)],
        # type 3: local [3,2,1,3,2,1]
        [(3, 0, 0, 0), (2, 0, 0, 0), (1,-1, 1, 0), (3,-1, 1, 0), (2,-1, 0, 1), (1,-1, 0, 1)],
    ]

    # With open boundaries only the hexagons lying completely inside the
    # lattice are kept; with periodic ones there are N of them.
    hexagons, type_per_hexagon = [], []
    edge_set = set(sorted(edges))  # for validation

    for lx in range(nx):
        for ly in range(ny):
            for lz in range(nz):
                for htype, template in enumerate(hex_templates):
                    raw = [(lx + ox, ly + oy, lz + oz) for (_, ox, oy, oz) in template]
                    if not periodic and not all(0 <= cx < nx and 0 <= cy < ny and 0 <= cz < nz
                                                for cx, cy, cz in raw):
                        continue
                    h = [((cz % nz) + (cy % ny) * nz + (cx % nx) * ny * nz) * 4 + ls
                         for (ls, _, _, _), (cx, cy, cz) in zip(template, raw)]
                    # validate: all consecutive pairs must be edges
                    for vi in range(6):
                        a, b = int(h[vi]), int(h[(vi + 1) % 6])
                        assert (min(a, b), max(a, b)) in edge_set, (
                            f"hexagon {len(hexagons)} (type {htype}, cell ({lx},{ly},{lz})): "
                            f"sites {a}-{b} are not connected by an edge"
                        )
                    hexagons.append(h)
                    type_per_hexagon.append(htype)
    hexagons = np.array(hexagons, dtype=np.int32).reshape(-1, 6)
    type_per_hexagon = np.array(type_per_hexagon, dtype=np.int32)

    # ── Cartesian coordinates ────────────────────────────────────────────
    # The pyrochlore lattice is the FCC lattice with a 4-site basis.  The
    # cell stride along the (lx, ly, lz) supercell axes uses the FCC
    # *primitive* vectors a1, a2, a3 (each of length √2/2 in units where the
    # conventional cubic cell has side 1):
    #     a1 = (1/2, 1/2,  0 ),  a2 = (1/2,  0 , 1/2),  a3 = ( 0 , 1/2, 1/2).
    # Sublattice offsets within a cell: pos[s] = a_s / 2 (with pos[0]=0).
    # All NN bonds then have length 1/(2√2) ≈ 0.354.
    a1 = np.array([0.5, 0.5, 0.0], dtype=np.float64)
    a2 = np.array([0.5, 0.0, 0.5], dtype=np.float64)
    a3 = np.array([0.0, 0.5, 0.5], dtype=np.float64)
    primitive = np.stack([a1, a2, a3], axis=0)         # (3, 3) rows = a1, a2, a3
    sublattice_offsets = np.array([
        [0.0,  0.0,  0.0 ],
        [0.25, 0.25, 0.0 ],
        [0.25, 0.0,  0.25],
        [0.0,  0.25, 0.25],
    ], dtype=np.float64)
    positions = np.zeros((N, 3), dtype=np.float64)
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                ci = z + y * nz + x * ny * nz
                cell_origin = x * a1 + y * a2 + z * a3
                positions[ci * 4 : ci * 4 + 4] = cell_origin + sublattice_offsets

    return Lattice(name="pyrochlore", N=N, edges=sorted(edges), extras={
        "hexagons":                  hexagons,
        "type_per_hexagon":          type_per_hexagon,
        "positions":                 positions,
        "supercell_dims":            np.array([nx, ny, nz], dtype=np.int32),
        "primitive_vectors":         primitive,        # (3, 3): rows are a1, a2, a3
        "sublattice_offsets":        sublattice_offsets,
    })


def kagome(Lx: int, Ly: int, periodic: bool = True) -> Lattice:
    """Kagome lattice of ``Lx`` x ``Ly`` unit cells (O(N)).

    A triangular Bravais lattice with primitive vectors ``a1 = (2, 0)``,
    ``a2 = (1, sqrt(3))`` and three sites per cell at ``0``, ``a1/2`` and
    ``a2/2`` (nearest-neighbour distance 1), so ``N = 3*Lx*Ly`` and
    ``n_edges = 2N`` (periodic).  Site index: ``3*(x + Lx*y) + b`` for cell
    ``(x, y)`` and sublattice ``b``.  Periodic boundaries require
    ``Lx, Ly >= 2``.

    ``extras`` carries ``positions`` ``(N, 2)``, ``primitive_vectors``,
    ``supercell_dims``, ``sublattice_offsets``, the torus vectors ``torus``
    (rows ``Lx*a1`` and ``Ly*a2``), and one hexagon per cell:
    ``hexagons`` ``(n_hex, 6)``, the site indices in counter-clockwise order
    starting at angle 0, and ``hexagon_centers`` ``(n_hex, 2)``.  With open
    boundaries only the hexagons fully inside the lattice are listed.
    """
    if periodic and (Lx < 2 or Ly < 2):
        raise ValueError(
            f"kagome requires Lx, Ly >= 2 for periodic boundary conditions "
            f"(got {Lx}, {Ly}); smaller tori wrap a bond onto itself.")

    a1 = np.array([2.0, 0.0])
    a2 = np.array([1.0, np.sqrt(3.0)])
    basis = np.array([[0.0, 0.0], a1 / 2, a2 / 2])
    # A point u*a1 + v*a2 is labelled by (U, V) = (2u, 2v); a site has
    # (U mod 2, V mod 2) = (0, 0), (1, 0) or (0, 1) for sublattice 0, 1, 2.
    sub_of_parity = {(0, 0): 0, (1, 0): 1, (0, 1): 2}

    def index(U, V):
        x, y = U // 2, V // 2
        if not periodic and not (0 <= x < Lx and 0 <= y < Ly):
            return None
        return 3 * ((x % Lx) + Lx * (y % Ly)) + sub_of_parity[(U % 2, V % 2)]

    # bonds: the up triangle inside the cell, and the down triangle that
    # joins sublattice 1 and 2 of one cell to sublattice 0 of the next
    templates = [((0, 0), (1, 0)), ((0, 0), (0, 1)), ((1, 0), (0, 1)),
                 ((1, 0), (2, 0)), ((0, 1), (0, 2)), ((1, 0), (2, -1))]
    edges: set[tuple[int, int]] = set()
    for x in range(Lx):
        for y in range(Ly):
            for (dU1, dV1), (dU2, dV2) in templates:
                i = index(2 * x + dU1, 2 * y + dV1)
                j = index(2 * x + dU2, 2 * y + dV2)
                if i is None or j is None:
                    continue
                edges.add((min(i, j), max(i, j)))

    # hexagon of cell (x, y): centred at (U, V) = (2x+1, 2y+1), vertices at
    # angles 0, 60, ..., 300 degrees
    hex_offsets = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    hexagons, centers = [], []
    for y in range(Ly):
        for x in range(Lx):
            h = [index(2 * x + 1 + dU, 2 * y + 1 + dV) for dU, dV in hex_offsets]
            if None in h:
                continue
            hexagons.append(h)
            centers.append(x * a1 + y * a2 + (a1 + a2) / 2)
    hexagons = np.array(hexagons, dtype=np.int32).reshape(-1, 6)
    for h in hexagons:                      # every hexagon side is a bond
        for k in range(6):
            a, b = int(h[k]), int(h[(k + 1) % 6])
            assert (min(a, b), max(a, b)) in edges, (a, b)

    positions = np.array([x * a1 + y * a2 + basis[b]
                          for y in range(Ly) for x in range(Lx) for b in range(3)])
    return Lattice(
        name="kagome", N=3 * Lx * Ly, edges=sorted(edges),
        extras=dict(positions=positions,
                    primitive_vectors=np.array([a1, a2]),
                    supercell_dims=np.array([Lx, Ly]),
                    sublattice_offsets=basis,
                    torus=np.array([Lx * a1, Ly * a2]),
                    hexagons=hexagons,
                    hexagon_centers=np.array(centers).reshape(-1, 2)))
