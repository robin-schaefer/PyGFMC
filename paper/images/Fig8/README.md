# Fig. 8 — geometries of the validation lattices

**Figure:** `images/Fig8.pdf` (`\label{fig:test-lattices}`, appendix).

Geometries used in the validation table (`tab:validation-models`).
(a)–(c): the randomly generated connected graphs R4, R9 and R13 with
`N = 4, 9, 13` sites. (d): the two-leg ladder L100 (`N = 100`) with open
boundary conditions; only the four rungs at each open end are drawn, and the
dotted bonds indicate the omitted 42 rungs in between.

## Data: `data.h5`

Sites are 0-based integers; each edge `(i, j)` with `i < j` is one bond.

| dataset | shape | meaning |
|---|---|---|
| `R4/edges`, `R9/edges`, `R13/edges` | `(n_edges, 2)` int64 | bond list of the random graph (6, 14, 20 bonds) |
| `R4/positions`, … | `(N, 2)` float64 | 2D drawing coordinates, row `k` = site `k`: Kamada–Kawai layout (networkx), centred and rescaled into `[-1, 1]^2`. Only for drawing — the graphs have no physical embedding. |
| `L100/edges` | `(148, 2)` int64 | full ladder bond list: legs `(i, i+1)` and `(L+i, L+i+1)` for `i < L−1`, rungs `(i, L+i)`, with `L = 50` |

Attributes: `R<N>` groups carry `N`, `n_edges`, `seed` and `generator`;
`L100` carries `N = 100`, `L_rungs = 50`, `n_edges`, `boundary = "open legs"`
and `site_convention` (top leg `0..L−1`, bottom leg `L..2L−1`).

The ladder panel is schematic: `plot.py` draws the first and last four rungs
of the ladder and reads only `N` from the file; the full edge list is stored so
the benchmarked geometry is documented.

## Provenance

The edge lists are exactly the ones used by the cross-library benchmark suite
(`tests/benchmark/helpers/graphs.py` and `ladder.py` in this package), where the Spin, SpinHalf and BoseHubbard models are compared to
QuSpin exact diagonalisation (random graphs) and TeNPy DMRG (ladder). Each
random graph is `networkx.gnm_random_graph(N, m, seed)` with
`m = max(round(1.5·N), N−1)`, taking the first connected draw starting from
`seed = 1000 + N` (the seed is advanced by one per disconnected draw); the
accepted seeds are 1004, 1009 and 1013, and regenerating the graphs from them
was checked to reproduce the stored edge lists. The layouts were computed with
networkx's deterministic Kamada–Kawai algorithm and stored, so the plot needs
neither networkx nor PyGFMC.

## Regenerate

```bash
python plot.py        # writes ../Fig8.pdf
```

Requires numpy, h5py and matplotlib with a LaTeX installation (`text.usetex = True`).

Width and font sizes come from the shared `../figstyle.py` (all text 8 pt, generated at the
printed width).
