# PyGFMC paper

LaTeX sources of the manuscript and, for every data figure, the data and the
script that plots it. Every figure can be regenerated from the files in this
folder alone, without rerunning any simulation.

## Building

```bash
make figures   # regenerate all data figures from images/Fig*/ (writes images/*.pdf)
make paper     # build the manuscript into out/ and copy PyGFMC.pdf here
make all       # both
make clean     # remove out/
```

`make paper` needs a TeX distribution with `latexmk` (the SciPost class and
bibliography style are included). `make figures` needs Python with `numpy`,
`h5py`, `matplotlib` and `scipy`; the figures use LaTeX for their text, so a TeX
installation is needed there too for the exact typography.

## Files

| File | Content |
|---|---|
| `PyGFMC.tex` | Manuscript |
| `PyGFMC.bib` | Bibliography |
| `table_rows.tex` | Rows of the benchmark-model table in the appendix (`\input` by the manuscript) |
| `SciPost.cls`, `SciPost_bibstyle.bst` | SciPost document class and bibliography style |
| `Makefile` | Build targets above |
| `images/` | `Fig3.pdf` … `Fig9.pdf` included by the manuscript, and one folder `Fig3/` … `Fig9/` per data figure |

## Figures

Figures 1 and 2 are TikZ diagrams drawn in `PyGFMC.tex` itself. They have no
data. Each data figure N has a folder `images/FigN/` with the plotted data,
a `plot.py` that reads the bundled data (Fig. 9 shares Fig. 6's data) and writes `images/FigN.pdf` (the file
the manuscript includes), and a `README.md` that describes every dataset, its
units and conventions, and where the data came from.

| Fig. | Label | Folder | Data file | Shows |
|---|---|---|---|---|
| 3 | `fig:throughput` | `images/Fig3/` | `data.csv` | Throughput (iterations/s) vs number of walkers for the L=100 Heisenberg chain on CPU, A100 and H200 |
| 4 | `fig:tfim_critical` | `images/Fig4/` | `data.h5` | Transverse-field Ising chain (N=100, periodic) at the critical point: effective sample size, ZZ correlator and transverse magnetization against the exact result |
| 5 | `fig:tfim_zz_xtot` | `images/Fig5/` | `data.h5` | Transverse-field Ising chain: deviation of the nearest-neighbour ZZ correlator and the transverse magnetization from the exact free-fermion result, vs h_x |
| 6 | `fig:pyro_Spm` | `images/Fig6/` | `data.h5` | Pyrochlore XXZ model: transverse structure factor S±(q=0) vs J_xy/J_z for several system sizes, GFMC and guiding wavefunction |
| 7 | `fig:ring_energy` | `images/Fig7/` | `data.h5` | Pyrochlore ring-exchange model: energy cost of a flux sector (φ_x, 0, 0) for several L (N = 4L³), with the quadratic fit |
| 8 | `fig:test-lattices` | `images/Fig8/` | `data.h5` | Geometries of the benchmark models (appendix table): the random graphs R4, R9, R13 and the L=100 ladder |

| 9 | `fig:pyro_tau` | `images/Fig9/` | `../Fig6/data.h5` | Projection-depth convergence and extrapolation of the N=1024 transverse structure factor |

Notes on the data:

- **Fig. 3:** the H200 points were recovered from the vector paths of the
  published figure, because the per-run H200 table was not available when this
  package was assembled. They carry no error bars. The CPU and A100 points are
  the measured means with standard deviations. `images/Fig3/README.md` gives
  the recovery procedure and its accuracy.
- **Fig. 8:** the stored edge lists are the ones the benchmark suite in
  `../tests/benchmark` uses, so the figure and the benchmark refer to the same
  graphs.
