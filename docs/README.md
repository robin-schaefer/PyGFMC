# PyGFMC documentation

Sphinx sources of the PyGFMC documentation website.

## Building and viewing locally

```bash
pip install -e "./codes" sphinx sphinx_rtd_theme nbsphinx   # from the repository root
cd docs
make html                              # writes _build/html/
xdg-open _build/html/index.html        # or: python -m http.server -d _build/html 8000
make clean                             # removes _build/
```

The build reads the `pygfmc` package from `../codes` (set in `conf.py`), so PyGFMC
itself need not be installed, but its dependencies (`jax`, `numpy`, `scipy`, `h5py`)
must be; installing `./codes` as above provides them. `nbsphinx` also needs
[pandoc](https://pandoc.org) (e.g. `sudo apt-get install pandoc`).

## Files

| File | Content |
|---|---|
| `conf.py` | Sphinx configuration: `sphinx_rtd_theme`, `nbsphinx` (notebooks, not executed during the build), MathJax, intersphinx links to Python, NumPy and JAX |
| `Makefile` | `make html`, `make clean` |
| `index.rst` | Start page: overview, the Bose-Hubbard quickstart, table of contents |
| `GFMC.rst` | About the algorithm |
| `paper.rst` | The paper, embedded as PDF with a download link |
| `installation.rst` | Installation and dependencies |
| `usage.rst` | Running a simulation: the spin-1 transverse-field Ising chain step by step (Hamiltonian, walkers, guide, observables, run, error bars, storing and resuming) |
| `good_simulation.rst` | Running a good simulation: choosing the step size, walkers and projection depth |
| `api.rst` | API reference, generated from the docstrings (`autodoc`, `autosummary`, `napoleon`) |
| `examples.rst` | Overview of the example notebooks in `../notebooks/`: model, guide and observables of each |
| `_static/` | `custom.css`: table layout, `.. container:: center`, and the review markup (`:red:`, `:del:`, `.. container:: red`) |
| `images/` | Figures of the pages (`convergence.*`, `neff_vs_tau.*`, `throughput.svg`, `pyrochlore_transition.svg`); `plot_convergence.py` and `plot_neff_vs_tau.py` regenerate the first two (see each script's docstring), `pyrochlore_transition.svg` is the paper's Fig. 6 converted with `pdftocairo -svg` |

## Publishing

`.github/workflows/docs.yml` builds the documentation with warnings treated as errors
on every push to `main` and publishes it on GitHub Pages at
https://robin-schaefer.github.io/PyGFMC/.

Planned: include the package READMEs (`myst-parser`).
