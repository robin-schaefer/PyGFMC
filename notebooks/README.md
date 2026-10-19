# PyGFMC notebooks

Runnable examples for the `pygfmc` package. They require PyGFMC
(`pip install pygfmc`, or `pip install -e "./codes"` from the repository root)
plus `matplotlib`; the notebooks that train a guide also need the NetKet extra
(`pip install "pygfmc[netket]"`, or `pip install -e "./codes[netket]"`).
Start Jupyter in `notebooks/` so relative data paths resolve.
Each notebook is stored with the outputs of its last full run.

| Notebook | Content | Runtime (4-core CPU) |
|---|---|---|
| `bose_hubbard_chain.ipynb` | The documentation quickstart: a 50-site Bose-Hubbard chain with a pair term, a `JastrowBoseHubbard` guide trained with NetKet, and a GFMC energy comparison with DMRG. | ~9 min |
| `transverse_field_ising_chain.ipynb` | The code of the documentation page *Running a simulation*: the critical spin-1 transverse-field Ising chain with 12 sites, a `JastrowSpin` guide trained with NetKet, diagonal and off-diagonal observables against exact diagonalization, their convergence with the projection depth with error bars, storing and resuming a run, and tuning the step size from the effective sample size versus the projection depth. | ~25 min |
| `xxz_chain.ipynb` | The paper's example: an XXZ chain with random couplings and a field on 20 sites, a `JastrowSpinHalf` guide trained with NetKet, observables against exact diagonalization, the raw per-step series, the final estimates with error bars, and the structure factors S^zz(k) and S^±(π). | ~14 min |
| `n_body_model.ipynb` | User-defined classes from the paper: an n-body Hamiltonian with a field on a 12-site chain with four-site clusters and a magnetization guide, the class protocol, a GFMC run without a guide checked against exact diagonalization, and training the guide with NetKet (the trained guide is not used in a GFMC run). | ~1 min |
| `kagome_xxz.ipynb` | The XXZ model on the kagome lattice with a `ClusterCPS` guide on star-of-David clusters, reduced by D6 and spin-flip symmetry, trained with NetKet, saved and loaded, and run with GFMC with and without the guide on 18 sites (against exact diagonalization) and 108 sites. | ~11 min |
| `toric_code.ipynb` | Kitaev's toric code: the pure code on a 10×10 torus, where GFMC is exact and why, and the convergence of the energy with the projection depth for fields on a 3×3 torus, against exact diagonalization. | ~6 min |
| `ring_model_pyrochlore.ipynb` | Ring exchange on the 128-site pyrochlore in the zero-flux ice sector with `multi_step`, from one ice configuration: energy and flippable fraction swept from pure ring exchange to the Rokhsar-Kivelson point, with the exact energies at both ends. Reads `pyrochlore.h5`. | ~6 min |
