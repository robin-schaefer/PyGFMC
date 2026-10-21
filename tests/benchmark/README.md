# Benchmark suite

End-to-end checks of `run_gfmc` against independent reference results for all
five Hamiltonian classes (`Spin`, `SpinHalf`, `BoseHubbard`, `ToricCode`,
`PlaquetteModel`).  Every physics case runs a real GFMC simulation without a guiding
wavefunction (`guide=None`) and compares the largest-p energy, and the
diagonal and off-diagonal (forward-walked) observables, against a stored
reference.  The suite is opt-in and takes about 140 minutes.

The additional multi-seed small-system tests described below construct their
own independent dense-ED references rather than reading stored files.

## Running

From the `tests/` folder:

```bash
RUN_GFMC_BENCH=1 pytest benchmark -q                   # all 72 cases, about 140 minutes
RUN_GFMC_BENCH=1 pytest benchmark -q -k graphN4        # a subset
pytest benchmark -q                                    # only the cheap ice-rule check
```

Without `RUN_GFMC_BENCH=1` every GFMC case is skipped.  The tests read only
the committed `reference_data/*.h5` files, so neither QuSpin nor TeNPy is
needed to run them; the random graphs are fixed edge lists in `helpers/graphs.py`.  Each broad regression case
uses one seed (`PRNGKey(0)`); `helpers/seed_scan.py` runs the models at
several seeds and separates reproducible GFMC bias from seed-to-seed spread,
which is how the tolerances were calibrated.

## Reference data

References come from three independent sources:

- **QuSpin exact diagonalization** (`helpers/generate_quspin_reference.py`):
  one reproducible random connected graph per size (N = 4, 9, 13;
  `helpers/graphs.py`) for Spin (S=1), SpinHalf and BoseHubbard
  (n_max = 2), and 2x2, 2x3, 2x4 tori for ToricCode.  QuSpin builds the
  Hamiltonian from the model's couplings, finds the ground state by sparse
  Lanczos and evaluates the observables on it.  The generator also checks the
  energy against PyGFMC's own `pygfmc.hamiltonian.exact_diagonalization`
  (agreement to 1e-6 relative).
- **TeNPy DMRG** (`helpers/generate_tenpy_reference.py`): the L = 50 two-leg
  ladder with open legs (N = 100), two-site DMRG at bond dimension up to 128
  (256 for `bosehubbard_ladderL50_superfluid` and
  `spin_ladderL50_fm_sign_isotropic`), bond observables on the
  middle rung.  The truncation data are stored with each file
  (`dmrg_max_chi`, `dmrg_chi_saturated`, `dmrg_max_trunc_err`).
- **Pyrochlore ring exchange** (`helpers/generate_pyrochlore_reference.py`):
  hexagon ring exchange on the N = 128 pyrochlore (2x2x2 cubic cells),
  restricted to the zero-flux sector of the spin-ice manifold, of dimension
  8.1e10.  At mu = 0 the energy comes from an external massively parallel
  Lanczos diagonalization inside the ice manifold, E0 = -24.63446378.  At the
  Rokhsar-Kivelson point mu = +1 the Hamiltonian is a sum of projectors that
  annihilate the equal-amplitude superposition over the sector, so E0 = 0
  exactly and the observables are classical sector averages, estimated with
  a uniform ring-flip Markov chain.  The walkers start from a stored ensemble
  of zero-flux ice configurations, since the flux is conserved.

The generators need QuSpin, TeNPy, or the external ED file respectively, and
are run by hand from `tests/`, e.g.
`python -m benchmark.helpers.generate_quspin_reference --only spinhalf`.

## Scoring

- Energy: per site, `|E_gfmc - E_ref| / N < 0.05` (QuSpin, TeNPy);
  `< 5e-3` per site for the pyrochlore at mu = 0, and `|E| < 1e-6` at the
  RK point, where the local energy vanishes for every configuration.
- Extensive observables (`*_tot`): per site, `< 0.08` (QuSpin) or `< 0.15`
  (TeNPy ladder, where guide-free GFMC carries a saturating bias).
- Bond correlators (`*_bond`): `|diff| < 0.08 + 0.03 |ref|`.
- Other intensive observables (`Bp_avg`, `As_avg`): `|diff| / max(|ref|, 1)`
  against the same 0.08.
- Pyrochlore observables: relative, `< 0.01`.
- Zero references and zero-field magnetizations are asserted too, using the
  same absolute/per-site bounds. No symmetry-based blanket exclusions apply.
- Missing measured outputs or required references fail. The sole explicit
  unavailable reference is `flippable_frac` at the pyrochlore ring-exchange
  point: the ED export does not contain it.
- RK energy is checked at every requested window. Pyrochlore observables are
  compared with their signs intact; stored ring flips already use the positive
  stoquastic convention.

Off-diagonal observables are measured on one representative model per
family on the small systems (the first of each family), on the RK point for
the pyrochlore, and not at all on the N = 100 ladder, because forward walking
costs about p/2 extra steps per step.  The SpinHalf phase points run under
all three propagators (`single_step`, `multi_step`, `cont_time`, at matched
projection depth), everything else under `multi_step`.

Observables: `z_tot` = Σ S^z, `x_tot` = Σ σ^x (SpinHalf, ToricCode),
`pm_tot` = Σ (S⁺ + S⁻) (Spin), `pm_bond` = S⁺S⁻ + h.c. on the first edge,
`n_tot` = Σ n, `n2_tot` = Σ n², `hop_bond` = b†b + h.c. and `pair_bond` =
b†b† + h.c. on the first edge, `Bp_avg`/`As_avg` = mean plaquette/star
stabilizer, `flippable_frac` = fraction of flippable hexagons,
`ring_flip_tot` = Σ_p (ring flip).

## Multi-seed accuracy and convergence

`test_small_system_accuracy.py` adds six cases: two four-site TFIM field
strengths under all three propagators. Each uses seeds 17, 29 and 43 at
256 and 1024 walkers, checking windows 8, 16 and 32 against independent dense
ED. Energy and magnetization have fixed per-site accuracy ceilings of 0.01
and 0.02, respectively, with separate seed-spread, window-plateau and
population-change checks. The four-SEM diagnostic is not a calibrated
confidence interval from only three seeds. These tests complement, rather
than tighten by fiat, the broad regression tolerances above. They are marked
`slow`, so `pytest benchmark -m "not slow"` skips them.

## Reference compatibility

Missing reference files fail when benchmarks are requested. Stored couplings,
identity and available geometry metadata are checked before each simulation.
`reference_data/contracts.json` additionally pins reference SHA-256 hashes,
geometry, initialization sectors and normalization conventions. This records
an audited pairing of existing files, not regenerated ED/DMRG evidence.
After regenerating and reviewing references, explicitly refresh the manifest:

```bash
python -m benchmark.helpers.reference_contract
```

The manifest also stores the SHA-256 of the source of
`helpers/gfmc_bench.py::measurement_operators` (`measurement_protocol_sha256`),
so any edit to that function, a comment or a blank line included, fails every
contract until the manifest is re-pinned with the command above.

The cheap contract tests in `core/test_benchmark_contracts.py` run in regular CI.

## The shift `lam`

`helpers/models.py` sets `lam` to the class's analytic bound on the diagonal
element plus `MARGIN = 3`.  The SpinHalf entries (TFIM and Heisenberg points,
on the random graphs and the ladder) pass `h_x` to `_lam_spinhalf` as
`extra_margin_field`: the SpinHalf diagonal holds no `h_x` term, so their
`lam` carries an extra `|h_x| N / 2` on top of `MARGIN`.  The tolerances above
were calibrated with this `lam`.

## Stored-reference test cases

| # | Test function | Parametrized id | Reference | N | Asserted observables | Unavailable reference | Energy tol. |
|---|---|---|---|---|---|---|---|
| 1 | `test_pyrochlore_reference_data_is_ice` | - | stored lattice file | 128 | ice rule, zero flux, 6-cycles | - | - |
| 2 | `test_gfmc_matches_pyrochlore_reference` | `pyrochlore_N128_ring_exchange` | constrained Lanczos | 128 | - | `flippable_frac` | 5e-3 / site |
| 3 | `test_gfmc_matches_pyrochlore_reference` | `pyrochlore_N128_rk_point` | RK analytics | 128 | `flippable_frac`, `ring_flip_tot` | - | \|E\| < 1e-6 |
| 4 | `test_gfmc_matches_quspin` | `spin_graphN4_fm_sign_isotropic-multi_step` | QuSpin ED | 4 | `pm_bond`, `pm_tot`, `z_tot` | - | 0.05 / site |
| 5 | `test_gfmc_matches_quspin` | `spin_graphN4_field_polarized-multi_step` | QuSpin ED | 4 | `z_tot` | - | 0.05 / site |
| 6 | `test_gfmc_matches_quspin` | `spin_graphN4_transverse_coverage-multi_step` | QuSpin ED | 4 | `z_tot` | - | 0.05 / site |
| 7 | `test_gfmc_matches_quspin` | `spin_graphN9_fm_sign_isotropic-multi_step` | QuSpin ED | 9 | `z_tot` | - | 0.05 / site |
| 8 | `test_gfmc_matches_quspin` | `spin_graphN9_field_polarized-multi_step` | QuSpin ED | 9 | `z_tot` | - | 0.05 / site |
| 9 | `test_gfmc_matches_quspin` | `spin_graphN9_transverse_coverage-multi_step` | QuSpin ED | 9 | `z_tot` | - | 0.05 / site |
| 10 | `test_gfmc_matches_quspin` | `spinhalf_graphN4_tfim_disordered-single_step` | QuSpin ED | 4 | `pm_bond`, `x_tot`, `z_tot` | - | 0.05 / site |
| 11 | `test_gfmc_matches_quspin` | `spinhalf_graphN4_tfim_disordered-multi_step` | QuSpin ED | 4 | `pm_bond`, `x_tot`, `z_tot` | - | 0.05 / site |
| 12 | `test_gfmc_matches_quspin` | `spinhalf_graphN4_tfim_disordered-cont_time` | QuSpin ED | 4 | `pm_bond`, `x_tot`, `z_tot` | - | 0.05 / site |
| 13 | `test_gfmc_matches_quspin` | `spinhalf_graphN4_field_polarized-single_step` | QuSpin ED | 4 | `z_tot` | - | 0.05 / site |
| 14 | `test_gfmc_matches_quspin` | `spinhalf_graphN4_field_polarized-multi_step` | QuSpin ED | 4 | `z_tot` | - | 0.05 / site |
| 15 | `test_gfmc_matches_quspin` | `spinhalf_graphN4_field_polarized-cont_time` | QuSpin ED | 4 | `z_tot` | - | 0.05 / site |
| 16 | `test_gfmc_matches_quspin` | `spinhalf_graphN4_fm_sign_isotropic-single_step` | QuSpin ED | 4 | `z_tot` | - | 0.05 / site |
| 17 | `test_gfmc_matches_quspin` | `spinhalf_graphN4_fm_sign_isotropic-multi_step` | QuSpin ED | 4 | `z_tot` | - | 0.05 / site |
| 18 | `test_gfmc_matches_quspin` | `spinhalf_graphN4_fm_sign_isotropic-cont_time` | QuSpin ED | 4 | `z_tot` | - | 0.05 / site |
| 19 | `test_gfmc_matches_quspin` | `spinhalf_graphN9_tfim_disordered-single_step` | QuSpin ED | 9 | `z_tot` | - | 0.05 / site |
| 20 | `test_gfmc_matches_quspin` | `spinhalf_graphN9_tfim_disordered-multi_step` | QuSpin ED | 9 | `z_tot` | - | 0.05 / site |
| 21 | `test_gfmc_matches_quspin` | `spinhalf_graphN9_tfim_disordered-cont_time` | QuSpin ED | 9 | `z_tot` | - | 0.05 / site |
| 22 | `test_gfmc_matches_quspin` | `spinhalf_graphN9_field_polarized-single_step` | QuSpin ED | 9 | `z_tot` | - | 0.05 / site |
| 23 | `test_gfmc_matches_quspin` | `spinhalf_graphN9_field_polarized-multi_step` | QuSpin ED | 9 | `z_tot` | - | 0.05 / site |
| 24 | `test_gfmc_matches_quspin` | `spinhalf_graphN9_field_polarized-cont_time` | QuSpin ED | 9 | `z_tot` | - | 0.05 / site |
| 25 | `test_gfmc_matches_quspin` | `spinhalf_graphN9_fm_sign_isotropic-single_step` | QuSpin ED | 9 | `z_tot` | - | 0.05 / site |
| 26 | `test_gfmc_matches_quspin` | `spinhalf_graphN9_fm_sign_isotropic-multi_step` | QuSpin ED | 9 | `z_tot` | - | 0.05 / site |
| 27 | `test_gfmc_matches_quspin` | `spinhalf_graphN9_fm_sign_isotropic-cont_time` | QuSpin ED | 9 | `z_tot` | - | 0.05 / site |
| 28 | `test_gfmc_matches_quspin` | `spinhalf_graphN13_tfim_disordered-single_step` | QuSpin ED | 13 | `z_tot` | - | 0.05 / site |
| 29 | `test_gfmc_matches_quspin` | `spinhalf_graphN13_tfim_disordered-multi_step` | QuSpin ED | 13 | `z_tot` | - | 0.05 / site |
| 30 | `test_gfmc_matches_quspin` | `spinhalf_graphN13_tfim_disordered-cont_time` | QuSpin ED | 13 | `z_tot` | - | 0.05 / site |
| 31 | `test_gfmc_matches_quspin` | `spinhalf_graphN13_field_polarized-single_step` | QuSpin ED | 13 | `z_tot` | - | 0.05 / site |
| 32 | `test_gfmc_matches_quspin` | `spinhalf_graphN13_field_polarized-multi_step` | QuSpin ED | 13 | `z_tot` | - | 0.05 / site |
| 33 | `test_gfmc_matches_quspin` | `spinhalf_graphN13_field_polarized-cont_time` | QuSpin ED | 13 | `z_tot` | - | 0.05 / site |
| 34 | `test_gfmc_matches_quspin` | `spinhalf_graphN13_fm_sign_isotropic-single_step` | QuSpin ED | 13 | `z_tot` | - | 0.05 / site |
| 35 | `test_gfmc_matches_quspin` | `spinhalf_graphN13_fm_sign_isotropic-multi_step` | QuSpin ED | 13 | `z_tot` | - | 0.05 / site |
| 36 | `test_gfmc_matches_quspin` | `spinhalf_graphN13_fm_sign_isotropic-cont_time` | QuSpin ED | 13 | `z_tot` | - | 0.05 / site |
| 37 | `test_gfmc_matches_quspin` | `bosehubbard_graphN4_superfluid-multi_step` | QuSpin ED | 4 | `hop_bond`, `n2_tot`, `n_tot`, `pair_bond` | - | 0.05 / site |
| 38 | `test_gfmc_matches_quspin` | `bosehubbard_graphN4_eta_coverage-multi_step` | QuSpin ED | 4 | `n2_tot`, `n_tot` | - | 0.05 / site |
| 39 | `test_gfmc_matches_quspin` | `bosehubbard_graphN9_superfluid-multi_step` | QuSpin ED | 9 | `n2_tot`, `n_tot` | - | 0.05 / site |
| 40 | `test_gfmc_matches_quspin` | `bosehubbard_graphN9_eta_coverage-multi_step` | QuSpin ED | 9 | `n2_tot`, `n_tot` | - | 0.05 / site |
| 41 | `test_gfmc_matches_quspin` | `toriccode_Lx2Ly2_pure_topological-multi_step` | QuSpin ED | 8 | `As_avg`, `Bp_avg`, `x_tot` | - | 0.05 / site |
| 42 | `test_gfmc_matches_quspin` | `toriccode_Lx2Ly2_weak_field_topological-multi_step` | QuSpin ED | 8 | `Bp_avg` | - | 0.05 / site |
| 43 | `test_gfmc_matches_quspin` | `toriccode_Lx2Ly2_repo_anchored_field-multi_step` | QuSpin ED | 8 | `Bp_avg` | - | 0.05 / site |
| 44 | `test_gfmc_matches_quspin` | `toriccode_Lx2Ly2_deep_polarized-multi_step` | QuSpin ED | 8 | `Bp_avg` | - | 0.05 / site |
| 45 | `test_gfmc_matches_quspin` | `toriccode_Lx2Ly3_pure_topological-multi_step` | QuSpin ED | 12 | `Bp_avg` | - | 0.05 / site |
| 46 | `test_gfmc_matches_quspin` | `toriccode_Lx2Ly3_weak_field_topological-multi_step` | QuSpin ED | 12 | `Bp_avg` | - | 0.05 / site |
| 47 | `test_gfmc_matches_quspin` | `toriccode_Lx2Ly3_repo_anchored_field-multi_step` | QuSpin ED | 12 | `Bp_avg` | - | 0.05 / site |
| 48 | `test_gfmc_matches_quspin` | `toriccode_Lx2Ly3_deep_polarized-multi_step` | QuSpin ED | 12 | `Bp_avg` | - | 0.05 / site |
| 49 | `test_gfmc_matches_quspin` | `toriccode_Lx2Ly4_pure_topological-multi_step` | QuSpin ED | 16 | `Bp_avg` | - | 0.05 / site |
| 50 | `test_gfmc_matches_quspin` | `toriccode_Lx2Ly4_weak_field_topological-multi_step` | QuSpin ED | 16 | `Bp_avg` | - | 0.05 / site |
| 51 | `test_gfmc_matches_quspin` | `toriccode_Lx2Ly4_repo_anchored_field-multi_step` | QuSpin ED | 16 | `Bp_avg` | - | 0.05 / site |
| 52 | `test_gfmc_matches_quspin` | `toriccode_Lx2Ly4_deep_polarized-multi_step` | QuSpin ED | 16 | `Bp_avg` | - | 0.05 / site |
| 53 | `test_gfmc_matches_tenpy` | `spin_ladderL50_fm_sign_isotropic-multi_step` | TeNPy DMRG | 100 | `z_tot` | - | 0.05 / site |
| 54 | `test_gfmc_matches_tenpy` | `spin_ladderL50_field_polarized-multi_step` | TeNPy DMRG | 100 | `z_tot` | - | 0.05 / site |
| 55 | `test_gfmc_matches_tenpy` | `spin_ladderL50_transverse_coverage-multi_step` | TeNPy DMRG | 100 | `z_tot` | - | 0.05 / site |
| 56 | `test_gfmc_matches_tenpy` | `spinhalf_ladderL50_tfim_disordered-single_step` | TeNPy DMRG | 100 | `z_tot` | - | 0.05 / site |
| 57 | `test_gfmc_matches_tenpy` | `spinhalf_ladderL50_tfim_disordered-multi_step` | TeNPy DMRG | 100 | `z_tot` | - | 0.05 / site |
| 58 | `test_gfmc_matches_tenpy` | `spinhalf_ladderL50_tfim_disordered-cont_time` | TeNPy DMRG | 100 | `z_tot` | - | 0.05 / site |
| 59 | `test_gfmc_matches_tenpy` | `spinhalf_ladderL50_field_polarized-single_step` | TeNPy DMRG | 100 | `z_tot` | - | 0.05 / site |
| 60 | `test_gfmc_matches_tenpy` | `spinhalf_ladderL50_field_polarized-multi_step` | TeNPy DMRG | 100 | `z_tot` | - | 0.05 / site |
| 61 | `test_gfmc_matches_tenpy` | `spinhalf_ladderL50_field_polarized-cont_time` | TeNPy DMRG | 100 | `z_tot` | - | 0.05 / site |
| 62 | `test_gfmc_matches_tenpy` | `spinhalf_ladderL50_fm_sign_isotropic-single_step` | TeNPy DMRG | 100 | `z_tot` | - | 0.05 / site |
| 63 | `test_gfmc_matches_tenpy` | `spinhalf_ladderL50_fm_sign_isotropic-multi_step` | TeNPy DMRG | 100 | `z_tot` | - | 0.05 / site |
| 64 | `test_gfmc_matches_tenpy` | `spinhalf_ladderL50_fm_sign_isotropic-cont_time` | TeNPy DMRG | 100 | `z_tot` | - | 0.05 / site |
| 65 | `test_gfmc_matches_tenpy` | `bosehubbard_ladderL50_superfluid-multi_step` | TeNPy DMRG | 100 | `n_tot`, `n2_tot` | - | 0.05 / site |
| 66 | `test_gfmc_matches_tenpy` | `bosehubbard_ladderL50_eta_coverage-multi_step` | TeNPy DMRG | 100 | `n_tot`, `n2_tot` | - | 0.05 / site |

GFMC settings per model (λ, step size, steps, walkers, p grid, walker
initialization) are in `helpers/models.py`.  Per-model overrides of the shared
settings are listed there with their reason: the three ladder models, and
`spin_graphN4_fm_sign_isotropic`, which runs with `step_size=4` and 512 walkers
because its `pm_tot` leaves the S^z_tot = 0 sector and converges only at a
projection depth τ ≈ 14 (p = 64).

## Files

Reference data (`reference_data/`):

| Path | Content |
|---|---|
| `quspin/<id>.h5` (31 files) | QuSpin E0 and observables, couplings as `param_*`, graph seed and edges |
| `tenpy/<id>.h5` (8 files) | DMRG E0, observables and truncation data for the L = 50 ladder |
| `pyrochlore/lattice_N128.h5` | hexagons, bonds, tetrahedra, sublattice labels and 2048 zero-flux ice configurations |
| `pyrochlore/pyrochlore_N128_ring_exchange.h5` | mu = 0: constrained-Lanczos E0 and per-hexagon ring-flip values |
| `pyrochlore/pyrochlore_N128_rk_point.h5` | mu = +1: E0 = 0 and the sector averages of `flippable_frac`, `ring_flip_tot` |

Tests and helpers:

| File | Role |
|---|---|
| `test_gfmc_vs_quspin.py` | GFMC vs QuSpin ED on small graphs and tori |
| `test_gfmc_vs_tenpy.py` | GFMC vs TeNPy DMRG on the L = 50 ladder |
| `test_gfmc_vs_pyrochlore_ed.py` | ice-rule check of the stored lattice, and GFMC vs the N = 128 pyrochlore references |
| `test_small_system_accuracy.py` | independent ED, three seeds, two walker counts and three projection windows |
| `helpers/models.py` | model registry: phase points, λ, GFMC settings, sector pinning |
| `helpers/gfmc_bench.py` | walker initialization, measurement operators, `run_gfmc` driver |
| `helpers/scoring.py` | tolerances and observable scoring for the QuSpin/TeNPy tests |
| `helpers/reference_contract.py` | reference metadata, geometry, sector, measurement-protocol and file-hash checks |
| `helpers/graphs.py` | the fixed random connected graphs R4, R9, R13 as edge lists |
| `helpers/native_lattices.py` | ToricCode torus sizes |
| `helpers/ladder.py` | two-leg ladder edges and the matching TeNPy models |
| `helpers/pyrochlore_n128.py` | N = 128 pyrochlore constants, lattice loader, flippability and ring-flip chain |
| `helpers/quspin_ref.py` | QuSpin Hamiltonians and observables per family |
| `helpers/generate_quspin_reference.py` | writes `reference_data/quspin/` (needs QuSpin) |
| `helpers/generate_tenpy_reference.py` | writes `reference_data/tenpy/` (needs TeNPy) |
| `helpers/generate_pyrochlore_reference.py` | writes `reference_data/pyrochlore/` from the external ED file |
| `helpers/seed_scan.py` | multi-seed runs to CSV, with `--analyze` for bias vs spread |
