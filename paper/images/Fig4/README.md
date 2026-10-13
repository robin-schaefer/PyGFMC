# Fig. 4 — TFIM at the critical point (`images/Fig4.pdf`)

Transverse-field Ising model
`H = J_z Σ_i σᶻ_i σᶻ_{i+1} − h_x Σ_i σˣ_i`, `J_z = −1`, `h_x = 1` (critical point),
periodic chain of `N = 100` sites. All observables are Pauli expectation values
(eigenvalues ±1). Continuous-time propagation with a nearest-neighbour Jastrow
guiding wavefunction.

- **(a)** Effective sample size `n_eff` (median over 16 independent runs, log colour
  scale) versus imaginary-time step `β` and projection depth `τ = mβ`. Dashed lines
  are `n_eff` contours; the triangles on the β axis mark the six step sizes used in (b)–(d).
- **(b)** Energy per site minus the exact value, `(E₀ − E_exact)/N` in units of 1e-5,
  versus τ for six step sizes (±SEM bands). Arrows give `n_eff` from (a) at selected τ
  for β = 1.5.
- **(c)** Bond correlator `(1/N) Σ_i ⟨σᶻ_i σᶻ_{i+1}⟩` versus τ; arrows give `n_eff` for β = 0.25.
- **(d)** Transverse magnetisation per site `(1/N) Σ_i ⟨σˣ_i⟩` from off-diagonal
  forward walking (solid, ±SEM bands; curves for β = 1.0 / 1.25 / 1.5 are cut at
  τ = 20 / 10 / 12 where the estimator breaks down). The red line is the same quantity
  derived from the diagonal data of (b)/(c) at β = 1.5 via the exact identity
  `Σ⟨σˣ⟩ = (J_z N ⟨σᶻσᶻ⟩ − E)/h_x` (not independent data; drawn without error bars).

The dashed black line in (b)–(d) is the exact free-fermion result.

## Conventions

- **τ = m·β = (p/2)·β**: projection depth applied to *each* side of the bra–ket;
  `p` is the estimator window length in branching steps, `m = p/2`, β the imaginary
  time per branching step (units of 1/|J_z|). The energy is on the same τ.
- Errors are the standard error of the mean over independent runs,
  `std(ddof=1)/√n_runs` (5 runs per β in (b)–(d)).
- `n_eff = N_w² / Σ_i c_i²` (Kish effective sample size over the descendant counts
  `c_i` of the walkers after m branching steps).

## `data.h5`

Root attributes: `hamiltonian`, `tau_convention`, `error_convention`, `N`, `J_z`, `h_x`.
Every dataset carries a `description` attribute.

`/neff` — panel (a); 131072 walkers; long layout, one row per measured `(β, m)` cell
(75 β values 0.02…1.50 in steps of 0.02 × every integer m with τ ≤ 100; 24478 rows).

| dataset | meaning |
|---|---|
| `beta` | imaginary-time step β (1/|J_z|) |
| `m` | branching steps traced back |
| `tau` | τ = m·β |
| `n_eff_median` | median n_eff over the runs (count) |
| `n_runs` | runs in the median (16) |

`/diagonal` — panels (b), (c) and the red curve of (d); 131072 walkers, 2·10⁵ steps
per run, 5 runs per β, β ∈ {0.25, 0.5, 0.75, 1.0, 1.25, 1.5}, τ ≤ 100. Long layout,
one row per `(β, τ)`; each β has its own τ grid.

| dataset | meaning |
|---|---|
| `beta`, `p`, `tau` | step size, window length, τ = (p/2)·β |
| `energy`, `energy_err` | total energy E of the chain (units |J_z|) and SEM |
| `zz`, `zz_err` | bond-averaged `(1/N) Σ⟨σᶻ_i σᶻ_{i+1}⟩` and SEM |
| `n_runs` | independent runs (5) |

`/offdiagonal` — panel (d); 32768 walkers, 6.4·10⁵ steps per run, off-diagonal
stride 64 (10⁴ measurements), 5 runs per β, same six β, τ ≤ 25. Same layout, with

| dataset | meaning |
|---|---|
| `sx_total`, `sx_total_err` | **extensive** `Σ_i ⟨σˣ_i⟩` and SEM (the plot divides by N) |
| `energy`, `energy_err` | total energy of the same runs (not plotted) |

`/exact` (attributes) — free-fermion (Jordan–Wigner) solution of the periodic N = 100
chain at h_x = 1: `energy` (total), `zz` (bond average), `sx_total` (extensive).

## Provenance

The GFMC data come from three PyGFMC campaigns at h_x = 1 run with the package's
continuous-time propagator: a dense `n_eff` map using the lineage statistics of
`statistics()` (16 seeds started from a stored equilibrated ensemble), a diagonal
β-scan measuring E and ⟨σᶻσᶻ⟩ with the ancestry (forward-walking) estimator, and an
off-diagonal β-scan measuring Σσˣ with shadow-chain forward walking. Per-run results
were combined (median / across-run mean and SEM) into a consolidated TFIM results file,
from which `data.h5` was extracted without further processing (the `n_eff` rows are the
finite, non-empty cells). The exact reference is the free-fermion solution, validated
against brute-force ED to < 1e-9 at N = 8, 10, 12.

## Regenerate

```bash
python plot.py            # writes ../Fig4.pdf (needs a LaTeX install)
python plot.py --no-latex # mathtext fallback, slightly different typography
```

Requires numpy, scipy, h5py, matplotlib. Width and font sizes come from the shared `../figstyle.py` (all text 8 pt, generated at the
printed width).
