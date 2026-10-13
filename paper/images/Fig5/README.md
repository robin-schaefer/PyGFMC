# Fig. 5 — TFIM observables versus transverse field (`images/Fig5.pdf`)

Transverse-field Ising model
`H = J_z Σ_i σᶻ_i σᶻ_{i+1} − h_x Σ_i σˣ_i`, `J_z = −1`, periodic chain of `N = 100`
sites, at 11 fields `h_x = 0.5, 0.6, …, 1.5`. Pauli expectation values (eigenvalues ±1).

- **(a)** Bond correlator `(1/N) Σ_i ⟨σᶻ_i σᶻ_{i+1}⟩` minus its exact value (units 1e-3)
  versus h_x, one curve per projection depth τ ∈ [4, 120]. Inset: absolute values,
  exact solution as open black squares.
- **(b)** Transverse magnetisation per site `(1/N) Σ_i ⟨σˣ_i⟩` minus its exact value
  (units 1e-3), from off-diagonal forward walking, τ ∈ [1, 10]. Red squares: the same
  quantity derived from the diagonal run of (a) at its deepest τ = 120 via the exact
  identity `Σ⟨σˣ⟩ = (J_z N ⟨σᶻσᶻ⟩ − E)/h_x` (not independent data). Inset: absolute values.

The colour of each curve encodes τ on one shared logarithmic colour bar. Error bars
are not drawn (median SEM 2·10⁻⁵ for ⟨σᶻσᶻ⟩, ≤ 7·10⁻⁴ per site for ⟨σˣ⟩; they are
stored in the data file). Same GFMC settings as Fig. 4.

## Conventions

- **τ = m·β = (p/2)·β**: projection depth applied to *each* side of the bra–ket;
  `p` is the estimator window length in branching steps, `m = p/2`, β the imaginary
  time per branching step (units 1/|J_z|). β = 1/M differs per field so that every
  field lands on the same τ grid; `p` therefore differs per field.
- Errors: standard error of the mean over 5 independent runs, `std(ddof=1)/√5`.

## `data.h5`

Root attributes: `hamiltonian`, `tau_convention`, `error_convention`, `layout`, `N`,
`J_z`. Every dataset carries a `description` attribute. 2-D arrays are
`(n_hx = 11, n_tau)`.

`/diagonal` — panel (a) and the red squares; 131072 walkers, 2·10⁵ steps per run.

| dataset | shape | meaning |
|---|---|---|
| `hx` | (11,) | transverse field |
| `beta` | (11,) | imaginary-time step per field (1/|J_z|) |
| `tau` | (30,) | τ = 4, 8, …, 120 |
| `p` | (11, 30) | window length per field and τ |
| `energy`, `energy_err` | (11, 30) | total energy E (units |J_z|) and SEM |
| `zz`, `zz_err` | (11, 30) | bond-averaged `(1/N) Σ⟨σᶻ_i σᶻ_{i+1}⟩` and SEM |
| `n_runs` | (11,) | independent runs (5) |

`/offdiagonal` — panel (b); 32768 walkers. Same layout with `tau` (10,) = 1…10 and

| dataset | shape | meaning |
|---|---|---|
| `sx_total`, `sx_total_err` | (11, 10) | **extensive** `Σ_i ⟨σˣ_i⟩` and SEM (plot divides by N) |
| `energy`, `energy_err` | (11, 10) | total energy of the same runs (not plotted) |

`/exact` — free-fermion (Jordan–Wigner) solution of the periodic N = 100 chain:
`hx`, `energy` (total), `zz` (bond average), `sx_total` (extensive), all (11,).

## Provenance

Two PyGFMC campaigns over the 11 fields with the continuous-time propagator and a
nearest-neighbour Jastrow guide (β_Jastrow capped at 0.3): a diagonal campaign
measuring E and ⟨σᶻσᶻ⟩ every step with the ancestry estimator (5 runs per field), and
an off-diagonal campaign measuring Σσˣ with shadow-chain forward walking (5 runs per
field). Per-run results were averaged (mean and SEM across runs) into a consolidated
TFIM results file, from which `data.h5` was extracted unchanged. The exact reference is
the free-fermion solution, validated against brute-force ED to < 1e-9 at N = 8, 10, 12.

## Regenerate

```bash
python plot.py            # writes ../Fig5.pdf (needs a LaTeX install)
python plot.py --no-latex # mathtext fallback
```

Requires numpy, h5py, matplotlib. Width and font sizes come from the shared `../figstyle.py` (all text 8 pt, generated at the
printed width).
