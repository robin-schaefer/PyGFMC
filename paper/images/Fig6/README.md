# Fig. 6 — transverse structure factor S^±(q=0) of the pyrochlore XXZ model

`\includegraphics{images/Fig6.pdf}` (label `fig:pyro_Spm`).

## What the figure shows

S^±(q=0) = (1/N) Σ_{i<j} ⟨S⁺ᵢS⁻ⱼ + S⁻ᵢS⁺ⱼ⟩ (paper eq. `Spm_q0`), i.e. `S0`/N with
`S0` as stored in `data.h5` (onsite term excluded), for the nearest-neighbour pyrochlore XXZ model (J_z = 1) versus
J_xy/J_z for N = 128, 256, 432, 500, 864, 1024. The inset (linear y-axis, inward ticks, no tick labels) zooms into the
transition region J_xy/J_z ∈ [−0.11, −0.10] (`INSET_X` in `plot.py`).
The red band in the inset marks the transition at J_xy/J_z = −31/300 ± 1/1200
(−0.103333 ± 0.000833, i.e. ± one grid spacing; `TRANSITION_C`, `TRANSITION_HW`
in `plot.py`). A red arrow labelled "transition" points from the lower-left corner
of the inset to the crossing point at J_xy/J_z = −31/300.

## Data

- **GFMC: `data.h5`**: all Spm(0) production cells (228 cells: 114 each in
  samples s0 and s1; layout below).
  The plotted value is `spm0/S0 / N` at the deepest τ rung of each cell, with
  error `4·spm0/sem / N`. The samples of a cell are averaged. The error is the larger of the
  combined blocked SEM (√(Σ sem²)/n_samples) and half the spread between the
  samples. Cells with one sample use only their own SEM.

## Layout of `data.h5`

```
/index                          table, one row per GFMC cell
/cells/s<sample>/<lattice>/jxy<J_xy>/
    attrs                       run parameters (N, j_xy, beta_imag, n_walkers, steps, seed, ...)
    spm0/   tau, value, S0, sem, neff, m_array, p_array   one entry per tau rung
    energy/ tau, p, value, sem, n_windows                 energy per tau rung
/guide_index                    table, one row per guide (VMC) evaluation
/guide/<lattice>/jxy<J_xy>/     attributes of each guide evaluation (spm0, spm0_err, S0, ...)
```

A cell is one (sample, lattice, J_xy) run; `<lattice>` is `cubicL2`–`cubicL4` or
`fccL4`–`fccL6`, and N is stored in `/index` and in the cell attributes.
`S0 = 4·value` is S^±(q=0)·N with the onsite term excluded, and `sem` is the blocked
standard error of `value`. τ is the projection depth per side, (p/2)·β_imag.

`/guide` holds the variational value of the guiding wavefunction alone (τ = 0),
measured only for N = 1024 (19 couplings; Fig. 9). It has no sample index,
because it depends on the guide only. Couplings that use the same guide share
one value. The value at J_xy/J_z = −0.07 is not reliable.

## Optional: τ → ∞ fit (`--fit`)

`python plot.py --fit` uses `fit_ladder()` instead of the deepest rung. It fits each
cell's ladder for τ ≥ 10 (`FIT_TAU_MIN`) to c·exp(−aτ) + b, weighted by the blocked
SEM, and plots b. Neighbouring rungs are strongly correlated (χ²_red ≪ 1), so the
per-cell error is max(fit error on b, SEM of the deepest rung). If the fit resolves
no decay, the deepest rung is used instead. That happens when a ends on its lower
bound 0.05 (`FIT_A_MIN`) or when the error on b exceeds 5× the deepest-rung SEM
(`FIT_ERR_MAX`). Fig. 9 imports this same function.

The fit shifts points by at most 1 % for N ≤ 864. For N = 1024 the largest shift is
−3.6 % at −31/300, where sample s0 stopped at τ ≈ 20. With the fit, the
N = 1024 curve crosses every smaller N at −0.10333 … −0.10349. With the deepest rung
the crossings are at −0.1030 … −0.1032. **The published figure uses the deepest rung
(the default).**

## Regenerate

```bash
python plot.py        # writes ../Fig6.pdf  (needs numpy, scipy, h5py, matplotlib, LaTeX)
```

Needs `h5py` in addition.

Width and font sizes come from the shared `../figstyle.py` (all text 8 pt, generated at the
printed width).
