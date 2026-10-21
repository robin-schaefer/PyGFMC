# Fig. 9: τ convergence of S^±(q=0) at N = 1024 (pyrochlore XXZ)

`\includegraphics{images/Fig9.pdf}` (label `fig:pyro_tau`, appendix `app:pyro_tau`).

## What the figure shows

S^±(q=0) = `S0`/N (paper eq. `Spm_q0`), with `S0` = Σ_{i<j} ⟨S⁺ᵢS⁻ⱼ + S⁻ᵢS⁺ⱼ⟩ (onsite term excluded),
for N = 1024 versus J_xy/J_z. There is one curve for each projection depth τ = 2, 4, …, 14, 17, 20, 25, 30, 35, 40
(`TAUS` in `plot.py`), coloured by τ (horizontal colorbar left of the inset; viridis 0.1–0.9 as in Fig. 6). It is the same quantity as Fig. 6 but at fixed τ instead of
each cell's deepest rung. The inset zooms into J_xy/J_z ∈ [−0.11, −0.10].

Red squares: τ → ∞ extrapolation, from `fit_ladder()` in `../Fig6/plot.py` (shared
with Fig. 6's `--fit` option; procedure described in `../Fig6/README.md`). Result:
a ≈ 0.15–0.2 in the transition region, and b agrees with τ = 40 within 2 %.

Filled black diamonds (solid line): expectation value of the guide (VMC, τ = 0) at N = 1024,
from `/guide_index` (`load_guide` in `plot.py`), S0/N with S0 = 4⟨spm0⟩. Couplings
that share a guide wavefunction share its value (−0.1025/−0.103333/−0.104167 and
−0.105/−0.105833), and the −0.07 guide value is flagged as unreliable in
`../Fig6/README.md`.

## Data

`../Fig6/data.h5` (shared with Fig. 6; see `../Fig6/README.md`).
For each cell the rung nearest to τ is used if it lies within `TAU_TOL` = 0.4,
which is larger than half the largest rung spacing (0.65). Otherwise the cell is skipped at that τ.
The samples that reach τ are averaged. The error is the larger of the combined
blocked SEM and half the spread between the samples, the same rule as in Fig. 6.

Coverage: τ ≤ 14 everywhere; τ = 20 for J_xy/J_z ≥ −0.11; τ = 30, 40 only for
−0.11 ≤ J_xy/J_z ≤ −0.10 (and partly from a single sample).

## Regenerate

```bash
python plot.py        # writes ../Fig9.pdf  (needs numpy, scipy, h5py, matplotlib, LaTeX)
```

Width and font sizes come from the shared `../figstyle.py` (all text 8 pt, generated at the
printed width).
