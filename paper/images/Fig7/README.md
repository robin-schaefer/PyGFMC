# Fig. 7 — flux-sector energy cost in the pyrochlore ring-exchange model

`\includegraphics{images/ring_energy}` (label `fig:ring_energy`).

## What the figure shows

Energy cost ΔE(Φ_x) = E(Φ_x,0,0) − E(0,0,0) of electric-flux sectors in the
ring-exchange model H = −g Σ_hexagons (S⁺S⁻S⁺S⁻S⁺S⁻ + h.c.) (eq. `H_ring`,
g = 1, no RK term) for linear sizes L = 4, 6, 8, 10 (N = 4L³ spins). The grey
line is a·Φ_x²/L with `a` fitted to the largest L; the inset shows L·ΔE/g,
which collapses onto a·Φ_x² (the |Φ|²/L signature of quantum spin ice).

## Data: `data.h5`

Root attributes: `description`, `p` (= 48, branching steps of the energy
estimator), `units`, `error_convention`, `L_values`.
One group per size, `/L{L}` (attrs `L`, `n_sites`):

| dataset | shape | meaning |
|---|---|---|
| `phi_x` | (n,) | flux Φ_x (Φ_y = Φ_z = 0), Φ_x > 0 |
| `delta_E` | (n,) | E(Φ_x) − E(0), units of g, total (not per site) |
| `delta_E_err` | (n,) | error: half the spread of 2 samples, combined in quadrature with the Φ_x = 0 reference error |

The quadratic coefficient is not stored; `plot.py` refits it (weighted least
squares on L·ΔE of the largest L), reproducing a = 1.9608.

## Provenance

GFMC runs of the ring-exchange plaquette model on the hexagons of the pyrochlore
lattice, one run set per (L, Φ_x) sector with walkers initialised in that flux
sector from a database of valid ice configurations (the dynamics conserve the
flux). single_step propagation, step size 15, λ = 0, no guide, 2 independent
samples per sector. The energy per sector is the mixed estimator at window
p = 48, computed from the stored per-step energies and weights; ΔE is taken
relative to the Φ_x = 0 sector of the same L. Only the p = 48 values of the
Φ_x > 0 sectors are kept here.

## Regenerate

```bash
python plot.py        # writes ../Fig7.pdf  (needs numpy, h5py, scipy, matplotlib, LaTeX)
```

Width and font sizes come from the shared `../figstyle.py` (all text 8 pt, generated at the
printed width).
