#!/usr/bin/env python3
"""Fig. 6: S^pm(q=0) of the pyrochlore XXZ model vs J_xy/J_z for several N.

Reads data.h5 (GFMC, layout in README.md; same folder) and writes
../Fig6.pdf. One curve per N with error bars; the inset zooms into the
transition region.

    python plot.py          # value at each cell's deepest tau rung
    python plot.py --fit    # tau -> infinity limit from fit_ladder()
-o writes elsewhere than ../Fig6.pdf.
"""
import argparse
import os
import sys
from collections import defaultdict

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figstyle  # noqa: E402  (shared width and font sizes, ../figstyle.py)
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data.h5")
OUT = os.path.join(HERE, "..", "Fig6.pdf")
INSET_X = (-0.11, -0.10)  # J_xy/J_z range shown in the inset
INSET_POS = [0.50, 0.38, 0.48, 0.59]  # inset in axes fraction: x0, y0, w, h
LEGEND_ANCHOR = (0.45, 1.0)  # upper-right corner of the legend, axes fraction
EXCLUDE_N = {1372}  # sizes left out of the figure
TRANSITION_C, TRANSITION_HW = -31 / 300, 1 / 1200  # centre and half-width of the band
TRANSITION = (TRANSITION_C - TRANSITION_HW, TRANSITION_C + TRANSITION_HW)
FIT_TAU_MIN = 10  # fit S(tau) = c exp(-a tau) + b on rungs tau >= FIT_TAU_MIN
FIT_A_MIN = 0.05  # lower bound on a; a fit stuck there resolves no decay
FIT_ERR_MAX = 5  # fit error on b above this many deepest-rung SEMs: unresolved


def prepare_standard_figure(aspect_ratio=16 / 9):
    """0.7\textwidth-wide figure in the paper-wide style (../figstyle.py)."""
    figstyle.apply()
    width = figstyle.WIDTH_COL
    return plt.subplots(figsize=(width, width / aspect_ratio))


def fit_ladder(tau, S, err):
    """Return (b, err_b), the tau -> infinity limit of one cell's ladder.

    Fits c exp(-a tau) + b to the rungs tau >= FIT_TAU_MIN, weighted by the
    blocked SEM. Rungs are strongly correlated, so the fit error alone
    underestimates; err_b is the larger of it and the deepest rung's SEM. If the
    fit resolves no decay (a on FIT_A_MIN, or the error on b above FIT_ERR_MAX
    deepest-rung SEMs) the ladder is flat within its reach and the deepest rung
    is returned instead.
    """
    m = tau >= FIT_TAU_MIN
    t, S, e = tau[m], S[m], err[m]
    p, cov = curve_fit(lambda t, c, a, b: c * np.exp(-a * t) + b, t, S,
                       sigma=e, absolute_sigma=True, p0=(S[0] - S[-1], 0.2, S[-1]),
                       bounds=([-np.inf, FIT_A_MIN, -np.inf], [np.inf, 5, np.inf]),
                       maxfev=20000)
    err_b = np.sqrt(cov[2, 2])
    if p[1] < FIT_A_MIN * 1.01 or err_b > FIT_ERR_MAX * e[-1]:
        return S[-1], e[-1]
    return p[2], max(err_b, e[-1])


def load_gfmc(fit=False, n_sites=None):
    """Return {N: (j_xy, S0/N, err)}, S0 as stored in the file.

    S0 = sum_{i<j} <S+_i S-_j + h.c.> (onsite term excluded), taken at the
    deepest tau rung, or with fit=True extrapolated to tau -> infinity by
    fit_ladder(). Samples are averaged; the error is the larger of the combined
    error and half the spread between samples. If n_sites is supplied, read
    and fit only that system size.
    """
    raw = defaultdict(lambda: defaultdict(list))
    with h5py.File(DATA, "r") as f:
        for r in f["index"][:]:
            N = int(r["N"])
            if n_sites is not None and N != n_sites:
                continue
            g = f["cells/s%d/%s/jxy%g" % (r["sample"], r["lattice"].decode(),
                                          r["j_xy"])]["spm0"]
            S, e = g["S0"][:] / N, 4 * g["sem"][:] / N
            raw[N][round(float(r["j_xy"]), 6)].append(
                fit_ladder(g["tau"][:], S, e) if fit else (S[-1], e[-1]))
    out = {}
    for N, d in raw.items():
        pts = []
        for jxy, samples in sorted(d.items()):
            v, e = np.array(samples).T
            err = np.sqrt(np.sum(e**2)) / len(e)
            if len(v) > 1:
                err = max(err, 0.5 * np.ptp(v))
            pts.append((jxy, v.mean(), err))
        out[N] = tuple(np.array(c) for c in zip(*pts))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fit", action="store_true",
                        help="plot the tau -> infinity fit instead of the "
                             "deepest rung")
    parser.add_argument("-o", "--out", default=OUT, help="output PDF")
    args = parser.parse_args()
    gfmc = load_gfmc(fit=args.fit)

    fig, ax = prepare_standard_figure()
    sizes = sorted(N for N in gfmc if N not in EXCLUDE_N)
    cmap = matplotlib.colormaps.get_cmap("viridis")
    colors = {N: cmap(0.1 + 0.8 * i / max(len(sizes) - 1, 1))
              for i, N in enumerate(sizes)}

    n_handles = []
    for N in sizes:
        jxy, S, err = gfmc[N]
        h = ax.errorbar(jxy, S, yerr=err, fmt="o-", ms=2.5, lw=0.8,
                        capsize=1.5, color=colors[N], label=fr"${N}$")
        n_handles.append(h)

    ax.set_xlabel(r"Transverse coupling $J_{xy}/J_z$")
    ax.set_ylabel(r"Structure factor $S^{\pm}(\mathbf{q}=0)$")
    ax.tick_params(top=False, right=False)
    ax.set_ylim(bottom=0)
    allj = np.concatenate([gfmc[N][0] for N in sizes])
    ax.set_xlim(allj.min(), allj.max())

    # inset: transition region
    axins = ax.inset_axes(INSET_POS)
    for N in sizes:
        jxy, S, err = gfmc[N]
        sel = (jxy >= INSET_X[0] - 1e-9) & (jxy <= INSET_X[1] + 1e-9)
        axins.errorbar(jxy[sel], S[sel], yerr=err[sel], fmt="o-", ms=1.5,
                       lw=0.6, capsize=1, color=colors[N])
    axins.axvspan(*TRANSITION, color="tab:red", alpha=0.25, lw=0, zorder=0)
    # arrow from the lower-left corner to the crossing point at TRANSITION_C
    y_c = np.mean([np.interp(TRANSITION_C, *gfmc[N][:2]) for N in sizes])
    axins.annotate("transition", xy=(TRANSITION_C, y_c), xycoords="data",
                   xytext=(0.05, 0.08), textcoords="axes fraction",
                   color="tab:red", fontsize=figstyle.FONT, ha="left", va="bottom",
                   arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.6,
                                   shrinkA=1, shrinkB=2))
    axins.set_xlim(*INSET_X)
    axins.set_ylim(bottom=0)
    axins.tick_params(which="both", direction="in", top=False, right=False,
                      labelbottom=False, labelleft=False)
    zoom = ax.indicate_inset_zoom(axins, edgecolor="k", lw=0.5, alpha=1)
    for artist in [zoom.rectangle, *zoom.connectors]:
        artist.set_linestyle((0, (3, 2)))
    # connectors: lower left, upper left, lower right, upper right. Fix the pair
    # so matplotlib cannot pick one that runs behind the inset.
    for k, c in enumerate(zoom.connectors):
        c.set_visible(k in (1, 2))

    ax.legend(handles=n_handles, title=r"System size $N$", ncol=2,
              loc="upper right", bbox_to_anchor=LEGEND_ANCHOR,
              columnspacing=0.8, handletextpad=0.3)
    fig.tight_layout()
    # no bbox_inches="tight": keep the canvas exactly figstyle.WIDTH_COL wide
    fig.savefig(args.out)
    print(f"wrote {os.path.normpath(args.out)}")


if __name__ == "__main__":
    main()
