#!/usr/bin/env python3
"""Fig. 9: S^pm(0)/N of the pyrochlore XXZ model at N = 1024 vs J_xy/J_z,
one curve per projection time tau.

Reads ../Fig6/data.h5 (layout in ../Fig6/README.md) and writes ../Fig9.pdf.
The inset zooms into the transition region, where the deepest rungs exist.
"""
import importlib.util
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

# Fig. 6's plot.py holds the shared tau -> infinity fit
_spec = importlib.util.spec_from_file_location(
    "fig6", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Fig6", "plot.py"))
fig6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fig6)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "Fig6", "data.h5")
OUT = os.path.join(HERE, "..", "Fig9.pdf")
N_SITES = 1024
TAUS = [2, 4, 6, 8, 10, 12, 14, 17, 20, 25, 30, 35, 40]  # projection depths shown (nearest rung)
TAU_TOL = 0.4  # max distance to the nearest rung; > half the largest rung spacing
INSET_X = (-0.11, -0.10)  # J_xy/J_z range shown in the inset
INSET_POS = [0.50, 0.38, 0.48, 0.59]  # inset in axes fraction: x0, y0, w, h
CBAR_POS = [0.21, 0.80, 0.21, 0.035]  # horizontal tau colorbar left of the inset


def prepare_standard_figure(aspect_ratio=16 / 9):
    """0.7\textwidth-wide figure in the paper-wide style (../figstyle.py)."""
    figstyle.apply()
    width = figstyle.WIDTH_COL
    return plt.subplots(figsize=(width, width / aspect_ratio))


def load():
    """Return {tau: (j_xy, S0/N, err)} for N = N_SITES.

    S0 = sum_{i<j} <S+_i S-_j + h.c.> (onsite term excluded) at the rung
    nearest to tau; cells whose ladder does not reach tau are skipped. Samples
    reaching tau are averaged; the error is the larger of the combined blocked
    SEM and half the spread between samples.
    """
    raw = defaultdict(lambda: defaultdict(list))
    with h5py.File(DATA, "r") as f:
        for r in f["index"][:]:
            if r["N"] != N_SITES:
                continue
            g = f["cells/s%d/%s/jxy%g" % (r["sample"], r["lattice"].decode(),
                                          r["j_xy"])]["spm0"]
            tau = g["tau"][:]
            values, errors = g["S0"][:] / N_SITES, 4 * g["sem"][:] / N_SITES
            for t in TAUS:
                k = np.argmin(abs(tau - t))
                if abs(tau[k] - t) <= TAU_TOL:
                    raw[t][round(float(r["j_xy"]), 6)].append(
                        (values[k], errors[k]))
    out = {}
    for t, d in raw.items():
        pts = []
        for jxy, samples in sorted(d.items()):
            v, e = np.array(samples).T
            err = np.sqrt(np.sum(e**2)) / len(e)
            if len(v) > 1:
                err = max(err, 0.5 * np.ptp(v))
            pts.append((jxy, v.mean(), err))
        out[t] = tuple(np.array(c) for c in zip(*pts))
    return out


def fit_tau_inf():
    """Return (j_xy, b, err): the tau -> infinity limit for N = N_SITES,
    from Fig. 6's fit_ladder() (one shared definition for both figures)."""
    return fig6.load_gfmc(fit=True, n_sites=N_SITES)[N_SITES]


def load_guide():
    """Return (j_xy, S0/N, err) of the guide (VMC, tau = 0) for N = N_SITES.

    From /guide_index; S0 = 4 <spm0> as for the GFMC cells. Couplings that
    share a guide wavefunction share its value (see ../Fig6/README.md).
    """
    with h5py.File(DATA, "r") as f:
        g = f["guide_index"][:]
    g = g[g["N"] == N_SITES]
    g = g[np.argsort(g["j_xy"])]
    return g["j_xy"], g["S0"] / N_SITES, 4 * g["spm0_err"] / N_SITES


def main():
    data = load()

    fig, ax = prepare_standard_figure()
    taus = [t for t in TAUS if t in data]
    # viridis over the same 0.1-0.9 range as Fig. 6, mapped linearly in tau
    cmap = matplotlib.colors.ListedColormap(
        matplotlib.colormaps["viridis"](np.linspace(0.1, 0.9, 256)))
    norm = matplotlib.colors.Normalize(min(taus), max(taus))
    colors = {t: cmap(norm(t)) for t in taus}

    for t in taus:
        jxy, S, err = data[t]
        ax.errorbar(jxy, S, yerr=err, fmt="o-", ms=1.8, lw=0.7,
                    capsize=1.2, color=colors[t])

    fit = fit_tau_inf()
    # fit values without error bars (see fit_ladder for the errors)
    fit_style = dict(marker="s", ls="none", color="tab:red", mfc="none", mew=0.7, zorder=5)
    h_fit, = ax.plot(fit[0], fit[1], ms=3.5,
                     label=r"$\tau\to\infty$ fit", **fit_style)

    guide = load_guide()
    guide_style = dict(marker="D", ls="-", lw=0.7, color="k", zorder=4)
    h_guide, = ax.plot(guide[0], guide[1], ms=2.8,
                       label=r"guide ($\tau=0$)", **guide_style)

    ax.set_xlabel(r"Transverse coupling $J_{xy}/J_z$")
    ax.set_ylabel(r"Structure factor $S^{\pm}(\mathbf{q}=0)$")
    ax.tick_params(top=False, right=False)
    ax.set_ylim(bottom=0)
    allj = np.concatenate([data[t][0] for t in taus])
    ax.set_xlim(allj.min(), allj.max())

    # inset: transition region
    axins = ax.inset_axes(INSET_POS)
    for t in taus:
        jxy, S, err = data[t]
        sel = (jxy >= INSET_X[0] - 1e-9) & (jxy <= INSET_X[1] + 1e-9)
        axins.errorbar(jxy[sel], S[sel], yerr=err[sel], fmt="o-", ms=1.5,
                       lw=0.6, capsize=1, color=colors[t])
    sel = (fit[0] >= INSET_X[0] - 1e-9) & (fit[0] <= INSET_X[1] + 1e-9)
    axins.plot(fit[0][sel], fit[1][sel], ms=3, **fit_style)
    sel = (guide[0] >= INSET_X[0] - 1e-9) & (guide[0] <= INSET_X[1] + 1e-9)
    axins.plot(guide[0][sel], guide[1][sel], ms=2.5, **guide_style)
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

    cax = ax.inset_axes(CBAR_POS)
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=cax, orientation="horizontal")
    cbar.set_ticks([t for t in (10, 20, 30, 40) if min(taus) <= t <= max(taus)])
    cbar.ax.tick_params(labelsize=figstyle.FONT, length=2, direction="in")
    cbar.outline.set_linewidth(0.5)
    cax.set_title(r"Projection depth $\tau$", fontsize=figstyle.FONT, pad=3)
    ax.legend(handles=[h_guide, h_fit], loc="lower left")
    fig.tight_layout()
    # no bbox_inches="tight": keep the canvas exactly figstyle.WIDTH_COL wide
    fig.savefig(OUT)
    print(f"wrote {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
