#!/usr/bin/env python3
"""Fig. 7: flux-sector energy cost in the pyrochlore ring-exchange model.

Reads data.h5 (same folder) and writes ../Fig7.pdf.
Main panel: Delta E(phi_x)/g for each linear size L, with the quadratic
a*phi_x^2/L_max fitted to the largest L. Inset: L*Delta E/g (1/L collapse).
"""
import os
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figstyle  # noqa: E402  (shared width and font sizes, ../figstyle.py)
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data.h5")
OUT = os.path.join(HERE, "..", "Fig7.pdf")


def prepare_standard_figure(aspect_ratio=16 / 9):
    """0.7\textwidth-wide figure in the paper-wide style (../figstyle.py)."""
    figstyle.apply()
    width = figstyle.WIDTH_COL
    return plt.subplots(figsize=(width, width / aspect_ratio))


# ── load ─────────────────────────────────────────────────────────────────────
with h5py.File(DATA, "r") as hf:
    p_plot = int(hf.attrs["p"])
    L_values = sorted((int(L) for L in hf.attrs["L_values"]), reverse=True)
    data = {L: (hf[f"L{L}/phi_x"][:], hf[f"L{L}/delta_E"][:],
                hf[f"L{L}/delta_E_err"][:]) for L in L_values}

# ── quadratic fit L*Delta E = a*phi_x^2 on the largest L ─────────────────────
ref_L = max(L_values)
phi, dE, err = data[ref_L]
(a_ref,), _ = curve_fit(lambda x, a: a * x**2, phi, dE * ref_L,
                        sigma=np.maximum(err * ref_L, 1e-10), absolute_sigma=True)

cmap = plt.get_cmap("viridis")
norm = plt.Normalize(vmin=min(L_values) - 2, vmax=max(L_values) + 2)
L_COLOR = {L: cmap(norm(L)) for L in L_values}

# ── main panel ───────────────────────────────────────────────────────────────
fig, ax = prepare_standard_figure()
legend_handles = []
for L in L_values:
    phi, dE, err = data[L]
    ax.errorbar(phi, dE, yerr=err, fmt="o-", markersize=3, lw=1.2,
                elinewidth=0.9, capsize=2, color=L_COLOR[L], zorder=3)
    legend_handles.append(Line2D([0], [0], color=L_COLOR[L], marker="o",
                                 markersize=4, lw=1.2, label=fr"${L}$"))

phi_fine = np.linspace(1, 20, 300)
ax.plot(phi_fine, (a_ref / ref_L) * phi_fine**2, color="k", lw=1.2, alpha=0.5)

ax.set_xlabel(r"Electric flux $\Phi_x$")
ax.set_ylabel(r"Energy cost $\Delta E / g$")
ax.set_xlim(0.5, None)
ax.xaxis.set_major_locator(plt.MultipleLocator(4))
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
ax.tick_params(top=False, right=False, axis="both", which="both")
ax.legend(handles=legend_handles, loc="upper left", title=r"Length $L$")

# ── inset: L * Delta E ───────────────────────────────────────────────────────
axins = ax.inset_axes([0.68, 0.07, 0.30, 0.32])
all_ys, all_yerrs = [], []
for L in L_values:
    phi, dE, err = data[L]
    y, yerr = dE * L, err * L
    axins.errorbar(phi, y, yerr=yerr, marker="o", markersize=2, lw=0.8,
                   capsize=1.5, color=L_COLOR[L])
    all_ys.append(y)
    all_yerrs.append(yerr)

ref_y = a_ref * phi_fine**2
axins.plot(phi_fine, ref_y, color="k", lw=1.2, alpha=0.5,
           label=fr"${a_ref:.2f}\,\Phi_x^2$")
axins.set_xlim(0.5, 20)
ys, ye = np.concatenate(all_ys), np.concatenate(all_yerrs)
ylo = min((ys - ye).min(), ref_y.min())
yhi = max((ys + ye).max(), ref_y.max())
pad = 0.05 * (yhi - ylo)
axins.set_ylim(ylo - pad, yhi + pad)

axins.set_ylabel(r"$L\,\Delta E / g$", fontsize=figstyle.FONT)
axins.tick_params(top=False, right=False, axis="both", which="both")
axins.tick_params(axis="x", bottom=False, labelbottom=False)
axins.tick_params(labelsize=figstyle.FONT)
axins.text(0.97, 0.03, fr"$p={p_plot}$", transform=axins.transAxes,
           ha="right", va="bottom", fontsize=figstyle.FONT)
axins.legend(loc="upper left", handlelength=1)

fig.tight_layout(pad=0.3)
fig.savefig(OUT)
print(f"wrote {os.path.normpath(OUT)}  (fit a = {a_ref:.10f})")
