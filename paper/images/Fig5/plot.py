#!/usr/bin/env python3
r"""Fig. 5 -- transverse-field Ising chain across the phase diagram (N = 100).

    (a) <sigma^z_i sigma^z_{i+1}> - exact  vs h_x, one curve per tau   left
    (b) <sigma^x_i>               - exact  vs h_x, one curve per tau   right

Main axes: difference to the exact free-fermion solution (units 1e-3); insets:
absolute values with the exact solution as open squares.  Red squares in (b):
<sigma^x_i> derived from the diagonal run of (a) at its deepest tau via
<sum X> = (J_z*N*<ZZ> - E)/h_x.  One shared logarithmic tau colour bar;
tau = m*beta = (p/2)*beta.

Reads ``data.h5`` in this folder and writes ``../Fig5.pdf``.
Sized for SciPost \textwidth (../figstyle.py); include at width=\textwidth.

    python plot.py [--no-latex]
"""
import argparse
import os
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figstyle  # noqa: E402  (shared width and font sizes, ../figstyle.py)
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data.h5")
OUT = os.path.join(os.path.dirname(HERE), "Fig5.pdf")

L = 100                     # sites
J_Z = -1.0                  # H = J_z sum ZZ - h_x sum X
ASPECT = 2.05
CMAP_TAU = "viridis"
TAU_COLOUR_LIM = (1.0, 120.0)
TAU_CBAR_TICKS = (1, 2, 5, 10, 20, 50, 100)
REF_MATH = r"\mathrm{exact}"
C_DERIVED = "tab:red"
DIFF_SCALE = 1e3
DIFF_EXP = r"$\times 10^{-3}$"
INSET_RECT_ZZ = (.640, .175, .320, .295)
INSET_RECT_XTOT = (.640, .635, .320, .295)
NOTE_GAP = 4.0
NAME_ZZ = "Bond correlator"
NAME_XTOT = "Transverse magnetisation"
MATH_ZZ = r"\langle \sigma^z_i \sigma^z_{i+1}\rangle"
MATH_XTOT = r"\langle \sigma^x_i \rangle"
INSET_TICK_SIZE = figstyle.FONT
INSET_YTICKS_ZZ = (0.4, 0.6, 0.8)
INSET_YTICKS_XTOT = (0.4, 0.6, 0.8)
INSET_XTICKS = (0.5, 1.0, 1.5)


def prepare_standard_figure(nrows, ncols, width, aspect_ratio, use_latex=True):
    """Paper-wide matplotlib style (../figstyle.py)."""
    figstyle.apply(use_latex)
    return plt.subplots(nrows=nrows, ncols=ncols,
                        figsize=(width, width / aspect_ratio),
                        constrained_layout=True)


def load():
    with h5py.File(DATA, "r") as h:
        def grab(g):
            return {k: h[g][k][:] for k in h[g].keys()}
        d = {"zz": grab("diagonal"), "xtot": grab("offdiagonal"),
             "ref": grab("exact")}
    # pointwise subtraction: all three share the same 11 fields
    assert np.allclose(d["ref"]["hx"], d["zz"]["hx"])
    assert np.allclose(d["xtot"]["hx"], d["zz"]["hx"])
    return d


# ------------------------------------------------------------------- helpers
def _panel_label(ax, s, color="k", note=None):
    """(a)/(b) upper left, with the panel's tau range beside it."""
    lbl = ax.text(.030, .955, s, transform=ax.transAxes, ha="left", va="top",
                  color=color, fontsize=figstyle.FONT)
    if note is not None:
        ax.annotate(note, (1.0, 0.5), xycoords=lbl,
                    xytext=(NOTE_GAP, 0), textcoords="offset points",
                    ha="left", va="center_baseline", color=color,
                    fontsize=figstyle.FONT)
    return lbl


def _headroom(ax, frac=.14, side="top"):
    lo, hi = ax.get_ylim()
    pad = frac * (hi - lo)
    ax.set_ylim(lo, hi + pad) if side == "top" else ax.set_ylim(lo - pad, hi)


def _tau_family(ax, hx, tau, Y, norm, lw=.7, ms=1.8):
    """One line per tau, coloured on the shared log norm."""
    sm = ScalarMappable(norm=norm, cmap=CMAP_TAU)
    for j, t in enumerate(tau):
        ax.plot(hx, Y[:, j], "-o", color=sm.to_rgba(t), lw=lw, ms=ms,
                zorder=2, alpha=.9)
    return sm


def _zero_line(ax):
    ax.axhline(0.0, color="k", lw=.7, ls="--", zorder=1)


def _inset(ax, rect, yticks):
    ins = ax.inset_axes(rect)
    ins.tick_params(which="both", top=False, right=False,
                    labelsize=INSET_TICK_SIZE, pad=1.5, length=2)
    for s in ins.spines.values():
        s.set_linewidth(.4)
    ins.set_xticks(list(INSET_XTICKS))
    ins.set_xticklabels([f"{v:.1f}" for v in INSET_XTICKS])
    ins.set_yticks(list(yticks))
    ins.set_yticklabels([f"{v:g}" for v in yticks])
    ins.set_xlim(.45, 1.55)
    return ins


def _inset_title(ins, s):
    ins.text(.5, 1.02, s, transform=ins.transAxes, ha="center", va="bottom",
             fontsize=INSET_TICK_SIZE)


def _reference_markers(ax, hx, y, ms=2.4):
    ax.plot(hx, y, "s", color="k", ms=ms, mfc="none", mew=.6, zorder=4)


def _finish(ax, ylabel):
    ax.set_xlabel(r"Transverse field $h_x$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(.45, 1.55)
    ax.text(0.0, 1.01, DIFF_EXP, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=figstyle.FONT)


# --------------------------------------------------------------- panel (a)
def panel_zz(ax, d, norm):
    z = d["zz"]
    _tau_family(ax, z["hx"], z["tau"],
                (z["zz"] - d["ref"]["zz"][:, None]) * DIFF_SCALE, norm)
    _zero_line(ax)
    _finish(ax, rf"{NAME_ZZ} ${MATH_ZZ}-{MATH_ZZ}_{REF_MATH}$")
    _headroom(ax, .12)
    note = rf"$\tau\in[{z['tau'].min():g},\,{z['tau'].max():g}]$"
    ins = _inset(ax, INSET_RECT_ZZ, INSET_YTICKS_ZZ)
    _tau_family(ins, z["hx"], z["tau"], z["zz"], norm, lw=.5, ms=0)
    _reference_markers(ins, d["ref"]["hx"], d["ref"]["zz"])
    _inset_title(ins, rf"${MATH_ZZ}$")
    _panel_label(ax, "(a)", note=note)


# --------------------------------------------------------------- panel (b)
def _derived_xtot(d):
    """<sum X> from the diagonal run at its deepest tau, and that tau."""
    z = d["zz"]
    j = int(np.argmax(z["tau"]))
    return (J_Z * L * z["zz"][:, j] - z["energy"][:, j]) / z["hx"], z["tau"][j]


def panel_xtot(ax, d, norm):
    x = d["xtot"]
    ref = d["ref"]["sx_total"] / L
    sm = _tau_family(ax, x["hx"], x["tau"],
                     (x["sx_total"] / L - ref[:, None]) * DIFF_SCALE, norm)
    derived, tau_d = _derived_xtot(d)
    ax.plot(d["zz"]["hx"], (derived / L - ref) * DIFF_SCALE, "s",
            color=C_DERIVED, ms=4.5, mfc="none", mew=.9, zorder=5,
            label=rf"diagonal, $\tau={tau_d:g}$")
    _zero_line(ax)
    _finish(ax, rf"{NAME_XTOT} ${MATH_XTOT}-{MATH_XTOT}_{REF_MATH}$")
    ax.legend(loc="lower left", borderaxespad=.3)
    _headroom(ax, .16, side="bottom")
    note = rf"$\tau\in[{x['tau'].min():g},\,{x['tau'].max():g}]$"
    ins = _inset(ax, INSET_RECT_XTOT, INSET_YTICKS_XTOT)
    _tau_family(ins, x["hx"], x["tau"], x["sx_total"] / L, norm, lw=.5, ms=0)
    ins.plot(d["zz"]["hx"], derived / L, "s", color=C_DERIVED, ms=2.4,
             mfc="none", mew=.6, zorder=5)
    _reference_markers(ins, d["ref"]["hx"], ref)
    _inset_title(ins, rf"${MATH_XTOT}$")
    _panel_label(ax, "(b)", note=note)
    return sm


def _check_ylabels(fig, axes):
    """Warn if a rotated y label is taller than its axes (it would be clipped)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    h_in = fig.get_size_inches()[1]
    for ax in np.ravel(axes):
        lab = ax.yaxis.label.get_window_extent(r).height / fig.dpi
        room = ax.get_position().height * h_in
        if lab > room:
            print(f"WARNING: y label is {lab:.2f} in against a {room:.2f} in "
                  f"axes and will be clipped", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=OUT)
    ap.add_argument("--no-latex", action="store_true")
    a = ap.parse_args()
    d = load()
    fig, axes = prepare_standard_figure(
        nrows=1, ncols=2, width=figstyle.WIDTH_FULL, aspect_ratio=ASPECT,
        use_latex=not a.no_latex)
    fig.get_layout_engine().set(w_pad=.04, h_pad=.04, wspace=.02)
    for ax in axes.ravel():
        ax.tick_params(which="both", top=False, right=False)
    norm = LogNorm(vmin=TAU_COLOUR_LIM[0], vmax=TAU_COLOUR_LIM[1])
    panel_zz(axes[0], d, norm)
    sm = panel_xtot(axes[1], d, norm)
    cb = fig.colorbar(sm, ax=axes.tolist(), pad=.02, fraction=.05, aspect=22,
                      ticks=list(TAU_CBAR_TICKS))
    cb.ax.set_yticklabels([f"{t:g}" for t in TAU_CBAR_TICKS])
    cb.ax.minorticks_off()
    cb.set_label(r"Projection depth $\tau$")
    _check_ylabels(fig, axes)
    # no bbox_inches="tight": keep the canvas exactly \textwidth wide
    fig.savefig(a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
