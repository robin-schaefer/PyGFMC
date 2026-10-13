#!/usr/bin/env python3
r"""Fig. 4 -- transverse-field Ising chain at the critical point (h_x = 1, N = 100).

    (a) median n_eff over (beta, tau)               upper left
    (b) (E - E_exact)/N vs tau, one curve per beta  upper right
    (c) <sigma^z_i sigma^z_{i+1}> vs tau            lower left
    (d) <sigma^x_i> vs tau (forward walking), plus the value derived from
        the diagonal run of (b)/(c) via the Hamiltonian at beta = 1.5  lower right

tau = m*beta = (p/2)*beta on every axis.  Reads ``data.h5`` in this folder and
writes ``../Fig4.pdf``.  Sized for SciPost \textwidth (../figstyle.py); include
at width=\textwidth with no scaling.

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
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data.h5")
OUT = os.path.join(os.path.dirname(HERE), "Fig4.pdf")

L = 100                     # sites
HX = 1.0                    # critical point
J_Z = -1.0                  # H = J_z sum ZZ - h_x sum X

BETA_LIM = (0.02, 1.50)     # x range of panel (a)
CMAP_BETA = "viridis"
CMAP_NEFF = "inferno"
TAU_LINE_LIM = (2.0, 100.0)     # tau window of (b), (c)
TAU_LINE_LIM_D = (2.0, 25.0)    # tau window of (d)
NEFF_TAU_LIM = (2.0, 98.8)      # tau window of (a)
NEFF_CONTOUR_SMOOTH = 2.5       # Gaussian blur (grid cells) for contours only
C_EXACT = "k"
BAND_ALPHA = .25
NEFF_LEVELS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
NEFF_DARK_LEVELS = (2048, 4096)
NEFF_LABEL_AT = 1.16
REF_LABEL, REF_MATH = "exact", r"\mathrm{exact}"
DERIVED_BETAS = (1.5,)
C_DERIVED = "tab:red"
# per-slice tau cutoffs of the direct forward-walking curves in (d)
XTOT_TAU_MAX = {1.0: 20.0, 1.25: 10.0, 1.5: 12.0}
# n_eff annotations: arrows onto the beta = 1.5 curve in (b) and the
# beta = 0.25 curve in (c), at measured depths of both tau ladders
NEFF_ARROW_BETA = 1.5
NEFF_ARROW_TAUS = (4.5, 7.5, 12.0, 21.0, 49.5)
NEFF_ARROW_H = 2.2
NEFF_ARROW_BETA_C = 0.25
NEFF_ARROW_TAUS_C = NEFF_ARROW_TAUS
NEFF_ARROW_H_C = 0.0055
NEFF_TITLE_AT = 9.0
NEFF_TITLE_AT_C = 50.0
NEFF_LABEL_DECIMALS = 1
LEFT_COL_FRAC = 1.036           # makes all four panels the same width
ENERGY_YMAX = 12.0
ZZ_YTICKS = (0.61, 0.62, 0.63, 0.64)
XTOT_YTICKS = (0.63, 0.64, 0.65, 0.66, 0.67)
NEFF_LABEL_PAD = 0.0
NEFF_LABEL_PAD_C = 3.5
NEFF_TITLE_OFFSET = (2.0, 17.0)
NEFF_TITLE_OFFSET_C = (14.0, 26.0)


def prepare_standard_figure(nrows, ncols, width, aspect_ratio, use_latex=True):
    """Paper-wide matplotlib style (../figstyle.py)."""
    figstyle.apply(use_latex)
    return plt.subplots(nrows=nrows, ncols=ncols,
                        figsize=(width, width / aspect_ratio))


def load():
    d = {}
    with h5py.File(DATA, "r") as h:
        def grab(g):
            return {k: h[g][k][:] for k in h[g].keys()}
        n = grab("neff")
        d["neff"] = {"beta_imag": n["beta"], "tau": n["tau"],
                     "n_eff": n["n_eff_median"]}
        z = grab("diagonal")
        d["zz"] = {"beta_imag": z["beta"], "tau": z["tau"],
                   "energy": z["energy"], "energy_err": z["energy_err"],
                   "zz": z["zz"], "zz_err": z["zz_err"]}
        x = grab("offdiagonal")
        d["xtot"] = {"beta_imag": x["beta"], "tau": x["tau"],
                     "xtot": x["sx_total"], "xtot_err": x["sx_total_err"]}
        e = h["exact"].attrs
        d["ref"] = {"energy": float(e["energy"]), "zz": float(e["zz"]),
                    "sx": float(e["sx_total"])}
    return d


# ------------------------------------------------------------------- helpers
def _panel_label(ax, s, color="k"):
    return ax.text(.030, .955, s, transform=ax.transAxes, ha="left", va="top",
                   color=color, fontsize=figstyle.FONT)


def _headroom(ax, frac=.14):
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + frac * (hi - lo))


def _beta_norm(g):
    """Discrete colour map over the six measured beta."""
    bs = np.unique(g["beta_imag"])
    cols = plt.get_cmap(CMAP_BETA)(np.linspace(.06, .94, len(bs)))
    step = np.diff(bs).min() if len(bs) > 1 else 1.
    edges = np.concatenate([bs - step / 2, [bs[-1] + step / 2]])
    sm = ScalarMappable(norm=BoundaryNorm(edges, len(bs)),
                        cmap=ListedColormap(cols))
    return bs, cols, sm


def _beta_rows(g, key, tau_lim, tau_max_by_beta=None):
    """(beta, colour, tau, value, err) per slice, sorted in tau."""
    bs, cols, _ = _beta_norm(g)
    b, t = g["beta_imag"], g["tau"]
    for bi, c in zip(bs, cols):
        hi = (tau_max_by_beta or {}).get(float(bi), tau_lim[1])
        k = np.where((b == bi) & (t >= tau_lim[0]) & (t <= min(hi, tau_lim[1])))[0]
        k = k[np.argsort(t[k])]
        yield bi, c, t[k], g[key][k], g[key + "_err"][k]


def _beta_family(ax, g, key, scale=1.0, offset=0.0, tau_lim=TAU_LINE_LIM,
                 tau_max_by_beta=None):
    """One convergence curve per beta: line + a +-SEM band."""
    _, _, sm = _beta_norm(g)
    for _, c, t, val, err in _beta_rows(g, key, tau_lim, tau_max_by_beta):
        v = (val - offset) * scale
        e = err * scale
        ax.fill_between(t, v - e, v + e, color=c, alpha=BAND_ALPHA, lw=0,
                        zorder=2)
        ax.plot(t, v, "-", color=c, lw=.9, zorder=3)
    return sm


def _exact_line(ax, y, where="below", label=REF_LABEL):
    ax.axhline(y, color=C_EXACT, lw=.7, ls="--", zorder=1)
    ax.annotate(label, (ax.get_xlim()[0], y), xycoords=("data", "data"),
                xytext=(3, 2 if where == "above" else -3),
                textcoords="offset points", fontsize=figstyle.FONT, color=C_EXACT,
                va="bottom" if where == "above" else "top", ha="left")


def _beta_legend(ax, g, sm, **kw):
    bs, cols, _ = _beta_norm(g)
    handles = [Line2D([], [], color=c, lw=1.4, label=f"{v:g}")
               for v, c in zip(bs, cols)]
    lg = ax.legend(handles=handles,
                   title=r"Imaginary-time step $\beta$", **kw)
    lg.get_title().set_fontsize(plt.rcParams["legend.fontsize"])
    return lg


def _finish_line_panel(ax, sm, ylabel, tau_lim=TAU_LINE_LIM):
    ax.set_xscale("log")
    ax.set_xlim(*tau_lim)
    ticks = [t for t in (2, 5, 10, 20, 50, 100) if tau_lim[0] <= t <= tau_lim[1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_xticks([], minor=True)
    ax.set_xlabel(r"Projection depth $\tau = m\,\beta$")
    ax.set_ylabel(ylabel)


# --------------------------------------------------------------- panel (a)
def _neff_columns(d):
    """{beta: (tau, n_eff)}, one entry per measured beta column."""
    n = d["neff"]
    ok = np.isfinite(n["n_eff"]) & (n["n_eff"] > 0)
    b, tau, v = n["beta_imag"][ok], n["tau"][ok], n["n_eff"][ok]
    out = {}
    for bi in np.unique(b):
        k = np.where(b == bi)[0]
        k = k[np.argsort(tau[k])]
        out[bi] = (tau[k], v[k])
    return out


def panel_neff(ax, d):
    """n_eff over (beta, tau): each beta column is interpolated alone in log tau."""
    cols = _neff_columns(d)
    b0, b1 = BETA_LIM
    t0, t1 = NEFF_TAU_LIM
    bs = np.array([bi for bi in cols if b0 - 1e-9 <= bi <= b1 + 1e-9])
    gt = np.linspace(np.log10(t0), np.log10(t1), 420)

    Z = np.full((len(bs), len(gt)), np.nan)
    for i, bi in enumerate(bs):
        tau, v = cols[bi]
        lt, lv = np.log10(tau), np.log10(v)
        inside = (gt >= lt.min()) & (gt <= lt.max())
        Z[i, inside] = np.interp(gt[inside], lt, lv)

    step = np.diff(bs).min() if len(bs) > 1 else .02
    be = np.concatenate([[bs[0] - step / 2], (bs[:-1] + bs[1:]) / 2,
                         [bs[-1] + step / 2]])
    te = 10 ** np.concatenate([[gt[0] - (gt[1] - gt[0]) / 2],
                               (gt[:-1] + gt[1:]) / 2,
                               [gt[-1] + (gt[1] - gt[0]) / 2]])
    BE, TE = np.meshgrid(be, te, indexing="ij")

    ax.set_facecolor("0.90")
    finite = np.isfinite(Z)
    norm = LogNorm(vmin=max(10 ** Z[finite].min(), 1.0),
                   vmax=10 ** Z[finite].max())
    pc = ax.pcolormesh(BE, TE, np.ma.masked_invalid(10 ** Z), cmap=CMAP_NEFF,
                       shading="flat", norm=norm)
    # contours on a lightly smoothed copy (readability only; colours are raw)
    Zs = gaussian_filter(np.where(finite, Z, np.nan), NEFF_CONTOUR_SMOOTH,
                         mode="nearest")
    BC, TC = np.meshgrid(bs, 10 ** gt, indexing="ij")
    lev_col = ["k" if lv in NEFF_DARK_LEVELS else "w" for lv in NEFF_LEVELS]
    cs = ax.contour(BC, TC, np.ma.masked_invalid(10 ** Zs),
                    levels=list(NEFF_LEVELS), colors=lev_col, linewidths=.9,
                    linestyles="--")
    # contour labels near beta = NEFF_LABEL_AT, kept clear of the frame
    tlo, thi = t0 * 1.3, t1 * 0.9
    manual = []
    for segs in cs.allsegs:
        best = None
        for sg in segs:
            if not len(sg):
                continue
            ok = sg[(sg[:, 1] >= tlo) & (sg[:, 1] <= thi)]
            if not len(ok):
                ok = sg[[int(np.argmax(sg[:, 1]))]]
            i = int(np.argmin(np.abs(ok[:, 0] - NEFF_LABEL_AT)))
            if best is None or (abs(ok[i, 0] - NEFF_LABEL_AT)
                                < abs(best[0] - NEFF_LABEL_AT)):
                best = ok[i]
        if best is not None:
            manual.append((best[0], best[1]))
    txts = ax.clabel(cs, fmt="%d", manual=manual, inline=True, inline_spacing=2,
                     fontsize=figstyle.FONT, colors="w")
    dark = {f"{lv:d}" for lv in NEFF_DARK_LEVELS}
    for t in txts or []:
        if t.get_text() in dark:
            t.set_color("k")
    # the six beta used in (b)-(d), in their curve colours
    bs, cols, _ = _beta_norm(d["zz"])
    for bi, c in zip(bs, cols):
        ax.plot([bi], [t0], "^", ms=5.5, color=c, mec="k", mew=.4,
                clip_on=False, zorder=7)
    ax.set_yscale("log")
    ax.set_xlim(*BETA_LIM)
    ax.set_ylim(t0, t1)
    ax.set_xticks([b0] + [v for v in bs if v > b0 + 1e-9])
    ax.set_xticklabels([f"{v:g}" for v in [b0] + [v for v in bs if v > b0 + 1e-9]])
    ax.set_xticks([], minor=True)
    ax.set_xlabel(r"Imaginary-time step $\beta$")
    ax.set_ylabel(r"Projection depth $\tau = m\,\beta$")
    cb = ax.figure.colorbar(pc, ax=ax, pad=.02, fraction=.05, aspect=22)
    cb.set_label(r"Median $n_\mathrm{eff}$")
    ax.set_yticks([2, 5, 10, 20, 50])
    ax.set_yticklabels(["2", "5", "10", "20", "50"])
    ax.set_yticks([], minor=True)
    _panel_label(ax, "(a)", color="w")
    return cb


# --------------------------------------------------------------- panel (b)
def _neff_at(d, beta, taus):
    """n_eff read off the panel-(a) surface at (beta, tau).

    Log-interpolated in log tau within a column; columns within 0.75 grid
    steps of `beta` are averaged in log.
    """
    g = d["neff"]
    cols = np.unique(g["beta_imag"])
    step = np.diff(cols).min() if len(cols) > 1 else 1.0
    sel = cols[np.abs(cols - beta) <= .75 * step]
    out = []
    for c in sel:
        k = np.where(g["beta_imag"] == c)[0]
        k = k[np.argsort(g["tau"][k])]
        out.append(np.interp(np.log10(taus), np.log10(g["tau"][k]),
                             np.log10(g["n_eff"][k])))
    return 10 ** np.mean(out, axis=0)


def _neff_arrows(ax, d, ys, taus, beta, h, colour, side="above", title=None,
                 title_at=NEFF_TITLE_AT, title_ha="center",
                 label_pad=NEFF_LABEL_PAD, title_offset=NEFF_TITLE_OFFSET):
    """Vertical arrows onto a curve, each labelled with its n_eff."""
    tau_c, y_c = ys
    sgn = 1.0 if side == "above" else -1.0
    va = "bottom" if side == "above" else "top"
    for t, n in zip(taus, _neff_at(d, beta, np.asarray(taus, float))):
        y = float(np.interp(t, tau_c, y_c))
        ax.annotate("", xy=(t, y), xytext=(t, y + sgn * h),
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=.8,
                                    shrinkA=0, shrinkB=0, mutation_scale=6),
                    zorder=6)
        ax.annotate(f"{n:.{NEFF_LABEL_DECIMALS}f}", (t, y + sgn * h),
                    xytext=(0, sgn * label_pad), textcoords="offset points",
                    color=colour, ha="center", va=va, fontsize=figstyle.FONT,
                    zorder=6)
    if title:
        y = float(np.interp(title_at, tau_c, y_c))
        ax.annotate(title, (title_at, y + sgn * h),
                    xytext=(title_offset[0], sgn * title_offset[1]),
                    textcoords="offset points",
                    color=colour, ha=title_ha, va=va, fontsize=figstyle.FONT,
                    zorder=6)


def panel_energy(ax, d):
    """Energy per site relative to the exact value, in units of 1e-5."""
    sm = _beta_family(ax, d["zz"], "energy", scale=1e5 / L,
                      offset=d["ref"]["energy"])
    _finish_line_panel(ax, sm,
                       rf"Energy difference $(E_0-E_{REF_MATH})/N$")
    ax.text(0.0, 1.01, r"$\times 10^{-5}$", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=figstyle.FONT)
    _exact_line(ax, 0.0, "below")
    bs, cols, _ = _beta_norm(d["zz"])
    row = next(r for r in _beta_rows(d["zz"], "energy", TAU_LINE_LIM)
               if r[0] == NEFF_ARROW_BETA)
    _neff_arrows(ax, d, (row[2], (row[3] - d["ref"]["energy"]) * 1e5 / L),
                 NEFF_ARROW_TAUS, NEFF_ARROW_BETA, NEFF_ARROW_H,
                 dict(zip(bs, cols))[NEFF_ARROW_BETA],
                 title=rf"$n_\mathrm{{eff}}$ for $\beta={NEFF_ARROW_BETA:g}$",
                 title_ha="left")
    ax.set_ylim(ax.get_ylim()[0], ENERGY_YMAX)
    _beta_legend(ax, d["zz"], sm, loc="upper center", ncol=3,
                 columnspacing=1.2, handlelength=1.3, handletextpad=.5,
                 borderaxespad=.3)
    _panel_label(ax, "(b)")


# --------------------------------------------------------------- panel (c)
def panel_zz(ax, d):
    """Bond correlator vs tau, same diagonal runs as (b)."""
    sm = _beta_family(ax, d["zz"], "zz")
    _finish_line_panel(
        ax, sm, r"Bond correlator $\langle \sigma^z_i \sigma^z_{i+1}\rangle$")
    ax.set_yticks(ZZ_YTICKS)
    ax.set_yticklabels([f"{v:.2f}" for v in ZZ_YTICKS])
    _exact_line(ax, d["ref"]["zz"], "below")
    bs, cols = _beta_norm(d["zz"])[:2]
    row = next(r for r in _beta_rows(d["zz"], "zz", TAU_LINE_LIM)
               if r[0] == NEFF_ARROW_BETA_C)
    _neff_arrows(ax, d, (row[2], row[3]), NEFF_ARROW_TAUS_C,
                 NEFF_ARROW_BETA_C, NEFF_ARROW_H_C,
                 dict(zip(bs, cols))[NEFF_ARROW_BETA_C], side="below",
                 title=rf"$n_\mathrm{{eff}}$ for $\beta={NEFF_ARROW_BETA_C:g}$",
                 title_at=NEFF_TITLE_AT_C, title_ha="right",
                 label_pad=NEFF_LABEL_PAD_C,
                 title_offset=NEFF_TITLE_OFFSET_C)
    _headroom(ax, .10)
    _panel_label(ax, "(c)")


# --------------------------------------------------------------- panel (d)
def _derived_xtot(d, beta, tau_lim):
    """<sum X> from the diagonal run: (J_z*N*<ZZ> - E)/h_x (exact identity)."""
    g = d["zz"]
    k = np.where((g["beta_imag"] == beta)
                 & (g["tau"] >= tau_lim[0]) & (g["tau"] <= tau_lim[1]))[0]
    k = k[np.argsort(g["tau"][k])]
    return g["tau"][k], (J_Z * L * g["zz"][k] - g["energy"][k]) / HX


def panel_xtot(ax, d):
    """Per-site transverse magnetisation: forward walking vs diagonal-derived."""
    g = d["xtot"]
    sm = _beta_family(ax, g, "xtot", scale=1.0 / L, tau_lim=TAU_LINE_LIM_D,
                      tau_max_by_beta=XTOT_TAU_MAX)
    for beta in DERIVED_BETAS:
        t, x = _derived_xtot(d, beta, TAU_LINE_LIM_D)
        ax.plot(t, x / L, "-", color=C_DERIVED, lw=1.2, zorder=5,
                label=rf"diagonal, $\beta={beta:g}$")

    _finish_line_panel(ax, sm,
                       r"Transverse magnetisation $\langle \sigma^x_i\rangle$",
                       tau_lim=TAU_LINE_LIM_D)
    _exact_line(ax, d["ref"]["sx"] / L, "below")
    shown = np.concatenate([v for *_, v, _ in
                            _beta_rows(g, "xtot", TAU_LINE_LIM_D,
                                       XTOT_TAU_MAX)])
    lo = min(shown.min(), d["ref"]["sx"]) / L
    hi = shown.max() / L
    ax.set_ylim(lo - .05 * (hi - lo), hi + .05 * (hi - lo))
    ax.set_yticks(XTOT_YTICKS)
    ax.set_yticklabels([f"{v:.2f}" for v in XTOT_YTICKS])
    ax.legend(loc="upper center", borderaxespad=.3)
    _headroom(ax, .10)
    _panel_label(ax, "(d)")


def _match_width(ax, ref):
    p, q = ax.get_position(), ref.get_position()
    ax.set_position([q.x0, p.y0, q.width, p.height])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=OUT)
    ap.add_argument("--no-latex", action="store_true")
    a = ap.parse_args()
    d = load()
    fig, axes = prepare_standard_figure(
        nrows=2, ncols=2, width=figstyle.WIDTH_FULL, aspect_ratio=1.38,
        use_latex=not a.no_latex)
    for ax in axes.ravel():
        ax.tick_params(which="both", top=False, right=False)
    # set before the colourbar steals width from panel (a)
    axes[0, 0].get_subplotspec().get_gridspec().set_width_ratios(
        [LEFT_COL_FRAC, 2.0 - LEFT_COL_FRAC])
    panel_neff(axes[0, 0], d)
    panel_energy(axes[0, 1], d)
    panel_zz(axes[1, 0], d)
    panel_xtot(axes[1, 1], d)
    fig.tight_layout(pad=.4, w_pad=.1, h_pad=.8)
    _match_width(axes[1, 0], axes[0, 0])
    # no bbox_inches="tight": keep the canvas exactly \textwidth wide
    fig.savefig(a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
