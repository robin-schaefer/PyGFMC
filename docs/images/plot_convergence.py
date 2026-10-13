"""Convergence in the projection depth for three step sizes (docs figure).

Energy, bond correlator and transverse magnetization of the critical
transverse-field Ising chain (N = 100, h_x = 1) versus the projection depth
tau = p*beta/2 at step sizes beta = 0.25, 1 and 1.5, each as the difference to the
exact free-fermion value.  Same diagonal data as Fig. 4(b)-(c) of the paper,
read from ../../paper/images/Fig4/data.h5.

The transverse magnetization is reconstructed from the energy and the bond
correlator through the exact identity of this Hamiltonian,
H = J_z sum_i sz_i sz_i+1 - h_x sum_i sx_i, i.e.
sum_i <sx_i> = (J_z N <sz sz> - E) / h_x, so it covers the same tau range
(as the red curve of Fig. 4(d)).  It is not independent data.  Its error band
propagates the two errors as independent; the energy error per site is ~40x
smaller than the correlator error, so the unknown covariance barely matters.

    python plot_convergence.py       # writes convergence.pdf and .svg
"""
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "paper", "images", "Fig4", "data.h5")
BETAS = {0.25: "tab:red", 1.0: "tab:blue", 1.5: "tab:green"}  # step size -> color
TAU_ZOOM = 25.0                                      # insets show tau >= TAU_ZOOM
INSET_YLIM = 7e-3                                    # same symmetric range in both insets
INSET_TICK = 5e-3

plt.rcParams.update({"font.family": "serif", "font.size": 12})


def rows(group, beta):
    """The rows of one data group at step size beta, sorted by tau."""
    sel = np.isclose(group["beta"][:], beta)
    order = np.argsort(group["tau"][sel])
    return {k: group[k][sel][order] for k in group.keys()}


with h5py.File(DATA, "r") as f:
    N = int(f.attrs["N"])
    J_z, h_x = float(f.attrs["J_z"]), float(f.attrs["h_x"])
    exact = dict(f["exact"].attrs)
    diag = {beta: rows(f["diagonal"], beta) for beta in BETAS}


def panels(d):
    """(value, error) per panel, all as differences to the exact value per site."""
    # sum_i <sx_i> from the exact identity sum_i <sx_i> = (J_z N <zz> - E) / h_x
    sx = (J_z * N * d["zz"] - d["energy"]) / h_x
    sx_err = np.hypot(J_z * N * d["zz_err"], d["energy_err"]) / abs(h_x)
    return [((d["energy"] - exact["energy"]) / N * 1e5, d["energy_err"] / N * 1e5),
            (d["zz"] - exact["zz"], d["zz_err"]),
            ((sx - exact["sx_total"]) / N, sx_err / N)]


YLABELS = [r"$(E - E_\mathrm{exact})/N \;[10^{-5}]$",
           r"$\frac{1}{N}\sum_i \langle \sigma^z_i \sigma^z_{i+1} \rangle"
           r" - \mathrm{exact}$",
           r"$\frac{1}{N}\sum_i \langle \sigma^x_i \rangle - \mathrm{exact}$"]
# panel label position inside each axis, in a corner the curves leave free
LABEL_AT = ((0.96, 0.93, "right", "top"), (0.04, 0.05, "left", "bottom"),
            (0.05, 0.93, "left", "top"))
# insets for (b) and (c): the late-tau region, where the differences are of
# order 1e-3, on its own linear scale, placed where each panel is empty
INSETS = ([0.42, 0.12, 0.53, 0.45], [0.42, 0.45, 0.53, 0.45])

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
insets = [ax.inset_axes(loc) for ax, loc in zip(axes[1:], INSETS)]
tau_min = min(d["tau"].min() for d in diag.values())
tau_max = max(d["tau"].max() for d in diag.values())

for beta, color in BETAS.items():
    tau = diag[beta]["tau"]
    zoom = tau >= TAU_ZOOM
    for i, (val, err) in enumerate(panels(diag[beta])):
        ax = axes[i]
        ax.fill_between(tau, val - err, val + err, color=color, alpha=0.3, lw=0)
        ax.plot(tau, val, "-", color=color, lw=1.2, label=f"{beta:g}")
        if i > 0:
            ins = insets[i - 1]
            assert np.all(np.abs(val[zoom]) + err[zoom] < INSET_YLIM)
            ins.fill_between(tau[zoom], (val - err)[zoom], (val + err)[zoom],
                             color=color, alpha=0.3, lw=0)
            ins.plot(tau[zoom], val[zoom], "-", color=color, lw=1.0)

for ax, label, ylabel, (lx, ly, lha, lva) in zip(axes, "abc", YLABELS, LABEL_AT):
    ax.axhline(0.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel(r"Projection depth $\tau = p\beta/2$")
    ax.set_xlim(tau_min, tau_max)
    ax.set_ylabel(ylabel)
    ax.text(lx, ly, f"({label})", transform=ax.transAxes, ha=lha, va=lva)
axes[0].legend(title=r"step size $\beta$", loc="upper center")

for ins in insets:
    ins.axhline(0.0, color="k", ls="--", lw=0.6)
    ins.set_xlim(TAU_ZOOM, tau_max)
    ins.set_ylim(-INSET_YLIM, INSET_YLIM)
    ins.set_xticks([])
    ins.set_yticks([-INSET_TICK, 0.0, INSET_TICK],
                   [r"$-5\cdot 10^{-3}$", r"$0$", r"$5\cdot 10^{-3}$"])
fig.tight_layout()

for ext in ("pdf", "svg"):
    fig.savefig(os.path.join(HERE, f"convergence.{ext}"))
print("wrote convergence.pdf/.svg")
