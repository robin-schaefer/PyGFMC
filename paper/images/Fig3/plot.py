#!/usr/bin/env python3
"""Fig. 3: GFMC throughput (iterations/s) vs walker population N_walkers.

Blue: CPU runs with 1-32 cores. Red: 1, 2, 4 GPUs, A100 (solid) and H200 (dashed; drawn with a fine dot pattern).
Reads data.csv from this folder and writes ../Fig3.pdf.
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figstyle  # noqa: E402  (shared width and font sizes, ../figstyle.py)

figstyle.apply()

MARKER_SIZE = 3
LINE_WIDTH = 0.9

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data.csv")
_ap = argparse.ArgumentParser(description="Plot Fig. 3 (throughput).")
_ap.add_argument("--out", default=os.path.join(HERE, "..", "Fig3.pdf"),
                 help="output file; the extension sets the format (default: ../Fig3.pdf)")
_ap.add_argument("--xlabel", default=r"Number of walkers $N_\mathrm{walkers}$",
                 help="x-axis label (the documentation uses 'Number of walkers')")
_args = _ap.parse_args()
OUTFILE = _args.out

CPU_CORES = [1, 2, 4, 8, 16, 32]
N_GPUS = [1, 2, 4]


def load(path):
    """Return {(device, n_devices): (n_walkers, mean it/s)} sorted by n_walkers."""
    with open(path) as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    curves = {}
    for r in rows:
        key = (r["device"], int(r["n_devices"]))
        curves.setdefault(key, []).append((int(r["n_walkers"]), float(r["it_per_sec_mean"])))
    return {k: tuple(np.array(c) for c in zip(*sorted(v))) for k, v in curves.items()}


curves = load(DATA)

width = figstyle.WIDTH_COL
fig, ax = plt.subplots(figsize=(width, width * 9 / 16))

# CPU curves -- Blues
cpu_colors = plt.get_cmap("Blues")(np.linspace(0.35, 0.9, len(CPU_CORES)))
cpu_handles = []
for n_cpu, color in zip(CPU_CORES, cpu_colors):
    nw, m = curves[("CPU", n_cpu)]
    ax.plot(nw, m, marker="o", markersize=MARKER_SIZE, linewidth=LINE_WIDTH, color=color)
    cpu_handles.append(Line2D([0], [0], color=color, marker="o", markersize=MARKER_SIZE,
                              linewidth=LINE_WIDTH, label=fr"${n_cpu}$"))

# GPU curves -- Reds; same shade per GPU count, A100 solid, H200 dashed
gpu_colors = plt.get_cmap("Reds")(np.linspace(0.45, 0.9, len(N_GPUS)))
gpu_curves = {}
for device, ls in (("A100", "-"), ("H200", (0, (1, 1)))):
    for n_gpu, color in zip(N_GPUS, gpu_colors):
        nw, m = curves[(device, n_gpu)]
        ax.plot(nw, m, marker="o", markersize=MARKER_SIZE, linewidth=LINE_WIDTH,
                color=color, linestyle=ls)
        gpu_curves[(device, n_gpu)] = (nw, m)

gpu_handles = [Line2D([0], [0], color=color, marker="o", markersize=MARKER_SIZE,
                      linewidth=LINE_WIDTH, label=fr"${n_gpu}$")
               for n_gpu, color in zip(N_GPUS, gpu_colors)]


def label_curve(device, n_gpu, label, n_walker_target, dy_factor):
    nw, m = gpu_curves[(device, n_gpu)]
    idx = int(np.argmin(np.abs(nw - n_walker_target)))
    ax.text(nw[idx], m[idx] * dy_factor, label, rotation=-30,
            color="k", ha="left", va="bottom", fontsize=figstyle.FONT)


label_curve("A100", 4, "A100 (solid)", 2**21, dy_factor=0.021)
label_curve("H200", 4, "H200 (dashed)", 2**21, dy_factor=0.30)

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel(_args.xlabel)
ax.set_ylabel(r"throughput [iter/s]")

leg_cpu = ax.legend(handles=cpu_handles, title=r"\# of CPUs", loc="lower left",
                    frameon=False, handlelength=1.5, ncol=2, columnspacing=1.0)
ax.add_artist(leg_cpu)
ax.legend(handles=gpu_handles, title=r"\# of GPUs", loc="upper right",
          frameon=False, handlelength=1.5, ncol=3, columnspacing=1.0)

fig.tight_layout()
plt.savefig(OUTFILE)
print(f"Saved -> {os.path.normpath(OUTFILE)}")
