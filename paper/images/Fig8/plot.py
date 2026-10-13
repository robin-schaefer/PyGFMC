#!/usr/bin/env python3
"""Fig. 8: geometries of the benchmark lattices.

Top row:    the random connected graphs R4, R9, R13 (N = 4, 9, 13).
Bottom row: the two-leg ladder L100 (L = 50 rungs, N = 100, open legs); only the
            first and last few rungs are drawn, dashed bonds mark the omitted bulk.

Reads data.h5 from this folder and writes ../Fig8.pdf.
"""
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figstyle  # noqa: E402  (shared width and font sizes, ../figstyle.py)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data.h5")
OUTFILE = os.path.join(HERE, "..", "Fig8.pdf")

FONT = figstyle.FONT
figstyle.apply()

NODE_KW = dict(s=22, c="white", edgecolors="black", linewidths=0.7, zorder=3)
EDGE_KW = dict(color="black", linewidth=0.7, zorder=1)

RANDOM_SIZES = [4, 9, 13]
LADDER_RUNGS_SHOWN = 8        # total rungs drawn (half at each open end)
PANEL = ["(a)", "(b)", "(c)", "(d)"]

width = figstyle.WIDTH_FULL
fig = plt.figure(figsize=(width, width * 0.40))
gs = fig.add_gridspec(2, len(RANDOM_SIZES), height_ratios=[1.0, 0.50],
                      hspace=0.05, wspace=0.05)


def draw_graph(ax, edges, pos):
    for i, j in edges:
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], **EDGE_KW)
    xy = np.array([pos[k] for k in sorted(pos)])
    ax.scatter(xy[:, 0], xy[:, 1], **NODE_KW)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(0.12)


with h5py.File(DATA, "r") as f:
    graphs = {N: (f[f"R{N}/edges"][()], f[f"R{N}/positions"][()]) for N in RANDOM_SIZES}
    N_ladder = int(f["L100"].attrs["N"])

# top row: the random connected graphs
for col, N in enumerate(RANDOM_SIZES):
    edges, xy = graphs[N]
    pos = {k: xy[k] for k in range(N)}
    ax = fig.add_subplot(gs[0, col])
    draw_graph(ax, [tuple(e) for e in edges], pos)
    ax.set_title(rf"{PANEL[col]} R{N} ($N={N}$)", fontsize=FONT, pad=1)

# bottom row: the two-leg ladder, both open ends with the bulk omitted
ax = fig.add_subplot(gs[1, :])
keep = LADDER_RUNGS_SHOWN // 2      # rungs drawn at each end
GAP = 1.8                           # horizontal width of the omitted bulk

x_left = [float(i) for i in range(keep)]
x_right = [x_left[-1] + GAP + 1.0 + i for i in range(keep)]
xs = x_left + x_right

# node index: 2*c (top leg) and 2*c+1 (bottom leg) for drawn column c
pos = {}
for c, x in enumerate(xs):
    pos[2 * c] = (x, 1.0)
    pos[2 * c + 1] = (x, 0.0)

sub = [(2 * c, 2 * c + 1) for c in range(len(xs))]                 # rungs
for block in (range(0, keep), range(keep, 2 * keep)):              # leg bonds
    b = list(block)
    for c0, c1 in zip(b[:-1], b[1:]):
        sub += [(2 * c0, 2 * c1), (2 * c0 + 1, 2 * c1 + 1)]

draw_graph(ax, sub, pos)
for y in (0.0, 1.0):
    ax.plot([x_left[-1], x_right[0]], [y, y], color="black", linewidth=0.7,
            linestyle=(0, (1.8, 1.8)), zorder=1)
ax.set_title(rf"{PANEL[len(RANDOM_SIZES)]} L{N_ladder} ($N={N_ladder}$)",
             fontsize=FONT, pad=1)
ax.set_xlim(x_left[0] - 0.6, x_right[-1] + 0.6)
ax.set_ylim(-0.4, 1.4)
ax.set_aspect("auto")

fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02)
plt.savefig(OUTFILE)
print(f"wrote {os.path.normpath(OUTFILE)}")
