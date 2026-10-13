"""Shared matplotlib style for all paper figures (images/Fig*/plot.py).

Every figure is generated at exactly the width it is printed at, so the font
size below is the size on the page (LaTeX scale factor 1):

    WIDTH_FULL  -> \\includegraphics[width=\\textwidth]      (Fig. 4, 5, 8)
    WIDTH_COL   -> \\includegraphics[width=0.7\\textwidth]   (Fig. 3, 6, 7, 9)

All text (labels, ticks, legends, insets, colour bars, annotations) uses the
single size FONT; do not set other font sizes in the plot scripts.

Import from a plot.py one directory down with
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import figstyle
"""
import cycler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEXTWIDTH = 426.79135 / 72.27   # SciPost \textwidth in inches
WIDTH_FULL = TEXTWIDTH
WIDTH_COL = 0.7 * TEXTWIDTH
FONT = 8          # pt on the page, for all text


def apply(use_latex=True):
    """Set the paper-wide rcParams (serif, inward ticks, frameless legends)."""
    if use_latex:
        plt.rc("text", usetex=True)
        plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
    plt.rc("font", family="serif", size=FONT)
    plt.rc("axes", linewidth=0.5, labelsize=FONT, titlesize=FONT)
    plt.rc("lines", linewidth=1)
    plt.rc("xtick", labelsize=FONT, direction="in", top=True)
    plt.rc("ytick", labelsize=FONT, direction="in", right=True)
    plt.rc("legend", fontsize=FONT, title_fontsize=FONT,
           numpoints=1, frameon=False, handlelength=1)
    plt.rc("errorbar", capsize=2)
    plt.rc("savefig", dpi=800)
    plt.rcParams["axes.prop_cycle"] = cycler.cycler(color=[
        "tab:blue", "tab:red", "tab:green", "0", "tab:orange", "tab:brown",
        "tab:purple", "tab:grey", "tab:olive", "tab:pink", "tab:cyan"])
