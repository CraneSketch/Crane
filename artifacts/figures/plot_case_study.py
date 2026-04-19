"""Fig. 9: downstream queries on the real workloads D1-D5 (ARE | nDCG@100 | Recall@100). data/case_study.csv -> out/casestudy/."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "case_study.csv"
OUT_DIR = BASE_DIR / "out" / "casestudy"

DATASETS = ["D1", "D2", "D3", "D4", "D5"]
METHODS = ["Crane", "Mayfly", "HourglassSketch"]
PANELS = [
    ("are", None, "ARE", "lower", "are"),
    ("ndcg", "100", "nDCG@100", "higher", "ndcg100"),
    ("recall", "100", "Recall@100", "higher", "recall100"),
]

AXIS_LABEL_FONT_SIZE = 46
TICK_FONT_SIZE = 38
LEGEND_FONT_SIZE = 42

sns.set_theme(style="whitegrid", font_scale=4)
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "serif",
    "font.serif": ["Linux Libertine O", "Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Keep method identity consistent with EdgeQuery/plot_main.py. Solid fills stay
# legible when the panels are reduced to publication size.
COLORS = {"Crane": "#d94701", "Mayfly": "#8c564b", "HourglassSketch": "#66CCEE"}



def load_tidy():
    return pd.read_csv(DATA_PATH)


def draw_panel(ax, tidy, metric_label, better):
    sub = tidy[tidy["metric"] == metric_label]
    dsets = [d for d in DATASETS if d in set(sub["dataset"])]
    x = np.arange(len(dsets))
    width = 0.27
    for i, method in enumerate(METHODS):
        cells = [sub[(sub["dataset"] == d) & (sub["method"] == method)]
                 for d in dsets]
        vals = [float(c["value"].iloc[0]) if len(c) else np.nan for c in cells]
        errs = [float(c["std"].iloc[0]) if len(c) else 0.0 for c in cells]
        # Upper-half error bars only: the lower whisker adds clutter on bars.
        ax.bar(x + (i - 1) * width, vals, width, color=COLORS[method],
               edgecolor="black", linewidth=1.4,
               yerr=np.vstack([np.zeros(len(errs)), errs]),
               error_kw=dict(ecolor="black", lw=2.2,
                             capsize=6, capthick=2.2),
               label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(dsets, fontsize=TICK_FONT_SIZE, fontweight="bold")
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    ax.set_xlabel("Dataset", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(metric_label + (" ↓" if better == "lower" else " ↑"),
                  fontsize=AXIS_LABEL_FONT_SIZE)
    if metric_label == "ARE":
        top = float(np.ceil((sub["value"] + sub["std"]).max() + 0.5))
        ax.set_ylim(0, top)
        ax.set_yticks(np.arange(0, top + 0.1, 4 if top > 8 else 2))
        ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    else:
        ax.set_ylim(0, 1.02)
    ax.grid(axis="x", visible=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def render_individual(tidy):
    out_dir = OUT_DIR
    for _, _, label, better, slug in PANELS:
        fig, ax = plt.subplots(figsize=(8.4, 9.0))
        draw_panel(ax, tidy, label, better)
        fig.tight_layout()
        out = out_dir / f"case_study_v2_{slug}.pdf"
        fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
        plt.close(fig)
        print("wrote", out)


def render_legend():
    fig, ax = plt.subplots(figsize=(20, 2.0))
    handles = [Patch(facecolor=COLORS[m], edgecolor="black",
                     linewidth=1.4, label=m) for m in METHODS]
    ax.legend(handles=handles, ncol=3, loc="center", frameon=False,
              fontsize=LEGEND_FONT_SIZE,
              handlelength=1.6, handleheight=1.1, columnspacing=1.8)
    ax.axis("off")
    out = OUT_DIR / "case_study_v2_legend.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("wrote", out)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tidy = load_tidy()
    render_individual(tidy)
    render_legend()


if __name__ == "__main__":
    main()
