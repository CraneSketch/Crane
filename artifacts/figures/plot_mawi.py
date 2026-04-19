"""Fig. 13: accuracy on the MAWI trace vs total memory budget. data/mawi.csv -> out/mawi/."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "mawi.csv"
OUT_DIR = BASE_DIR / "out" / "mawi"

BUDGETS_MB = [1.0, 2.0, 5.0, 10.0, 30.0, 100.0]

LEGEND_PATH = OUT_DIR / "mawi_legend.pdf"


sns.set_theme(style="whitegrid", font_scale=4)
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.family": "serif",
        "font.serif": [
            "Linux Libertine O",
            "Liberation Serif",
            "Times New Roman",
            "DejaVu Serif",
        ],
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    }
)

COLORBLIND = sns.color_palette("colorblind")
METHOD_STYLES = {
    "Crane": {
        "color": COLORBLIND[1],
        "marker": "o",
        "linewidth": 4.0,
        "markersize": 18,
        "zorder": 10,
    },
    "Mayfly": {
        "color": COLORBLIND[0],
        "marker": "P",
        "linewidth": 3.2,
        "markersize": 17,
        "zorder": 6,
    },
    "HourglassSketch": {
        "color": COLORBLIND[2],
        "marker": "^",
        "linewidth": 3.2,
        "markersize": 17,
        "zorder": 4,
    },
}
METHOD_ORDER = ["Crane", "Mayfly", "HourglassSketch"]
METHOD_LABELS = {
    "Crane": "Crane",
    "Mayfly": "Mayfly",
    "HourglassSketch": "HourglassSketch",
}


def load_summary() -> pd.DataFrame:
    summary = pd.read_csv(DATA_PATH)
    for method in METHOD_ORDER:
        budgets = sorted(summary.loc[summary["method"] == method, "budget_mb"].tolist())
        if budgets != BUDGETS_MB:
            raise ValueError(f"{method}: expected budgets {BUDGETS_MB}, found {budgets}")
    return summary


def _budget_formatter(value: float, _position: int) -> str:
    if value in BUDGETS_MB:
        return f"{value:g}"
    return ""


def plot_metric(
    ax: plt.Axes,
    summary: pd.DataFrame,
    metric: str,
) -> None:
    """Draw one EdgeQuery-style memory-budget accuracy curve."""
    mean_column = f"{metric}_mean"
    std_column = f"{metric}_std"

    for method in METHOD_ORDER:
        method_data = (
            summary[summary["method"] == method]
            .set_index("budget_mb")
            .reindex(BUDGETS_MB)
        )
        style = METHOD_STYLES[method]
        ax.errorbar(
            BUDGETS_MB,
            method_data[mean_column].to_numpy(),
            yerr=method_data[std_column].to_numpy(),
            color=style["color"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            zorder=style["zorder"],
            capsize=5,
            capthick=2,
            elinewidth=2,
            markeredgecolor="black",
            markeredgewidth=1.2,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.8, 125)
    ax.set_xticks(BUDGETS_MB)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_budget_formatter))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
    ax.yaxis.set_minor_locator(
        ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20)
    )
    ax.set_xlabel("Memory Budget (MB)")
    ax.set_ylabel(metric.upper())
    ax.grid(True, which="major", linewidth=1.2, alpha=0.65)
    ax.grid(True, which="minor", axis="y", linewidth=0.6, alpha=0.25)


def plot_individual(summary: pd.DataFrame, metric: str) -> None:
    """Save one standalone vector plot for a metric."""
    figure, axis = plt.subplots(figsize=(9.5, 9))
    plot_metric(axis, summary, metric)
    figure.tight_layout()

    stem = OUT_DIR / f"mawi_{metric}"
    figure.savefig(
        f"{stem}.pdf", dpi=300, bbox_inches="tight", transparent=True
    )
    plt.close(figure)


def plot_legend() -> None:
    """Save one shared legend serving both the ARE and AAE plots."""
    figure, axis = plt.subplots(figsize=(21, 2.2))
    axis.set_axis_off()
    for method in METHOD_ORDER:
        style = METHOD_STYLES[method]
        axis.plot(
            [],
            [],
            color=style["color"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            markeredgecolor="black",
            markeredgewidth=1.2,
            label=METHOD_LABELS[method],
        )
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="center",
        ncol=len(METHOD_ORDER),
        frameon=False,
    )
    figure.savefig(
        LEGEND_PATH, dpi=300, bbox_inches="tight", transparent=True
    )
    plt.close(figure)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    plot_individual(summary, "are")
    plot_individual(summary, "aae")
    plot_legend()
    print(f"Saved: {OUT_DIR / 'mawi_are.pdf'}")
    print(f"Saved: {OUT_DIR / 'mawi_aae.pdf'}")
    print(f"Saved: {LEGEND_PATH}")


if __name__ == "__main__":
    main()
