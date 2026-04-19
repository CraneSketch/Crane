"""Fig. 12: query-weighted ARE at gamma in {-0.5, 0.5, 1.0} (query popularity ~ f(e)^gamma; NotreDame omitted
because all its weights are one). data/weighted_are.csv -> out/weighted_are/."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "out" / "weighted_are"
SUMMARY_PATH = BASE_DIR / "data" / "weighted_are.csv"

QUERY_WEIGHT_ALPHAS = [-0.5, 0.5, 1.0]
DATASETS = ["Lkml", "CAIDA2018", "WiKiTalk", "StackOverflow"]
DATASET_SIZES = {
    "Lkml": [20_000, 40_000, 80_000, 200_000, 400_000, 800_000],
    "CAIDA2018": [2_000_000, 8_000_000, 16_000_000, 27_100_000],
    "WiKiTalk": [2_000_000, 8_000_000, 16_000_000, 25_000_000],
    "StackOverflow": [2_000_000, 8_000_000, 16_000_000, 63_500_000],
}
DATASET_LABELS = {
    "Lkml": ["20K", "40K", "80K", "200K", "400K", "800K"],
    "CAIDA2018": ["2M", "8M", "16M", "27.1M"],
    "WiKiTalk": ["2M", "8M", "16M", "25.0M"],
    "StackOverflow": ["2M", "8M", "16M", "63.5M"],
}
METHOD_ORDER = ["Crane", "TCM", "GSS", "Auxo", "HourglassSketch", "Mayfly"]

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

# Match EdgeQuery/plot_main.py exactly so each method keeps the same visual
# identity across the main experiment and this weighted-ARE extension. Six
# method colors are justified here by that cross-figure identity requirement;
# markers and black outlines provide a second, grayscale-friendly encoding.
METHOD_STYLES = {
    "Crane": {
        "color": "#d94701",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 4.0,
        "markersize": 18,
        "zorder": 10,
    },
    "TCM": {
        "color": "#4477AA",
        "marker": "s",
        "linestyle": "-",
        "linewidth": 2.5,
        "markersize": 16,
        "zorder": 5,
    },
    "GSS": {
        "color": "#228833",
        "marker": "^",
        "linestyle": "-",
        "linewidth": 2.5,
        "markersize": 16,
        "zorder": 5,
    },
    "Auxo": {
        "color": "#AA3377",
        "marker": "D",
        "linestyle": "-",
        "linewidth": 2.5,
        "markersize": 16,
        "zorder": 5,
    },
    "HourglassSketch": {
        "color": "#66CCEE",
        "marker": "v",
        "linestyle": "-",
        "linewidth": 2.5,
        "markersize": 16,
        "zorder": 5,
    },
    "Mayfly": {
        "color": "#8c564b",
        "marker": "P",
        "linestyle": "-",
        "linewidth": 2.5,
        "markersize": 16,
        "zorder": 5,
    },
}


def _alpha_tag(alpha: float) -> str:
    """0.5 -> 'alpha0p5', 1.0 -> 'alpha1p0', -0.5 -> 'alpham0p5'.

    Positive tags are unchanged from the originally published filenames;
    negatives use the leading 'm' for minus that the per_run pivot columns
    already use (are_am0p5).
    """
    sign = "m" if alpha < 0 else ""
    return "alpha" + sign + f"{abs(alpha):.1f}".replace(".", "p")


def load_alpha_summary(alpha: float) -> pd.DataFrame:
    """Rows of one weighting alpha with columns dataset, method, task_index, are_mean, are_std."""
    frame = pd.read_csv(SUMMARY_PATH)
    frame = frame[np.isclose(frame["alpha"], alpha)].copy()
    frame["task_index"] = [DATASET_SIZES[d].index(int(s)) for d, s in zip(frame["dataset"], frame["stream_size"])]
    return frame.rename(columns={"mean": "are_mean", "std": "are_std"})


def plot_dataset(frame: pd.DataFrame, dataset: str, alpha: float) -> None:
    """Save one weighted-ARE curve against stream size."""
    dataset_frame = frame[frame["dataset"] == dataset]
    x_values = np.arange(len(DATASET_LABELS[dataset]))

    figure, axis = plt.subplots(figsize=(8.4, 9.0))
    for method in METHOD_ORDER:
        series = (
            dataset_frame[dataset_frame["method"] == method]
            .set_index("task_index")
            .reindex(x_values)
        )
        means = series["are_mean"].to_numpy()
        stds = series["are_std"].fillna(0.0).to_numpy()
        lower_errors = np.minimum(stds, means * 0.95)
        style = METHOD_STYLES[method]

        axis.errorbar(
            x_values,
            means,
            yerr=np.vstack([lower_errors, stds]),
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            zorder=style["zorder"],
            capsize=5,
            capthick=2,
            elinewidth=2,
            markeredgecolor="black",
            markeredgewidth=1.2,
        )

    axis.set_yscale("log")
    axis.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=12))
    axis.yaxis.set_minor_locator(
        ticker.LogLocator(
            base=10,
            subs=np.arange(2, 10) * 0.1,
            numticks=20,
        )
    )
    axis.set_xticks(x_values)
    axis.set_xticklabels(
        DATASET_LABELS[dataset],
        rotation=45,
        ha="right",
    )
    axis.set_xlabel("Stream Size")
    axis.set_ylabel("ARE")
    axis.grid(True, which="major", linewidth=1.2, alpha=0.65)
    axis.grid(True, which="minor", axis="y", linewidth=0.6, alpha=0.25)

    figure.tight_layout()
    output_path = OUTPUT_DIR / f"weighted_are_{_alpha_tag(alpha)}_{dataset}.pdf"
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(figure)
    print(f"Saved: {output_path}")


def plot_legend() -> None:
    """Save one shared legend for the four dataset plots."""
    figure, axis = plt.subplots(figsize=(42, 2.2))
    axis.set_axis_off()
    for method in METHOD_ORDER:
        style = METHOD_STYLES[method]
        axis.plot(
            [],
            [],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            markeredgecolor="black",
            markeredgewidth=1.2,
            label=method,
        )
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="center",
        ncol=len(METHOD_ORDER),
        frameon=False,
    )
    output_path = OUTPUT_DIR / "weighted_are_alpha0p5_legend.pdf"
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(figure)
    print(f"Saved: {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for alpha in QUERY_WEIGHT_ALPHAS:
        frame = load_alpha_summary(alpha)
        for dataset in DATASETS:
            plot_dataset(frame, dataset, alpha)
    plot_legend()


if __name__ == "__main__":
    main()
