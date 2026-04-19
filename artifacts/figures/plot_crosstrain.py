"""Fig. 6: Crane / Mayfly cross-training heatmaps over Zipf exponents. data/crosstrain.csv -> out/robustness/.
Diagonal cells: matched-distribution mean ARE and s.d.; off-diagonal cells: ARE relative to the row diagonal."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "out" / "robustness"

ALPHAS = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]

DATA_PATH = DATA_DIR / "crosstrain.csv"
CRANE_PDF_PATH = OUT_DIR / "robustness_v2_crane.pdf"
MAYFLY_PDF_PATH = OUT_DIR / "robustness_v2_mayfly.pdf"
COLORBAR_PDF_PATH = OUT_DIR / "robustness_v2_shared_colorbar.pdf"
DIAGONAL_LEGEND_PDF_PATH = OUT_DIR / "robustness_v2_diagonal_legend.pdf"

AXIS_LABEL_FONT_SIZE = 69
TICK_FONT_SIZE = 54
CELL_FONT_SIZE = 42
DIAGONAL_CELL_FONT_SIZE = 39
LEGEND_FONT_SIZE = 45
COLORBAR_LABEL_FONT_SIZE = 45
COLORBAR_TICK_FONT_SIZE = 45


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


def _matrix(frame: pd.DataFrame, value: str, method: str) -> pd.DataFrame:
    subset = frame[frame["method"] == method]
    return (
        subset.pivot(index="train_alpha", columns="test_alpha", values=value)
        .reindex(index=ALPHAS, columns=ALPHAS)
    )


def _format_absolute(value: float) -> str:
    """Format a diagonal absolute ARE compactly."""
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _format_multiplier(value: float) -> str:
    """Format an off-diagonal relative ARE multiplier compactly."""
    if value < 0.1:
        return f"{value:.2f}×"
    if value >= 100:
        return f"{value:.0f}×"
    if value >= 10:
        return f"{value:.1f}×"
    return f"{value:.1f}×"


def _outline_diagonal(ax: plt.Axes) -> None:
    for index in range(len(ALPHAS)):
        ax.add_patch(
            Rectangle(
                (index, index),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=3.2,
            )
        )


def _style_heatmap_axis(
    ax: plt.Axes,
    show_ylabel: bool,
    show_xlabel: bool,
) -> None:
    labels = [f"{alpha:.1f}" for alpha in ALPHAS]
    ax.set_xticklabels(
        labels, rotation=0, fontsize=TICK_FONT_SIZE, fontweight="bold"
    )
    ax.set_yticklabels(
        labels, rotation=0, fontsize=TICK_FONT_SIZE, fontweight="bold"
    )
    ax.set_xlabel(
        r"Test Zipf Coefficient $\alpha$" if show_xlabel else "",
        fontsize=AXIS_LABEL_FONT_SIZE,
        labelpad=28,
    )
    ax.set_ylabel(
        r"Train Zipf Coefficient $\alpha$" if show_ylabel else "",
        fontsize=AXIS_LABEL_FONT_SIZE,
        labelpad=28,
    )
    _outline_diagonal(ax)


def _style_colorbar(colorbar, label: str, fontsize: int = 18) -> None:
    colorbar.set_label(label, fontsize=fontsize, fontweight="bold", labelpad=28)
    colorbar.ax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE)
    for tick_label in colorbar.ax.get_yticklabels():
        tick_label.set_fontweight("bold")


def _plot_hybrid_heatmap(
    mean_matrix: pd.DataFrame,
    std_matrix: pd.DataFrame,
    relative_matrix: pd.DataFrame,
    norm: TwoSlopeNorm,
    output_path: Path,
) -> None:
    """Save raw diagonal ARE and relative off-diagonal multipliers."""
    log_relative = np.log10(relative_matrix)
    diagonal_mask = np.eye(len(ALPHAS), dtype=bool)
    palette = plt.get_cmap("vlag")

    figure, axis = plt.subplots(figsize=(16.0, 15.0))
    sns.heatmap(
        log_relative,
        mask=diagonal_mask,
        ax=axis,
        cmap=palette,
        norm=norm,
        annot=False,
        linewidths=1.2,
        linecolor="white",
        square=True,
        cbar=False,
    )

    for row in range(len(ALPHAS)):
        for col in range(len(ALPHAS)):
            if row == col:
                axis.add_patch(
                    Rectangle(
                        (col, row),
                        1,
                        1,
                        facecolor="0.91",
                        edgecolor="black",
                        linewidth=3.0,
                        zorder=4,
                    )
                )
                label = (
                    f"{_format_absolute(float(mean_matrix.iloc[row, col]))}\n"
                    f"±{_format_absolute(float(std_matrix.iloc[row, col]))}"
                )
                text_color = "0.10"
            else:
                value = float(relative_matrix.iloc[row, col])
                label = _format_multiplier(value)
                rgba = palette(norm(float(log_relative.iloc[row, col])))
                luminance = (
                    0.2126 * rgba[0]
                    + 0.7152 * rgba[1]
                    + 0.0722 * rgba[2]
                )
                text_color = "white" if luminance < 0.48 else "black"

            axis.text(
                col + 0.5,
                row + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=(
                    CELL_FONT_SIZE
                    if row != col
                    else DIAGONAL_CELL_FONT_SIZE
                ),
                fontweight="bold",
                color=text_color,
                zorder=5,
            )

    _style_heatmap_axis(
        axis,
        show_ylabel=True,
        show_xlabel=True,
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(figure)


def _plot_shared_colorbar(
    norm: TwoSlopeNorm,
    colorbar_ticks: np.ndarray,
    output_path: Path,
) -> None:
    """Save the shared relative multiplier scale as a standalone PDF."""
    palette = plt.get_cmap("vlag")
    figure = plt.figure(figsize=(6.0, 12.5))
    colorbar_axis = figure.add_axes([0.12, 0.05, 0.12, 0.90])
    colorbar = figure.colorbar(
        ScalarMappable(norm=norm, cmap=palette),
        cax=colorbar_axis,
        orientation="vertical",
    )
    colorbar.set_ticks(colorbar_ticks)
    colorbar.set_ticklabels(
        [f"{10 ** tick:g}×" for tick in colorbar_ticks]
    )
    _style_colorbar(
        colorbar,
        "Off-diagonal ARE / row-diagonal ARE",
        fontsize=COLORBAR_LABEL_FONT_SIZE,
    )
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(figure)


def _plot_diagonal_legend(output_path: Path) -> None:
    """Save the diagonal-cell explanation as a standalone PDF."""
    figure, axis = plt.subplots(figsize=(12.5, 2.4))
    axis.axis("off")
    axis.legend(
        handles=[
            Patch(
                facecolor="0.91",
                edgecolor="black",
                linewidth=2.0,
                label=r"Diagonal: mean ARE $\pm$ s.d.",
            )
        ],
        loc="center",
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.0,
    )
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(figure)


def plot_individual_figures(frame: pd.DataFrame) -> None:
    """Render one hybrid robustness heatmap for each method."""
    means = {method: _matrix(frame, "mean", method) for method in ["Crane", "Mayfly"]}
    stds = {method: _matrix(frame, "std", method) for method in ["Crane", "Mayfly"]}
    relative = {method: _matrix(frame, "over_diagonal", method) for method in ["Crane", "Mayfly"]}
    shared_relative_values = np.concatenate(
        [
            np.log10(relative["Crane"].values.ravel()),
            np.log10(relative["Mayfly"].values.ravel()),
        ]
    )
    relative_limit = max(
        float(np.nanmax(np.abs(shared_relative_values))),
        0.1,
    )
    relative_norm = TwoSlopeNorm(
        vmin=-relative_limit,
        vcenter=0.0,
        vmax=relative_limit,
    )
    relative_ticks = np.arange(
        np.ceil(-relative_limit),
        np.floor(relative_limit) + 1,
    )
    if len(relative_ticks) == 0:
        relative_ticks = np.array([0.0])

    _plot_hybrid_heatmap(
        means["Crane"],
        stds["Crane"],
        relative["Crane"],
        relative_norm,
        CRANE_PDF_PATH,
    )
    _plot_hybrid_heatmap(
        means["Mayfly"],
        stds["Mayfly"],
        relative["Mayfly"],
        relative_norm,
        MAYFLY_PDF_PATH,
    )
    _plot_shared_colorbar(
        relative_norm,
        relative_ticks,
        COLORBAR_PDF_PATH,
    )
    _plot_diagonal_legend(DIAGONAL_LEGEND_PDF_PATH)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_individual_figures(pd.read_csv(DATA_PATH))
    for path in (CRANE_PDF_PATH, MAYFLY_PDF_PATH, COLORBAR_PDF_PATH, DIAGONAL_LEGEND_PDF_PATH):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
