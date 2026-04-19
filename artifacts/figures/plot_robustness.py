"""Fig. 5: ARE of Crane / Mayfly trained on Zipfian vs Uniform weights. data/robustness.csv -> out/robustness/."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=4)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Linux Libertine O', 'Liberation Serif', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'robustness.csv')
OUT_DIR = os.path.join(BASE_DIR, 'out', 'robustness')

DATASETS = ['Lkml', 'NotreDame', 'CAIDA2018', 'WiKiTalk', 'StackOverflow']

METHOD_ORDER = ['Crane (zipf)', 'Crane (uniform)', 'Mayfly (zipf)', 'Mayfly (uniform)']
METHOD_PALETTE = {
    'Crane (zipf)':    '#d94701',
    'Crane (uniform)': '#fd8d3c',
    'Mayfly (zipf)':   '#8c564b',
    'Mayfly (uniform)':'#c49c94',
}


def format_size(x):
    if x >= 1e6:
        val = x / 1e6
        return f'{int(val)}M' if val == int(val) else f'{val:.1f}M'
    elif x >= 1e3:
        val = x / 1e3
        return f'{int(val)}K' if val == int(val) else f'{val:.1f}K'
    return str(int(x))


def load_data():
    """Aggregated frame: Method, Dataset, StreamLength, mean, std."""
    df = pd.read_csv(DATA_PATH)
    return df.rename(columns={'method': 'Method', 'dataset': 'Dataset', 'stream_length': 'StreamLength'})


def plot_single(ax, agg_all, dataset, show_ylabel=True, show_legend=False):
    """Plot a single dataset on the given axes."""
    agg = agg_all[agg_all['Dataset'] == dataset]
    if agg.empty:
        return

    sizes = sorted(agg['StreamLength'].unique())
    size_labels = [format_size(s) for s in sizes]
    x_pos = np.arange(len(sizes))
    n_methods = len(METHOD_ORDER)
    bar_width = 0.8 / n_methods

    for j, method in enumerate(METHOD_ORDER):
        m_agg = agg[agg['Method'] == method].set_index('StreamLength').reindex(sizes)
        offset = (j - (n_methods - 1) / 2) * bar_width
        std_vals = m_agg['std'].values
        ax.bar(x_pos + offset, m_agg['mean'].values, bar_width,
               yerr=[np.zeros_like(std_vals), std_vals], capsize=4,
               color=METHOD_PALETTE[method], label=method if show_legend else None,
               edgecolor='black', linewidth=1.2)

    ax.set_yscale('log')
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=20))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xticks(x_pos)
    ax.set_xticklabels(size_labels, rotation=45, ha='right')
    ax.set_xlabel('Stream Size')
    if show_ylabel:
        ax.set_ylabel('ARE')
    else:
        ax.set_ylabel('')


def plot_individual(agg_all):
    """Plot each dataset as a separate PDF without legend."""
    for dataset in DATASETS:
        fig, ax = plt.subplots(figsize=(8.4, 9))
        plot_single(ax, agg_all, dataset, show_ylabel=True, show_legend=False)
        fig.tight_layout()
        out = os.path.join(OUT_DIR, f'robustness_{dataset}.pdf')
        fig.savefig(out, dpi=300, bbox_inches='tight', transparent=True)
        print(f'Saved: {out}')
        plt.close(fig)


def plot_legend():
    """Export a standalone legend as a separate PDF."""
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(42, 2))
    ax.set_axis_off()
    handles = [Patch(facecolor=METHOD_PALETTE[m], edgecolor='black', linewidth=1.2, label=m)
               for m in METHOD_ORDER]
    fig.legend(handles=handles, loc='center', ncol=4, frameon=False)
    out = os.path.join(OUT_DIR, 'robustness_legend.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight', transparent=True)
    print(f'Saved: {out}')
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    agg_all = load_data()
    plot_individual(agg_all)
    plot_legend()


if __name__ == '__main__':
    main()
