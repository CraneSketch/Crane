"""Fig. 4: edge weight estimation under a 64 KB budget. data/edge_query.csv -> out/edge_weights/."""
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
DATA_PATH = os.path.join(BASE_DIR, 'data', 'edge_query.csv')
OUT_DIR = os.path.join(BASE_DIR, 'out', 'edge_weights')

DATASETS = ['Lkml', 'NotreDame', 'CAIDA2018', 'WiKiTalk', 'StackOverflow']

DATASET_SIZES = {
    'Lkml':          [20_000, 40_000, 80_000, 200_000, 400_000, 800_000],
    'NotreDame':     [200_000, 500_000, 1_000_000, 1_500_000],
    'CAIDA2018':     [2_000_000, 8_000_000, 16_000_000, 27_100_000],
    'WiKiTalk':      [2_000_000, 8_000_000, 16_000_000, 25_000_000],
    'StackOverflow': [2_000_000, 8_000_000, 16_000_000, 63_500_000],
}

# Stream size labels for x-axis
DATASET_LABELS = {
    'Lkml':          ['20K', '40K', '80K', '200K', '400K', '800K'],
    'NotreDame':     ['200K', '500K', '1M', '1.5M'],
    'CAIDA2018':     ['2M', '8M', '16M', '27.1M'],
    'WiKiTalk':      ['2M', '8M', '16M', '25.0M'],
    'StackOverflow': ['2M', '8M', '16M', '63.5M'],
}

METHOD_STYLES = {
    'Crane':           {'color': '#d94701', 'marker': 'o', 'linewidth': 4, 'markersize': 18, 'zorder': 10},
    'TCM':             {'color': '#4477AA', 'marker': 's', 'linewidth': 2.5, 'markersize': 16, 'zorder': 5},
    'GSS':             {'color': '#228833', 'marker': '^', 'linewidth': 2.5, 'markersize': 16, 'zorder': 5},
    'Auxo':            {'color': '#AA3377', 'marker': 'D', 'linewidth': 2.5, 'markersize': 16, 'zorder': 5},
    'HourglassSketch': {'color': '#66CCEE', 'marker': 'v', 'linewidth': 2.5, 'markersize': 16, 'zorder': 5},
    'Mayfly':          {'color': '#8c564b', 'marker': 'P', 'linewidth': 2.5, 'markersize': 16, 'zorder': 5},
}
METHOD_ORDER = ['Crane', 'TCM', 'GSS', 'Auxo', 'HourglassSketch', 'Mayfly']


def load_data(metric):
    """all_methods[method][dataset] = {'means', 'stds'} in stream-size order."""
    df = pd.read_csv(DATA_PATH)
    df = df[df['metric'] == metric]
    all_methods = {}
    for method in METHOD_ORDER:
        all_methods[method] = {}
        for dataset in DATASETS:
            sub = (df[(df['method'] == method) & (df['dataset'] == dataset)]
                   .set_index('stream_size').reindex(DATASET_SIZES[dataset]))
            all_methods[method][dataset] = {'means': sub['mean'].to_numpy(), 'stds': sub['std'].to_numpy()}
    return all_methods


def plot_single(ax, all_methods, dataset, metric_label, show_ylabel=True):
    """Plot a single dataset on the given axes."""
    labels = DATASET_LABELS[dataset]
    x = np.arange(len(labels))

    for method in METHOD_ORDER:
        data = all_methods[method][dataset]
        style = METHOD_STYLES[method]
        ax.errorbar(x, data['means'], yerr=data['stds'], color=style['color'],
                    marker=style['marker'], linewidth=style['linewidth'],
                    markersize=style['markersize'], zorder=style['zorder'],
                    label=method, capsize=5, capthick=2,
                    markeredgecolor='black', markeredgewidth=1.2)

    ax.set_yscale('log')
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=20))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs='auto', numticks=20))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title(dataset, visible=False)
    ax.set_xlabel('Stream Size')
    if show_ylabel:
        ax.set_ylabel(metric_label)
    else:
        ax.set_ylabel('')


def plot_legend():
    """Export a standalone legend as a separate PDF."""
    fig, ax = plt.subplots(figsize=(42, 2))
    ax.set_axis_off()
    for method in METHOD_ORDER:
        style = METHOD_STYLES[method]
        ax.plot([], [], color=style['color'], marker=style['marker'],
                linewidth=style['linewidth'], markersize=style['markersize'],
                markeredgecolor='black', markeredgewidth=1.2,
                label=method)
    handles, labels_ = ax.get_legend_handles_labels()
    fig.legend(handles, labels_, loc='center', ncol=6, frameon=False)
    output_path = os.path.join(OUT_DIR, 'edge_query_legend.pdf')
    fig.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Saved to: {output_path}")
    plt.close(fig)


def plot_individual(all_methods, metric, metric_label):
    """Plot each dataset as a separate PDF without legend."""
    for dataset in DATASETS:
        fig, ax = plt.subplots(figsize=(8.4, 9))
        plot_single(ax, all_methods, dataset, metric_label)
        fig.tight_layout()
        output_path = os.path.join(OUT_DIR, f'edge_query_{metric}_{dataset}.pdf')
        fig.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"Saved to: {output_path}")
        plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for metric, label in [('are', 'ARE'), ('aae', 'AAE')]:
        plot_individual(load_data(metric), metric, label)
    plot_legend()


if __name__ == '__main__':
    main()
