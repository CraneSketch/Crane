"""Fig. 10-11: Flink optimisation ablation and local / cluster throughput. data/flink_*.csv -> out/throughput/."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=4)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Linux Libertine O', 'Liberation Serif', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR = os.path.join(BASE_DIR, 'out', 'throughput')

# Method display names and styles
METHOD_COLORS = {
    'crane':        '#d94701',
    'crane-native': '#fd8d3c',
    'stateful':     '#4477AA',
}
METHOD_LABELS = {
    'crane':        'Crane',
    'crane-native': 'Crane (Native)',
    'stateful':     'Stateful',
}
METHOD_MARKERS = {
    'crane':        'o',
    'crane-native': 'D',
    'stateful':     's',
}

# Ablation step colors (sequential gradient)
ABLATION_CMAP = sns.color_palette("YlOrRd", n_colors=7)


def load_csv(name):
    return pd.read_csv(os.path.join(DATA_DIR, name))


METHODS_ORDER = ['crane-native', 'crane', 'stateful']


def _plot_throughput(df, x_col, x_label, metric, output_name):
    xs = sorted(df[x_col].unique())
    x = np.arange(len(xs))
    n = len(METHODS_ORDER)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(9, 9))
    for i, m in enumerate(METHODS_ORDER):
        sub = df[df['method'] == m].sort_values(x_col)
        vals = [sub[sub[x_col] == v][metric].values[0] for v in xs]
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=METHOD_LABELS[m],
               color=METHOD_COLORS[m], edgecolor='black', linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in xs])
    ax.set_xlabel(x_label)
    ax.set_ylabel('Throughput (Mops)')
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, output_name)
    fig.savefig(out, dpi=300, bbox_inches='tight', transparent=True)
    print(f'Saved: {out}')
    plt.close(fig)


def plot_method_legend():
    fig, ax = plt.subplots(figsize=(14, 2))
    ax.set_axis_off()
    handles = [Patch(facecolor=METHOD_COLORS[m], edgecolor='black', linewidth=1.2,
                     label=METHOD_LABELS[m]) for m in METHODS_ORDER]
    fig.legend(handles=handles, loc='center', ncol=len(METHODS_ORDER), frameon=False)
    out = os.path.join(OUT_DIR, 'flink_method_legend.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight', transparent=True)
    print(f'Saved: {out}')
    plt.close(fig)


# ─── Local comparison ────────────────────────────────────────────────────────

def plot_local():
    df = load_csv('flink_local.csv')
    _plot_throughput(df, 'parallelism', 'Parallelism', 'store_mips',
                     'flink_local_store.pdf')
    _plot_throughput(df, 'parallelism', 'Parallelism', 'query_mips',
                     'flink_local_query.pdf')


# ─── Cluster comparison ──────────────────────────────────────────────────────

def plot_cluster():
    df = load_csv('flink_cluster.csv')
    _plot_throughput(df, 'workers', 'Workers', 'store_mips',
                     'flink_cluster_store.pdf')
    _plot_throughput(df, 'workers', 'Workers', 'query_mips',
                     'flink_cluster_query.pdf')


# ─── Ablation study ──────────────────────────────────────────────────────────

def plot_ablation():
    df = load_csv('flink_ablation.csv')
    steps = df['label'].tolist()
    store_vals = df['store_mips'].values
    query_vals = df['query_mips'].values

    x = np.arange(len(steps))
    width = 0.35

    fig, ax = plt.subplots(figsize=(21, 10))

    ax.bar(x - width / 2, store_vals, width, label='Store',
           color=ABLATION_CMAP, edgecolor='black', linewidth=1.2)
    ax.bar(x + width / 2, query_vals, width, label='Query',
           color=ABLATION_CMAP, edgecolor='black', linewidth=1.2,
           hatch='///', alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(steps, rotation=30, ha='right')
    ax.set_ylabel('Throughput (Mops)')
    ax.legend(fontsize=22, loc='upper left')
    plt.tight_layout(pad=1.5)
    out = os.path.join(OUT_DIR, 'flink_ablation.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight', transparent=True)
    print(f'Saved: {out}')
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_local()
    plot_cluster()
    plot_ablation()
    plot_method_legend()


if __name__ == '__main__':
    main()
