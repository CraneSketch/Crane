"""Fig. 7: ARE vs carry threshold and mini-batch size. data/hyperparams.csv -> out/parameter/."""
import os
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
DATA_PATH = os.path.join(BASE_DIR, 'data', 'hyperparams.csv')
OUT_DIR = os.path.join(BASE_DIR, 'out', 'parameter')

DATASETS = ['Lkml', 'NotreDame', 'CAIDA2018', 'WiKiTalk', 'StackOverflow']
PALETTE = sns.color_palette("colorblind", n_colors=len(DATASETS))
DATASET_COLORS = dict(zip(DATASETS, PALETTE))
DATASET_MARKERS = dict(zip(DATASETS, ['o', 's', '^', 'D', 'P']))


def load_data(param):
    """Aggregated frame for one parameter: columns param(value), dataset, mean, std."""
    df = pd.read_csv(DATA_PATH)
    df = df[df['param'] == param].rename(columns={'value': 'param_value'})
    return df


def _plot_param(ax, df, xlabel):
    params = sorted(df['param_value'].unique())
    for ds in DATASETS:
        sub = df[df['dataset'] == ds]
        if sub.empty:
            continue
        agg = sub.set_index('param_value')[['mean', 'std']].reindex(params)
        ax.errorbar(
            params, agg['mean'].values, yerr=agg['std'].values,
            marker=DATASET_MARKERS[ds], color=DATASET_COLORS[ds],
            markersize=18, linewidth=3, capsize=6, capthick=2,
            label=ds, markeredgecolor='black', markeredgewidth=1.2,
        )
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(params)
    ax.set_xticklabels([str(p) for p in params])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel(xlabel)
    ax.set_ylabel('ARE')


def plot_carry_threshold(ax, df):
    _plot_param(ax, df, 'Carry Threshold')


def plot_mini_batch(ax, df):
    _plot_param(ax, df, 'Mini Batch Size')


def _save_single(plot_fn, df, output_name):
    fig, ax = plt.subplots(figsize=(11, 9))
    plot_fn(ax, df)
    ax.legend_ = None  # don't embed legend in the single figure
    plt.tight_layout()
    out = os.path.join(OUT_DIR, output_name)
    fig.savefig(out, bbox_inches='tight', transparent=True)
    print(f'Saved: {out}')
    plt.close(fig)


def plot_dataset_legend():
    fig, ax = plt.subplots(figsize=(28, 2))
    ax.set_axis_off()
    for ds in DATASETS:
        ax.plot([], [], marker=DATASET_MARKERS[ds], color=DATASET_COLORS[ds],
                markersize=18, linewidth=3, markeredgecolor='black',
                markeredgewidth=1.2, label=ds)
    handles, labels_ = ax.get_legend_handles_labels()
    fig.legend(handles, labels_, loc='center', ncol=len(DATASETS), frameon=False)
    out = os.path.join(OUT_DIR, 'hyperparams_legend.pdf')
    fig.savefig(out, bbox_inches='tight', transparent=True)
    print(f'Saved: {out}')
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _save_single(plot_carry_threshold, load_data('carry_threshold'), 'hyperparams_carry_threshold.pdf')
    _save_single(plot_mini_batch, load_data('mini_batch'), 'hyperparams_mini_batch.pdf')
    plot_dataset_legend()


if __name__ == '__main__':
    main()
