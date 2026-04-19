"""Fig. 8: store / query throughput on Lkml. data/throughput_{crane,baselines}.csv -> out/throughput/."""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
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


def load_data():
    """baselines: {'GSS CPU': {'Store': items/s, 'Query': items/s}, ...};
    cpu_data / gpu_data: {mini_batch: (store items/s, query items/s)}."""
    b = pd.read_csv(os.path.join(DATA_DIR, 'throughput_baselines.csv'))
    baselines = {f"{r['method']} {r['device'].upper()}": {'Store': float(r['store_items_s']),
                                                          'Query': float(r['query_items_s'])}
                 for _, r in b.iterrows()}
    c = pd.read_csv(os.path.join(DATA_DIR, 'throughput_crane.csv'))
    cpu_data = {int(r['mini_batch']): (float(r['store_items_s']), float(r['query_items_s']))
                for _, r in c[c['device'] == 'cpu'].iterrows()}
    gpu_data = {int(r['mini_batch']): (float(r['store_items_s']), float(r['query_items_s']))
                for _, r in c[c['device'] == 'gpu'].iterrows()}
    return baselines, cpu_data, gpu_data


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    baselines, cpu_data, gpu_data = load_data()

    baseline_colors = {
        'GSS CPU': '#228833',
        'TCM CPU': '#4477AA',
        'Auxo CPU': '#AA3377',
        'HourglassSketch CPU': '#66CCEE',
        'Mayfly CPU': '#8c564b',
        'Mayfly GPU': '#c49c94',
    }
    crane_styles = {
        'Crane CPU': {'color': '#d94701', 'marker': 'o'},
        'Crane GPU': {'color': '#fd8d3c', 'marker': 'D'},
    }
    crane_order = ['Crane CPU', 'Crane GPU']

    def mops_formatter(x, p):
        v = x / 1e6
        if v >= 1:
            return f'{v:.0f}'
        elif v >= 0.1:
            return f'{v:.1f}'
        else:
            return f'{v:.2f}'

    def plot_metric(metric_idx, title, output_name):
        fig, ax = plt.subplots(figsize=(11, 10.5))

        for method, data in baselines.items():
            y = data['Store'] if metric_idx == 0 else data['Query']
            color = baseline_colors.get(method, '#7f7f7f')
            ax.axhline(y=y, color=color, linestyle='--', linewidth=1.5, alpha=0.7)

        if cpu_data:
            x = sorted(cpu_data.keys())
            y = [cpu_data[k][metric_idx] for k in x]
            style = crane_styles['Crane CPU']
            ax.plot(x, y, marker=style['marker'], color=style['color'],
                    linewidth=4, markersize=18,
                    markeredgecolor='black', markeredgewidth=1.2)

        if gpu_data:
            x = sorted(gpu_data.keys())
            y = [gpu_data[k][metric_idx] for k in x]
            style = crane_styles['Crane GPU']
            ax.plot(x, y, marker=style['marker'], color=style['color'],
                    linewidth=4, markersize=18,
                    markeredgecolor='black', markeredgewidth=1.2)

        ax.set_xlabel('Mini Batch Size')
        ax.set_ylabel('Throughput (Mops)')
        ax.set_title(title)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(mops_formatter))

        plt.tight_layout()
        out = os.path.join(OUT_DIR, output_name)
        fig.savefig(out, bbox_inches='tight', transparent=True)
        print(f"Saved: {out}")
        plt.close(fig)

    def plot_legend():
        top_row = ['GSS CPU', 'TCM CPU', 'Auxo CPU', 'HourglassSketch CPU']
        bottom_row = ['Mayfly CPU', 'Mayfly GPU']
        baseline_handles = [
            Line2D([], [], color=baseline_colors[m], linestyle='--',
                   linewidth=1.5, alpha=0.7, label=m)
            for m in top_row if m in baselines
        ]
        crane_handles = [
            Line2D([], [], color=baseline_colors[m], linestyle='--',
                   linewidth=1.5, alpha=0.7, label=m)
            for m in bottom_row if m in baselines
        ] + [
            Line2D([], [], color=crane_styles[m]['color'],
                   marker=crane_styles[m]['marker'],
                   linewidth=4, markersize=18,
                   markeredgecolor='black', markeredgewidth=1.2, label=m)
            for m in crane_order
        ]

        fig, ax = plt.subplots(figsize=(24, 3))
        ax.set_axis_off()
        leg1 = fig.legend(handles=baseline_handles, loc='center',
                          bbox_to_anchor=(0.5, 0.72),
                          ncol=len(baseline_handles), frameon=False,
                          handlelength=2.5, columnspacing=1.5)
        leg2 = fig.legend(handles=crane_handles, loc='center',
                          bbox_to_anchor=(0.5, 0.28),
                          ncol=len(crane_handles), frameon=False,
                          handlelength=2.5, columnspacing=2)
        fig.add_artist(leg1)
        out = os.path.join(OUT_DIR, 'throughput_legend.pdf')
        fig.savefig(out, bbox_inches='tight', transparent=True)
        print(f"Saved: {out}")
        plt.close(fig)

    plot_metric(0, 'Store Throughput', 'throughput_store.pdf')
    plot_metric(1, 'Query Throughput', 'throughput_query.pdf')
    plot_legend()


if __name__ == '__main__':
    main()
