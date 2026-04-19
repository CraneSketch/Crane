"""Tables 1 (memory-layer ablation), 2 (loss ablation), and 3 (node-flow directions) as TeX,
plus the numbers quoted in the text.
data/*.csv -> out/tables/{memory_layer_ablation,loss_ablation,nodeflow_directions}.tex and stdout."""
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR = os.path.join(BASE_DIR, 'out', 'tables')

DATASETS = ['Lkml', 'NotreDame', 'CAIDA2018', 'WiKiTalk', 'StackOverflow']


def load(name):
    return pd.read_csv(os.path.join(DATA_DIR, name))


def memory_layer_table():
    df = load('memory_layer_ablation.csv')
    lines = [r'\begin{tabular}{lccc}', r'\toprule', r'Dataset & ML=1 & ML=4 & Reduction \\', r'\midrule']
    for ds in DATASETS:
        r1 = df[(df['dataset'] == ds) & (df['memory_layer'] == 1)].iloc[0]
        r4 = df[(df['dataset'] == ds) & (df['memory_layer'] == 4)].iloc[0]
        lines.append(f"{ds} & {r1['mean']:.2f}$\\pm${r1['std']:.2f} & {r4['mean']:.2f}$\\pm${r4['std']:.2f} "
                     f"& {round(r1['mean'], 2) / round(r4['mean'], 2):.1f}$\\times$ \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines) + '\n'


def loss_table():
    df = load('loss_ablation.csv')
    lines = [r'\begin{tabular}{llccccc}', r'\toprule',
             ' & Objective & Lkml & NotreDame & CAIDA & WiKiTalk & StackOverflow \\\\', r'\midrule']
    for metric, label in (('are', 'ARE'), ('aae', 'AAE')):
        for i, (objective, name) in enumerate((('MAE', 'MAE (default)'), ('Balanced MSE', 'Balanced MSE'))):
            cells = []
            for ds in DATASETS:
                r = df[(df['objective'] == objective) & (df['dataset'] == ds) & (df['metric'] == metric)].iloc[0]
                cells.append(f"{r['mean']:.2f} $\\pm$ {r['std']:.2f}")
            head = f"\\multirow{{2}}{{*}}{{{label}}}" if i == 0 else ''
            lines.append(f"{head} & {name} & " + ' & '.join(cells) + ' \\\\')
        if metric == 'are':
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines) + '\n'


def nodeflow_directions_table():
    df = load('case_study_directions.csv')
    fmt = {'ARE': '{:.2f}', 'nDCG@100': '{:.3f}', 'Recall@100': '{:.3f}'}
    lines = [r'\begin{tabular}{llcccc}', r'\toprule',
             r' & Method & D1 Out & D1 In & D2 Out & D2 In \\', r'\midrule']
    for mi, metric in enumerate(('ARE', 'nDCG@100', 'Recall@100')):
        for i, method in enumerate(('Crane', 'Mayfly', 'HourglassSketch')):
            cells = []
            for ds in ('D1', 'D2'):
                for direction in ('out', 'in'):
                    r = df[(df['dataset'] == ds) & (df['direction'] == direction)
                           & (df['method'] == method) & (df['metric'] == metric)].iloc[0]
                    f = fmt[metric]
                    cells.append(f"{f.format(r['mean'])} $\\pm$ {f.format(r['std'])}")
            head = f"\\multirow{{3}}{{*}}{{{metric}}}" if i == 0 else ''
            lines.append(f"{head} & {method} & " + ' & '.join(cells) + ' \\\\')
        lines.append(r'\midrule' if mi < 2 else r'\bottomrule')
    lines.append(r'\end{tabular}')
    return '\n'.join(lines) + '\n'


def print_text_numbers():
    eq = load('edge_query.csv')
    eq = eq[eq['metric'] == 'are']
    print('Sec. 6.2  Crane vs strongest baseline (ARE ratio range over stream lengths):')
    for ds in DATASETS:
        sub = eq[eq['dataset'] == ds]
        ratios = [g[g['method'] != 'Crane']['mean'].min() / g[g['method'] == 'Crane']['mean'].iloc[0]
                  for _, g in sub.groupby('stream_size')]
        print(f'  {ds:<14} {min(ratios):5.1f}x .. {max(ratios):5.1f}x')

    ct = load('crosstrain.csv')
    print('Sec. 6.3  Cross-training grid:')
    for m in ('Crane', 'Mayfly'):
        s = ct[ct['method'] == m]
        diag = s[s['train_alpha'] == s['test_alpha']]
        print(f"  {m:<7} max ARE {s['mean'].max():.1f}, worst matched ARE {diag['mean'].max():.1f}, "
              f"max off-diagonal / diagonal {s['over_diagonal'].max():.1f}x")
    piv = ct.pivot_table(index=['train_alpha', 'test_alpha'], columns='method', values='mean')
    print(f"  Crane lower in all {len(piv)} cells: {(piv['Mayfly'] > piv['Crane']).all()}")

    tt = load('training_time.csv')
    print('Sec. 6.5  Training cost: ' + ', '.join(f"{r['method']} {r['hours']:.1f} h" for _, r in tt.iterrows()))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, tex in (('memory_layer_ablation.tex', memory_layer_table()), ('loss_ablation.tex', loss_table()),
                      ('nodeflow_directions.tex', nodeflow_directions_table())):
        path = os.path.join(OUT_DIR, name)
        with open(path, 'w') as f:
            f.write(tex)
        print(f'Saved: {path}\n{tex}')
    print_text_numbers()


if __name__ == '__main__':
    main()
