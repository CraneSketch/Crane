#!/bin/bash
# Regenerate every paper figure and table from figures/data/*.csv into figures/out/.
# Usage: bash make_all.sh          (any cwd; needs numpy pandas matplotlib seaborn; PYTHON=... to pick the interpreter)
set -e
cd "$(dirname "$0")"
for script in plot_edge_query.py plot_robustness.py plot_crosstrain.py plot_hyperparams.py \
              plot_throughput.py plot_case_study.py plot_flink.py plot_weighted_are.py plot_mawi.py \
              make_tables.py; do
    echo "== $script"
    "${PYTHON:-python3}" "$script"
done
echo "== done: $(find out -name '*.pdf' | wc -l | tr -d ' ') PDFs, $(find out -name '*.tex' | wc -l | tr -d ' ') TeX tables under out/"
