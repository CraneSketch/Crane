# Crane

**Crane: An Accurate and Scalable Neural Sketch for Graph Stream Summarization**

This repository is the anonymous code release accompanying the Crane submission. It contains the reference PyTorch implementation of the neural sketch together with a PyFlink deployment for distributed streaming benchmarks.

## Part 1 — Crane Implementation (`crane/`)

### Installation

```bash
cd Crane
pip install -r requirements.txt
```

### Edge weight estimation

Training:

```bash
PYTHONPATH=${PWD} python -m crane.run.run_train \
    --config crane/configs/edge_query_config.yaml 
```

Evaluation on the five real datasets:

```bash
PYTHONPATH=${PWD} python -m crane.run.run_eval \
    --config crane/configs/edge_query_config.yaml \
    --load-model-path /path/to/basic/best_model.pth
```

### Case studies

| ID | Workload | Query                   | Query model |
|---|---|-------------------------|---|
| D1 | FinBench SF10 | node out-flow / in-flow | `CraneForNodeFlow` |
| D2 | CAIDA host accounting | node out-flow / in-flow | `CraneForNodeFlow` |
| D3 | RouteViews AS paths | path flow               | `CraneForPathFlow` |
| D4 | CAIDA prefix matrix | subgraph flow          | `CraneForSubGraphFlow` |
| D5 | RouteViews AS footprint | subgraph flow          | `CraneForSubGraphFlow` |

Generate the adaptation data from the first 25% of each stream:

```bash
DATA_ROOT=artifacts/datasets/DownstreamDatasets
PYTHONPATH=${PWD} python -m crane.data.generate_finetune_splits \
    --dataset-root "${DATA_ROOT}/NodeFlow" \
    --include FinBench_SF10_TSR2 CAIDA2018_HostAccounting \
    --stream-prefix 0.25 --neg-ratio 0.15 --overwrite
PYTHONPATH=${PWD} python -m crane.data.generate_finetune_splits \
    --dataset-root "${DATA_ROOT}/PathQuery" \
    --include RouteViews_ASPaths \
    --stream-prefix 0.25 --neg-ratio 0.15 --overwrite
PYTHONPATH=${PWD} python -m crane.data.generate_finetune_splits \
    --dataset-root "${DATA_ROOT}/SubgraphQuery" \
    --include CAIDA2018_PrefixMatrix RouteViews_ASFootprint \
    --stream-prefix 0.25 --neg-ratio 0.15 --overwrite
```

Run D1 or D2

```bash
DATASET=${DATA_ROOT}/NodeFlow/FinBench_SF10_TSR2
BASIC_CKPT=/path/to/basic/best_model.pth
SEED=42

PYTHONPATH=${PWD} python -m crane.run.run_finetune \
    --config crane/configs/nodeflow_config.yaml \
    --load-model-path "${BASIC_CKPT}" \
    -- \
    seed=${SEED} \
    task_type=NodeFlowOut \
    "eval.dataset_path_list=[\"${DATASET}\"]"

PYTHONPATH=${PWD} python -m crane.run.run_finetune \
    --config crane/configs/nodeflow_config.yaml \
    --load-model-path "${BASIC_CKPT}" \
    -- \
    seed=${SEED} \
    task_type=NodeFlowIn \
    "eval.dataset_path_list=[\"${DATASET}\"]"

OUT_CKPT=/path/to/out/best_model.pth
IN_CKPT=/path/to/in/best_model.pth
PYTHONPATH=${PWD} python -m crane.run.run_eval_nodeflow \
    --config crane/configs/nodeflow_config.yaml \
    --out-model-path "${OUT_CKPT}" \
    --in-model-path "${IN_CKPT}" \
    -- \
    seed=${SEED} \
    "eval.dataset_path_list=[\"${DATASET}\"]"
```

Run D3, D4, or D5

```bash
DATASET=${DATA_ROOT}/PathQuery/RouteViews_ASPaths
BASIC_CKPT=/path/to/basic/best_model.pth
QW_MODEL=CraneForPathFlow
SEED=42

PYTHONPATH=${PWD} python -m crane.run.run_finetune \
    --config crane/configs/path_subgraph_adapt_config.yaml \
    --load-model-path "${BASIC_CKPT}" \
    -- \
    seed=${SEED} \
    model.name=${QW_MODEL} \
    "eval.dataset_path_list=[\"${DATASET}\"]"

ADAPTED_CKPT=/path/to/adapted/best_model.pth
PYTHONPATH=${PWD} python -m crane.run.run_eval_downstream \
    --config crane/configs/path_subgraph_eval_config.yaml \
    --qw-model-path "${ADAPTED_CKPT}" \
    -- \
    seed=${SEED} \
    qw_model.name=${QW_MODEL} \
    "eval.dataset_path_list=[\"${DATASET}\"]"
```

## Part 2 — Flink Implementation (`flink/`)

The `flink/` subproject packages Crane as a PyFlink job and measures store/query throughput (in Mops) and average relative error (ARE) against a declarative exact-counting baseline (`stateful`) implemented with Flink keyed state.

Layout:

- `flink/crane/` — Crane model plus the Triton write/query kernels used inside Flink workers
- `flink/src/` — PyFlink entrypoint (`main.py`), the three `KeyedCoProcessFunction` operators (Crane, Crane-Native, Stateful), and `prepare_data.py` / `hash_utils.py` for building the batched support and query streams

### Preparing the streams

```bash
PYTHONPATH=flink/src python flink/src/prepare_data.py \
    --dataset-path <path to Datasets/ForCrane/<NAME>> \
    --task-dir <task subdirectory> \
    --output-dir flink/data \
    --batch-size 512
```

### Running a job

```bash
PYTHONPATH=flink/src python flink/src/main.py \
    --method {crane,crane-native,stateful} \
    --model-path weights/best_model.pth \
    --support-stream-file flink/data/support_stream.txt \
    --query-stream-file flink/data/query_stream.txt \
    --meta-dir flink/data \
    --parallelism 4 \
    --device cuda:0
```

`--method crane` uses the fused-kernel, CUDA-stream-pipelined implementation; `crane-native` is the vanilla PyTorch version without fused kernels or pipelining; `stateful` is the declarative exact-counter. The stateful baseline consumes the `baseline_support_stream.txt` / `baseline_query_stream.txt` variants.

## Part 3 — Paper Artifacts (`artifacts/`)

The paper data, plotting code, and standalone dataset builders are documented in [`artifacts/README.md`](artifacts/README.md).

## License

All rights reserved. This code is provided for review purposes only and will be released under an open-source license upon paper publication.
