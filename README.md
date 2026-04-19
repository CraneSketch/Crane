# Crane

**Crane: An Accurate and Scalable Neural Sketch for Graph Stream Summarization**

This repository is the anonymous code release accompanying the Crane submission. It contains the reference PyTorch implementation of the neural sketch together with a PyFlink deployment for distributed streaming benchmarks.

## Part 1 — Crane Implementation (`crane/`)

### Installation

```bash
cd Crane
pip install -r requirements.txt
```

### Training

Edge frequency estimation:

```bash
PYTHONPATH=${PWD} python -m crane.run.run_train \
    --config crane/configs/basic_config.yaml \
    --workdir ./workdir
```

Case Studies:

```bash
PYTHONPATH=${PWD} python -m crane.run.run_train \
    --config crane/configs/degree_config.yaml \
    --workdir ./workdir
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

## License

All rights reserved. This code is provided for review purposes only and will be released under an open-source license upon paper publication.
