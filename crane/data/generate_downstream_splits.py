"""Generate path/subgraph query splits as downstream.npz per task dir.

Mirrors the on-the-fly generation in crane.eval.evaluate_downstream and
persists the result so evaluation is reproducible and fast.

Storage format (downstream.npz):
    path_edges   : [sum_K_i, 2*d]     float32, concatenated path edges
    path_offsets : [num_path + 1]     int64,   cumulative offsets into path_edges
    path_targets : [num_path]         float32, ground-truth max-flow values
    sg_edges     : [sum_M_i, 2*d]     float32, concatenated subgraph edges
    sg_offsets   : [num_sg + 1]       int64
    sg_targets   : [num_sg]           float32

Usage:
    python -m crane.data.generate_downstream_splits \\
        --dataset-root ../../Datasets/ForCrane --seed 42
"""
import os
import argparse

import numpy as np
import torch

from crane.eval.evaluate_downstream import (
    build_adjacency_from_support,
    generate_path_queries,
    generate_subgraph_queries,
)


DEFAULT_NODE_BINARY_DIM = 32
DEFAULT_PATH_NUM_QUERIES = 500
DEFAULT_PATH_MIN_LEN = 3
DEFAULT_PATH_MAX_LEN = 8
DEFAULT_PATH_NEG_RATIO = 0.3
DEFAULT_SG_NUM_QUERIES = 500
DEFAULT_SG_MIN_SIZE = 5
DEFAULT_SG_MAX_SIZE = 20
DEFAULT_SG_NONEXIST_RATIO = 0.2


def _scan_and_sort_task_dirs(root_path):
    dir_list = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    try:
        dir_list = sorted(dir_list, key=lambda s: int(os.path.basename(s).split("_")[1]))
    except (IndexError, ValueError):
        dir_list = sorted(dir_list)
    return dir_list


def _pack_variable_length(tensor_list, feature_dim):
    lengths = [t.shape[0] for t in tensor_list]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    if tensor_list:
        edges = torch.cat(tensor_list, dim=0).cpu().numpy().astype(np.float32)
    else:
        edges = np.zeros((0, feature_dim), dtype=np.float32)
    return edges, offsets


def generate_one_task(npz_path, out_path, sample_idx, args):
    with np.load(npz_path, allow_pickle=False) as data:
        support_x = torch.tensor(data["support_x"], dtype=torch.float32)
        support_y = torch.tensor(data["support_y"], dtype=torch.long)

    feature_dim = support_x.shape[1]

    adjacency, weight_dict, edge_set = build_adjacency_from_support(
        support_x, support_y, args.node_binary_dim
    )

    path_seed = args.seed + sample_idx
    path_queries, path_gt = generate_path_queries(
        adjacency, weight_dict, args.node_binary_dim,
        num_queries=args.path_num_queries,
        min_path_length=args.path_min_length,
        max_path_length=args.path_max_length,
        negative_ratio=args.path_negative_ratio,
        seed=path_seed,
        device="cpu",
    )

    sg_seed = args.seed + sample_idx + 10000
    sg_queries, sg_gt = generate_subgraph_queries(
        weight_dict, edge_set, args.node_binary_dim,
        num_queries=args.sg_num_queries,
        min_subgraph_size=args.sg_min_size,
        max_subgraph_size=args.sg_max_size,
        nonexistent_ratio=args.sg_nonexistent_ratio,
        seed=sg_seed,
        device="cpu",
    )

    path_edges, path_offsets = _pack_variable_length(path_queries, feature_dim)
    sg_edges, sg_offsets = _pack_variable_length(sg_queries, feature_dim)

    np.savez(
        out_path,
        path_edges=path_edges,
        path_offsets=path_offsets,
        path_targets=path_gt.cpu().numpy().astype(np.float32),
        sg_edges=sg_edges,
        sg_offsets=sg_offsets,
        sg_targets=sg_gt.cpu().numpy().astype(np.float32),
    )
    return len(path_queries), len(sg_queries)


def generate_all(args):
    for dataset_name in sorted(os.listdir(args.dataset_root)):
        dataset_dir = os.path.join(args.dataset_root, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue
        task_dirs = _scan_and_sort_task_dirs(dataset_dir)
        for sample_idx, task_name in enumerate(task_dirs):
            task_dir = os.path.join(dataset_dir, task_name)
            npz_path = os.path.join(task_dir, "0.npz")
            if not os.path.isfile(npz_path):
                continue
            out_path = os.path.join(task_dir, "downstream.npz")
            if not args.overwrite and os.path.isfile(out_path):
                print(f"  {dataset_name}/{task_name}: skip (already exists)")
                continue
            n_path, n_sg = generate_one_task(npz_path, out_path, sample_idx, args)
            print(f"  {dataset_name}/{task_name}: path={n_path}, subgraph={n_sg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate downstream query splits")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--node-binary-dim", type=int, default=DEFAULT_NODE_BINARY_DIM)
    parser.add_argument("--path-num-queries", type=int, default=DEFAULT_PATH_NUM_QUERIES)
    parser.add_argument("--path-min-length", type=int, default=DEFAULT_PATH_MIN_LEN)
    parser.add_argument("--path-max-length", type=int, default=DEFAULT_PATH_MAX_LEN)
    parser.add_argument("--path-negative-ratio", type=float, default=DEFAULT_PATH_NEG_RATIO)
    parser.add_argument("--sg-num-queries", type=int, default=DEFAULT_SG_NUM_QUERIES)
    parser.add_argument("--sg-min-size", type=int, default=DEFAULT_SG_MIN_SIZE)
    parser.add_argument("--sg-max-size", type=int, default=DEFAULT_SG_MAX_SIZE)
    parser.add_argument("--sg-nonexistent-ratio", type=float, default=DEFAULT_SG_NONEXIST_RATIO)
    args = parser.parse_args()
    generate_all(args)
