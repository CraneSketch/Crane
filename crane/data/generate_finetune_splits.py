"""Generate finetune_train.npz / finetune_test.npz from 0.npz files.

Splits each query type (edge, node) independently into 10% train / 90% test.
Support data is shared across both splits.

Usage:
    python -m crane.data.generate_finetune_splits --dataset-root ../../Datasets/ForCrane
"""
import os
import argparse
import logging

import numpy as np

logger = logging.getLogger("crane")

QUERY_GROUPS = [
    ("query_edge_x", "query_edge_y"),
    ("query_node_x", "query_node_y"),
]

TRAIN_RATIO = 0.1


def split_one_npz(npz_path: str, seed: int = 42):
    with np.load(npz_path, allow_pickle=False) as data:
        keys = list(data.keys())
        support_x = data["support_x"]
        support_y = data["support_y"]

        train_dict = {"support_x": support_x, "support_y": support_y}
        test_dict = {"support_x": support_x, "support_y": support_y}

        rng = np.random.RandomState(seed)

        for qx_key, qy_key in QUERY_GROUPS:
            if qx_key not in keys:
                continue
            qx = data[qx_key]
            qy = data[qy_key]
            n = qx.shape[0]
            perm = rng.permutation(n)
            split = max(1, int(n * TRAIN_RATIO))
            train_idx, test_idx = perm[:split], perm[split:]

            train_dict[qx_key] = qx[train_idx]
            train_dict[qy_key] = qy[train_idx]
            test_dict[qx_key] = qx[test_idx]
            test_dict[qy_key] = qy[test_idx]

    parent = os.path.dirname(npz_path)
    train_path = os.path.join(parent, "finetune_train.npz")
    test_path = os.path.join(parent, "finetune_test.npz")
    np.savez(train_path, **train_dict)
    np.savez(test_path, **test_dict)
    return train_path, test_path


def generate_all(dataset_root: str, seed: int = 42, overwrite: bool = False):
    for dataset_name in sorted(os.listdir(dataset_root)):
        dataset_dir = os.path.join(dataset_root, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue
        for task_name in sorted(os.listdir(dataset_dir)):
            task_dir = os.path.join(dataset_dir, task_name)
            npz_path = os.path.join(task_dir, "0.npz")
            if not os.path.isfile(npz_path):
                continue
            train_path = os.path.join(task_dir, "finetune_train.npz")
            test_path = os.path.join(task_dir, "finetune_test.npz")
            if not overwrite and os.path.isfile(train_path) and os.path.isfile(test_path):
                print(f"  {dataset_name}/{task_name}: skip (already exists)")
                continue
            train_path, test_path = split_one_npz(npz_path, seed=seed)
            print(f"  {dataset_name}/{task_name}: train={os.path.basename(train_path)}, test={os.path.basename(test_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate finetune splits from 0.npz")
    parser.add_argument("--dataset-root", type=str, required=True,
                        help="Root directory containing dataset folders (e.g. ../../Datasets/ForCrane)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate splits even if they already exist")
    args = parser.parse_args()
    generate_all(args.dataset_root, seed=args.seed, overwrite=args.overwrite)
