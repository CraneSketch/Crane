"""Generate standard or temporal-prefix fine-tuning splits."""
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


def _decode_edges(edge_x):
    packed = np.packbits(np.asarray(edge_x, dtype=np.uint8), axis=1)
    if packed.shape[1] != 8:
        raise ValueError(f"expected 64-bit edges, got {edge_x.shape[1]} bits")
    return packed.view(">u4").astype(np.int64)


def _encode_ids(ids):
    shifts = np.arange(31, -1, -1, dtype=np.int64)
    return ((np.asarray(ids, dtype=np.int64)[:, None] >> shifts) & 1).astype(np.uint8)


def split_stream_prefix_task(task_dir: str, prefix_ratio: float = 0.25,
                             neg_ratio: float = 0.15, seed: int = 42):
    if not 0 < prefix_ratio <= 1:
        raise ValueError("stream prefix must be in (0, 1]")
    if neg_ratio < 0:
        raise ValueError("negative ratio must be non-negative")

    with np.load(os.path.join(task_dir, "0.npz"), allow_pickle=False) as data:
        n = int(data["support_x"].shape[0] * prefix_ratio)
        if n == 0:
            raise ValueError(f"stream prefix is empty: {task_dir}")
        support_x = data["support_x"][:n]
        support_y = data["support_y"][:n].astype(np.float32)

    ids = _decode_edges(support_x)
    rng = np.random.RandomState(seed)
    nodeflow = os.path.isfile(os.path.join(task_dir, "nodeflow.npz"))

    if nodeflow:
        nodes = np.unique(ids)
        out_y = np.zeros(len(nodes), dtype=np.float64)
        in_y = np.zeros(len(nodes), dtype=np.float64)
        np.add.at(out_y, np.searchsorted(nodes, ids[:, 0]), support_y)
        np.add.at(in_y, np.searchsorted(nodes, ids[:, 1]), support_y)
        query_x = _encode_ids(nodes)
        targets = {
            "query_node_y": (out_y + in_y).astype(np.float32),
            "query_out_y": out_y.astype(np.float32),
            "query_in_y": in_y.astype(np.float32),
            "query_freq": np.ones(len(nodes), dtype=np.float32),
        }
    else:
        keys = (ids[:, 0] << 32) | ids[:, 1]
        unique, inverse = np.unique(keys, return_inverse=True)
        totals = np.zeros(len(unique), dtype=np.float64)
        np.add.at(totals, inverse, support_y)
        n_neg = int(len(unique) * neg_ratio)
        nodes = np.unique(ids)
        edge_set = set(unique.tolist())
        if n_neg and len(edge_set) == len(nodes) ** 2:
            raise ValueError(f"not enough absent pairs in prefix: {task_dir}")
        negatives = []
        attempts = 0
        while len(negatives) < n_neg and attempts < 100:
            src = nodes[rng.randint(0, len(nodes), n_neg)]
            dst = nodes[rng.randint(0, len(nodes), n_neg)]
            attempts += 1
            for key in ((src << 32) | dst).tolist():
                if key not in edge_set:
                    negatives.append(key)
                    if len(negatives) == n_neg:
                        break
        if len(negatives) != n_neg:
            raise ValueError(f"failed to sample absent pairs in prefix: {task_dir}")
        unique = np.concatenate((unique, np.asarray(negatives, dtype=np.int64)))
        totals = np.concatenate((totals, np.zeros(n_neg, dtype=np.float64)))
        query_x = np.concatenate((_encode_ids(unique >> 32),
                                  _encode_ids(unique & 0xFFFFFFFF)), axis=1)
        targets = {"query_edge_y": totals.astype(np.float32)}

    train_idx = np.arange(len(query_x))
    val_idx = np.sort(rng.permutation(len(query_x))[:max(1, int(len(query_x) * 0.05))])
    for filename, idx in (("finetune_train.npz", train_idx),
                          ("finetune_val.npz", val_idx)):
        arrays = {"support_x": support_x, "support_y": support_y,
                  "query_node_x" if nodeflow else "query_edge_x": query_x[idx]}
        arrays.update({key: value[idx] for key, value in targets.items()})
        np.savez(os.path.join(task_dir, filename), **arrays)
    return len(train_idx), len(val_idx), "nodes" if nodeflow else "edges"


def generate_stream_prefix(dataset_root: str, prefix_ratio: float = 0.25,
                           neg_ratio: float = 0.15, seed: int = 42,
                           include=None, overwrite: bool = False):
    processed = 0
    dataset_names = sorted(os.listdir(dataset_root))
    if include is not None:
        dataset_names = [name for name in dataset_names if name in set(include)]
    for dataset_name in dataset_names:
        dataset_dir = os.path.join(dataset_root, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue
        for task_name in sorted(os.listdir(dataset_dir)):
            task_dir = os.path.join(dataset_dir, task_name)
            if not os.path.isfile(os.path.join(task_dir, "downstream.npz")) and not os.path.isfile(os.path.join(task_dir, "nodeflow.npz")):
                continue
            train_path = os.path.join(task_dir, "finetune_train.npz")
            val_path = os.path.join(task_dir, "finetune_val.npz")
            if not overwrite and os.path.isfile(train_path) and os.path.isfile(val_path):
                print(f"  {dataset_name}/{task_name}: skip (already exists)")
                processed += 1
                continue
            n_train, n_val, kind = split_stream_prefix_task(
                task_dir, prefix_ratio=prefix_ratio, neg_ratio=neg_ratio, seed=seed)
            print(f"  {dataset_name}/{task_name}: {kind} train={n_train}, val={n_val}")
            processed += 1
    if processed == 0:
        raise ValueError(f"no downstream tasks found under {dataset_root}")


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
    parser.add_argument("--stream-prefix", type=float)
    parser.add_argument("--neg-ratio", type=float, default=0.15)
    parser.add_argument("--include", nargs="+")
    args = parser.parse_args()
    if args.stream_prefix is None:
        generate_all(args.dataset_root, seed=args.seed, overwrite=args.overwrite)
    else:
        generate_stream_prefix(args.dataset_root, prefix_ratio=args.stream_prefix,
                               neg_ratio=args.neg_ratio, seed=args.seed,
                               include=args.include, overwrite=args.overwrite)
