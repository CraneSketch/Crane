import os
import argparse
import base64

import numpy as np

from hash_utils import bucket_edge_bytes, bucket_edge_ints, MAX_BUCKETS


def scan_task_dirs(root):
    dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    try:
        dirs = sorted(dirs, key=lambda s: int(s.split("_")[1]))
    except (IndexError, ValueError):
        dirs = sorted(dirs)
    return dirs


def _b64(arr: np.ndarray) -> str:
    """Encode contiguous numpy array as base64 string."""
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _write_batched_stream(path, bucket_ids, arrays, batch_size):
    """Write a batched stream file.

    Each line: bucket_id<TAB>base64(arr0_batch)<TAB>base64(arr1_batch)...
    Arrays are grouped by bucket, then chunked into batches of batch_size.
    """
    n_written = 0
    with open(path, "w") as f:
        for bid in range(MAX_BUCKETS):
            mask = bucket_ids == bid
            n_bucket = int(mask.sum())
            if n_bucket == 0:
                continue
            arrs_bucket = [a[mask] for a in arrays]
            for i in range(0, n_bucket, batch_size):
                parts = [str(bid)]
                for a in arrs_bucket:
                    parts.append(_b64(a[i:i + batch_size]))
                f.write("\t".join(parts) + "\n")
                n_written += min(batch_size, n_bucket - i)
    return n_written


def _decode_binary_edges(edges_bits: np.ndarray):
    """Decode [N, 2*d] binary-encoded edges back to 0-based (src, dst) int64 arrays."""
    d = edges_bits.shape[1] // 2
    bin_weights = (1 << np.arange(d - 1, -1, -1, dtype=np.int64))
    ids = (edges_bits.astype(np.int64).reshape(-1, 2, d) * bin_weights).sum(axis=-1)
    # Encoder uses 1-based IDs; shift back to 0-based.
    return ids[:, 0] - 1, ids[:, 1] - 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--task-dir", default=None)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    task_dirs = scan_task_dirs(args.dataset_path)
    task_dir = args.task_dir if args.task_dir else task_dirs[0]
    npz_path = os.path.join(args.dataset_path, task_dir, "0.npz")

    data = np.load(npz_path, allow_pickle=False)
    os.makedirs(args.output_dir, exist_ok=True)

    BS = args.batch_size

    # ── Crane support stream ──
    support_x = data["support_x"]       # (N, 64) uint8
    support_y = data["support_y"]       # (N,) float32
    print(f"Computing bucket IDs for {len(support_x)} support edges...")
    s_buckets = np.array([bucket_edge_bytes(support_x[i])
                          for i in range(len(support_x))], dtype=np.int32)

    path = os.path.join(args.output_dir, "support_stream.txt")
    n = _write_batched_stream(path, s_buckets, [support_x, support_y], BS)
    print(f"Wrote {n} support edges in batched stream to {path}")

    # ── Crane query stream ──
    query_x = data["query_edge_x"]     # (M, 64) uint8
    query_y = data["query_edge_y"]     # (M, 1) float32
    query_y_flat = query_y.squeeze(-1) if query_y.ndim > 1 else query_y
    n_queries = len(query_x)
    print(f"Computing bucket IDs for {n_queries} query edges...")
    q_buckets = np.array([bucket_edge_bytes(query_x[i])
                          for i in range(n_queries)], dtype=np.int32)

    path = os.path.join(args.output_dir, "query_stream.txt")
    n = _write_batched_stream(path, q_buckets, [query_x, query_y_flat], BS)
    print(f"Wrote {n} query edges in batched stream to {path}")

    # ── meta.txt ──
    meta_path = os.path.join(args.output_dir, "meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"{n_queries}\n")

    # ── Baseline (Stateful) streams ──
    # The Stateful operator works on raw (src, dst, w) tuples, so we decode the
    # binary-encoded support_x / query_edge_x back to integer node IDs and reuse
    # support_y as the per-edge weight. query_edge_y already carries the ground-
    # truth cumulative weight, so we pass it through as the query truth.
    print(f"Decoding {len(support_x)} support edges for baseline stream...")
    src_arr, dst_arr = _decode_binary_edges(support_x)
    w_arr = support_y.astype(np.int32)
    bs_buckets = np.array([bucket_edge_ints(int(s), int(d))
                           for s, d in zip(src_arr, dst_arr)], dtype=np.int32)

    path = os.path.join(args.output_dir, "baseline_support_stream.txt")
    n = _write_batched_stream(path, bs_buckets, [src_arr, dst_arr, w_arr], BS)
    print(f"Wrote {n} baseline support edges in batched stream to {path}")

    print(f"Decoding {n_queries} query edges for baseline stream...")
    qsrc_arr, qdst_arr = _decode_binary_edges(query_x)
    qw_arr = np.asarray(query_y_flat, dtype=np.int64)
    bq_buckets = np.array([bucket_edge_ints(int(s), int(d))
                           for s, d in zip(qsrc_arr, qdst_arr)], dtype=np.int32)

    path = os.path.join(args.output_dir, "baseline_query_stream.txt")
    n = _write_batched_stream(path, bq_buckets,
                              [qsrc_arr, qdst_arr, qw_arr], BS)
    print(f"Wrote {n} baseline query edges in batched stream to {path}")

    bl_meta_path = os.path.join(args.output_dir, "baseline_meta.txt")
    with open(bl_meta_path, "w") as f:
        f.write(f"{n_queries}\n")

    print(f"  dataset: {task_dir}, crane queries: {n_queries}")


if __name__ == "__main__":
    main()
