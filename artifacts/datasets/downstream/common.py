"""Shared helpers for downstream candidate-dataset builders.

Conventions match DatasetBuild/generate_crane.py and crane/eval/evaluate_downstream.py:
  - node IDs are remapped to sequential integers starting at 1
  - each node is encoded as NODE_BITS-bit binary vector (uint8), edge = [src_bits, dst_bits]
  - 0.npz always stores support_x/y; NodeFlow also stores query_node_x/y
  - downstream.npz keys: path_edges/offsets/targets, sg_edges/offsets/targets
"""
import os
import numpy as np

NODE_BITS = 32


def encode_ids(ids: np.ndarray) -> np.ndarray:
    """Vectorized: [N] int64 (1-based) -> [N, NODE_BITS] uint8."""
    shifts = np.arange(NODE_BITS - 1, -1, -1, dtype=np.int64)
    return ((ids[:, None] >> shifts) & 1).astype(np.uint8)


def encode_edges(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """[N] , [N] -> [N, 2*NODE_BITS] uint8."""
    return np.concatenate([encode_ids(src), encode_ids(dst)], axis=1)


def build_standard_npz(out_dir: str, src: np.ndarray, dst: np.ndarray,
                       seed: int = 42, weights: np.ndarray = None,
                       compress: bool = True, include: tuple = ()) -> dict:
    """Build 0.npz from a (src, dst[, weight]) update stream (remapped 1-based
    IDs). weights=None -> weight 1 per update.

    Only task-relevant keys are stored: support_x/y always (the stream the
    sketch is built from); query_node_x/y only when include contains "node"
    (NodeFlow task). Path/Subgraph queries live in downstream.npz.
    Scales to tens of millions of updates (chunked encode). Returns stats."""
    n = len(src)
    w = np.ones(n, np.float64) if weights is None else np.asarray(weights, np.float64)

    def enc_chunked(a, b):
        out = np.empty((len(a), 2 * NODE_BITS), np.uint8)
        for i in range(0, len(a), 1_000_000):
            out[i:i + 1_000_000] = encode_edges(a[i:i + 1_000_000], b[i:i + 1_000_000])
        return out

    arrays = {
        "support_x": enc_chunked(src, dst),
        "support_y": w.astype(np.float32),
    }

    mod = int(max(src.max(), dst.max())) + 1
    keys = src.astype(np.int64) * mod + dst.astype(np.int64)
    num_uniq = len(np.unique(keys))
    del keys
    node_ids = np.unique(np.concatenate([src, dst]))

    if "node" in include:
        # node totals (in + out), the input of the existing Degree pipeline;
        # directional out/in targets live in nodeflow.npz
        out_w = np.zeros(len(node_ids), np.float64)
        in_w = np.zeros(len(node_ids), np.float64)
        np.add.at(out_w, np.searchsorted(node_ids, src), w)
        np.add.at(in_w, np.searchsorted(node_ids, dst), w)
        arrays["query_node_x"] = encode_ids(node_ids)
        arrays["query_node_y"] = (out_w + in_w).astype(np.float32)[:, None]

    os.makedirs(out_dir, exist_ok=True)
    saver = np.savez_compressed if compress else np.savez
    saver(os.path.join(out_dir, "0.npz"), **arrays)
    return {"stream_len": n, "unique_edges": int(num_uniq), "nodes": int(len(node_ids))}


def save_downstream_npz(out_dir: str, path_queries, path_targets, sg_queries, sg_targets):
    """path_queries / sg_queries: list of [K_i, 2*NODE_BITS] uint8 arrays."""
    def pack(qs):
        lengths = [q.shape[0] for q in qs]
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        edges = (np.concatenate(qs, axis=0).astype(np.float32)
                 if qs else np.zeros((0, 2 * NODE_BITS), dtype=np.float32))
        return edges, offsets

    path_edges, path_offsets = pack(path_queries)
    sg_edges, sg_offsets = pack(sg_queries)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, "downstream.npz"),
        path_edges=path_edges, path_offsets=path_offsets,
        path_targets=np.asarray(path_targets, dtype=np.float32),
        sg_edges=sg_edges, sg_offsets=sg_offsets,
        sg_targets=np.asarray(sg_targets, dtype=np.float32),
    )
