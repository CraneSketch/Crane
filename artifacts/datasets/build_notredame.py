#!/usr/bin/env python3
"""Build ForCrane tasks from the SNAP web-NotreDame edge list."""
import argparse
import os
import time

import numpy as np


DATASET = "NotreDame"
RAW_FILE = "NotreDame.txt"
PAPER_SIZES = [200_000, 500_000, 1_000_000, 1_500_000]
NODE_BITS = 32
HERE = os.path.dirname(os.path.abspath(__file__))


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] [{DATASET}] {message}", flush=True)


def read_edges(path, limit):
    src = np.empty(limit, dtype=np.int64)
    dst = np.empty(limit, dtype=np.int64)
    count = 0
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            if count >= limit:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2 or not (parts[0].isdigit() and parts[1].isdigit()):
                continue
            src[count] = int(parts[0])
            dst[count] = int(parts[1])
            count += 1
    log(f"read {count} edges")
    return src[:count], dst[:count]


def remap_to_sequential(src, dst):
    endpoints = np.empty(2 * len(src), dtype=np.int64)
    endpoints[0::2] = src
    endpoints[1::2] = dst
    unique, first = np.unique(endpoints, return_index=True)
    order = np.argsort(first, kind="stable")
    ids = np.empty(len(unique), dtype=np.int64)
    ids[order] = np.arange(1, len(unique) + 1, dtype=np.int64)
    return ids[np.searchsorted(unique, src)], ids[np.searchsorted(unique, dst)]


def encode_bits(values, bit_count, out=None):
    values = np.ascontiguousarray(values, dtype=">u8")
    bits = np.unpackbits(values.view(np.uint8).reshape(-1, 8),
                         axis=1, bitorder="big")[:, -bit_count:]
    if out is None:
        return np.ascontiguousarray(bits)
    out[:] = bits
    return out


def encode_edges(keys, chunk_size=1 << 22):
    encoded = np.empty((len(keys), 2 * NODE_BITS), dtype=np.uint8)
    for start in range(0, len(keys), chunk_size):
        chunk = keys[start:start + chunk_size]
        encode_bits(chunk & np.uint64(0xFFFFFFFF), NODE_BITS,
                    out=encoded[start:start + len(chunk), :NODE_BITS])
        encode_bits(chunk >> np.uint64(32), NODE_BITS,
                    out=encoded[start:start + len(chunk), NODE_BITS:])
    return encoded


def build_task(src, dst, out_root):
    total = len(src)
    if total == 0:
        raise ValueError("input contains no edges")
    src, dst = remap_to_sequential(src, dst)
    edge_keys = src.astype(np.uint64) | (dst.astype(np.uint64) << np.uint64(32))
    num_nodes = int(max(src.max(), dst.max()))
    log(f"{total} edges, {num_nodes} nodes")

    support_x = encode_edges(edge_keys)
    support_y = np.ones(total, dtype=np.float32)
    unique_keys, first, counts = np.unique(edge_keys, return_index=True,
                                            return_counts=True)
    order = np.argsort(first, kind="stable")
    query_keys = unique_keys[order]
    query_edge_x = encode_edges(query_keys)
    query_edge_y = counts[order].astype(np.float64).astype(np.float32)[:, None]
    num_unique = len(query_keys)

    out_dir = os.path.join(out_root, DATASET,
                           f"size_{total}_unique_edge_{num_unique}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "0.npz")
    np.savez_compressed(out_path, support_x=support_x, support_y=support_y,
                        query_edge_x=query_edge_x, query_edge_y=query_edge_y)
    log(f"saved {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=os.path.join(HERE, "raw"))
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=PAPER_SIZES)
    args = parser.parse_args()
    src, dst = read_edges(os.path.join(args.raw_dir, RAW_FILE), max(args.sizes))
    for size in args.sizes:
        build_task(src[:size], dst[:size], args.out_root)


if __name__ == "__main__":
    main()
