#!/usr/bin/env python3
"""Build Datasets/ForCrane/Lkml from Mayfly's sampled Lkml task files.

Input (one directory per task, extracted under raw/Lkml_mayfly_tasks/):
    <src-root>/<task_dir>/0.npz with support_x (n,32) f32, support_y (n,) f32,
    query_x (m,32) f32, query_y (m,1) f32 -- 16 bits per node, MSB-first.

Output (task directory names mirrored):
    <out-root>/<task_dir>/0.npz in the ForCrane multi-task format: node ids re-encoded
    MSB-first and zero-extended to 32 bits (src in columns 0-31, dst in 32-63);
    query_edge_* is the re-encoded query set.

Usage:
    python build_lkml.py --out-root <out> [--src-root raw/Lkml_mayfly_tasks]
"""
import argparse
import os

import numpy as np

NODE_BITS = 32
HERE = os.path.dirname(os.path.abspath(__file__))


def decode_msb(bits):
    width = bits.shape[1]
    weights = 2 ** np.arange(width - 1, -1, -1, dtype=np.int64)
    return bits.astype(np.int64) @ weights


def encode_msb(values, nbits):
    u64 = np.ascontiguousarray(values, dtype=">u8")
    bits = np.unpackbits(u64.view(np.uint8).reshape(-1, 8), axis=1, bitorder="big")
    return np.ascontiguousarray(bits[:, -nbits:])


def reencode_edges(bits_mat):
    half = bits_mat.shape[1] // 2
    src = decode_msb(bits_mat[:, :half])
    dst = decode_msb(bits_mat[:, half:])
    return np.concatenate([encode_msb(src, NODE_BITS), encode_msb(dst, NODE_BITS)], axis=1)


def augment_task(src_npz, dst_npz):
    with np.load(src_npz) as data:
        support_y = data["support_y"].astype(np.float32)
        query_y = data["query_y"].astype(np.float32).reshape(-1)
        support_x = reencode_edges(data["support_x"])
        query_edge_x = reencode_edges(data["query_x"])
    query_edge_y = query_y[:, None].astype(np.float32)
    num_edges = query_edge_x.shape[0]

    os.makedirs(os.path.dirname(dst_npz), exist_ok=True)
    np.savez_compressed(
        dst_npz,
        support_x=support_x,
        support_y=support_y,
        query_edge_x=query_edge_x,
        query_edge_y=query_edge_y,
    )
    print(f"{dst_npz}: support {support_x.shape[0]}, edges {num_edges}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src-root", default=os.path.join(HERE, "raw", "Lkml_mayfly_tasks"))
    parser.add_argument("--out-root", required=True, help="output root, e.g. Datasets/ForCrane/Lkml")
    parser.add_argument("--file-name", default="0.npz")
    args = parser.parse_args()

    for sub in sorted(os.listdir(args.src_root)):
        src_npz = os.path.join(args.src_root, sub, args.file_name)
        if not os.path.isfile(src_npz):
            continue
        augment_task(src_npz, os.path.join(args.out_root, sub, args.file_name))


if __name__ == "__main__":
    main()
