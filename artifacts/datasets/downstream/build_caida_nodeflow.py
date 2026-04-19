#!/usr/bin/env python3
"""Build the CAIDA host-accounting NodeFlow workload."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_standard_npz, encode_ids


RECORD_LEN = 21
SLICES = [100_000, 200_000, 400_000, 800_000, 2_000_000]


def read_caida_ips(path, limit):
    with open(path, "rb") as stream:
        raw = stream.read(RECORD_LEN * limit)
    count = len(raw) // RECORD_LEN
    records = np.frombuffer(raw[:count * RECORD_LEN], dtype=np.uint8).reshape(count,
                                                                              RECORD_LEN)
    src = np.ascontiguousarray(records[:, :4]).view(">u4").reshape(-1).astype(np.uint32)
    dst = np.ascontiguousarray(records[:, 4:8]).view(">u4").reshape(-1).astype(np.uint32)
    return src, dst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caida", default="CAIDA2018.dat")
    parser.add_argument("--out-root", default="DownstreamDatasets")
    parser.add_argument("--slices", default=None)
    args = parser.parse_args()
    slices = [int(value) for value in args.slices.split(",")] if args.slices else SLICES

    src_ip, dst_ip = read_caida_ips(args.caida, max(max(SLICES), max(slices)))
    all_ips, inverse = np.unique(np.concatenate((src_ip, dst_ip)), return_inverse=True)
    ids = (inverse + 1).astype(np.int64)
    src_ids, dst_ids = ids[:len(src_ip)], ids[len(src_ip):]

    family_root = os.path.join(args.out_root, "NodeFlow")
    output_root = os.path.join(family_root, "CAIDA2018_HostAccounting")
    os.makedirs(family_root, exist_ok=True)
    np.savez_compressed(os.path.join(family_root, "CAIDA2018_HostAccounting_ipmap.npz"),
                        ip=all_ips)

    for size in slices:
        src, dst = src_ids[:size], dst_ids[:size]
        modulus = int(max(src.max(), dst.max())) + 1
        unique_count = len(np.unique(src * modulus + dst))
        task = f"size_{size}_unique_edge_{unique_count}"
        task_dir = os.path.join(output_root, task)
        stats = build_standard_npz(task_dir, src, dst, include=("node",))

        out_nodes, out_counts = np.unique(src, return_counts=True)
        in_nodes, in_counts = np.unique(dst, return_counts=True)
        nodes = np.union1d(out_nodes, in_nodes)
        out_y = np.zeros(len(nodes), np.float32)
        in_y = np.zeros(len(nodes), np.float32)
        out_y[np.searchsorted(nodes, out_nodes)] = out_counts
        in_y[np.searchsorted(nodes, in_nodes)] = in_counts
        np.savez_compressed(os.path.join(task_dir, "nodeflow.npz"),
                            node_x=encode_ids(nodes), out_y=out_y, in_y=in_y,
                            freq=out_y.copy())
        print(f"[nodeflow] {task}: {stats}", flush=True)


if __name__ == "__main__":
    main()
