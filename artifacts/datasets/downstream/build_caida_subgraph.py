#!/usr/bin/env python3
"""Build the CAIDA /16 prefix-matrix SubgraphQuery workload."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_standard_npz, encode_edges, save_downstream_npz


RECORD_LEN = 21
SLICES = [100_000, 200_000, 400_000, 800_000, 2_000_000]
MIN_CELL_EDGES = 3
MAX_CELL_EDGES = 200
TOP_CELLS = 1000


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
    output_root = os.path.join(args.out_root, "SubgraphQuery",
                               "CAIDA2018_PrefixMatrix")

    for size in slices:
        src, dst = src_ids[:size], dst_ids[:size]
        src_raw, dst_raw = src_ip[:size], dst_ip[:size]
        modulus = int(max(src.max(), dst.max())) + 1
        keys = src * modulus + dst
        unique_keys, first, counts = np.unique(keys, return_index=True,
                                                return_counts=True)
        task = f"size_{size}_unique_edge_{len(unique_keys)}"
        task_dir = os.path.join(output_root, task)
        build_standard_npz(task_dir, src, dst)

        unique_src, unique_dst = src[first], dst[first]
        unique_src_raw, unique_dst_raw = src_raw[first], dst_raw[first]
        cell_keys = ((unique_src_raw >> 16).astype(np.uint64) << np.uint64(16))
        cell_keys |= (unique_dst_raw >> 16).astype(np.uint64)
        order = np.argsort(cell_keys, kind="stable")
        boundaries = np.flatnonzero(np.diff(cell_keys[order])) + 1
        groups = np.split(order, boundaries)
        candidates = [(float(counts[group].sum()), group) for group in groups
                      if MIN_CELL_EDGES <= len(group) <= MAX_CELL_EDGES]
        candidates.sort(key=lambda item: -item[0])

        queries, targets = [], []
        meta_src, meta_dst, meta_edges = [], [], []
        for target, group in candidates[:TOP_CELLS]:
            queries.append(encode_edges(unique_src[group], unique_dst[group]))
            targets.append(target)
            meta_src.append(int(unique_src_raw[group[0]]) >> 16)
            meta_dst.append(int(unique_dst_raw[group[0]]) >> 16)
            meta_edges.append(len(group))

        save_downstream_npz(task_dir, [], [], queries, targets)
        np.savez_compressed(os.path.join(task_dir, "cells_meta.npz"),
                            src_pfx16=np.asarray(meta_src, np.uint32),
                            dst_pfx16=np.asarray(meta_dst, np.uint32),
                            num_edges=np.asarray(meta_edges, np.int32),
                            gt=np.asarray(targets, np.float32))
        print(f"[subgraph] {task}: cells={len(targets)}", flush=True)

    os.makedirs(output_root, exist_ok=True)
    with open(os.path.join(output_root, "provenance.json"), "w") as output:
        json.dump({"source": "CAIDA2018.dat (anonymized passive trace, Crypto-PAn prefix-preserving)",
                   "cell_granularity": "/16 x /16 directed prefix pairs",
                   "cell_filter": f"unique edges in [{MIN_CELL_EDGES},{MAX_CELL_EDGES}], top {TOP_CELLS} by weight",
                   "slices": slices}, output, indent=2)


if __name__ == "__main__":
    main()
