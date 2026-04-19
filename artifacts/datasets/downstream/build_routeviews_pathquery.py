"""Build candidate PathQuery downstream dataset from RouteViews BGP updates.

Real workload: BGP route validation (ARTEMIS/Argus-style). The stream is the
AS-link activity graph from a build window of real BGP announcements; the path
queries are the AS_PATHs actually announced in a held-out query window, ranked
by their real announcement frequency. Bottleneck GT = min accumulated link
weight over the stream prefix; 0 when the route contains an unseen link
(novel-link route = the real detection target).

Output:
  DownstreamDatasets/PathQuery/RouteViews_ASPaths/size_<N>_unique_edge_<U>/
    0.npz            AS-link update stream (standard keys)
    downstream.npz   path_* filled with real AS_PATH queries (sg_* empty)
    paths_meta.npz   freq (announcements in query window), novel (bool:
                     contains link unseen in this stream prefix), path_len
  DownstreamDatasets/PathQuery/RouteViews_ASPaths_asnmap.npz  (index+1 = node id)
  DownstreamDatasets/PathQuery/RouteViews_ASPaths/provenance.json

Usage:
  python3 build_routeviews_pathquery.py --mrt-dir /tmp/bgp \
      --build-files updates.20260701.0800.bz2,... --query-files updates.20260701.1000.bz2,...
"""
import os
import sys
import json
import argparse
from collections import Counter

import numpy as np
from mrtparse import Reader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import encode_edges, build_standard_npz, save_downstream_npz

SLICES = [250_000, 500_000, 1_000_000, 2_000_000]
TOP_PATHS = 1000
MAX_PATH_LEN = 12


def iter_as_paths(mrt_file):
    """Yield collapsed AS_PATH (tuple of int ASNs) per announcement message."""
    for entry in Reader(mrt_file):
        bgp = entry.data.get("bgp_message") or {}
        nlri = bgp.get("nlri") or []
        if not nlri:
            continue
        as_path = None
        for a in bgp.get("path_attributes") or []:
            t = a.get("type")
            if (isinstance(t, dict) and 2 in t) or t == 2:
                as_path = a.get("value")
        if not as_path:
            continue
        asns = []
        ok = True
        for seg in as_path:
            st = seg.get("type")
            is_seq = (isinstance(st, dict) and 2 in st) or st == 2  # AS_SEQUENCE
            if not is_seq:
                ok = False  # skip AS_SET paths (aggregates, not a concrete route)
                break
            asns.extend(int(x) for x in seg.get("value", []))
        if not ok or len(asns) < 2:
            continue
        collapsed = [asns[0]]
        for x in asns[1:]:
            if x != collapsed[-1]:  # remove prepending
                collapsed.append(x)
        if 2 <= len(collapsed) <= MAX_PATH_LEN + 1:
            yield tuple(collapsed)


def paths_to_links(paths):
    src, dst = [], []
    for p in paths:
        for i in range(len(p) - 1):
            src.append(p[i])
            dst.append(p[i + 1])
    return np.array(src, np.int64), np.array(dst, np.int64)


def save_paths_npz(paths, out):
    flat = np.array([a for p in paths for a in p], np.int64)
    offs = np.concatenate([[0], np.cumsum([len(p) for p in paths])]).astype(np.int64)
    np.savez_compressed(out, flat=flat, offsets=offs)


def load_paths_npz(files):
    paths = []
    for fn in files:
        with np.load(fn) as d:
            flat, offs = d["flat"], d["offsets"]
        paths.extend(tuple(flat[offs[i]:offs[i + 1]].tolist())
                     for i in range(len(offs) - 1))
    return paths


def cmd_parse(args):
    """Parse MRT files -> checkpoint npz of announcement AS paths (in order)."""
    import time
    t0 = time.time()
    for f in args.files.split(","):
        if args.max_seconds and time.time() - t0 > args.max_seconds:
            print(f"[parse] time budget reached; rerun to continue")
            break
        src = os.path.join(args.mrt_dir, f)
        out = os.path.join(args.mrt_dir, f + ".paths.npz")
        if os.path.exists(out):
            print(f"[parse] skip {f} (exists)")
            continue
        paths = list(iter_as_paths(src))
        save_paths_npz(paths, out)
        print(f"[parse] {f}: {len(paths)} announcements", flush=True)


def stream_links_vectorized(files):
    """Chronological AS-link updates from parsed-path npz files (vectorized)."""
    srcs, dsts, n_ann = [], [], 0
    for fn in files:
        with np.load(fn) as d:
            flat, offs = d["flat"], d["offsets"]
        n_ann += len(offs) - 1
        valid = np.ones(len(flat) - 1, bool)
        valid[offs[1:-1] - 1] = False  # pairs crossing path boundaries
        srcs.append(flat[:-1][valid])
        dsts.append(flat[1:][valid])
    return np.concatenate(srcs), np.concatenate(dsts), n_ann


def cmd_build(args):
    build_files = [os.path.join(args.mrt_dir, f + ".paths.npz") for f in args.build_files.split(",")]
    query_files = [os.path.join(args.mrt_dir, f + ".paths.npz") for f in args.query_files.split(",")]

    # ---------- stream: AS-link updates from build window ----------
    s_asn, d_asn, n_ann = stream_links_vectorized(build_files)
    print(f"[build] {n_ann} announcements -> {len(s_asn)} link updates")

    # ---------- query workload from held-out window ----------
    qc = Counter(load_paths_npz(query_files))
    top = qc.most_common(TOP_PATHS)
    q_paths = [p for p, _ in top]
    q_freq = np.array([c for _, c in top], np.int64)
    print(f"[query] using top {len(q_paths)} paths by real announcement frequency")

    # ---------- 1-based ASN remap over stream + query ASNs ----------
    q_s, q_d = paths_to_links(q_paths)
    all_asn = np.unique(np.concatenate([s_asn, d_asn, q_s, q_d]))
    lut = {int(a): i + 1 for i, a in enumerate(all_asn)}
    root = os.path.join(args.out_root, "PathQuery")
    os.makedirs(root, exist_ok=True)
    np.savez_compressed(os.path.join(root, "RouteViews_ASPaths_asnmap.npz"), asn=all_asn)

    s_id = np.searchsorted(all_asn, s_asn) + 1
    d_id = np.searchsorted(all_asn, d_asn) + 1

    ds_root = os.path.join(root, "RouteViews_ASPaths")
    mod = int(all_asn.shape[0]) + 2

    slices = ([int(x) for x in args.slices.split(",")] if args.slices else SLICES)
    for n in slices:
        if n > len(s_id):
            print(f"[warn] slice {n} > stream {len(s_id)}; using full stream")
            n = len(s_id)
        s, d = s_id[:n], d_id[:n]
        keys = s * mod + d
        uniq, counts = np.unique(keys, return_counts=True)
        w = dict(zip(uniq.tolist(), counts.astype(float).tolist()))
        task = f"size_{n}_unique_edge_{len(uniq)}"
        out_dir = os.path.join(ds_root, task)

        stats = build_standard_npz(out_dir, s, d, seed=args.seed,
                                   compress=(n <= 4_000_000))
        print(f"[pathquery] {task}: {stats}", flush=True)

        path_queries, path_targets, novel, plens = [], [], [], []
        for p in q_paths:
            pid = [lut[a] for a in p]
            ps = np.array(pid[:-1], np.int64)
            pd = np.array(pid[1:], np.int64)
            path_queries.append(encode_edges(ps, pd))
            ws = [w.get(int(a) * mod + int(b), 0.0) for a, b in zip(ps, pd)]
            gt = min(ws)
            path_targets.append(gt)
            novel.append(gt == 0.0)
            plens.append(len(ps))
        save_downstream_npz(out_dir, path_queries, path_targets, [], [])
        np.savez_compressed(os.path.join(out_dir, "paths_meta.npz"),
                            freq=q_freq, novel=np.array(novel, bool),
                            path_len=np.array(plens, np.int32))
        nz = int(np.sum(np.array(path_targets) > 0))
        print(f"[pathquery] {task}: {len(path_targets)} real AS_PATH queries, "
              f"{nz} fully-observed, {len(path_targets)-nz} novel-link", flush=True)

    with open(os.path.join(ds_root, "provenance.json"), "w") as f:
        json.dump({
            "source": "RouteViews archive (archive.routeviews.org), route-views2 collector",
            "stream": "one edge update per adjacent AS pair per announcement message (AS_SET skipped, prepending collapsed)",
            "build_files": [os.path.basename(x) for x in build_files],
            "query_files": [os.path.basename(x) for x in query_files],
            "query_workload": f"top {TOP_PATHS} announced AS_PATHs of the query window, ranked by real announcement frequency",
            "gt": "bottleneck = min accumulated link weight in stream prefix; 0 if any link unseen (novel-link route)",
            "slices": slices,
        }, f, indent=2)
    print("DONE")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("parse")
    p1.add_argument("--mrt-dir", required=True)
    p1.add_argument("--files", required=True, help="comma-separated MRT files")
    p1.add_argument("--max-seconds", type=float, default=35.0)
    p2 = sub.add_parser("build")
    p2.add_argument("--mrt-dir", required=True)
    p2.add_argument("--build-files", required=True, help="comma-separated, chronological")
    p2.add_argument("--query-files", required=True)
    p2.add_argument("--out-root", default="DownstreamDatasets")
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--slices", default=None, help="optional comma-separated subset of stream slices")
    args = ap.parse_args()
    if args.cmd == "parse":
        cmd_parse(args)
    else:
        cmd_build(args)


if __name__ == "__main__":
    main()
