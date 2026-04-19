"""Build the RouteViews per-AS footprint SubgraphQuery dataset.

Real workload: the "routing footprint of an AS" -- all AS links adjacent to a
given AS in the routing-activity graph. Entity-defined subgraphs (one per real
AS), the aggregate an operator inspects when auditing a network's connectivity
or a hijacker's blast radius. Complements the CAIDA prefix-matrix variant with
zero extra data cost: it reuses the RouteViews_ASPaths stream.

For each existing RouteViews_ASPaths/<task_dir>:
  - hard-link its 0.npz into RouteViews_ASFootprint/<task_dir>/
  - sg queries = for each selected AS X: all unique links (X,*) and (*,X)
    observed in this stream prefix; GT = sum of their accumulated weights
  - AS selection: 3..MAX_EGO_EDGES adjacent links, top TOP_AS by activity
    (activity-ranked sweep, not sampled); labels in ego_meta.npz
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import encode_edges, save_downstream_npz

MAX_EGO_EDGES = 300
MIN_EGO_EDGES = 3
TOP_AS = 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="DownstreamDatasets")
    ap.add_argument("--task-dirs", default=None, help="optional comma-separated subset of task dir names")
    args = ap.parse_args()

    src_root = os.path.join(args.out_root, "PathQuery", "RouteViews_ASPaths")
    dst_root = os.path.join(args.out_root, "SubgraphQuery", "RouteViews_ASFootprint")
    asn_map = np.load(os.path.join(args.out_root, "PathQuery", "RouteViews_ASPaths_asnmap.npz"))["asn"]

    tasks = sorted([t for t in os.listdir(src_root)
                    if os.path.isdir(os.path.join(src_root, t))],
                   key=lambda s: int(s.split("_")[1]))
    if args.task_dirs:
        keep = set(args.task_dirs.split(","))
        tasks = [t for t in tasks if t in keep]

    for task in tasks:
        with np.load(os.path.join(src_root, task, "0.npz")) as d:
            sup = np.packbits(d["support_x"], axis=1).view(">u4").astype(np.int64)
            sup_w = d["support_y"].astype(np.float64)
        mod = int(sup.max()) + 1
        keys = sup[:, 0] * mod + sup[:, 1]
        uniq, inv = np.unique(keys, return_inverse=True)
        qey = np.zeros(len(uniq))
        np.add.at(qey, inv, sup_w)
        u_s = (uniq // mod).astype(np.int64)
        u_d = (uniq % mod).astype(np.int64)
        del sup, sup_w, keys, inv

        out_dir = os.path.join(dst_root, task)
        os.makedirs(out_dir, exist_ok=True)
        dst_npz = os.path.join(out_dir, "0.npz")
        if not os.path.exists(dst_npz):
            try:
                os.link(os.path.join(src_root, task, "0.npz"), dst_npz)
            except OSError:
                import shutil
                shutil.copyfile(os.path.join(src_root, task, "0.npz"), dst_npz)

        # group unique links by adjacent AS (both directions)
        deg_nodes = np.concatenate([u_s, u_d])
        deg_w = np.concatenate([qey, qey]).astype(np.float64)
        edge_idx = np.concatenate([np.arange(len(u_s)), np.arange(len(u_s))])
        order = np.argsort(deg_nodes, kind="stable")
        nodes_sorted = deg_nodes[order]
        bounds = np.flatnonzero(np.diff(nodes_sorted)) + 1
        groups = np.split(order, bounds)

        cand = []
        for g in groups:
            e = np.unique(edge_idx[g])  # dedup links seen from both directions
            if MIN_EGO_EDGES <= len(e) <= MAX_EGO_EDGES:
                cand.append((float(qey[e].sum()), int(deg_nodes[g[0]]), e))
        cand.sort(key=lambda x: -x[0])
        cand = cand[:TOP_AS]

        sg_queries, sg_targets, meta_as, meta_ne = [], [], [], []
        for gt, node_id, e in cand:
            sg_queries.append(encode_edges(u_s[e], u_d[e]))
            sg_targets.append(gt)
            meta_as.append(int(asn_map[node_id - 1]))
            meta_ne.append(len(e))
        save_downstream_npz(out_dir, [], [], sg_queries, sg_targets)
        np.savez_compressed(os.path.join(out_dir, "ego_meta.npz"),
                            asn=np.array(meta_as, np.int64),
                            num_edges=np.array(meta_ne, np.int32),
                            gt=np.array(sg_targets, np.float32))
        print(f"[asfootprint] {task}: {len(sg_targets)} AS egos "
              f"gt[min,max]=({min(sg_targets):.0f},{max(sg_targets):.0f})", flush=True)

    with open(os.path.join(dst_root, "provenance.json"), "w") as f:
        json.dump({"source": "derived from PathQuery/RouteViews_ASPaths stream (hard-linked 0.npz)",
                   "subgraph": "per-AS ego (all adjacent AS links, both directions)",
                   "selection": f"ASes with {MIN_EGO_EDGES}..{MAX_EGO_EDGES} adjacent links, top {TOP_AS} by summed link weight"},
                  f, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
