"""Independent sanity check of the candidate downstream datasets.

Decodes the binary-encoded arrays back to integer IDs and recomputes every
ground truth from the support stream, in a codepath separate from the builders.
"""
import os
import sys
import numpy as np

ROOT = os.environ.get("DOWNSTREAM_ROOT", "DownstreamDatasets")
BITS = 32


def decode(x):  # [N, k*32] uint8/float(0/1) -> [N, k] int64, memory-light
    x = np.asarray(x)
    if x.dtype != np.uint8:
        x = x.astype(np.uint8)
    packed = np.packbits(x, axis=1)          # [N, 4k] bytes
    return packed.view(">u4").astype(np.int64)  # [N, k]


def check_nodeflow(task_dir, weighted=False):
    with np.load(os.path.join(task_dir, "0.npz")) as d:
        sup = decode(d["support_x"])
        sup_w = d["support_y"].astype(np.float64)
        qn = decode(d["query_node_x"])[:, 0]
        qny = d["query_node_y"][:, 0]
    with np.load(os.path.join(task_dir, "nodeflow.npz")) as d:
        nx = decode(d["node_x"])[:, 0]
        out_y, in_y = d["out_y"], d["in_y"]
    s, t = sup[:, 0], sup[:, 1]
    w = sup_w if weighted else np.ones(len(s))
    nodes = np.unique(np.concatenate([s, t]))
    og = np.zeros(len(nodes)); ig = np.zeros(len(nodes))
    np.add.at(og, np.searchsorted(nodes, s), w)
    np.add.at(ig, np.searchsorted(nodes, t), w)
    rtol = 1e-4 if weighted else 0
    assert np.allclose(out_y, og[np.searchsorted(nodes, nx)], rtol=rtol), "out_y mismatch"
    assert np.allclose(in_y, ig[np.searchsorted(nodes, nx)], rtol=rtol), "in_y mismatch"
    assert np.allclose(qny, (og + ig)[np.searchsorted(nodes, qn)], rtol=rtol), "query_node_y mismatch"
    extra = ""
    tsr = os.path.join(task_dir, "tsr2_params.npz")
    if os.path.isfile(tsr):
        with np.load(tsr) as d:
            tx = decode(d["node_x"])[:, 0]
            to, ti = d["out_y"], d["in_y"]
        idx = np.searchsorted(nodes, tx)
        assert np.allclose(to, og[idx], rtol=1e-4) and np.allclose(ti, ig[idx], rtol=1e-4), \
            "tsr2 param GT mismatch"
        extra = f", {len(tx)} TSR2 param queries verified"
    print(f"  [OK] nodeflow {os.path.basename(task_dir)}: "
          f"{len(nx)} nodes, directional+combined GT verified{extra}")


def check_downstream(task_dir, kind, weighted=False):
    with np.load(os.path.join(task_dir, "0.npz")) as d:
        sup = decode(d["support_x"])
        sup_w = d["support_y"].astype(np.float64)
    s, t = sup[:, 0], sup[:, 1]
    wt = sup_w if weighted else np.ones(len(s))
    mod = int(max(s.max(), t.max())) + 1
    k, inv = np.unique(s * mod + t, return_inverse=True)
    acc = np.zeros(len(k)); np.add.at(acc, inv, wt)
    w = dict(zip(k.tolist(), acc.tolist()))
    with np.load(os.path.join(task_dir, "downstream.npz")) as d:
        pe, po, pt = d["path_edges"], d["path_offsets"], d["path_targets"]
        se, so, st = d["sg_edges"], d["sg_offsets"], d["sg_targets"]
    if kind == "path":
        assert len(po) - 1 == len(pt) and len(st) == 0
        ids = decode(pe)
        bad = 0
        for i in range(len(pt)):
            seg = ids[po[i]:po[i + 1]]
            gt = min(w.get(int(a) * mod + int(b), 0.0) for a, b in seg)
            if abs(gt - pt[i]) > 1e-3 * max(1.0, abs(gt)):
                bad += 1
        assert bad == 0, f"{bad} path GT mismatches"
        nz = int((pt > 0).sum())
        print(f"  [OK] path {os.path.basename(task_dir)}: {len(pt)} queries "
              f"({nz} observed / {len(pt)-nz} novel-link), bottleneck GT verified")
    else:
        assert len(so) - 1 == len(st) and len(pt) == 0
        ids = decode(se)
        for i in range(len(st)):
            seg = ids[so[i]:so[i + 1]]
            gt = sum(w.get(int(a) * mod + int(b), 0.0) for a, b in seg)
            assert abs(gt - st[i]) < 1e-3 * max(1.0, abs(gt)), \
                f"sg GT mismatch at {i}: {gt} vs {st[i]}"
        print(f"  [OK] subgraph {os.path.basename(task_dir)}: {len(st)} cells, sum GT verified")


def task_dirs(p):
    return sorted([os.path.join(p, d) for d in os.listdir(p)
                   if os.path.isdir(os.path.join(p, d))],
                  key=lambda s: int(os.path.basename(s).split("_")[1]))


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    suites = []
    for d in sorted(os.listdir(os.path.join(ROOT, "NodeFlow"))):
        if os.path.isdir(os.path.join(ROOT, "NodeFlow", d)):
            w = d.startswith("FinBench")
            suites.append((f"NodeFlow/{d}", lambda td, w=w: check_nodeflow(td, weighted=w)))
    for d in sorted(os.listdir(os.path.join(ROOT, "PathQuery"))):
        if os.path.isdir(os.path.join(ROOT, "PathQuery", d)) and d != "raw_mrt":
            suites.append((f"PathQuery/{d}", lambda td: check_downstream(td, "path", weighted=False)))
    for d in sorted(os.listdir(os.path.join(ROOT, "SubgraphQuery"))):
        if os.path.isdir(os.path.join(ROOT, "SubgraphQuery", d)):
            suites.append((f"SubgraphQuery/{d}", lambda td: check_downstream(td, "subgraph", weighted=False)))
    for rel, fn in suites:
        p = os.path.join(ROOT, *rel.split("/"))
        if only and only not in rel:
            continue
        if not os.path.isdir(p):
            print(f"{rel}: (absent, skipped)")
            continue
        print(f"{rel}:")
        for td in task_dirs(p):
            fn(td)
    print("ALL CHECKS PASSED")
