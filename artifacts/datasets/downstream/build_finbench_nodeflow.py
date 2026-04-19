"""Build the FinBench-based NodeFlow dataset (LDBC FinBench SF1, TSR2 workload).

Real workload rationale (C1/R2.O3): LDBC FinBench is the industry-derived
financial graph benchmark; its simple read TSR2 -- "given an account, find the
sum and max of fund amount in transfer-ins and transfer-outs" -- is exactly the
S3 node-flow query over the account-transfer stream. Both the data (skewed,
hub-heavy by design) and the query workload (driver-curated parameter accounts)
come from the benchmark, answering R2's "if benchmarks exist ..." clause.

Output: DownstreamDatasets/NodeFlow/FinBench_SF1_TSR2/size_<N>_unique_edge_<U>/
    0.npz          standard keys; support_y = transfer AMOUNT (weighted stream)
    nodeflow.npz   node_x (all active accounts), out_y/in_y (TSR2 sum GT,
                   directional), freq (transfer activity)
    tsr2_params.npz  driver-curated workload: the Account ids appearing in the
                   official params/complex_*_param.csv files (with multiplicity
                   = curation frequency), restricted to accounts; plus their
                   directional GT. These are the benchmark's own query params.
  DownstreamDatasets/NodeFlow/FinBench_SF1_TSR2_idmap.npz (index+1 -> account id)

Source: sf1.tar.xz (md5 09049e0dd9982075157ee0f2d6e70508) from the LDBC
FinBench reference-implementation dataset release (v0.2.0, Aliyun OSS).
Stream order = createTime over snapshot + incremental transfer files.
Amounts are kept in currency units (float32).
"""
import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import encode_ids, build_standard_npz

SLICES = [40_000, 80_000, None]  # None = full stream


def _read_transfer_csv(path, ts_is_epoch):
    """Stream one transfer CSV with pyarrow (low memory). Returns 4 arrays."""
    import pyarrow.csv as pacsv
    ropts = pacsv.ReadOptions(block_size=1 << 24)
    popts = pacsv.ParseOptions(delimiter="|")
    copts = pacsv.ConvertOptions(
        include_columns=["fromId", "toId", "amount", "createTime"],
        column_types={"createTime": "string"})
    fid, tid, amt, ts = [], [], [], []
    with pacsv.open_csv(path, read_options=ropts, parse_options=popts,
                        convert_options=copts) as reader:
        for batch in reader:
            fid.append(batch.column("fromId").to_numpy(zero_copy_only=False))
            tid.append(batch.column("toId").to_numpy(zero_copy_only=False))
            amt.append(batch.column("amount").to_numpy(zero_copy_only=False))
            ct = batch.column("createTime").to_numpy(zero_copy_only=False)
            if ts_is_epoch:
                ts.append(ct.astype(np.int64))
            else:
                ts.append(pd.to_datetime(pd.Series(ct)).astype("int64").to_numpy() // 10**6)
    return (np.concatenate(fid).astype(np.int64), np.concatenate(tid).astype(np.int64),
            np.concatenate(amt).astype(np.float64), np.concatenate(ts))


def load_stream(fb_root):
    """Return src_id, dst_id (raw account ids), amount, ts (ms) sorted by time.
    Uses the snapshot transfer history; incremental files are appended when
    present (large-SF archives are extracted snapshot-only)."""
    parts = [_read_transfer_csv(os.path.join(fb_root, "snapshot", "AccountTransferAccount.csv"),
                                ts_is_epoch=False)]
    for f in sorted(glob.glob(os.path.join(fb_root, "incremental",
                                           "AddAccountTransferAccountReadWrite*.csv"))):
        parts.append(_read_transfer_csv(f, ts_is_epoch=True))
    fid = np.concatenate([p[0] for p in parts])
    tid = np.concatenate([p[1] for p in parts])
    amt = np.concatenate([p[2] for p in parts])
    ts = np.concatenate([p[3] for p in parts])
    order = np.argsort(ts, kind="stable")
    return fid[order], tid[order], amt[order], ts[order]


def load_param_accounts(fb_root, account_ids):
    """Account ids referenced by the official driver parameter files."""
    acc = set(account_ids.tolist())
    hits, prov = [], {}
    for f in sorted(glob.glob(os.path.join(fb_root, "params", "complex_*_param.csv"))):
        df = pd.read_csv(f, sep="|")
        ids = [c for c in df.columns if c in ("id", "id1", "id2")]
        n = 0
        for c in ids:
            for v in df[c].to_numpy(np.int64):
                if int(v) in acc:
                    hits.append(int(v))
                    n += 1
        prov[os.path.basename(f)] = n
    return hits, prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finbench-root", default="/tmp/finbench/sf1")
    ap.add_argument("--out-root", default="DownstreamDatasets")
    ap.add_argument("--name", default="FinBench_SF1_TSR2", help="output dataset dir name")
    ap.add_argument("--full-only", action="store_true", help="build only the full-stream task dir")
    ap.add_argument("--stream-cache", default=None,
                    help="npz with fid/tid/amt/ts (pre-parsed snapshot); skips CSV parsing")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.stream_cache:
        with np.load(args.stream_cache) as d:
            src_raw, dst_raw, amount, ts = d["fid"], d["tid"], d["amt"], d["ts"]
        order = np.argsort(ts, kind="stable")
        src_raw, dst_raw, amount, ts = (src_raw[order], dst_raw[order],
                                        amount[order], ts[order])
    else:
        src_raw, dst_raw, amount, ts = load_stream(args.finbench_root)
    print(f"[finbench] stream: {len(src_raw)} transfers, "
          f"span {ts.min()}..{ts.max()}")

    all_ids = np.unique(np.concatenate([src_raw, dst_raw]))
    root = os.path.join(args.out_root, "NodeFlow")
    os.makedirs(root, exist_ok=True)
    np.savez_compressed(os.path.join(root, f"{args.name}_idmap.npz"), account_id=all_ids)
    lut_pos = {int(a): i + 1 for i, a in enumerate(all_ids)}
    src = np.searchsorted(all_ids, src_raw) + 1
    dst = np.searchsorted(all_ids, dst_raw) + 1

    param_ids_raw, prov = load_param_accounts(args.finbench_root, all_ids)
    print(f"[finbench] driver-curated account params: {len(param_ids_raw)} "
          f"({len(set(param_ids_raw))} unique) from {prov}")

    ds_root = os.path.join(root, args.name)
    slices = [None] if args.full_only else SLICES
    for sl in slices:
        n = len(src) if sl is None else min(sl, len(src))
        s, d, w = src[:n], dst[:n], amount[:n]
        mod = int(max(s.max(), d.max())) + 1
        uniq = np.unique(s.astype(np.int64) * mod + d.astype(np.int64))
        task = f"size_{n}_unique_edge_{len(uniq)}"
        out_dir = os.path.join(ds_root, task)
        stats = build_standard_npz(out_dir, s, d, seed=args.seed, weights=w,
                                   include=("node",), compress=(n <= 4_000_000))
        print(f"[finbench] {task}: {stats}", flush=True)

        # directional GT over all active accounts (TSR2 semantics, full history)
        nodes = np.unique(np.concatenate([s, d]))
        out_y = np.zeros(len(nodes), np.float64)
        in_y = np.zeros(len(nodes), np.float64)
        cnt = np.zeros(len(nodes), np.float64)
        np.add.at(out_y, np.searchsorted(nodes, s), w)
        np.add.at(in_y, np.searchsorted(nodes, d), w)
        np.add.at(cnt, np.searchsorted(nodes, s), 1.0)
        np.add.at(cnt, np.searchsorted(nodes, d), 1.0)
        np.savez_compressed(os.path.join(out_dir, "nodeflow.npz"),
                            node_x=encode_ids(nodes),
                            out_y=out_y.astype(np.float32),
                            in_y=in_y.astype(np.float32),
                            freq=cnt.astype(np.float32))

        # benchmark-curated TSR2 workload (param accounts active in this slice)
        active = set(nodes.tolist())
        q = [lut_pos[a] for a in param_ids_raw if lut_pos[a] in active]
        if q:
            q = np.array(q, np.int64)
            qi = np.searchsorted(nodes, q)
            np.savez_compressed(os.path.join(out_dir, "tsr2_params.npz"),
                                node_x=encode_ids(q),
                                out_y=out_y[qi].astype(np.float32),
                                in_y=in_y[qi].astype(np.float32))
        print(f"[finbench] {task}: tsr2 param queries active: {len(q)}")

    with open(os.path.join(ds_root, "provenance.json"), "w") as f:
        json.dump({"source": "LDBC FinBench SF1 v0.2.0 (Aliyun OSS release, md5 09049e0dd9982075157ee0f2d6e70508)",
                   "stream": "AccountTransferAccount snapshot + incremental, createTime order, weight = amount",
                   "workload": "TSR2 (sum of transfer-ins/outs per account); query accounts = official driver params (complex_*_param.csv) + full active-account sweep",
                   "param_provenance": prov,
                   "note": "time-window params map to full history under the cumulative setting (S3/C2)"},
                  f, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
