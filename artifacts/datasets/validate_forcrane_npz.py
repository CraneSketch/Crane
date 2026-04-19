#!/usr/bin/env python3
"""Validate the four-array ForCrane edge-query format."""
import argparse
import sys
import zipfile

import numpy as np
from numpy.lib import format as npf


KEYS = ["support_x", "support_y", "query_edge_x", "query_edge_y"]
CHUNK_BYTES = 256 << 20


def read_header(stream):
    major, minor = npf.read_magic(stream)
    reader = getattr(npf, f"read_array_header_{major}_{minor}")
    shape, fortran, dtype = reader(stream)
    if fortran:
        raise ValueError("Fortran-ordered arrays are not supported")
    return shape, dtype


def member(archive, key):
    return archive.open(key + ".npy")


def members(archive):
    return {name[:-4] for name in archive.namelist() if name.endswith(".npy")}


def headers(archive):
    result = {}
    for key in KEYS:
        with member(archive, key) as stream:
            result[key] = read_header(stream)
    return result


def iter_chunks(archive, key):
    with member(archive, key) as stream:
        shape, dtype = read_header(stream)
        row_bytes = (int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
                     if len(shape) > 1 else dtype.itemsize)
        rows_per_chunk = max(1, CHUNK_BYTES // max(row_bytes, 1))
        offset = 0
        while offset < shape[0]:
            rows = min(rows_per_chunk, shape[0] - offset)
            raw = stream.read(rows * row_bytes)
            if len(raw) != rows * row_bytes:
                raise ValueError(f"truncated array member: {key}")
            yield np.frombuffer(raw, dtype=dtype).reshape((rows,) + tuple(shape[1:]))
            offset += rows


def first_rows(archive, key, count):
    with member(archive, key) as stream:
        shape, dtype = read_header(stream)
        rows = min(count, shape[0])
        row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
        raw = stream.read(rows * row_bytes)
        return np.frombuffer(raw, dtype=dtype).reshape((rows,) + tuple(shape[1:]))


def decode_edges(bits):
    packed = np.packbits(bits.astype(np.uint8), axis=1)
    return packed.view(">u4").astype(np.uint64)


def check_quick(archive):
    actual = members(archive)
    expected = set(KEYS)
    if actual != expected:
        print(f"  schema mismatch: missing={sorted(expected - actual)}, "
              f"extra={sorted(actual - expected)}")
        return False

    hdr = headers(archive)
    for key, (shape, dtype) in hdr.items():
        print(f"  {key:14s} shape={shape} dtype={dtype}")
    edge_count = hdr["support_x"][0][0]
    query_count = hdr["query_edge_x"][0][0]
    ok = True
    ok &= hdr["support_x"] == ((edge_count, 64), np.dtype("uint8"))
    ok &= hdr["support_y"] == ((edge_count,), np.dtype("float32"))
    ok &= hdr["query_edge_x"] == ((query_count, 64), np.dtype("uint8"))
    ok &= hdr["query_edge_y"] == ((query_count, 1), np.dtype("float32"))

    sample = first_rows(archive, "support_x", 5)
    decoded = decode_edges(sample)
    print("  first stream edges (src,dst):",
          [(int(row[0]), int(row[1])) for row in decoded])
    return bool(ok)


def check_full(archive):
    ok = check_quick(archive)
    if not ok:
        return False

    support_sum = query_sum = 0.0
    support_values_ok = query_values_ok = True
    for chunk in iter_chunks(archive, "support_y"):
        values = chunk.astype(np.float64)
        support_sum += float(values.sum())
        support_values_ok &= bool(np.isfinite(values).all() and (values >= 0).all())
    for chunk in iter_chunks(archive, "query_edge_y"):
        values = chunk.astype(np.float64)
        query_sum += float(values.sum())
        query_values_ok &= bool(np.isfinite(values).all() and (values >= 0).all())

    binary_ok = True
    for key in ("support_x", "query_edge_x"):
        for chunk in iter_chunks(archive, key):
            binary_ok &= bool(np.logical_or(chunk == 0, chunk == 1).all())

    totals_ok = abs(query_sum - support_sum) < 0.5
    print(f"  sum(support_y)={support_sum:.0f}, sum(query_edge_y)={query_sum:.0f}")
    print(f"  CHECK nonnegative finite weights: {support_values_ok and query_values_ok}")
    print(f"  CHECK binary edge features: {binary_ok}")
    print(f"  CHECK query total == support total: {totals_ok}")
    return bool(ok and support_values_ok and query_values_ok and binary_ok and totals_ok)


def compare(left, right):
    if members(left) != set(KEYS) or members(right) != set(KEYS):
        print("  reference comparison requires the four-array schema")
        return False
    left_headers, right_headers = headers(left), headers(right)
    result = True
    for key in KEYS:
        if left_headers[key] != right_headers[key]:
            print(f"  {key:14s} DIFF header")
            result = False
            continue
        same = all(np.array_equal(a, b)
                   for a, b in zip(iter_chunks(left, key), iter_chunks(right, key)))
        print(f"  {key:14s} {'identical' if same else 'DIFF'}")
        result &= same
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--reference")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    print("validating", args.path)
    with zipfile.ZipFile(args.path) as archive:
        ok = check_quick(archive) if args.quick else check_full(archive)
        if args.reference:
            with zipfile.ZipFile(args.reference) as reference:
                ok &= compare(archive, reference)
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
