#!/usr/bin/env python3
"""Parse and build the MAWI ForCrane dataset."""
import argparse
import contextlib
import gzip
import os
import struct
import sys
import time
import zipfile

import numpy as np
from numpy.lib import format as npf


PAPER_STAMPS = ["202512011400", "202512021400", "202512031400", "202512041400"]


def parse_pcap(stream, out_prefix, packet_chunk=1 << 23):
    if packet_chunk < 1:
        raise ValueError("packet_chunk must be positive")
    global_header = stream.read(24)
    if len(global_header) != 24:
        raise ValueError("truncated pcap global header")
    magic = global_header[:4]
    if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
        endian = "<"
    elif magic in (b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"):
        endian = ">"
    else:
        raise ValueError(f"unsupported pcap magic: {magic.hex()}")
    linktype = struct.unpack(endian + "I", global_header[20:24])[0]
    if linktype != 1:
        raise ValueError(f"expected EN10MB(1), got {linktype}")

    record = struct.Struct(endian + "IIII")
    chunks_src, chunks_dst = [], []
    buffer_src = np.empty(packet_chunk, dtype=np.uint32)
    buffer_dst = np.empty(packet_chunk, dtype=np.uint32)
    buffered = packets = ipv4 = ipv6 = other = 0
    started = time.time()

    while True:
        header = stream.read(16)
        if not header:
            break
        if len(header) < 16:
            raise ValueError("truncated pcap packet header")
        _, _, captured, _ = record.unpack(header)
        data = stream.read(captured)
        if len(data) < captured:
            raise ValueError("truncated pcap packet body")
        packets += 1
        if captured < 14:
            continue
        ethertype = (data[12] << 8) | data[13]
        offset = 14
        if ethertype == 0x8100 and captured >= 18:
            ethertype = (data[16] << 8) | data[17]
            offset = 18
        if ethertype == 0x0800 and captured >= offset + 20:
            buffer_src[buffered] = struct.unpack_from(">I", data, offset + 12)[0]
            buffer_dst[buffered] = struct.unpack_from(">I", data, offset + 16)[0]
            buffered += 1
            ipv4 += 1
            if buffered == packet_chunk:
                chunks_src.append(buffer_src)
                chunks_dst.append(buffer_dst)
                buffer_src = np.empty(packet_chunk, dtype=np.uint32)
                buffer_dst = np.empty(packet_chunk, dtype=np.uint32)
                buffered = 0
        elif ethertype == 0x86DD:
            ipv6 += 1
        else:
            other += 1
        if packets % 50_000_000 == 0:
            rate = packets / max(time.time() - started, 1e-9)
            print(f"{packets:,} packets, ipv4={ipv4:,}, {rate:,.0f}/s", flush=True)

    if buffered:
        chunks_src.append(buffer_src[:buffered])
        chunks_dst.append(buffer_dst[:buffered])
    src = np.concatenate(chunks_src) if chunks_src else np.empty(0, np.uint32)
    dst = np.concatenate(chunks_dst) if chunks_dst else np.empty(0, np.uint32)
    parent = os.path.dirname(os.path.abspath(out_prefix))
    os.makedirs(parent, exist_ok=True)
    np.save(out_prefix + "_src.npy", src)
    np.save(out_prefix + "_dst.npy", dst)
    elapsed = time.time() - started
    print(f"parsed packets={packets:,}, ipv4={ipv4:,}, ipv6={ipv6:,}, "
          f"other={other:,}, time={elapsed:.1f}s", flush=True)
    return src, dst


def parse_file(input_path, out_prefix, packet_chunk=1 << 23):
    if input_path == "-":
        manager = contextlib.nullcontext(sys.stdin.buffer)
    elif input_path.endswith(".gz"):
        manager = gzip.open(input_path, "rb")
    else:
        manager = open(input_path, "rb")
    with manager as stream:
        return parse_pcap(stream, out_prefix, packet_chunk)


def bits(values, width):
    if width not in (32, 64):
        raise ValueError("bit width must be 32 or 64")
    blocks = []
    for shift in range(0, width, 32):
        block = np.ascontiguousarray(values >> np.uint64(shift), dtype=">u8")
        encoded = np.unpackbits(block.view(np.uint8).reshape(-1, 8),
                                axis=1, bitorder="big")[:, -32:]
        blocks.append(encoded)
    return np.ascontiguousarray(np.concatenate(blocks, axis=1))


def stream_bits(archive, name, values, width, chunk):
    rows = len(values)
    with archive.open(name + ".npy", "w", force_zip64=True) as output:
        npf.write_array_header_1_0(output, {"descr": "|u1", "fortran_order": False,
                                           "shape": (rows, width)})
        for start in range(0, rows, chunk):
            output.write(bits(values[start:start + chunk], width).tobytes())


def write_array(archive, name, array):
    with archive.open(name + ".npy", "w", force_zip64=True) as output:
        npf.write_array(output, np.ascontiguousarray(array))


def edge_chunks(src_arrays, dst_arrays, chunk):
    offset = 0
    for src, dst in zip(src_arrays, dst_arrays):
        for start in range(0, len(src), chunk):
            src_chunk = np.asarray(src[start:start + chunk])
            dst_chunk = np.asarray(dst[start:start + chunk])
            yield offset, src_chunk, dst_chunk
            offset += len(src_chunk)


def build_big(src_arrays, dst_arrays, out_path, chunk=1 << 26,
              compresslevel=1, verbose=True):
    if chunk < 1:
        raise ValueError("chunk must be positive")
    if not isinstance(src_arrays, (list, tuple)):
        src_arrays, dst_arrays = [src_arrays], [dst_arrays]
    edge_count = sum(len(array) for array in src_arrays)
    if edge_count != sum(len(array) for array in dst_arrays):
        raise ValueError("source and destination lengths differ")
    if verbose:
        print(f"E={edge_count:,}", flush=True)

    table = np.zeros(1 << 32, dtype=np.uint32)
    next_id = 1
    for _, src, dst in edge_chunks(src_arrays, dst_arrays, chunk):
        endpoints = np.empty(2 * len(src), dtype=np.uint32)
        endpoints[0::2], endpoints[1::2] = src, dst
        unseen = endpoints[table[endpoints] == 0]
        if unseen.size:
            unique, first = np.unique(unseen, return_index=True)
            new_values = unique[np.argsort(first, kind="stable")]
            table[new_values] = np.arange(next_id, next_id + len(new_values),
                                          dtype=np.uint32)
            next_id += len(new_values)
    node_count = next_id - 1

    edges = np.empty(edge_count, dtype=np.uint64)
    for offset, src, dst in edge_chunks(src_arrays, dst_arrays, chunk):
        src_ids = table[src].astype(np.uint64)
        dst_ids = table[dst].astype(np.uint64)
        edges[offset:offset + len(src_ids)] = src_ids | (dst_ids << np.uint64(32))
    del table
    unique_edges, counts = np.unique(edges, return_counts=True)
    unique_count = len(unique_edges)
    edge_totals = counts.astype(np.float32)

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=compresslevel, allowZip64=True) as archive:
        stream_bits(archive, "support_x", edges, 64, chunk)
        with archive.open("support_y.npy", "w", force_zip64=True) as output:
            npf.write_array_header_1_0(output, {"descr": "<f4", "fortran_order": False,
                                               "shape": (edge_count,)})
            ones = np.ones(min(chunk, edge_count), np.float32)
            written = 0
            while written < edge_count:
                count = min(chunk, edge_count - written)
                output.write(ones[:count].tobytes())
                written += count
        del edges

        stream_bits(archive, "query_edge_x", unique_edges, 64, chunk)
        write_array(archive, "query_edge_y", edge_totals.reshape(-1, 1))

    if verbose:
        print(f"N={node_count:,}, Ue={unique_count:,}, saved {out_path}", flush=True)
    return {"N": node_count, "E": edge_count, "Ue": unique_count}


def build_dataset(parsed_dir, out_root, stamps, chunk=1 << 26, compresslevel=1):
    src = [np.load(os.path.join(parsed_dir, f"t_{stamp}_src.npy"), mmap_mode="r")
           for stamp in stamps]
    dst = [np.load(os.path.join(parsed_dir, f"t_{stamp}_dst.npy"), mmap_mode="r")
           for stamp in stamps]
    out_dir = os.path.join(out_root, "MAWI")
    os.makedirs(out_dir, exist_ok=True)
    temporary = os.path.join(out_dir, "_tmp_build.npz")
    stats = build_big(src, dst, temporary, chunk=chunk,
                      compresslevel=compresslevel)
    task_dir = os.path.join(out_dir, f"size_{stats['E']}_unique_edge_{stats['Ue']}")
    os.makedirs(task_dir, exist_ok=True)
    final = os.path.join(task_dir, "0.npz")
    os.replace(temporary, final)
    print(f"-> {final}", flush=True)
    return final


def parsed_arrays_valid(prefix):
    try:
        src = np.load(prefix + "_src.npy", mmap_mode="r")
        dst = np.load(prefix + "_dst.npy", mmap_mode="r")
    except (OSError, ValueError):
        return False
    return (src.dtype == np.uint32 and dst.dtype == np.uint32 and
            src.ndim == dst.ndim == 1 and len(src) == len(dst))


def add_build_arguments(parser):
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--stamps", nargs="+", default=PAPER_STAMPS)
    parser.add_argument("--chunk", type=int, default=1 << 26)
    parser.add_argument("--compresslevel", type=int, default=1)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    parse = commands.add_parser("parse")
    parse.add_argument("--input", required=True)
    parse.add_argument("--out-prefix", required=True)
    parse.add_argument("--packet-chunk", type=int, default=1 << 23)

    build = commands.add_parser("build")
    build.add_argument("--parsed-dir", required=True)
    add_build_arguments(build)

    all_steps = commands.add_parser("all")
    all_steps.add_argument("--workdir", required=True)
    all_steps.add_argument("--force-parse", action="store_true")
    all_steps.add_argument("--packet-chunk", type=int, default=1 << 23)
    add_build_arguments(all_steps)
    return parser


def main():
    args = make_parser().parse_args()
    if args.command == "parse":
        parse_file(args.input, args.out_prefix, args.packet_chunk)
    elif args.command == "build":
        build_dataset(args.parsed_dir, args.out_root, args.stamps,
                      args.chunk, args.compresslevel)
    else:
        for stamp in args.stamps:
            prefix = os.path.join(args.workdir, f"t_{stamp}")
            if args.force_parse or not parsed_arrays_valid(prefix):
                parse_file(os.path.join(args.workdir, f"{stamp}.pcap.gz"),
                           prefix, args.packet_chunk)
            else:
                print(f"{stamp}: parsed arrays already exist", flush=True)
        build_dataset(args.workdir, args.out_root, args.stamps,
                      args.chunk, args.compresslevel)


if __name__ == "__main__":
    main()
