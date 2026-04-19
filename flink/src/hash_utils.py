"""Deterministic hash functions for edge partitioning.

Uses zlib.crc32 (not Python's hash()) because hash() is randomized per
process in Python 3.3+. CRC32 is deterministic across processes, so
prepare_data.py (host) and operators.py (container) produce identical hashes.
"""

import struct
import zlib

import numpy as np

MAX_BUCKETS = 16  # Coarse bucket count for keyBy; must be >= max parallelism


def hash_edge_bytes(edge_vec: np.ndarray) -> int:
    """CRC32 hash of a uint8 edge vector. Returns non-negative 32-bit int."""
    return zlib.crc32(edge_vec.tobytes()) & 0xFFFFFFFF


def bucket_edge_bytes(edge_vec: np.ndarray) -> int:
    """Hash edge vector to a bucket ID in [0, MAX_BUCKETS)."""
    return hash_edge_bytes(edge_vec) % MAX_BUCKETS


def hash_edge_ints(src: int, dst: int) -> int:
    """CRC32 hash of packed (src, dst) integer pair. Returns non-negative 32-bit int."""
    raw = struct.pack('<QQ', src & 0xFFFFFFFFFFFFFFFF, dst & 0xFFFFFFFFFFFFFFFF)
    return zlib.crc32(raw) & 0xFFFFFFFF


def bucket_edge_ints(src: int, dst: int) -> int:
    """Hash integer edge to a bucket ID in [0, MAX_BUCKETS)."""
    return hash_edge_ints(src, dst) % MAX_BUCKETS
