#!/usr/bin/env python3
"""Generate a binary test case for ggml's cube_memory_retrieve op.

Computes a deterministic random query / slot_keys / slot_values, runs
the algorithm in pure NumPy (the canonical reference), and writes
both the inputs and the gold output to a single binary file.

The matching C++ test (`tests/test-cube-memory-retrieve-roundtrip.cpp`
in the llama.cpp `cube-memory-op` branch) reads this file, runs the
same computation through ggml's CPU op, and asserts byte-level parity
on inputs (layout check) and fp tolerance on the output (algorithm
check).

Layout (little-endian, packed):
    [16 bytes] magic = b"CUBE_MEM_RTRT_01"
    [int64]    d_key
    [int64]    n_slots
    [int64]    d_value
    [int32]    top_k
    [int32]    pad (alignment)
    [d_key * 4 bytes]                query (f32)
    [n_slots * d_key * 4 bytes]      slot_keys (f32, row-major)
    [n_slots * d_value * 4 bytes]    slot_values (f32, row-major)
    [d_value * 4 bytes]              gold output (f32)
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


MAGIC = b"CUBE_MEM_RTRT_01"


def reference_retrieve(
    query: np.ndarray,
    slot_keys: np.ndarray,
    slot_values: np.ndarray,
    top_k: int,
) -> np.ndarray:
    """Pure NumPy reference, identical algorithm to the ggml CPU op,
    rust-gpu shader, and Rust CPU reference."""
    sims = slot_keys @ query  # (n_slots,)
    topk_idx = np.argsort(-sims)[:top_k]
    topk_sims = sims[topk_idx]
    smax = float(topk_sims.max())
    weights = np.exp(topk_sims - smax)
    weights /= max(weights.sum(), 1e-8)
    out = np.zeros(slot_values.shape[1], dtype=np.float32)
    for t, j in enumerate(topk_idx):
        out += weights[t].astype(np.float32) * slot_values[j]
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path)
    ap.add_argument("--seed", type=int, default=0xCAFE_F00D)
    ap.add_argument("--d-key", type=int, default=32)
    ap.add_argument("--n-slots", type=int, default=128)
    ap.add_argument("--d-value", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=4)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    query = rng.standard_normal(args.d_key).astype(np.float32)
    slot_keys = rng.standard_normal((args.n_slots, args.d_key)).astype(np.float32)
    slot_values = rng.standard_normal((args.n_slots, args.d_value)).astype(np.float32)

    gold = reference_retrieve(query, slot_keys, slot_values, args.top_k)

    with args.output.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack(
            "<qqqii",
            args.d_key, args.n_slots, args.d_value, args.top_k, 0,
        ))
        f.write(query.tobytes())
        f.write(slot_keys.tobytes())
        f.write(slot_values.tobytes())
        f.write(gold.tobytes())

    print(f"wrote {args.output} "
          f"(d_key={args.d_key} n_slots={args.n_slots} "
          f"d_value={args.d_value} top_k={args.top_k})")
    print(f"gold[:4] = {gold[:4].tolist()}")


if __name__ == "__main__":
    main()
