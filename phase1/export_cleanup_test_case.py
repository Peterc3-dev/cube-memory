#!/usr/bin/env python3
"""Generate a binary test case for ggml's cube_memory_cleanup op.

Mirror of `export_retrieve_test_case.py` for the cleanup op. Writes
deterministic random query + codebook, the gold output (= the
codebook entry that wins the argmax of the real Hermitian inner
product), and matching shape metadata.

Companion C++ test: `tests/test-cube-memory-cleanup-roundtrip.cpp`
in the cube-memory-op llama.cpp branch.

Layout (little-endian, packed):
    [16 bytes] magic = b"CUBE_MEM_RTCL_01"
    [int64]    d
    [int64]    m
    [int64]    pad0  (8 bytes alignment to 32)
    [int32]    pad1  (4 bytes)
    [int32]    pad2  (4 bytes — keep header at 48 bytes total)
    [d * 4 bytes]      query
    [m * d * 4 bytes]  codebook (row-major: row j at offset j*d)
    [d * 4 bytes]      gold output
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


MAGIC = b"CUBE_MEM_RTCL_01"


def reference_cleanup(query: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Argmax cosine match against codebook.

    query    (d,)        — phasors as interleaved (re, im) F32
    codebook (m, d)      — m phasor rows
    returns  (d,)        — winning codebook row, byte-identical
    """
    sims = codebook @ query  # real Hermitian inner product real part
    best = int(np.argmax(sims))
    return codebook[best].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path)
    ap.add_argument("--seed", type=int, default=0xDEADBEEF)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--m", type=int, default=32)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    # Random phasors as (re, im) interleaved real f32 — for the
    # cleanup op the only constraint is that d be even.
    assert args.d % 2 == 0, "d must be even (interleaved re/im)"
    query    = rng.standard_normal(args.d).astype(np.float32)
    codebook = rng.standard_normal((args.m, args.d)).astype(np.float32)
    gold     = reference_cleanup(query, codebook)

    with args.output.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<qqqii", args.d, args.m, 0, 0, 0))
        f.write(query.tobytes())
        f.write(codebook.tobytes())
        f.write(gold.tobytes())

    print(f"wrote {args.output} (d={args.d} m={args.m})")
    print(f"gold[:4] = {gold[:4].tolist()}")


if __name__ == "__main__":
    main()
