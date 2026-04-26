#!/usr/bin/env python3
"""Generate a binary test case for the simplified CubeMemoryLayer
forward pass (the smallest useful wedge toward integrating the
layer into llama.cpp's model loader).

The "layer" here is a degenerate single-axis (p=1) variant that
skips the to_phasor/bind/unitize algebra and just exercises the
data flow through the two new ggml ops in sequence:

    qreal    = role_proj  @ h                   (d_in,)        -> (2*d_codebook,)
    cleaned  = cube_memory_cleanup(qreal, codebook)            -> (2*d_codebook,)
    gathered = cube_memory_retrieve(cleaned,
                                    slot_keys,
                                    slot_values, top_k)        -> (d_value,)
    delta    = out_proj @ gathered              (d_value,)     -> (d_in,)

Companion C++ test:
    tests/test-cube-memory-layer.cpp
in the cube-memory-op llama.cpp branch.

Tensor layout convention: matches the existing cleanup/retrieve
roundtrip exporters and ggml's standard numpy-bytes-to-ggml-ne
mapping (a numpy (rows, cols) row-major tensor maps to a ggml
tensor with ne[0]=cols, ne[1]=rows). We pre-orient role_proj and
out_proj in numpy so that their on-disk bytes drop directly into
the ggml tensors that ggml_mul_mat expects:

    role_proj : numpy (d_key, d_in)    -> ggml (d_in, d_key)
        ggml_mul_mat(role_proj, h)         -- h has ne[0] = d_in
        produces (d_key, 1).
    out_proj  : numpy (d_in, d_value)  -> ggml (d_value, d_in)
        ggml_mul_mat(out_proj, gathered)   -- gathered has ne[0] = d_value
        produces (d_in, 1).

NumPy reference therefore expresses these as
    qreal = role_proj @ h     with role_proj.shape == (d_key, d_in)
    delta = out_proj  @ gath  with out_proj.shape  == (d_in, d_value)

Binary layout (little-endian, packed):
    [16 bytes] magic = b"CUBE_MEM_RTLA_01"
    [int64]    d_in
    [int64]    d_codebook        # half of d_key (interleaved re/im)
    [int64]    m
    [int64]    n_slots
    [int64]    d_value
    [int32]    top_k
    [int32]    pad
    [d_in]                          h            (f32)
    [d_key   * d_in]                role_proj    (f32, numpy row-major (d_key, d_in))
    [m       * d_key]               codebook     (f32, numpy row-major (m, d_key))
    [n_slots * d_key]               slot_keys    (f32, numpy row-major (n_slots, d_key))
    [n_slots * d_value]             slot_values  (f32, numpy row-major (n_slots, d_value))
    [d_in    * d_value]             out_proj     (f32, numpy row-major (d_in, d_value))
    [d_in]                          gold delta   (f32)
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


MAGIC = b"CUBE_MEM_RTLA_01"


def reference_cleanup(query: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    sims = codebook @ query
    best = int(np.argmax(sims))
    return codebook[best].astype(np.float32)


def reference_retrieve(
    query: np.ndarray,
    slot_keys: np.ndarray,
    slot_values: np.ndarray,
    top_k: int,
) -> np.ndarray:
    sims = slot_keys @ query
    topk_idx = np.argsort(-sims)[:top_k]
    topk_sims = sims[topk_idx]
    smax = float(topk_sims.max())
    weights = np.exp(topk_sims - smax)
    weights /= max(weights.sum(), 1e-8)
    out = np.zeros(slot_values.shape[1], dtype=np.float32)
    for t, j in enumerate(topk_idx):
        out += weights[t].astype(np.float32) * slot_values[j]
    return out.astype(np.float32)


def reference_layer(
    h: np.ndarray,
    role_proj: np.ndarray,
    codebook: np.ndarray,
    slot_keys: np.ndarray,
    slot_values: np.ndarray,
    out_proj: np.ndarray,
    top_k: int,
) -> np.ndarray:
    qreal    = role_proj @ h                                 # (d_key,)
    cleaned  = reference_cleanup(qreal, codebook)            # (d_key,)
    gathered = reference_retrieve(cleaned, slot_keys, slot_values, top_k)  # (d_value,)
    delta    = out_proj @ gathered                           # (d_in,)
    return delta.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path)
    ap.add_argument("--seed",       type=int, default=0xBEEFCAFE)
    ap.add_argument("--d-in",       type=int, default=32)
    ap.add_argument("--d-codebook", type=int, default=8)   # 2*d_codebook = 16 reals
    ap.add_argument("--m",          type=int, default=16)
    ap.add_argument("--n-slots",    type=int, default=32)
    ap.add_argument("--d-value",    type=int, default=8)
    ap.add_argument("--top-k",      type=int, default=4)
    args = ap.parse_args()

    d_key = 2 * args.d_codebook
    rng = np.random.default_rng(args.seed)

    h           = rng.standard_normal(args.d_in).astype(np.float32)
    role_proj   = rng.standard_normal((d_key,        args.d_in )).astype(np.float32)
    codebook    = rng.standard_normal((args.m,       d_key     )).astype(np.float32)
    slot_keys   = rng.standard_normal((args.n_slots, d_key     )).astype(np.float32)
    slot_values = rng.standard_normal((args.n_slots, args.d_value)).astype(np.float32)
    out_proj    = rng.standard_normal((args.d_in,    args.d_value)).astype(np.float32)

    gold = reference_layer(
        h, role_proj, codebook, slot_keys, slot_values, out_proj, args.top_k,
    )

    with args.output.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack(
            "<qqqqqii",
            args.d_in, args.d_codebook, args.m,
            args.n_slots, args.d_value,
            args.top_k, 0,
        ))
        f.write(h.tobytes())
        f.write(role_proj.tobytes())
        f.write(codebook.tobytes())
        f.write(slot_keys.tobytes())
        f.write(slot_values.tobytes())
        f.write(out_proj.tobytes())
        f.write(gold.tobytes())

    print(
        f"wrote {args.output} "
        f"(d_in={args.d_in} d_codebook={args.d_codebook} "
        f"m={args.m} n_slots={args.n_slots} "
        f"d_value={args.d_value} top_k={args.top_k})"
    )
    print(f"gold[:4] = {gold[:4].tolist()}")


if __name__ == "__main__":
    main()
