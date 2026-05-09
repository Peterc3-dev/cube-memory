#!/usr/bin/env python3
"""Export a small CubeMemoryLayer + gold forward-pass output to GGUF.

This is the Python half of the Phase 2 round-trip acceptance test.
It creates a deterministic CubeMemoryLayer, runs its full forward pass
(phasor projection, per-axis cleanup, FHRR bind, unitize, top-k
retrieval, output projection) on a fixed input, and writes every
tensor plus the gold output to a GGUF file.

The companion C++ test loads this GGUF, rebuilds the same computation
in ggml, and asserts numerical parity with the gold output.

GGUF tensor naming and layout
------------------------------

All tensors are F32.  NumPy (rows, cols) row-major maps to ggml
(ne[0]=cols, ne[1]=rows), so PyTorch weight tensors are exported
directly — no transpose needed.

Tensors:
    cube_memory.input                — (d_in,)
    cube_memory.role_proj.weight     — (p*d_codebook, d_in)
    cube_memory.codebook_0 .. _{p-1} — (m, 2*d_codebook)  [complex -> interleaved re,im]
    cube_memory.slot_keys            — (n_slots, 2*d_codebook)
    cube_memory.slot_values          — (n_slots, d_value)
    cube_memory.out_proj.weight      — (d_in, d_value)
    cube_memory.gold_output          — (d_in,)

KV metadata (all uint32):
    cube_memory.{d_in, d_codebook, d_value, m, p, n_slots, top_k, seed}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# gguf-py from the local llama.cpp build
sys.path.insert(0, "/home/raz/builds/llama-cpp-vulkan/gguf-py")
from gguf import GGUFWriter  # noqa: E402

# CubeMemoryLayer from the same package
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cube_memory_layer import CubeMemoryLayer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export a CubeMemoryLayer round-trip test case to GGUF.",
    )
    ap.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("cube_memory_roundtrip.gguf"),
        help="Output GGUF path (default: cube_memory_roundtrip.gguf)",
    )
    args = ap.parse_args()

    # ── Test parameters (small for fast round-trip) ──────────────
    d_in = 32
    d_codebook = 8
    d_value = 8
    m = 16
    p = 3
    n_slots = 32
    top_k = 4
    seed = 42

    # ── Build the layer deterministically ────────────────────────
    torch.manual_seed(seed)
    layer = CubeMemoryLayer(
        d_in=d_in,
        d_codebook=d_codebook,
        d_value=d_value,
        m=m,
        p=p,
        n_slots=n_slots,
        top_k=top_k,
        seed=seed,
    )

    # out_proj is initialized to zeros, which would give an
    # all-zero gold output — perturb it so the test is meaningful.
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        layer.out_proj.weight.copy_(torch.randn_like(layer.out_proj.weight) * 0.02)

    layer.eval()

    # ── Generate a deterministic input ───────────────────────────
    torch.manual_seed(seed + 2)
    h = torch.randn(1, 1, d_in, dtype=torch.float32)  # (B=1, T=1, D)

    # ── Run the full forward pass for the gold output ────────────
    with torch.no_grad():
        gold = layer(h)  # (1, 1, d_in)

    # Squeeze batch/time dims for export
    h_vec = h.squeeze(0).squeeze(0)          # (d_in,)
    gold_vec = gold.squeeze(0).squeeze(0)    # (d_in,)

    # ── Print diagnostics ────────────────────────────────────────
    gold_np = gold_vec.numpy()
    print(f"gold[:4] = {gold_np[:4].tolist()}")
    print(f"gold norm = {float(np.linalg.norm(gold_np)):.8f}")

    # ── Convert codebooks from complex64 to interleaved (re,im) ─
    codebook_reals = []
    for ax in range(p):
        cb_complex = getattr(layer, f"codebook_{ax}")  # (m, d_codebook) complex64
        # torch.view_as_real: (m, d_codebook) complex -> (m, d_codebook, 2) float
        cb_ri = torch.view_as_real(cb_complex)          # (m, d_codebook, 2)
        # Reshape to interleaved: (m, 2*d_codebook)
        # For each row, the layout is [re0, im0, re1, im1, ...]
        cb_interleaved = cb_ri.reshape(m, 2 * d_codebook)
        codebook_reals.append(cb_interleaved)

    # ── Write GGUF ───────────────────────────────────────────────
    writer = GGUFWriter(args.output, arch="cube-memory")

    # KV metadata
    writer.add_uint32("cube_memory.d_in", d_in)
    writer.add_uint32("cube_memory.d_codebook", d_codebook)
    writer.add_uint32("cube_memory.d_value", d_value)
    writer.add_uint32("cube_memory.m", m)
    writer.add_uint32("cube_memory.p", p)
    writer.add_uint32("cube_memory.n_slots", n_slots)
    writer.add_uint32("cube_memory.top_k", top_k)
    writer.add_uint32("cube_memory.seed", seed)

    # Tensors — all contiguous f32 numpy arrays
    def t2np(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().contiguous().numpy().astype(np.float32)

    writer.add_tensor("cube_memory.input", t2np(h_vec))
    writer.add_tensor("cube_memory.role_proj.weight", t2np(layer.role_proj.weight.data))

    for ax in range(p):
        writer.add_tensor(f"cube_memory.codebook_{ax}", t2np(codebook_reals[ax]))

    writer.add_tensor("cube_memory.slot_keys", t2np(layer.slot_keys.data))
    writer.add_tensor("cube_memory.slot_values", t2np(layer.slot_values.data))
    writer.add_tensor("cube_memory.out_proj.weight", t2np(layer.out_proj.weight.data))
    writer.add_tensor("cube_memory.gold_output", t2np(gold_vec))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # ── Summary ──────────────────────────────────────────────────
    n_tensors = 2 + p + 4  # input, role_proj, p codebooks, slot_keys, slot_values, out_proj, gold
    print(
        f"wrote {args.output} "
        f"({n_tensors} tensors, "
        f"d_in={d_in} d_codebook={d_codebook} d_value={d_value} "
        f"m={m} p={p} n_slots={n_slots} top_k={top_k} seed={seed})"
    )


if __name__ == "__main__":
    main()
