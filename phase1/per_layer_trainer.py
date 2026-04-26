#!/usr/bin/env python3
"""Per-layer trainer for cube-memory consortium pipeline (Phase B).

Consumes activation chunks dumped by `llama-dump-activations` and fits a
single `CubeMemoryLayer` to map ffn_inp -> ffn_out under MSE.

Chunk file format (little-endian):
    [magic="CACT" 4 bytes]
    [version u32 = 1]
    [n_tokens u32]
    [n_embd u32]
    [n_tokens * n_embd * uint16  bf16 raw payload]

Usage:
    ~/rocm-gpu-test/venv/bin/python phase1/per_layer_trainer.py \\
        --activations-dir ~/cube-memory-cache/activations \\
        --layer 3 \\
        --output ~/cube-memory-cache/trained-layers/layer_3.safetensors \\
        --steps 5000 --batch-size 32 --lr 1e-3 --seed 42

The Qwen3.6-27B residual width (n_embd) is 5120; this trainer asserts
the chunk's embedded n_embd matches --d-in (default 5120). The inner
FFN dim n_ff=17408 is irrelevant — we match the residual stream.
"""
from __future__ import annotations

import argparse
import logging
import struct
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

CACT_MAGIC = b"CACT"
CACT_VERSION = 1
HEADER_STRUCT = struct.Struct("<4sIII")  # magic, version, n_tokens, n_embd
HEADER_SIZE = HEADER_STRUCT.size  # 16 bytes


def _read_chunk(path: Path, expected_d_in: int) -> torch.Tensor:
    """Read one CACT chunk and return a float32 tensor of shape
    (n_tokens, n_embd). Reads bf16 raw bytes, reinterprets as torch
    bfloat16, then up-casts to float32 for the trainer."""
    with path.open("rb") as f:
        header = f.read(HEADER_SIZE)
        if len(header) != HEADER_SIZE:
            raise ValueError(f"{path}: truncated header")
        magic, version, n_tokens, n_embd = HEADER_STRUCT.unpack(header)
        if magic != CACT_MAGIC:
            raise ValueError(f"{path}: bad magic {magic!r} (expected {CACT_MAGIC!r})")
        if version != CACT_VERSION:
            raise ValueError(f"{path}: unsupported version {version}")
        if n_embd != expected_d_in:
            raise ValueError(
                f"{path}: n_embd={n_embd} does not match expected d_in={expected_d_in}"
            )
        payload = f.read(n_tokens * n_embd * 2)
        if len(payload) != n_tokens * n_embd * 2:
            raise ValueError(
                f"{path}: payload size {len(payload)} != "
                f"{n_tokens * n_embd * 2} expected"
            )

    # bf16 is uint16 wire format; reinterpret bytes as bf16 then up-cast.
    raw = torch.frombuffer(bytearray(payload), dtype=torch.uint16)
    bf16 = raw.view(torch.bfloat16).reshape(n_tokens, n_embd)
    return bf16.to(torch.float32)


def _load_side(activations_dir: Path, layer: int, side: str, d_in: int) -> torch.Tensor:
    """Concatenate every chunk for one side ('in' or 'out') of one layer."""
    side_dir = activations_dir / f"layer_{layer}_{side}"
    if not side_dir.exists():
        raise FileNotFoundError(f"missing {side_dir}")
    chunk_paths = sorted(side_dir.glob("chunk_*.bin"))
    if not chunk_paths:
        raise FileNotFoundError(f"no chunk_*.bin under {side_dir}")
    parts = [_read_chunk(p, d_in) for p in chunk_paths]
    full = torch.cat(parts, dim=0)
    logger.info("loaded %s/%s: %d chunks, %d tokens, %d-dim",
                side_dir.name, side, len(parts), full.shape[0], full.shape[1])
    return full


def load_layer_pairs(
    activations_dir: Path, layer: int, d_in: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (ffn_in, ffn_out) tensors of shape (N, d_in)."""
    x_in = _load_side(activations_dir, layer, "in", d_in)
    x_out = _load_side(activations_dir, layer, "out", d_in)
    if x_in.shape != x_out.shape:
        raise ValueError(
            f"shape mismatch in vs out: {x_in.shape} vs {x_out.shape}"
        )
    return x_in, x_out


def train(
    activations_dir: Path,
    layer: int,
    output: Path,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    d_in: int,
    d_codebook: int,
    m: int,
    n_slots: int,
    d_value: int,
    top_k: int,
    p: int,
    val_split: float,
    log_every: int,
    val_every: int,
    device: str,
) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cube_memory_layer import CubeMemoryLayer  # local import after path fix
    from safetensors.torch import save_file

    torch.manual_seed(seed)

    x_in, x_out = load_layer_pairs(activations_dir, layer, d_in)
    n = x_in.shape[0]
    n_val = max(1, int(n * val_split))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    logger.info("split: %d train / %d val (val_frac=%.3f)",
                train_idx.numel(), val_idx.numel(), val_split)

    model = CubeMemoryLayer(
        d_in=d_in, d_codebook=d_codebook, d_value=d_value,
        m=m, p=p, n_slots=n_slots, top_k=top_k, seed=seed,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    x_in = x_in.to(device)
    x_out = x_out.to(device)

    rng = torch.Generator(device="cpu").manual_seed(seed + 1)

    train_loss_history: List[float] = []
    val_loss_history: List[Tuple[int, float]] = []
    initial_loss = None

    t0 = time.time()
    for step in range(steps):
        # CubeMemoryLayer expects (B, T, D); we flatten per-token, so use
        # B=1, T=batch_size to match the (..., D) contract.
        idx = train_idx[
            torch.randint(
                0, train_idx.numel(), (batch_size,), generator=rng
            )
        ]
        batch_in = x_in[idx].unsqueeze(0)   # (1, B, D)
        batch_out = x_out[idx].unsqueeze(0)  # (1, B, D)

        optim.zero_grad()
        pred = model(batch_in)
        loss = F.mse_loss(pred, batch_out)
        loss.backward()
        optim.step()

        train_loss_history.append(loss.item())
        if initial_loss is None:
            initial_loss = loss.item()

        if step % log_every == 0 or step == steps - 1:
            logger.info("step %5d  train_mse=%.6f", step, loss.item())

        if val_every > 0 and (step % val_every == 0 or step == steps - 1):
            model.eval()
            with torch.no_grad():
                v_in = x_in[val_idx].unsqueeze(0)
                v_out = x_out[val_idx].unsqueeze(0)
                v_pred = model(v_in)
                v_loss = F.mse_loss(v_pred, v_out).item()
            val_loss_history.append((step, v_loss))
            logger.info("step %5d  val_mse=%.6f", step, v_loss)
            model.train()

    elapsed = time.time() - t0
    logger.info("training done: %d steps in %.1fs (%.1f step/s)",
                steps, elapsed, steps / max(elapsed, 1e-6))

    # Save state_dict via safetensors. Codebook buffers are complex,
    # which safetensors does not support; split into real/imag pairs.
    output.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    for k, v in model.state_dict().items():
        if torch.is_complex(v):
            state[f"{k}.real"] = v.real.contiguous()
            state[f"{k}.imag"] = v.imag.contiguous()
        else:
            state[k] = v.contiguous()
    metadata = {
        "layer": str(layer),
        "d_in": str(d_in),
        "d_codebook": str(d_codebook),
        "d_value": str(d_value),
        "m": str(m),
        "p": str(p),
        "n_slots": str(n_slots),
        "top_k": str(top_k),
        "seed": str(seed),
        "steps": str(steps),
        "final_train_mse": f"{train_loss_history[-1]:.6e}",
    }
    save_file(state, str(output), metadata=metadata)
    logger.info("saved %s", output)

    return {
        "initial_train_mse": initial_loss,
        "final_train_mse": train_loss_history[-1],
        "val_history": val_loss_history,
        "wall_seconds": elapsed,
        "n_train": int(train_idx.numel()),
        "n_val": int(val_idx.numel()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations-dir", type=Path,
                    default=Path.home() / "cube-memory-cache" / "activations")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    # CubeMemoryLayer hyperparams; conservative defaults.
    ap.add_argument("--d-in", type=int, default=5120,
                    help="Qwen3.6-27B residual width.")
    ap.add_argument("--d-codebook", type=int, default=256)
    ap.add_argument("--m", type=int, default=64,
                    help="Codebook size per role-axis.")
    ap.add_argument("--p", type=int, default=2,
                    help="Number of role axes (bind depth).")
    ap.add_argument("--n-slots", type=int, default=4096)
    ap.add_argument("--d-value", type=int, default=2048)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--val-split", type=float, default=0.05)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--device", default="cpu",
                    help="cpu or cuda; default cpu (iGPU may be busy).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    train(
        activations_dir=args.activations_dir,
        layer=args.layer,
        output=args.output,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        d_in=args.d_in,
        d_codebook=args.d_codebook,
        m=args.m,
        n_slots=args.n_slots,
        d_value=args.d_value,
        top_k=args.top_k,
        p=args.p,
        val_split=args.val_split,
        log_every=args.log_every,
        val_every=args.val_every,
        device=args.device,
    )


if __name__ == "__main__":
    main()
