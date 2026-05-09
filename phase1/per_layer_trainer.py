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
from typing import Dict, List, Tuple

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
    version: int = 1,
    n_heads: int = 4,
    tau_init: float = 1.0,
    tau_final: float = 0.1,
) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
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

    if version == 2:
        from cube_memory_layer_v2 import CubeMemoryLayerV2
        model = CubeMemoryLayerV2(
            d_in=d_in, d_codebook=d_codebook, d_value=d_value,
            m=m, p=p, n_slots=n_slots, top_k=top_k, seed=seed,
            n_heads=n_heads, tau_init=tau_init,
        ).to(device)
    else:
        from cube_memory_layer import CubeMemoryLayer
        model = CubeMemoryLayer(
            d_in=d_in, d_codebook=d_codebook, d_value=d_value,
            m=m, p=p, n_slots=n_slots, top_k=top_k, seed=seed,
        ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Keep x_in/x_out on CPU; only move per-batch tensors to device.
    # At Qwen3.6-27B scale (5120 dim × 100k tokens × 2 sides × 4 bytes
    # = ~4 GB per layer per side) the full residual stream blows the
    # 16 GB VRAM budget when pinned to GPU.
    x_in = x_in.cpu().pin_memory() if device != "cpu" else x_in
    x_out = x_out.cpu().pin_memory() if device != "cpu" else x_out

    rng = torch.Generator(device="cpu").manual_seed(seed + 1)

    # Pre-allocated pinned scratch buffers. x_in[idx] is advanced-indexing
    # into a pinned tensor, which produces an UNPINNED copy — so the
    # subsequent .to(device, non_blocking=True) silently blocks. Copying
    # into a pre-pinned scratch first restores async H2D.
    if device != "cpu":
        scratch_in = torch.empty(batch_size, x_in.shape[-1], dtype=x_in.dtype, pin_memory=True)
        scratch_out = torch.empty(batch_size, x_out.shape[-1], dtype=x_out.dtype, pin_memory=True)
    else:
        scratch_in = scratch_out = None

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
        if scratch_in is not None:
            torch.index_select(x_in, 0, idx, out=scratch_in)
            torch.index_select(x_out, 0, idx, out=scratch_out)
            batch_in = scratch_in.unsqueeze(0).to(device, non_blocking=True)
            batch_out = scratch_out.unsqueeze(0).to(device, non_blocking=True)
        else:
            batch_in = x_in[idx].unsqueeze(0)
            batch_out = x_out[idx].unsqueeze(0)

        optim.zero_grad()
        pred = model(batch_in)
        loss = F.mse_loss(pred, batch_out)
        loss.backward()
        optim.step()

        # Tau annealing for V2 Gumbel-softmax cleanup.
        if version == 2:
            frac = step / max(steps - 1, 1)
            tau = tau_init + (tau_final - tau_init) * frac
            model.set_tau(tau)

        train_loss_history.append(loss.item())
        if initial_loss is None:
            initial_loss = loss.item()

        if step % log_every == 0 or step == steps - 1:
            if version == 2:
                logger.info("step %5d  train_mse=%.6f  tau=%.4f", step, loss.item(), tau)
            else:
                logger.info("step %5d  train_mse=%.6f", step, loss.item())

        if val_every > 0 and (step % val_every == 0 or step == steps - 1):
            model.eval()
            with torch.no_grad():
                v_in_full = x_in[val_idx]
                v_out_full = x_out[val_idx]
                # Batched val to bound peak VRAM at production scale
                # (n_val × d_in × top_k × n_slots intermediates compound).
                acc, n_seen = 0.0, 0
                for vb_in, vb_out in zip(
                    v_in_full.split(batch_size),
                    v_out_full.split(batch_size),
                ):
                    vb_in = vb_in.unsqueeze(0).to(device, non_blocking=True)
                    vb_out = vb_out.unsqueeze(0).to(device, non_blocking=True)
                    vp = model(vb_in)
                    acc += F.mse_loss(vp, vb_out, reduction="sum").item()
                    n_seen += vb_in.numel()
                v_loss = acc / max(n_seen, 1)
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
        "version": str(version),
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
    if version == 2:
        metadata["n_heads"] = str(n_heads)
        metadata["tau_init"] = str(tau_init)
        metadata["tau_final"] = str(tau_final)
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


def load_layer(path: Path) -> "torch.nn.Module":
    """Inverse of the save path: reconstruct CubeMemoryLayer (V1) or
    CubeMemoryLayerV2 from a safetensors checkpoint, recombining
    `.real`/`.imag` keys back into complex codebook tensors and reading
    hyperparams from metadata.

    The checkpoint version is determined by the ``version`` metadata key
    (defaults to ``"1"`` for legacy checkpoints).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from safetensors.torch import load_file, safe_open

    with safe_open(str(path), framework="pt") as f:
        meta = dict(f.metadata() or {})
    raw = load_file(str(path))

    # Recombine .real/.imag pairs into complex tensors.
    merged: Dict[str, torch.Tensor] = {}
    bases = {k[: -len(".real")] for k in raw if k.endswith(".real")}
    for b in bases:
        merged[b] = torch.complex(raw[f"{b}.real"], raw[f"{b}.imag"])
    for k, v in raw.items():
        if not (k.endswith(".real") or k.endswith(".imag")):
            merged[k] = v

    version = int(meta.get("version", "1"))

    if version == 2:
        from cube_memory_layer_v2 import CubeMemoryLayerV2
        model = CubeMemoryLayerV2(
            d_in=int(meta["d_in"]),
            d_codebook=int(meta["d_codebook"]),
            d_value=int(meta["d_value"]),
            m=int(meta["m"]),
            p=int(meta["p"]),
            n_slots=int(meta["n_slots"]),
            top_k=int(meta["top_k"]),
            seed=int(meta["seed"]),
            n_heads=int(meta.get("n_heads", "4")),
            tau_init=float(meta.get("tau_init", "1.0")),
        )
    else:
        from cube_memory_layer import CubeMemoryLayer
        model = CubeMemoryLayer(
            d_in=int(meta["d_in"]),
            d_codebook=int(meta["d_codebook"]),
            d_value=int(meta["d_value"]),
            m=int(meta["m"]),
            p=int(meta["p"]),
            n_slots=int(meta["n_slots"]),
            top_k=int(meta["top_k"]),
            seed=int(meta["seed"]),
        )

    model.load_state_dict(merged, strict=True)
    return model


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
    ap.add_argument("--d-value", type=int, default=None,
                    help="Slot value dim. Default: 2048 for v1, d_in for v2.")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--val-split", type=float, default=0.05)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--device", default="cpu",
                    help="cpu or cuda; default cpu (iGPU may be busy).")
    # V2 support
    ap.add_argument("--version", type=int, default=2, choices=[1, 2],
                    help="Layer version: 1=CubeMemoryLayer, 2=CubeMemoryLayerV2 (default).")
    ap.add_argument("--n-heads", type=int, default=4,
                    help="Number of retrieval heads (V2 only).")
    ap.add_argument("--tau-init", type=float, default=1.0,
                    help="Initial Gumbel-softmax temperature (V2 only).")
    ap.add_argument("--tau-final", type=float, default=0.1,
                    help="Final Gumbel-softmax temperature after annealing (V2 only).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.d_value is None:
        args.d_value = args.d_in if args.version == 2 else 2048

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
        version=args.version,
        n_heads=args.n_heads,
        tau_init=args.tau_init,
        tau_final=args.tau_final,
    )


if __name__ == "__main__":
    main()
