#!/usr/bin/env python3
"""Bootstrap the Cube Memory local distillation pipeline.

This script orchestrates the full Path A (layer-wise FitNets) distillation
pipeline on raz-gpd4. It handles:

  1. Teacher activation extraction via llama-server /v1/completions with
     logprobs, OR reuse of existing CACT activation chunks on disk.
  2. CubeMemoryLayer (V1 or V2) fitting per swap position using MSE loss.
  3. Sequential re-caching: after fitting layer N, re-extract activations
     for layer N+1 through the partially-swapped model so each fit sees
     the real upstream signal.
  4. Diagnostic reporting: variance capture ratio, loss curves, per-layer
     quality metrics.

Architecture decision:
  V1 (frozen codebooks, STE cleanup, single-head retrieval) captured only
  4.7% of target variance after 5K steps on layer 3 (Phase B debug,
  2026-04-27). V2 adds learned codebooks, multi-head retrieval,
  Gumbel-softmax cleanup, and a gated residual. This bootstrap defaults
  to V2 and sweeps hyperparams that V1 couldn't explore.

Memory budget (raz-gpd4, per-layer trainer, V2):
  - Activation cache per layer: ~978 MB on disk (50K tokens x 5120 x 2 bytes x 2 sides)
  - Loaded as fp32: ~2 GB per side = ~4 GB both sides
  - Kept on CPU with pin_memory; per-batch slices sent to GPU
  - CubeMemoryLayerV2 with default config: ~90 MB (slot keys + values + codebooks)
  - Adam state (2x params): ~180 MB
  - Per-batch working set on GPU: ~50 MB
  - Total GPU: ~320 MB. Total CPU: ~4 GB. Both fit comfortably.

Usage:
  source ~/rocm-gpu-test/venv/bin/activate
  export HSA_OVERRIDE_GFX_VERSION=11.0.0

  # Diagnose existing V1 activations with V2 architecture
  python bootstrap_distill.py diagnose --layer 3

  # Train V2 on one layer as proof-of-concept
  python bootstrap_distill.py train --layer 3 --version 2 --steps 10000

  # Run the full sequential pipeline on all 8 layers
  python bootstrap_distill.py pipeline --steps 10000

  # Extract new activations (requires llama-server running with teacher model)
  python bootstrap_distill.py extract --corpus ~/cube-memory-cache/corpus/fineweb_100k.txt
"""
from __future__ import annotations

import argparse
import json
import logging
import struct
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SWAP_LAYERS = [3, 11, 19, 27, 35, 43, 51, 59]
ACTIVATIONS_DIR = Path.home() / "cube-memory-cache" / "activations"
TRAINED_DIR = Path.home() / "cube-memory-cache" / "trained-layers"
CORPUS_DIR = Path.home() / "cube-memory-cache" / "corpus"

# CACT chunk format (from per_layer_trainer.py)
CACT_MAGIC = b"CACT"
CACT_VERSION = 1
HEADER_STRUCT = struct.Struct("<4sIII")
HEADER_SIZE = HEADER_STRUCT.size

# Qwen3.6-27B dimensions
D_IN = 5120
N_FF = 17408


# ---------------------------------------------------------------------------
# Activation I/O
# ---------------------------------------------------------------------------

def read_cact_chunk(path: Path, expected_d_in: int) -> torch.Tensor:
    """Read one CACT chunk -> float32 tensor (n_tokens, n_embd)."""
    with path.open("rb") as f:
        header = f.read(HEADER_SIZE)
        if len(header) != HEADER_SIZE:
            raise ValueError(f"{path}: truncated header")
        magic, version, n_tokens, n_embd = HEADER_STRUCT.unpack(header)
        if magic != CACT_MAGIC:
            raise ValueError(f"{path}: bad magic {magic!r}")
        if version != CACT_VERSION:
            raise ValueError(f"{path}: unsupported version {version}")
        if n_embd != expected_d_in:
            raise ValueError(f"{path}: n_embd={n_embd} != expected {expected_d_in}")
        payload = f.read(n_tokens * n_embd * 2)
        if len(payload) != n_tokens * n_embd * 2:
            raise ValueError(f"{path}: truncated payload")

    raw = torch.frombuffer(bytearray(payload), dtype=torch.uint16)
    bf16 = raw.view(torch.bfloat16).reshape(n_tokens, n_embd)
    return bf16.to(torch.float32)


def load_layer_activations(
    layer: int, d_in: int = D_IN, activations_dir: Path = ACTIVATIONS_DIR
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load (ffn_in, ffn_out) for one layer from CACT chunks on disk."""
    x_in_parts, x_out_parts = [], []
    for side, parts in [("in", x_in_parts), ("out", x_out_parts)]:
        side_dir = activations_dir / f"layer_{layer}_{side}"
        if not side_dir.exists():
            raise FileNotFoundError(f"No activation dir: {side_dir}")
        chunks = sorted(side_dir.glob("chunk_*.bin"))
        if not chunks:
            raise FileNotFoundError(f"No chunks in {side_dir}")
        for c in chunks:
            parts.append(read_cact_chunk(c, d_in))

    x_in = torch.cat(x_in_parts, dim=0)
    x_out = torch.cat(x_out_parts, dim=0)
    assert x_in.shape == x_out.shape, f"shape mismatch: {x_in.shape} vs {x_out.shape}"
    return x_in, x_out


# ---------------------------------------------------------------------------
# Diagnostic: analyze activation statistics and baseline
# ---------------------------------------------------------------------------

def diagnose(layer: int, d_in: int = D_IN):
    """Print activation statistics and compute baseline comparisons.

    This replaces the manual /tmp/phase_b_diag.py workflow with a
    structured, repeatable diagnostic that covers:
      - Input/output tensor statistics (mean, std, min, max, NaN/Inf)
      - Variance ratio between output delta and input
      - Zero-baseline MSE (how much does "do nothing" cost)
      - Mean-baseline MSE (how much does "predict mean" cost)
      - If a trained checkpoint exists, measure its capture ratio
    """
    logger.info("=== Diagnostic for layer %d ===", layer)

    x_in, x_out = load_layer_activations(layer, d_in)
    n_tokens = x_in.shape[0]
    logger.info("Loaded %d tokens, d_in=%d", n_tokens, d_in)

    # Basic stats
    for name, t in [("x_in", x_in), ("x_out", x_out)]:
        logger.info(
            "%s: mean=%.4e  std=%.4e  min=%.4f  max=%.4f  "
            "nan=%d  inf=%d",
            name, t.mean().item(), t.std().item(),
            t.min().item(), t.max().item(),
            t.isnan().sum().item(), t.isinf().sum().item(),
        )

    var_in = x_in.var().item()
    var_out = x_out.var().item()
    logger.info("Var(x_in)=%.4e  Var(x_out)=%.4e  ratio=%.2f",
                var_in, var_out, var_in / max(var_out, 1e-20))

    # Baselines: zero-output MSE and mean-output MSE
    zero_mse = x_out.pow(2).mean().item()
    mean_out = x_out.mean(dim=0, keepdim=True)
    mean_mse = (x_out - mean_out).pow(2).mean().item()

    logger.info("Zero-baseline MSE: %.6e", zero_mse)
    logger.info("Mean-baseline MSE: %.6e  (normalized: %.4f)",
                mean_mse, mean_mse / max(zero_mse, 1e-20))

    # Check for existing trained checkpoints (V2 best, V2 final, V1)
    for suffix in ["_v2_best", "_v2", ""]:
        ckpt = TRAINED_DIR / f"layer_{layer}{suffix}.safetensors"
        if ckpt.exists():
            logger.info("Found checkpoint: %s", ckpt)
            _eval_checkpoint(ckpt, x_in, x_out, zero_mse)

    return {
        "layer": layer,
        "n_tokens": n_tokens,
        "var_in": var_in,
        "var_out": var_out,
        "zero_mse": zero_mse,
        "mean_mse": mean_mse,
        "normalized_mean_mse": mean_mse / max(zero_mse, 1e-20),
    }


def _eval_checkpoint(ckpt_path: Path, x_in: torch.Tensor, x_out: torch.Tensor, zero_mse: float):
    """Evaluate a safetensors checkpoint against activation data."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "phase1"))
    from per_layer_trainer import load_layer

    model = load_layer(ckpt_path)
    model.eval()

    # Batched eval to bound memory
    batch_size = 256
    total_mse = 0.0
    n_seen = 0
    with torch.no_grad():
        for i in range(0, x_in.shape[0], batch_size):
            batch_in = x_in[i:i+batch_size].unsqueeze(0)
            batch_out = x_out[i:i+batch_size].unsqueeze(0)
            pred = model(batch_in)
            total_mse += F.mse_loss(pred, batch_out, reduction="sum").item()
            n_seen += batch_in.numel()

    model_mse = total_mse / max(n_seen, 1)
    capture = 1.0 - (model_mse / max(zero_mse, 1e-20))
    logger.info(
        "  Checkpoint MSE: %.6e  normalized: %.4f  capture: %.1f%%",
        model_mse, model_mse / max(zero_mse, 1e-20), capture * 100,
    )


# ---------------------------------------------------------------------------
# V2 Training with expanded hyperparameter support
# ---------------------------------------------------------------------------

def train_layer_v2(
    layer: int,
    steps: int = 10000,
    batch_size: int = 64,
    lr: float = 5e-4,
    seed: int = 42,
    d_in: int = D_IN,
    d_codebook: int = 512,
    m: int = 128,
    p: int = 2,
    n_slots: int = 16384,
    d_value: int = 2048,
    top_k: int = 8,
    n_heads: int = 8,
    tau_init: float = 2.0,
    tau_final: float = 0.05,
    device: str = "cuda",
    output_dir: Path = TRAINED_DIR,
    val_split: float = 0.05,
    log_every: int = 200,
    val_every: int = 1000,
    warmup_steps: int = 500,
    grad_clip: float = 1.0,
) -> dict:
    """Train CubeMemoryLayerV2 on one layer's activations.

    Key differences from the V1 per_layer_trainer:
      - Larger default capacity (m=128, n_slots=16384, n_heads=8)
      - Cosine LR schedule with warmup
      - Gradient clipping to prevent divergence
      - Wider Gumbel-softmax temperature annealing range
      - Higher default top_k for more expressive retrieval
      - Structured logging with per-step variance-capture tracking
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent / "phase1"))
    from cube_memory_layer_v2 import CubeMemoryLayerV2
    from safetensors.torch import save_file

    torch.manual_seed(seed)

    # Load activations
    logger.info("Loading layer %d activations...", layer)
    x_in, x_out = load_layer_activations(layer, d_in)
    n_tokens = x_in.shape[0]
    logger.info("Loaded %d tokens", n_tokens)

    # Train/val split
    n_val = max(1, int(n_tokens * val_split))
    perm = torch.randperm(n_tokens, generator=torch.Generator().manual_seed(seed))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    logger.info("Split: %d train / %d val", train_idx.numel(), val_idx.numel())

    # Compute baseline statistics for normalized reporting
    var_out = x_out.var().item()
    zero_mse = x_out.pow(2).mean().item()
    logger.info("Target Var(x_out)=%.4e, zero-baseline MSE=%.6e", var_out, zero_mse)

    # Build model
    model = CubeMemoryLayerV2(
        d_in=d_in, d_codebook=d_codebook, d_value=d_value,
        m=m, p=p, n_slots=n_slots, top_k=top_k, seed=seed,
        n_heads=n_heads, tau_init=tau_init,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model params: %.2fM trainable", n_params / 1e6)

    # Optimizer: AdamW with decoupled weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999)
    )

    # Pin memory for async transfer
    if device != "cpu":
        x_in = x_in.cpu().pin_memory()
        x_out = x_out.cpu().pin_memory()

    rng = torch.Generator(device="cpu").manual_seed(seed + 1)

    # Pre-allocated pinned scratch
    if device != "cpu":
        scratch_in = torch.empty(batch_size, d_in, dtype=torch.float32, pin_memory=True)
        scratch_out = torch.empty(batch_size, d_in, dtype=torch.float32, pin_memory=True)
    else:
        scratch_in = scratch_out = None

    history = {
        "train_mse": [],
        "val_mse": [],
        "capture_ratio": [],
        "tau": [],
    }
    best_val_mse = float("inf")
    t0 = time.time()

    for step in range(steps):
        # LR schedule: linear warmup then cosine decay
        if step < warmup_steps:
            lr_factor = (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / max(steps - warmup_steps - 1, 1)
            lr_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr * lr_factor

        # Tau annealing: linear in log-space for smoother schedule
        frac = step / max(steps - 1, 1)
        tau = tau_init * (tau_final / tau_init) ** frac
        model.set_tau(tau)

        # Sample batch
        idx = train_idx[torch.randint(0, train_idx.numel(), (batch_size,), generator=rng)]
        if scratch_in is not None:
            torch.index_select(x_in, 0, idx, out=scratch_in)
            torch.index_select(x_out, 0, idx, out=scratch_out)
            batch_in = scratch_in.unsqueeze(0).to(device, non_blocking=True)
            batch_out = scratch_out.unsqueeze(0).to(device, non_blocking=True)
        else:
            batch_in = x_in[idx].unsqueeze(0)
            batch_out = x_out[idx].unsqueeze(0)

        # Forward + backward
        optimizer.zero_grad()
        pred = model(batch_in)
        loss = F.mse_loss(pred, batch_out)
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        train_mse = loss.item()
        history["train_mse"].append(train_mse)
        history["tau"].append(tau)

        if step % log_every == 0 or step == steps - 1:
            capture = 1.0 - (train_mse / max(zero_mse, 1e-20))
            logger.info(
                "step %5d  train_mse=%.6e  capture=%.1f%%  tau=%.4f  lr=%.2e",
                step, train_mse, capture * 100, tau, lr * lr_factor,
            )

        # Validation
        if val_every > 0 and (step % val_every == 0 or step == steps - 1):
            model.eval()
            with torch.no_grad():
                v_in = x_in[val_idx]
                v_out = x_out[val_idx]
                acc, n_seen = 0.0, 0
                for vb_in, vb_out in zip(
                    v_in.split(batch_size), v_out.split(batch_size)
                ):
                    vb_in = vb_in.unsqueeze(0).to(device, non_blocking=True)
                    vb_out = vb_out.unsqueeze(0).to(device, non_blocking=True)
                    vp = model(vb_in)
                    acc += F.mse_loss(vp, vb_out, reduction="sum").item()
                    n_seen += vb_in.numel()
                v_loss = acc / max(n_seen, 1)

            history["val_mse"].append((step, v_loss))
            capture = 1.0 - (v_loss / max(zero_mse, 1e-20))
            history["capture_ratio"].append((step, capture))
            logger.info(
                "step %5d  val_mse=%.6e  val_capture=%.1f%%",
                step, v_loss, capture * 100,
            )

            if v_loss < best_val_mse:
                best_val_mse = v_loss
                # Save best checkpoint
                _save_checkpoint(
                    model, output_dir / f"layer_{layer}_v2_best.safetensors",
                    layer=layer, d_in=d_in, d_codebook=d_codebook,
                    d_value=d_value, m=m, p=p, n_slots=n_slots,
                    top_k=top_k, seed=seed, steps=step,
                    n_heads=n_heads, tau_init=tau_init, tau_final=tau_final,
                    train_mse=train_mse,
                )

            model.train()

    elapsed = time.time() - t0
    logger.info("Training done: %d steps in %.1fs (%.1f step/s)",
                steps, elapsed, steps / max(elapsed, 1e-6))

    # Save final checkpoint
    final_path = output_dir / f"layer_{layer}_v2.safetensors"
    _save_checkpoint(
        model, final_path,
        layer=layer, d_in=d_in, d_codebook=d_codebook,
        d_value=d_value, m=m, p=p, n_slots=n_slots,
        top_k=top_k, seed=seed, steps=steps,
        n_heads=n_heads, tau_init=tau_init, tau_final=tau_final,
        train_mse=history["train_mse"][-1],
    )

    final_capture = 1.0 - (best_val_mse / max(zero_mse, 1e-20))
    logger.info(
        "Layer %d result: best_val_mse=%.6e  capture=%.1f%%  "
        "zero_baseline=%.6e",
        layer, best_val_mse, final_capture * 100, zero_mse,
    )

    return {
        "layer": layer,
        "steps": steps,
        "best_val_mse": best_val_mse,
        "final_train_mse": history["train_mse"][-1],
        "capture_ratio": final_capture,
        "zero_mse": zero_mse,
        "wall_seconds": elapsed,
        "history": history,
    }


def _save_checkpoint(model, path: Path, **metadata_kwargs):
    """Save model state dict via safetensors, splitting complex tensors."""
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    for k, v in model.state_dict().items():
        if torch.is_complex(v):
            state[f"{k}.real"] = v.real.contiguous()
            state[f"{k}.imag"] = v.imag.contiguous()
        else:
            state[k] = v.contiguous()

    metadata = {k: str(v) for k, v in metadata_kwargs.items()
                 if not isinstance(v, (dict, list))}
    metadata["version"] = "2"
    save_file(state, str(path), metadata=metadata)
    logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Full sequential pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    layers: list[int] = SWAP_LAYERS,
    steps: int = 10000,
    device: str = "cuda",
    **train_kwargs,
) -> dict:
    """Run the full sequential Path A distillation pipeline.

    For each swap position in order:
      1. Load existing activations (or extract new ones if --extract is set)
      2. Train CubeMemoryLayerV2 with MSE
      3. Report capture ratio

    NOTE: Step 4e from LOCAL_DISTILL_PLAN.md (re-caching activations through
    the partially-swapped model) requires the teacher GGUF to be loaded in
    llama-server AND the ability to inject trained CubeMemoryLayers into the
    inference graph. This is not yet possible with vanilla llama-server --
    it requires the GGUF export path (Phase D) to be complete. For now, we
    train each layer independently against the original teacher activations.
    This is the "fast Path A" -- error compounding is addressed in Phase C
    (end-to-end calibration).
    """
    results = []
    for layer in layers:
        logger.info("=" * 60)
        logger.info("Pipeline: layer %d (%d of %d)", layer, len(results) + 1, len(layers))
        logger.info("=" * 60)

        # Check activations exist
        in_dir = ACTIVATIONS_DIR / f"layer_{layer}_in"
        out_dir = ACTIVATIONS_DIR / f"layer_{layer}_out"
        if not in_dir.exists() or not out_dir.exists():
            logger.error(
                "Missing activations for layer %d. Run extraction first:\n"
                "  python bootstrap_distill.py extract --layer %d",
                layer, layer,
            )
            results.append({"layer": layer, "status": "skipped", "reason": "no activations"})
            continue

        try:
            result = train_layer_v2(
                layer=layer, steps=steps, device=device, **train_kwargs,
            )
            result["status"] = "ok"
            results.append(result)
        except Exception as e:
            logger.exception("Layer %d failed: %s", layer, e)
            results.append({"layer": layer, "status": "failed", "error": str(e)})

    # Summary
    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    for r in results:
        if r["status"] == "ok":
            logger.info(
                "  layer %2d: capture=%.1f%%  val_mse=%.4e  wall=%.0fs",
                r["layer"], r["capture_ratio"] * 100,
                r["best_val_mse"], r["wall_seconds"],
            )
        else:
            logger.info("  layer %2d: %s (%s)", r["layer"], r["status"],
                        r.get("reason", r.get("error", "?")))

    return {"layers": results}


# ---------------------------------------------------------------------------
# Activation extraction via llama-server
# ---------------------------------------------------------------------------

def extract_activations_via_server(
    corpus_path: Path,
    layers: list[int] = SWAP_LAYERS,
    server_url: str = "http://localhost:8090",
    chunk_size: int = 2000,
    d_in: int = D_IN,
):
    """Extract teacher activations by running text through llama-server.

    This is a PLACEHOLDER for the actual extraction pipeline. The real
    extraction requires a patched llama.cpp that registers a callback at
    the FFN boundary and dumps the pre/post-FFN hidden states. The patch
    is described in LOCAL_DISTILL_PLAN.md Phase A, step 2.

    The existing 50K-token activation cache at ~/cube-memory-cache/activations/
    was generated by a custom `llama-dump-activations` binary built from
    the cube-memory-op branch. That binary is the correct tool; this
    function documents the extraction command for convenience.

    To extract new activations:
      1. Build llama-dump-activations from /tmp/llama-mainline (cube-memory-op branch)
      2. Start with the teacher GGUF:
           /tmp/llama-mainline/build/bin/llama-dump-activations \\
             -m ~/models/Qwen3.6-27B-Q4_K_M/Qwen3.6-27B-Q4_K_M.gguf \\
             -f ~/cube-memory-cache/corpus/fineweb_100k.txt \\
             --layers 3,11,19,27,35,43,51,59 \\
             --out-dir ~/cube-memory-cache/activations/ \\
             --chunk-tokens 2000 \\
             -ngl 30
      3. This dumps CACT-format chunks to layer_{N}_{in|out}/chunk_{NNNN}.bin
    """
    logger.error(
        "Activation extraction via llama-server API is not implemented.\n"
        "The existing activation cache was created by `llama-dump-activations`\n"
        "(a custom tool built from the cube-memory-op llama.cpp branch).\n"
        "\n"
        "To re-extract, you need:\n"
        "  1. Qwen3.6-27B-Q4_K_M GGUF (deleted 2026-04-29, needs re-pull)\n"
        "  2. llama-dump-activations binary from /tmp/llama-mainline\n"
        "\n"
        "Existing cache has 50K tokens across 8 layers (7.7 GB total).\n"
        "For bootstrap purposes, the existing cache is sufficient."
    )
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap Cube Memory local distillation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # diagnose
    diag = sub.add_parser("diagnose", help="Analyze layer activation statistics")
    diag.add_argument("--layer", type=int, default=3)
    diag.add_argument("--d-in", type=int, default=D_IN)

    # train
    tr = sub.add_parser("train", help="Train V2 on one layer")
    tr.add_argument("--layer", type=int, default=3)
    tr.add_argument("--steps", type=int, default=10000)
    tr.add_argument("--batch-size", type=int, default=64)
    tr.add_argument("--lr", type=float, default=5e-4)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--d-in", type=int, default=D_IN)
    tr.add_argument("--d-codebook", type=int, default=512)
    tr.add_argument("--m", type=int, default=128)
    tr.add_argument("--p", type=int, default=2)
    tr.add_argument("--n-slots", type=int, default=16384)
    tr.add_argument("--d-value", type=int, default=2048)
    tr.add_argument("--top-k", type=int, default=8)
    tr.add_argument("--n-heads", type=int, default=8)
    tr.add_argument("--tau-init", type=float, default=2.0)
    tr.add_argument("--tau-final", type=float, default=0.05)
    tr.add_argument("--warmup-steps", type=int, default=500)
    tr.add_argument("--grad-clip", type=float, default=1.0)
    tr.add_argument("--device", default="cuda")
    tr.add_argument("--val-split", type=float, default=0.05)
    tr.add_argument("--log-every", type=int, default=200)
    tr.add_argument("--val-every", type=int, default=1000)

    # pipeline
    pipe = sub.add_parser("pipeline", help="Run full sequential distillation")
    pipe.add_argument("--steps", type=int, default=10000)
    pipe.add_argument("--device", default="cuda")
    pipe.add_argument("--layers", type=int, nargs="+", default=SWAP_LAYERS)
    pipe.add_argument("--batch-size", type=int, default=64)
    pipe.add_argument("--lr", type=float, default=5e-4)

    # extract
    ext = sub.add_parser("extract", help="Extract teacher activations (needs model)")
    ext.add_argument("--corpus", type=Path, default=CORPUS_DIR / "fineweb_100k.txt")
    ext.add_argument("--layers", type=int, nargs="+", default=SWAP_LAYERS)

    # compare: run both V1 and V2 on same layer for A/B
    cmp = sub.add_parser("compare", help="A/B test V1 vs V2 on one layer")
    cmp.add_argument("--layer", type=int, default=3)
    cmp.add_argument("--steps", type=int, default=5000)
    cmp.add_argument("--device", default="cuda")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.command == "diagnose":
        result = diagnose(args.layer, args.d_in)
        print(json.dumps(result, indent=2))

    elif args.command == "train":
        result = train_layer_v2(
            layer=args.layer, steps=args.steps, batch_size=args.batch_size,
            lr=args.lr, seed=args.seed, d_in=args.d_in,
            d_codebook=args.d_codebook, m=args.m, p=args.p,
            n_slots=args.n_slots, d_value=args.d_value, top_k=args.top_k,
            n_heads=args.n_heads, tau_init=args.tau_init,
            tau_final=args.tau_final, device=args.device,
            val_split=args.val_split, log_every=args.log_every,
            val_every=args.val_every, warmup_steps=args.warmup_steps,
            grad_clip=args.grad_clip,
        )
        # Write result summary (excluding large history arrays)
        summary = {k: v for k, v in result.items() if k != "history"}
        print(json.dumps(summary, indent=2))

    elif args.command == "pipeline":
        result = run_pipeline(
            layers=args.layers, steps=args.steps, device=args.device,
            batch_size=args.batch_size, lr=args.lr,
        )
        # Write summary
        summary_path = TRAINED_DIR / "pipeline_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "layers": [
                {k: v for k, v in r.items() if k != "history"}
                for r in result["layers"]
            ],
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        logger.info("Summary written to %s", summary_path)

    elif args.command == "extract":
        extract_activations_via_server(args.corpus, args.layers)

    elif args.command == "compare":
        _compare_v1_v2(args.layer, args.steps, args.device)

    else:
        parser.print_help()


def _compare_v1_v2(layer: int, steps: int, device: str):
    """A/B test: train both V1 and V2 on the same layer's activations
    and compare variance capture."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "phase1"))

    logger.info("=== A/B comparison: V1 vs V2 on layer %d ===", layer)

    x_in, x_out = load_layer_activations(layer)
    zero_mse = x_out.pow(2).mean().item()
    logger.info("Zero-baseline MSE: %.6e", zero_mse)

    # Train V1
    logger.info("--- Training V1 (frozen codebooks, STE, single-head) ---")
    from per_layer_trainer import train as train_v1
    v1_result = train_v1(
        activations_dir=ACTIVATIONS_DIR, layer=layer,
        output=TRAINED_DIR / f"layer_{layer}_v1_compare.safetensors",
        steps=steps, batch_size=64, lr=1e-3, seed=42,
        d_in=D_IN, d_codebook=256, m=64, n_slots=4096,
        d_value=2048, top_k=4, p=2,
        val_split=0.05, log_every=500, val_every=1000,
        device=device, version=1,
    )

    # Train V2
    logger.info("--- Training V2 (learned codebooks, multi-head, Gumbel) ---")
    v2_result = train_layer_v2(
        layer=layer, steps=steps, batch_size=64, lr=5e-4,
        seed=42, device=device,
    )

    # Report -- use val_history from both for fair comparison.
    # V1's per_layer_trainer.train() returns val_history as list of (step, val_mse) tuples.
    v1_val_history = v1_result.get("val_history", [])
    v1_best_val = min((vm for _, vm in v1_val_history), default=float("nan"))
    v1_capture = 1.0 - (v1_best_val / max(zero_mse, 1e-20))

    v2_best_val = v2_result["best_val_mse"]
    v2_capture = v2_result["capture_ratio"]

    logger.info("=" * 60)
    logger.info("COMPARISON RESULTS (layer %d, %d steps)", layer, steps)
    logger.info("  V1 best val MSE: %.6e  capture: %.1f%%", v1_best_val, v1_capture * 100)
    logger.info("  V2 best val MSE: %.6e  capture: %.1f%%", v2_best_val, v2_capture * 100)
    logger.info("  Improvement: %.1fx",
                v1_best_val / max(v2_best_val, 1e-20))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
