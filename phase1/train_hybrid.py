#!/usr/bin/env python3
"""Staged hybrid trainer: low-rank linear + cube memory on residual.

Stage 1: Compute optimal rank-r linear via truncated SVD (closed-form).
Stage 2: Freeze linear. Train cube memory to capture the non-linear residual.
Stage 3: Unfreeze all. Joint fine-tune with small LR.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

logger = logging.getLogger(__name__)


def load_data(activations_dir: Path, layer: int, d_in: int, val_split: float, seed: int):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from per_layer_trainer import load_layer_pairs

    x_in, x_out = load_layer_pairs(activations_dir, layer, d_in)
    n = x_in.shape[0]
    n_val = max(1, int(n * val_split))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    logger.info("split: %d train / %d val", train_idx.numel(), val_idx.numel())
    return x_in, x_out, train_idx, val_idx


def compute_val_mse(model, x_in, x_out, val_idx, device, batch_size=256):
    model.eval()
    acc, n_seen = 0.0, 0
    with torch.no_grad():
        for start in range(0, val_idx.numel(), batch_size):
            idx = val_idx[start:start + batch_size]
            b_in = x_in[idx].unsqueeze(0).to(device)
            b_out = x_out[idx].unsqueeze(0).to(device)
            pred = model(b_in)
            acc += F.mse_loss(pred, b_out, reduction="sum").item()
            n_seen += b_in.numel()
    model.train()
    return acc / max(n_seen, 1)


def compute_zero_baseline_mse(x_out, val_idx):
    vals = x_out[val_idx]
    return (vals ** 2).mean().item()


def variance_captured(val_mse, zero_mse):
    return (1.0 - val_mse / zero_mse) * 100.0


def fit_linear_svd(x_in, x_out, train_idx, rank, device):
    """Compute optimal rank-r linear map via truncated SVD of cross-covariance.

    The optimal rank-r linear predictor Y_hat = X @ W_r minimizes ||Y - X @ W_r||^2.
    W_r = V_x @ diag(1/S_x) @ U_x^T @ Y, then truncate W_r to rank r via its own SVD.

    For efficiency with 50K samples × 5120 dims, we work with X^T Y directly.
    """
    logger.info("computing SVD-based rank-%d linear fit...", rank)
    t0 = time.time()

    X = x_in[train_idx].to(torch.float64)
    Y = x_out[train_idx].to(torch.float64)

    XtX = X.T @ X  # (d, d)
    XtY = X.T @ Y  # (d, d)

    # Regularized solve: W = (X^T X + λI)^{-1} X^T Y
    lam = 1e-6 * XtX.diagonal().mean()
    XtX.diagonal().add_(lam)
    W_full = torch.linalg.solve(XtX, XtY)  # (d_in, d_in)

    # Truncated SVD of W_full to rank r
    U, S, Vt = torch.linalg.svd(W_full, full_matrices=False)
    U_r = U[:, :rank].to(torch.float32)      # (d_in, r)
    S_r = S[:rank].to(torch.float32)           # (r,)
    Vt_r = Vt[:rank, :].to(torch.float32)     # (r, d_in)

    # nn.Linear(d_in, r).weight is (r, d_in), computes x @ W.T
    # nn.Linear(r, d_in).weight is (d_in, r), computes z @ W.T
    # Composition: x @ W_up.T @ W_down.T = x @ U_r @ diag(S_r) @ Vt_r
    # So W_up.T = U_r @ diag(sqrt(S)), W_down.T = diag(sqrt(S)) @ Vt_r
    sqrt_S = S_r.sqrt()
    W_up = (sqrt_S.unsqueeze(1) * U_r.T)      # (r, d_in)
    W_down = (Vt_r.T * sqrt_S.unsqueeze(0))   # (d_in, r)

    elapsed = time.time() - t0
    logger.info("SVD done in %.1fs, top singular values: %s", elapsed,
                S[:min(10, rank)].tolist())

    return W_up, W_down


def train_loop(model, params, x_in, x_out, train_idx, val_idx,
               zero_mse, steps, batch_size, lr, device, stage_name,
               log_every=100, val_every=500):
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    rng = torch.Generator(device="cpu").manual_seed(42)

    if device != "cpu":
        scratch_in = torch.empty(batch_size, x_in.shape[-1], dtype=x_in.dtype, pin_memory=True)
        scratch_out = torch.empty(batch_size, x_out.shape[-1], dtype=x_out.dtype, pin_memory=True)
    else:
        scratch_in = scratch_out = None

    t0 = time.time()
    best_val = float("inf")

    for step in range(steps):
        idx = train_idx[torch.randint(0, train_idx.numel(), (batch_size,), generator=rng)]

        if scratch_in is not None:
            torch.index_select(x_in, 0, idx, out=scratch_in)
            torch.index_select(x_out, 0, idx, out=scratch_out)
            b_in = scratch_in.unsqueeze(0).to(device, non_blocking=True)
            b_out = scratch_out.unsqueeze(0).to(device, non_blocking=True)
        else:
            b_in = x_in[idx].unsqueeze(0)
            b_out = x_out[idx].unsqueeze(0)

        optim.zero_grad()
        pred = model(b_in)
        loss = F.mse_loss(pred, b_out)
        loss.backward()
        optim.step()

        if step % log_every == 0 or step == steps - 1:
            logger.info("[%s] step %5d  train_mse=%.6f", stage_name, step, loss.item())

        if val_every > 0 and (step % val_every == 0 or step == steps - 1):
            v_mse = compute_val_mse(model, x_in, x_out, val_idx, device, batch_size)
            vc = variance_captured(v_mse, zero_mse)
            best_val = min(best_val, v_mse)
            logger.info("[%s] step %5d  val_mse=%.6f  var_captured=%.1f%%", stage_name, step, v_mse, vc)

    elapsed = time.time() - t0
    logger.info("[%s] done: %d steps in %.1fs", stage_name, steps, elapsed)
    return best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations-dir", type=Path, default=Path.home() / "cube-memory-cache" / "activations")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--output", type=Path, default=Path.home() / "cube-memory-cache" / "trained-layers" / "layer_3_v3.safetensors")
    ap.add_argument("--d-in", type=int, default=5120)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--d-codebook", type=int, default=256)
    ap.add_argument("--d-value", type=int, default=512)
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--n-slots", type=int, default=4096)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-split", type=float, default=0.05)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--stage2-steps", type=int, default=5000)
    ap.add_argument("--stage2-lr", type=float, default=1e-3)
    ap.add_argument("--stage3-steps", type=int, default=2000)
    ap.add_argument("--stage3-lr", type=float, default=1e-4)
    ap.add_argument("--skip-stage3", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cube_memory_layer_v3 import HybridCubeMemoryLayer

    torch.manual_seed(args.seed)

    x_in, x_out, train_idx, val_idx = load_data(
        args.activations_dir, args.layer, args.d_in, args.val_split, args.seed)

    zero_mse = compute_zero_baseline_mse(x_out, val_idx)
    logger.info("zero baseline val_mse=%.6f", zero_mse)

    model = HybridCubeMemoryLayer(
        d_in=args.d_in, rank=args.rank, d_codebook=args.d_codebook,
        d_value=args.d_value, m=args.m, p=args.p, n_slots=args.n_slots,
        top_k=args.top_k, seed=args.seed,
    ).to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    linear_params = sum(p.numel() for p in model.linear_params())
    cube_params = sum(p.numel() for p in model.cube_params())
    logger.info("params: total=%d, linear=%d (rank-%d), cube=%d",
                total_params, linear_params, args.rank, cube_params)

    # --- Stage 1: SVD-based linear initialization (closed-form) ---
    logger.info("=== STAGE 1: SVD rank-%d linear (closed-form) ===", args.rank)
    W_up, W_down = fit_linear_svd(x_in, x_out, train_idx, args.rank, args.device)

    with torch.no_grad():
        model.linear_up.weight.copy_(W_up)
        model.linear_down.weight.copy_(W_down)

    s1_val = compute_val_mse(model, x_in, x_out, val_idx, args.device, args.batch_size)
    s1_vc = variance_captured(s1_val, zero_mse)
    logger.info("Stage 1 final: val_mse=%.6f, var_captured=%.1f%%", s1_val, s1_vc)

    x_in_pinned = x_in.cpu().pin_memory() if args.device != "cpu" else x_in
    x_out_pinned = x_out.cpu().pin_memory() if args.device != "cpu" else x_out

    # --- Stage 2: Freeze linear, train cube on residual ---
    for p in model.linear_params():
        p.requires_grad_(False)
    for p in model.cube_params():
        p.requires_grad_(True)

    logger.info("=== STAGE 2: Cube memory on residual ===")
    train_loop(model, model.cube_params(), x_in_pinned, x_out_pinned, train_idx, val_idx,
               zero_mse, args.stage2_steps, args.batch_size, args.stage2_lr,
               args.device, "S2-cube-residual")

    s2_val = compute_val_mse(model, x_in, x_out, val_idx, args.device, args.batch_size)
    s2_vc = variance_captured(s2_val, zero_mse)
    cube_contribution = s2_vc - s1_vc
    logger.info("Stage 2 final: val_mse=%.6f, var_captured=%.1f%%", s2_val, s2_vc)
    logger.info("Cube memory contribution on residual: +%.1f%% variance", cube_contribution)

    # --- Stage 3: Joint fine-tune ---
    s3_val = s2_val
    s3_vc = s2_vc
    if not args.skip_stage3:
        for p in model.parameters():
            p.requires_grad_(True)

        logger.info("=== STAGE 3: Joint fine-tune ===")
        train_loop(model, list(model.parameters()), x_in_pinned, x_out_pinned, train_idx, val_idx,
                   zero_mse, args.stage3_steps, args.batch_size, args.stage3_lr,
                   args.device, "S3-joint")

        s3_val = compute_val_mse(model, x_in, x_out, val_idx, args.device, args.batch_size)
        s3_vc = variance_captured(s3_val, zero_mse)

    # --- Save ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    for k, v in model.state_dict().items():
        if torch.is_complex(v):
            state[f"{k}.real"] = v.real.contiguous()
            state[f"{k}.imag"] = v.imag.contiguous()
        else:
            state[k] = v.contiguous()

    final_val = s3_val
    final_vc = s3_vc

    metadata = {
        "version": "3",
        "layer": str(args.layer),
        "d_in": str(args.d_in),
        "rank": str(args.rank),
        "d_codebook": str(args.d_codebook),
        "d_value": str(args.d_value),
        "m": str(args.m),
        "p": str(args.p),
        "n_slots": str(args.n_slots),
        "top_k": str(args.top_k),
        "seed": str(args.seed),
        "zero_baseline_mse": f"{zero_mse:.6e}",
        "final_val_mse": f"{final_val:.6e}",
        "variance_captured_pct": f"{final_vc:.1f}",
        "s1_linear_var_pct": f"{s1_vc:.1f}",
        "cube_contribution_pct": f"{cube_contribution:.1f}",
    }
    save_file(state, str(args.output), metadata=metadata)
    logger.info("saved %s", args.output)

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("  Zero baseline:     val_mse=%.6f", zero_mse)
    logger.info("  Stage 1 (SVD):     val_mse=%.6f  var=%.1f%%  (%d params)", s1_val, s1_vc, linear_params)
    logger.info("  Stage 2 (+cube):   val_mse=%.6f  var=%.1f%%  (+%.1f%% from cube)", s2_val, s2_vc, cube_contribution)
    if not args.skip_stage3:
        logger.info("  Stage 3 (joint):   val_mse=%.6f  var=%.1f%%", s3_val, s3_vc)
    logger.info("  Total params:      %d", total_params)
    logger.info("=" * 60)

    if final_vc >= 30.0:
        logger.info("GATE PASSED: %.1f%% >= 30%% — proceed to multi-layer", final_vc)
    elif final_vc >= 10.0:
        logger.info("GATE MARGINAL: %.1f%% — consider rank or slot scaling", final_vc)
    else:
        logger.info("GATE FAILED: %.1f%% < 10%% — rethink architecture", final_vc)


if __name__ == "__main__":
    main()
