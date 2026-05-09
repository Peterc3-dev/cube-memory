#!/usr/bin/env python3
"""Diagnostic: is the FFN residual (after linear) noise or learnable non-linearity?

Fits a 2-layer MLP on the residual after removing the rank-r linear component.
If the MLP captures significant additional variance, the non-linearity is real
and the VSA pipeline is the bottleneck. If even the MLP fails, the residual is noise.

Also runs the analysis on multiple layers to find which have the most non-linear structure.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def load_data(activations_dir, layer, d_in, val_split, seed):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from per_layer_trainer import load_layer_pairs
    x_in, x_out = load_layer_pairs(activations_dir, layer, d_in)
    n = x_in.shape[0]
    n_val = max(1, int(n * val_split))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    return x_in, x_out, perm[n_val:], perm[:n_val]


def fit_linear_svd(x_in, x_out, train_idx, rank):
    X = x_in[train_idx].to(torch.float64)
    Y = x_out[train_idx].to(torch.float64)
    XtX = X.T @ X
    XtY = X.T @ Y
    lam = 1e-6 * XtX.diagonal().mean()
    XtX.diagonal().add_(lam)
    W_full = torch.linalg.solve(XtX, XtY)
    U, S, Vt = torch.linalg.svd(W_full, full_matrices=False)
    W_r = (U[:, :rank] * S[:rank]) @ Vt[:rank, :]
    return W_r.to(torch.float32)


def compute_residual(x_in, x_out, W_r, idx):
    pred = x_in[idx] @ W_r
    return x_out[idx] - pred


class ResidualMLP(nn.Module):
    def __init__(self, d_in, hidden):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, d_in)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))


class ResidualMemoryLayer(nn.Module):
    """Plain learned-key memory layer (Meta Memory Layers style, no VSA)."""
    def __init__(self, d_in, n_slots=4096, d_key=256, top_k=8):
        super().__init__()
        self.key_proj = nn.Linear(d_in, d_key, bias=False)
        self.slot_keys = nn.Parameter(torch.empty(n_slots, d_key))
        self.slot_values = nn.Parameter(torch.empty(n_slots, d_in))
        self.top_k = top_k
        nn.init.normal_(self.key_proj.weight, std=0.02)
        nn.init.normal_(self.slot_keys, std=0.02)
        nn.init.normal_(self.slot_values, std=0.02)

    def forward(self, x):
        q = self.key_proj(x)
        sims = q @ self.slot_keys.T
        topk_sims, topk_idx = sims.topk(self.top_k, dim=-1)
        weights = topk_sims.softmax(dim=-1)
        gathered = self.slot_values[topk_idx]
        return (weights.unsqueeze(-1) * gathered).sum(dim=-2)


def train_model(model, x_in, residual_train, residual_val, train_idx, val_idx,
                steps, batch_size, lr, name):
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = torch.Generator(device="cpu").manual_seed(42)

    residual_var = (residual_val ** 2).mean().item()
    logger.info("[%s] residual_var=%.6f (this is 100%% of what we try to capture)", name, residual_var)

    for step in range(steps):
        idx = torch.randint(0, train_idx.numel(), (batch_size,), generator=rng)
        b_in = x_in[train_idx[idx]]
        b_tgt = residual_train[idx]

        optim.zero_grad()
        pred = model(b_in)
        loss = F.mse_loss(pred, b_tgt)
        loss.backward()
        optim.step()

        if step % 500 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                val_pred = []
                for s in range(0, val_idx.numel(), 256):
                    batch = x_in[val_idx[s:s+256]]
                    val_pred.append(model(batch))
                val_pred = torch.cat(val_pred)
                val_mse = F.mse_loss(val_pred, residual_val).item()
                vc_of_residual = max(0, (1 - val_mse / residual_var) * 100)
            logger.info("[%s] step %5d  train=%.6f  val=%.6f  residual_var_captured=%.1f%%",
                        name, step, loss.item(), val_mse, vc_of_residual)
            model.train()

    model.eval()
    with torch.no_grad():
        val_pred = []
        for s in range(0, val_idx.numel(), 256):
            batch = x_in[val_idx[s:s+256]]
            val_pred.append(model(batch))
        val_pred = torch.cat(val_pred)
        final_mse = F.mse_loss(val_pred, residual_val).item()
        final_vc = max(0, (1 - final_mse / residual_var) * 100)
    return final_mse, final_vc


def analyze_layer(activations_dir, layer, d_in, rank, val_split, seed, steps, batch_size):
    logger.info("=" * 50)
    logger.info("LAYER %d", layer)
    logger.info("=" * 50)

    x_in, x_out, train_idx, val_idx = load_data(activations_dir, layer, d_in, val_split, seed)

    zero_mse = (x_out[val_idx] ** 2).mean().item()
    logger.info("zero baseline: %.6f", zero_mse)

    t0 = time.time()
    W_r = fit_linear_svd(x_in, x_out, train_idx, rank)
    logger.info("SVD rank-%d fit in %.1fs", rank, time.time() - t0)

    with torch.no_grad():
        linear_pred_val = x_in[val_idx] @ W_r
        linear_mse = F.mse_loss(linear_pred_val, x_out[val_idx]).item()
    linear_vc = max(0, (1 - linear_mse / zero_mse) * 100)
    logger.info("linear rank-%d: val_mse=%.6f, var_captured=%.1f%%", rank, linear_mse, linear_vc)

    residual_train = compute_residual(x_in, x_out, W_r, train_idx)
    residual_val = compute_residual(x_in, x_out, W_r, val_idx)
    residual_var = (residual_val ** 2).mean().item()
    logger.info("residual variance: %.6f (%.1f%% of total)", residual_var, residual_var / zero_mse * 100)

    # MLP on residual
    mlp = ResidualMLP(d_in, hidden=512)
    mlp_mse, mlp_vc = train_model(mlp, x_in, residual_train, residual_val,
                                   train_idx, val_idx, steps, batch_size, 1e-3,
                                   f"L{layer}-MLP-h512")

    # Memory layer on residual (Meta style, no VSA)
    mem = ResidualMemoryLayer(d_in, n_slots=4096, d_key=256, top_k=8)
    mem_mse, mem_vc = train_model(mem, x_in, residual_train, residual_val,
                                   train_idx, val_idx, steps, batch_size, 1e-3,
                                   f"L{layer}-MemLayer")

    total_mlp_vc = linear_vc + mlp_vc * (1 - linear_vc / 100)
    total_mem_vc = linear_vc + mem_vc * (1 - linear_vc / 100)

    logger.info("-" * 50)
    logger.info("LAYER %d RESULTS:", layer)
    logger.info("  Linear rank-%d:     %.1f%% total var", rank, linear_vc)
    logger.info("  MLP on residual:    %.1f%% of residual → %.1f%% total", mlp_vc, total_mlp_vc)
    logger.info("  MemLayer on res:    %.1f%% of residual → %.1f%% total", mem_vc, total_mem_vc)
    logger.info("  MLP params:         %d", sum(p.numel() for p in mlp.parameters()))
    logger.info("  MemLayer params:    %d", sum(p.numel() for p in mem.parameters()))

    return {
        "layer": layer,
        "zero_mse": zero_mse,
        "linear_vc": linear_vc,
        "mlp_residual_vc": mlp_vc,
        "mem_residual_vc": mem_vc,
        "total_mlp_vc": total_mlp_vc,
        "total_mem_vc": total_mem_vc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations-dir", type=Path, default=Path.home() / "cube-memory-cache" / "activations")
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 27, 43])
    ap.add_argument("--d-in", type=int, default=5120)
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--val-split", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    results = []
    for layer in args.layers:
        try:
            r = analyze_layer(args.activations_dir, layer, args.d_in, args.rank,
                              args.val_split, args.seed, args.steps, args.batch_size)
            results.append(r)
        except FileNotFoundError as e:
            logger.warning("skipping layer %d: %s", layer, e)

    logger.info("\n" + "=" * 60)
    logger.info("CROSS-LAYER SUMMARY (rank-%d linear base)", args.rank)
    logger.info("%-8s %-12s %-12s %-12s %-12s", "Layer", "Linear%", "MLP+Lin%", "Mem+Lin%", "Non-linear?")
    for r in results:
        nonlinear = "YES" if r["total_mlp_vc"] > r["linear_vc"] + 3 else "marginal" if r["total_mlp_vc"] > r["linear_vc"] + 1 else "no"
        logger.info("%-8d %-12.1f %-12.1f %-12.1f %-12s",
                     r["layer"], r["linear_vc"], r["total_mlp_vc"], r["total_mem_vc"], nonlinear)


if __name__ == "__main__":
    main()
