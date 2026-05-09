#!/usr/bin/env python3
"""Sweep SVD linear rank to find the variance ceiling and the optimal rank
for a cube memory base layer."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from per_layer_trainer import load_layer_pairs

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    act_dir = Path.home() / "cube-memory-cache" / "activations"
    d_in = 5120
    seed = 42

    for layer in [3, 27, 43]:
        logger.info("=" * 60)
        logger.info("LAYER %d — RANK SWEEP", layer)
        logger.info("=" * 60)

        x_in, x_out = load_layer_pairs(act_dir, layer, d_in)
        n = x_in.shape[0]
        n_val = max(1, int(n * 0.05))
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        zero_mse = (x_out[val_idx] ** 2).mean().item()
        logger.info("zero baseline: %.6e", zero_mse)

        X = x_in[train_idx].to(torch.float64)
        Y = x_out[train_idx].to(torch.float64)
        XtX = X.T @ X
        XtY = X.T @ Y
        lam = 1e-6 * XtX.diagonal().mean()
        XtX.diagonal().add_(lam)

        t0 = time.time()
        W_full = torch.linalg.solve(XtX, XtY)
        U, S, Vt = torch.linalg.svd(W_full, full_matrices=False)
        logger.info("full SVD in %.1fs", time.time() - t0)

        ranks = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 5120]
        logger.info("%-8s %-12s %-12s %-12s", "Rank", "Val MSE", "Var%", "Params")

        for r in ranks:
            if r > d_in:
                break
            W_r = ((U[:, :r] * S[:r]) @ Vt[:r, :]).to(torch.float32)
            with torch.no_grad():
                pred = x_in[val_idx] @ W_r
                val_mse = F.mse_loss(pred, x_out[val_idx]).item()
            vc = max(0, (1 - val_mse / zero_mse) * 100)
            params = 2 * d_in * r
            logger.info("%-8d %-12.6e %-12.1f %-12d", r, val_mse, vc, params)


if __name__ == "__main__":
    main()
