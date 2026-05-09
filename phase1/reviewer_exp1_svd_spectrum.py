#!/usr/bin/env python3
"""Reviewer Experiment 1: SVD spectrum of FFN activation mapping.

Computes W_full = (X^T X)^{-1} X^T Y (the optimal linear map from FFN
input to output), then reports the singular value spectrum. Slow decay =
high effective rank = FFN is nearly linear and full-rank.

Also computes effective rank at various thresholds (fraction of total
singular value energy) and the rank needed to capture X% of variance.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from per_layer_trainer import load_layer_pairs

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    act_dir = Path.home() / "cube-memory-cache" / "activations"
    d_in = 5120
    seed = 42
    results = {}

    for layer in [3, 27, 43]:
        logger.info("=" * 60)
        logger.info("LAYER %d — SVD SPECTRUM", layer)

        x_in, x_out = load_layer_pairs(act_dir, layer, d_in)
        n = x_in.shape[0]
        n_val = max(1, int(n * 0.05))
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        train_idx = perm[n_val:]

        X = x_in[train_idx].to(torch.float64)
        Y = x_out[train_idx].to(torch.float64)

        XtX = X.T @ X
        XtY = X.T @ Y
        lam = 1e-6 * XtX.diagonal().mean()
        XtX.diagonal().add_(lam)

        t0 = time.time()
        W_full = torch.linalg.solve(XtX, XtY)
        U, S, Vt = torch.linalg.svd(W_full, full_matrices=False)
        elapsed = time.time() - t0
        logger.info("SVD done in %.1fs, %d singular values", elapsed, S.numel())

        S_np = S.cpu().numpy()
        energy = (S ** 2).cumsum(0) / (S ** 2).sum()
        energy_np = energy.cpu().numpy()

        logger.info("Top 20 singular values:")
        for i in range(min(20, len(S_np))):
            logger.info("  σ[%4d] = %.6f  (cumulative energy: %.4f)", i, S_np[i], energy_np[i])

        logger.info("Singular value decay milestones:")
        for frac in [0.5, 0.75, 0.9, 0.95, 0.99]:
            rank_at = int((energy >= frac).nonzero(as_tuple=True)[0][0].item()) + 1
            logger.info("  %.0f%% energy at rank %d (of %d)", frac * 100, rank_at, d_in)

        ratio_10_1 = S_np[9] / S_np[0] if S_np[0] > 0 else 0
        ratio_100_1 = S_np[99] / S_np[0] if S_np[0] > 0 else 0
        ratio_1000_1 = S_np[999] / S_np[0] if S_np[0] > 0 else 0
        logger.info("Decay ratios: σ10/σ1=%.4f  σ100/σ1=%.4f  σ1000/σ1=%.4f",
                     ratio_10_1, ratio_100_1, ratio_1000_1)

        effective_rank = float(torch.exp(-torch.sum(
            (S / S.sum()) * torch.log(S / S.sum() + 1e-30)
        )).item())
        logger.info("Shannon effective rank (exp entropy): %.1f / %d", effective_rank, d_in)

        results[f"layer_{layer}"] = {
            "singular_values_top20": S_np[:20].tolist(),
            "singular_values_bottom5": S_np[-5:].tolist(),
            "energy_at_rank": {
                str(r): float(energy_np[r - 1]) for r in [4, 16, 64, 256, 512, 1024, 2048, 5120]
                if r <= len(energy_np)
            },
            "rank_for_energy": {
                f"{int(f*100)}pct": int((energy >= f).nonzero(as_tuple=True)[0][0].item()) + 1
                for f in [0.5, 0.75, 0.9, 0.95, 0.99]
            },
            "effective_rank_shannon": round(effective_rank, 1),
            "decay_ratios": {
                "s10_s1": round(ratio_10_1, 4),
                "s100_s1": round(ratio_100_1, 4),
                "s1000_s1": round(ratio_1000_1, 4),
            },
            "total_dims": d_in,
        }

    out_path = Path(__file__).resolve().parent.parent / "reviewer_results" / "exp1_svd_spectrum.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
