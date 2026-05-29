#!/usr/bin/env python3
"""Case Study 1 analysis for a second model: SVD spectrum + linear rank-sweep
variance, computed from cached CACT activations (see hf_extract_activations.py).

Mirrors reviewer_exp1_svd_spectrum.py (SVD/effective-rank) and the linear rows
of reviewer_exp3_flops.py (variance explained vs rank), parameterized for any
d_in / layer set / cache dir so it runs on a model other than Qwen3.6-27B.

Usage:
  python phase1/svd_for_model.py \
    --act-dir ~/cube-memory-cache/activations-qwen3-4b \
    --d-in 2560 --layers 3 18 32 --out reviewer_results/qwen3-4b_svd.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def effective_rank_shannon(s: torch.Tensor) -> float:
    """Roy-Vetterli effective rank: exp(-sum p_i ln p_i), p_i = s_i / sum s_j."""
    p = s / s.sum()
    p = p[p > 0]
    h = -(p * p.log()).sum()
    return float(h.exp().item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-dir", required=True)
    ap.add_argument("--d-in", type=int, required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ranks", type=int, nargs="+",
                    default=[4, 16, 64, 256, 512, 1024, 2048])
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from per_layer_trainer import load_layer_pairs

    act_dir = Path(args.act_dir).expanduser()
    results: dict = {"_meta": {"act_dir": str(act_dir), "d_in": args.d_in}}

    for layer in args.layers:
        logger.info("=" * 60)
        logger.info("LAYER %d", layer)
        x_in, x_out = load_layer_pairs(act_dir, layer, args.d_in)
        n = x_in.shape[0]
        n_val = max(1, int(n * 0.05))
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        X = x_in[train_idx].to(torch.float64)
        Y = x_out[train_idx].to(torch.float64)
        Xv = x_in[val_idx].to(torch.float64)
        Yv = x_out[val_idx].to(torch.float64)

        # optimal ridge-regularized linear map X -> Y
        XtX = X.T @ X
        XtY = X.T @ Y
        lam = 1e-6 * XtX.diagonal().mean()
        XtX.diagonal().add_(lam)
        W_full = torch.linalg.solve(XtX, XtY)
        U, S, Vt = torch.linalg.svd(W_full, full_matrices=False)

        energy = (S ** 2).cumsum(0) / (S ** 2).sum()
        energy_at = {str(k): float(energy[min(k, len(energy)) - 1])
                     for k in [4, 16, 64, 256, 512, 1024, 2048] if k <= len(energy)}
        rank_for = {}
        for frac in [0.5, 0.75, 0.9, 0.95, 0.99]:
            idx = (energy >= frac).nonzero(as_tuple=True)[0]
            rank_for[f"{int(frac*100)}pct"] = int(idx[0].item()) + 1 if len(idx) else None

        # variance explained by best rank-r LINEAR map (truncated optimal W) on val
        mse_zero = float((Yv ** 2).mean().item())
        var_by_rank = {}
        for r in args.ranks:
            if r > len(S):
                continue
            W_r = (U[:, :r] * S[:r]) @ Vt[:r, :]
            Yhat = Xv @ W_r
            mse_r = float(((Yv - Yhat) ** 2).mean().item())
            var_by_rank[str(r)] = round((1 - mse_r / mse_zero) * 100, 3)

        erank = effective_rank_shannon(S)
        results[f"layer_{layer}"] = {
            "n_tokens": int(n),
            "total_dims": args.d_in,
            "singular_values_top20": [float(v) for v in S[:20].cpu()],
            "energy_at_rank": energy_at,
            "rank_for_energy": rank_for,
            "effective_rank_shannon": round(erank, 1),
            "linear_var_pct_by_rank": var_by_rank,
        }
        logger.info("layer %d: eff_rank=%.1f / %d, energy@4=%.3f%%, "
                    "rank@90%%=%s, rank-16 linear var=%.2f%%",
                    layer, erank, args.d_in, 100 * energy_at.get("4", 0),
                    rank_for["90pct"], var_by_rank.get("16", float("nan")))

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
