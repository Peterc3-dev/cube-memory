#!/usr/bin/env python3
"""Reviewer Experiment 2: top_k scaling hypothesis.

If VSA retrieval is a rank-k bottleneck, variance captured should scale
roughly with top_k until saturating the codebook. Test top_k = 4, 16,
64, 256, 1024 on layer 27.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def train_vsa(model, x_in, x_out, train_idx, val_idx, zero_mse, steps=5000, bs=64, lr=1e-3, name=""):
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = torch.Generator(device="cpu").manual_seed(42)
    best_vc = 0.0

    for step in range(steps):
        idx = train_idx[torch.randint(0, train_idx.numel(), (bs,), generator=rng)]
        b_in, b_out = x_in[idx].unsqueeze(0), x_out[idx].unsqueeze(0)
        optim.zero_grad()
        loss = F.mse_loss(model(b_in), b_out)
        loss.backward()
        optim.step()

        if step % 1000 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                vp = model(x_in[val_idx].unsqueeze(0))
                vm = F.mse_loss(vp, x_out[val_idx].unsqueeze(0)).item()
            vc = max(0, (1 - vm / zero_mse) * 100)
            best_vc = max(best_vc, vc)
            logger.info("[%s] step %5d  val=%.6f  var=%.1f%%", name, step, vm, vc)
            model.train()

    return best_vc


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from per_layer_trainer import load_layer_pairs
    from cube_memory_layer import CubeMemoryLayer

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(42)

    act_dir = Path.home() / "cube-memory-cache" / "activations"
    layer = 27
    d_in = 5120

    x_in, x_out = load_layer_pairs(act_dir, layer, d_in)
    n = x_in.shape[0]
    n_val = max(1, int(n * 0.05))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    zero_mse = (x_out[val_idx] ** 2).mean().item()
    logger.info("Layer %d, zero baseline: %.6f, %d train, %d val", layer, zero_mse, len(train_idx), len(val_idx))

    topk_values = [4, 16, 64, 256]
    results = []

    for topk in topk_values:
        n_slots = max(4096, topk * 16)
        model = CubeMemoryLayer(
            d_in=d_in, d_codebook=256, d_value=d_in,
            m=64, p=2, n_slots=n_slots, top_k=topk, seed=0,
        )
        tp = sum(p.numel() for p in model.parameters())
        name = f"VSA-topk{topk}-slots{n_slots}"
        logger.info("=== %s === (%d params)", name, tp)
        vc = train_vsa(model, x_in, x_out, train_idx, val_idx, zero_mse, steps=5000, name=name)
        results.append({"top_k": topk, "n_slots": n_slots, "params": tp, "var_pct": round(vc, 1)})
        logger.info("RESULT: %s -> %.1f%%", name, vc)

    logger.info("=" * 60)
    logger.info("TOP-K SCALING SUMMARY (layer %d)", layer)
    logger.info("%-8s %-10s %-12s %-8s", "top_k", "n_slots", "params", "var%")
    for r in results:
        logger.info("%-8d %-10d %-12d %-8.1f", r["top_k"], r["n_slots"], r["params"], r["var_pct"])

    out_path = Path(__file__).resolve().parent.parent / "reviewer_results" / "exp2_topk_scaling.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"layer": layer, "results": results}, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
