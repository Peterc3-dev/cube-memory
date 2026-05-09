#!/usr/bin/env python3
"""Control: rank-1024 SVD linear → SGD fine-tune WITHOUT cube memory.
Tests whether stage 3 gains are from the linear escaping SVD optimum."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LinearOnly(nn.Module):
    def __init__(self, d_in, rank):
        super().__init__()
        self.up = nn.Linear(d_in, rank, bias=False)
        self.down = nn.Linear(rank, d_in, bias=False)

    def forward(self, x):
        return self.down(self.up(x))


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from per_layer_trainer import load_layer_pairs
    from train_hybrid import fit_linear_svd, compute_val_mse, variance_captured

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(42)

    act_dir = Path.home() / "cube-memory-cache" / "activations"

    for layer in [27, 43]:
        logger.info("=" * 50)
        logger.info("CONTROL: LAYER %d — Linear-only SGD fine-tune", layer)

        x_in, x_out = load_layer_pairs(act_dir, layer, 5120)
        n = x_in.shape[0]
        n_val = max(1, int(n * 0.05))
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        zero_mse = (x_out[val_idx] ** 2).mean().item()
        logger.info("zero baseline: %.6f", zero_mse)

        model = LinearOnly(5120, 1024)

        W_up, W_down = fit_linear_svd(x_in, x_out, train_idx, 1024, "cpu")
        with torch.no_grad():
            model.up.weight.copy_(W_up)
            model.down.weight.copy_(W_down)

        svd_mse = compute_val_mse(model, x_in, x_out, val_idx, "cpu", 256)
        logger.info("SVD init: val_mse=%.6f, var=%.1f%%", svd_mse, variance_captured(svd_mse, zero_mse))

        # SGD fine-tune (same as hybrid stage 3)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        rng = torch.Generator(device="cpu").manual_seed(42)
        bs = 64

        for step in range(3000):
            idx = train_idx[torch.randint(0, train_idx.numel(), (bs,), generator=rng)]
            b_in = x_in[idx].unsqueeze(0)
            b_out = x_out[idx].unsqueeze(0)
            optim.zero_grad()
            pred = model(b_in)
            loss = F.mse_loss(pred, b_out)
            loss.backward()
            optim.step()

            if step % 500 == 0 or step == 2999:
                v = compute_val_mse(model, x_in, x_out, val_idx, "cpu", 256)
                vc = variance_captured(v, zero_mse)
                logger.info("step %5d  val_mse=%.6f  var=%.1f%%", step, v, vc)

        final = compute_val_mse(model, x_in, x_out, val_idx, "cpu", 256)
        logger.info("CONTROL RESULT layer %d: SVD=%.1f%% → SGD=%.1f%%",
                     layer, variance_captured(svd_mse, zero_mse), variance_captured(final, zero_mse))


if __name__ == "__main__":
    main()
