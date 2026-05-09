#!/usr/bin/env python3
"""Reviewer Experiment 4: Codebook quality ablation.

Tests three codebook initialization strategies to prove the rank
bottleneck is structural (inherent to the retrieval pipeline), not
caused by poor codebook quality:

1. Frozen random codebooks (V1 baseline)
2. Learned codebooks (V2 - trained end-to-end)
3. SVD-derived codebooks (optimal linear basis of activations)

If all three perform similarly (~5%), the bottleneck is the retrieval
architecture itself, not codebook quality.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _to_phasor(x):
    return torch.complex(x.cos(), x.sin())


def _unitize_complex(z):
    return z / z.abs().clamp(min=1e-8)


class CubeMemoryWithCodebook(nn.Module):
    """Simplified VSA layer that accepts external codebook init."""

    def __init__(self, d_in, d_cb, m, p, n_slots, top_k, codebook_init, learn_codebook=False):
        super().__init__()
        self.d_in = d_in
        self.d_cb = d_cb
        self.m = m
        self.p = p
        self.top_k = top_k

        for ax in range(p):
            cb = codebook_init[ax]  # (m, d_cb) complex
            if learn_codebook:
                self.register_parameter(f"cb_{ax}_real", nn.Parameter(cb.real.clone()))
                self.register_parameter(f"cb_{ax}_imag", nn.Parameter(cb.imag.clone()))
            else:
                self.register_buffer(f"cb_{ax}", cb)

        self.learn_codebook = learn_codebook
        self.proj = nn.Linear(d_in, p * d_cb, bias=False)
        self.slot_keys = nn.Parameter(torch.randn(n_slots, 2 * d_cb) * 0.02)
        self.slot_values = nn.Parameter(torch.randn(n_slots, d_in) * 0.02)
        self.out_proj = nn.Linear(d_in, d_in, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.weight)

    def _get_cb(self, ax):
        if self.learn_codebook:
            r = getattr(self, f"cb_{ax}_real")
            i = getattr(self, f"cb_{ax}_imag")
            return _unitize_complex(torch.complex(r, i))
        return getattr(self, f"cb_{ax}")

    def forward(self, x):
        h = self.proj(x)
        h = h.view(*x.shape[:-1], self.p, self.d_cb)
        q = _to_phasor(h.float())

        addr = q[..., 0, :]
        for ax in range(1, self.p):
            cb = self._get_cb(ax)
            q_ax = q[..., ax, :]
            sims = (q_ax.unsqueeze(-2) * cb.conj()).sum(-1).real / self.d_cb
            best = cb[sims.argmax(dim=-1)]
            q_ax_ste = q_ax + (best - q_ax).detach()
            addr = addr * q_ax_ste

        addr = _unitize_complex(addr)
        q_real = torch.cat([addr.real, addr.imag], dim=-1)

        sims = q_real.reshape(-1, q_real.shape[-1]) @ self.slot_keys.T
        topk_sims, topk_idx = sims.topk(self.top_k, dim=-1)
        weights = topk_sims.softmax(dim=-1)
        gathered = self.slot_values[topk_idx]
        out = (weights.unsqueeze(-1) * gathered).sum(dim=-2)
        out = out.reshape(*x.shape[:-1], self.d_in)
        return self.out_proj(out)


def train_and_eval(model, x_in, x_out, train_idx, val_idx, zero_mse, steps, bs, lr, name):
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

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(42)

    act_dir = Path.home() / "cube-memory-cache" / "activations"
    layer = 27
    d_in = 5120
    d_cb = 256
    m = 64
    p = 2
    n_slots = 4096
    top_k = 4

    x_in, x_out = load_layer_pairs(act_dir, layer, d_in)
    n = x_in.shape[0]
    n_val = max(1, int(n * 0.05))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    zero_mse = (x_out[val_idx] ** 2).mean().item()
    logger.info("Layer %d, zero baseline: %.6f", layer, zero_mse)

    # --- Codebook 1: Frozen random (V1 baseline) ---
    g = torch.Generator().manual_seed(0)
    random_cbs = []
    for ax in range(p):
        phases = (torch.rand(m, d_cb, generator=g) * 2 - 1) * torch.pi
        random_cbs.append(_to_phasor(phases))

    # --- Codebook 3: SVD-derived (optimal linear basis of activations) ---
    logger.info("Computing SVD of activation matrix for optimal codebooks...")
    X_train = x_in[train_idx[:10000]].to(torch.float64)
    # U columns are principal directions in feature space (5120-d)
    U, S, Vt = torch.linalg.svd(X_train.T, full_matrices=False)
    svd_cbs = []
    for ax in range(p):
        # U is (5120, 5120); columns are principal directions
        # Take m directions per axis, slice to d_cb dims -> (m, d_cb)
        basis = U[:d_cb, ax * m:(ax + 1) * m].T.to(torch.float32)
        svd_cbs.append(_to_phasor(basis))
    logger.info("SVD codebooks computed from top-%d activation directions", m * p)

    configs = [
        ("Frozen-random (V1)", random_cbs, False),
        ("Learned (V2)", random_cbs, True),
        ("SVD-optimal", svd_cbs, False),
        ("SVD-optimal+learned", svd_cbs, True),
    ]

    results = []
    for name, cbs, learn in configs:
        model = CubeMemoryWithCodebook(
            d_in=d_in, d_cb=d_cb, m=m, p=p,
            n_slots=n_slots, top_k=top_k,
            codebook_init=cbs, learn_codebook=learn,
        )
        tp = sum(pp.numel() for pp in model.parameters())
        logger.info("=== %s === (%d params)", name, tp)
        vc = train_and_eval(model, x_in, x_out, train_idx, val_idx, zero_mse,
                            steps=5000, bs=64, lr=1e-3, name=name)
        results.append({"codebook": name, "params": tp, "var_pct": round(vc, 1)})

    logger.info("=" * 60)
    logger.info("CODEBOOK ABLATION SUMMARY (layer %d)", layer)
    for r in results:
        logger.info("  %-25s params=%d  var=%.1f%%", r["codebook"], r["params"], r["var_pct"])

    out_path = Path(__file__).resolve().parent.parent / "reviewer_results" / "exp4_codebook_ablation.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"layer": layer, "results": results}, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
