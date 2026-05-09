#!/usr/bin/env python3
"""VSA-routed Mixture of Experts for FFN replacement.

Instead of using VSA for direct value retrieval (which is a rank-4
bottleneck), use it as a compositional routing function that selects
which small linear experts to activate. Each expert is a rank-r linear.

Architecture:
    output = sum_i(w_i * expert_i(x))
    where w_i = softmax(VSA_address . expert_key_i)[top_k]

Control: standard learned-gate MoE (no VSA) with identical experts.
"""
from __future__ import annotations

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


class VSARouter(nn.Module):
    """Compositional VSA routing: project → phasor → cleanup → bind → dot with expert keys."""
    def __init__(self, d_in, d_addr, n_experts, p=2, m=64, top_k=2):
        super().__init__()
        self.d_addr = d_addr
        self.p = p
        self.m = m
        self.top_k = top_k
        self.proj = nn.Linear(d_in, p * d_addr, bias=False)
        g = torch.Generator().manual_seed(0)
        for ax in range(p):
            phases = (torch.rand(m, d_addr, generator=g) * 2 - 1) * torch.pi
            self.register_buffer(f"cb_{ax}", _to_phasor(phases))
        self.expert_keys = nn.Parameter(torch.randn(n_experts, 2 * d_addr) * 0.02)

    def forward(self, x):
        h = self.proj(x)
        h = h.view(*x.shape[:-1], self.p, self.d_addr)
        q = _to_phasor(h.float())
        addr = q[..., 0, :]
        for ax in range(1, self.p):
            cb = getattr(self, f"cb_{ax}")
            q_ax = q[..., ax, :]
            sims = (q_ax.unsqueeze(-2) * cb.conj()).sum(-1).real / self.d_addr
            best = cb[sims.argmax(dim=-1)]
            addr = addr * best
        addr = _unitize_complex(addr)
        q_real = torch.cat([addr.real, addr.imag], dim=-1)
        logits = q_real @ self.expert_keys.T
        topk_logits, topk_idx = logits.topk(self.top_k, dim=-1)
        weights = topk_logits.softmax(dim=-1)
        return weights, topk_idx


class LearnedRouter(nn.Module):
    """Standard learned gate (no VSA)."""
    def __init__(self, d_in, n_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(d_in, n_experts, bias=False)
        self.top_k = top_k

    def forward(self, x):
        logits = self.gate(x)
        topk_logits, topk_idx = logits.topk(self.top_k, dim=-1)
        weights = topk_logits.softmax(dim=-1)
        return weights, topk_idx


class Expert(nn.Module):
    def __init__(self, d_in, rank):
        super().__init__()
        self.up = nn.Linear(d_in, rank, bias=False)
        self.down = nn.Linear(rank, d_in, bias=False)
        nn.init.normal_(self.up.weight, std=0.02)
        nn.init.zeros_(self.down.weight)

    def forward(self, x):
        return self.down(self.up(x))


class MoELayer(nn.Module):
    def __init__(self, d_in, n_experts, expert_rank, router):
        super().__init__()
        self.experts = nn.ModuleList([Expert(d_in, expert_rank) for _ in range(n_experts)])
        self.router = router
        self.n_experts = n_experts

    def forward(self, x):
        weights, topk_idx = self.router(x)
        batch_shape = x.shape[:-1]
        flat_x = x.reshape(-1, x.shape[-1])
        flat_w = weights.reshape(-1, weights.shape[-1])
        flat_idx = topk_idx.reshape(-1, topk_idx.shape[-1])
        N = flat_x.shape[0]
        out = torch.zeros_like(flat_x)
        for i, expert in enumerate(self.experts):
            mask = (flat_idx == i).any(dim=-1)
            if mask.any():
                expert_out = expert(flat_x[mask])
                positions = (flat_idx[mask] == i).float()
                w = (flat_w[mask] * positions).sum(dim=-1, keepdim=True)
                out[mask] += w * expert_out
        return out.reshape(*batch_shape, x.shape[-1])


def train_and_eval(model, x_in, x_out, train_idx, val_idx, zero_mse, steps, bs, lr, name):
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = torch.Generator(device="cpu").manual_seed(42)
    best_vc = 0.0

    for step in range(steps):
        idx = train_idx[torch.randint(0, train_idx.numel(), (bs,), generator=rng)]
        b_in, b_out = x_in[idx], x_out[idx]
        optim.zero_grad()
        loss = F.mse_loss(model(b_in), b_out)
        loss.backward()
        optim.step()

        if step % 1000 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                vp = model(x_in[val_idx])
                vm = F.mse_loss(vp, x_out[val_idx]).item()
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

    for layer in [27]:
        x_in, x_out = load_layer_pairs(act_dir, layer, 5120)
        n = x_in.shape[0]
        nv = max(1, int(n * 0.05))
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
        vi, ti = perm[:nv], perm[nv:]
        zmse = (x_out[vi] ** 2).mean().item()
        logger.info("Layer %d, zero baseline: %.6f", layer, zmse)

        configs = [
            ("VSA-MoE-8x256-top2", 8, 256, "vsa", 2),
            ("Learned-MoE-8x256-top2", 8, 256, "learned", 2),
            ("VSA-MoE-16x128-top4", 16, 128, "vsa", 4),
            ("Learned-MoE-16x128-top4", 16, 128, "learned", 4),
        ]

        results = []
        for name, n_exp, rank, rtype, topk in configs:
            if rtype == "vsa":
                router = VSARouter(5120, 256, n_exp, p=2, m=64, top_k=topk)
            else:
                router = LearnedRouter(5120, n_exp, top_k=topk)
            model = MoELayer(5120, n_exp, rank, router)
            tp = sum(p.numel() for p in model.parameters())
            logger.info("=== %s === (%d params)", name, tp)
            vc = train_and_eval(model, x_in, x_out, ti, vi, zmse, 5000, 64, 1e-3, name)
            results.append((name, vc, tp))

        logger.info("=" * 60)
        logger.info("MOE COMPARISON (layer %d)", layer)
        for name, vc, tp in results:
            logger.info("  %-35s var=%.1f%%  params=%d", name, vc, tp)


if __name__ == "__main__":
    main()
