#!/usr/bin/env python3
"""Reviewer Experiment 3: FLOPs and wall-clock per-token cost comparison.

Analytical FLOPs + measured wall-clock for each architecture on a
single-token forward pass.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

D = 5120
D_FFN = 13824


def flops_full_ffn():
    # SwiGLU: gate_proj (D->D_FFN) + up_proj (D->D_FFN) + down_proj (D_FFN->D)
    return 3 * 2 * D * D_FFN


def flops_linear(rank):
    return 2 * D * rank + 2 * rank * D  # up (D->r) + down (r->D)


def flops_linear_plus_mlp(rank, hidden):
    linear = flops_linear(rank)
    mlp = 2 * D * hidden + 2 * hidden * D  # 2-layer MLP
    return linear + mlp


def flops_vsa(d_cb, m, p, n_slots, top_k, d_value):
    proj = 2 * D * (p * d_cb)
    cleanup = p * m * d_cb * 2  # cosine sim per axis
    bind = (p - 1) * d_cb * 6  # complex multiply
    key_match = n_slots * 2 * d_cb * 2  # inner product
    topk_sort = n_slots * 10  # approximate
    retrieve = top_k * d_value  # weighted sum
    out_proj = 2 * d_value * D
    return proj + cleanup + bind + key_match + topk_sort + retrieve + out_proj


def flops_moe(n_experts, expert_rank, top_k):
    routing = 2 * D * n_experts  # gate linear
    expert_fwd = top_k * flops_linear(expert_rank)
    return routing + expert_fwd


def params_linear(rank):
    return 2 * D * rank


def params_linear_mlp(rank, hidden):
    return 2 * D * rank + D * hidden + hidden * D


def params_vsa(d_cb, p, n_slots, d_value):
    proj = D * p * d_cb
    slot_keys = n_slots * 2 * d_cb
    slot_values = n_slots * d_value
    out_proj = d_value * D
    return proj + slot_keys + slot_values + out_proj


def params_moe(n_experts, expert_rank, d_in=D):
    gate = d_in * n_experts
    experts = n_experts * 2 * d_in * expert_rank
    return gate + experts


def measure_wallclock(model, x, warmup=50, runs=200):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize() if x.is_cuda else None
        t0 = time.perf_counter()
        for _ in range(runs):
            model(x)
        torch.cuda.synchronize() if x.is_cuda else None
        elapsed = (time.perf_counter() - t0) / runs
    return elapsed * 1e6  # microseconds


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    architectures = [
        {
            "name": "Full FFN (Qwen3.6-27B)",
            "flops": flops_full_ffn(),
            "params": 3 * D * D_FFN,
            "var_pct": "100 (reference)",
        },
        {
            "name": "Rank-16 linear",
            "flops": flops_linear(16),
            "params": params_linear(16),
            "var_pct": 5.9,
        },
        {
            "name": "Rank-256 linear",
            "flops": flops_linear(256),
            "params": params_linear(256),
            "var_pct": 14.2,
        },
        {
            "name": "Rank-1024 linear",
            "flops": flops_linear(1024),
            "params": params_linear(1024),
            "var_pct": 28.1,
        },
        {
            "name": "Rank-2048 linear",
            "flops": flops_linear(2048),
            "params": params_linear(2048),
            "var_pct": 36.6,
        },
        {
            "name": "Rank-2048 + MLP-512",
            "flops": flops_linear_plus_mlp(2048, 512),
            "params": params_linear_mlp(2048, 512),
            "var_pct": 38.4,
        },
        {
            "name": "Cube Memory V2 (VSA)",
            "flops": flops_vsa(256, 64, 2, 4096, 4, D),
            "params": params_vsa(256, 2, 4096, D),
            "var_pct": 4.8,
        },
        {
            "name": "VSA-MoE 16x128 top-4",
            "flops": flops_moe(16, 128, 4) + flops_vsa(256, 64, 2, 16, 4, 0),
            "params": params_moe(16, 128) + params_vsa(256, 2, 0, 0),
            "var_pct": 14.2,
        },
        {
            "name": "Learned-MoE 8x256 top-2",
            "flops": flops_moe(8, 256, 2),
            "params": params_moe(8, 256),
            "var_pct": 16.2,
        },
    ]

    logger.info("=" * 80)
    logger.info("FLOPS & PARAMETER COMPARISON (per token, layer 27)")
    logger.info("=" * 80)
    logger.info("%-30s %12s %12s %10s %8s", "Architecture", "FLOPs", "Params", "FLOPs/FFN", "Var%")

    full_flops = flops_full_ffn()
    for a in architectures:
        ratio = a["flops"] / full_flops if isinstance(a["var_pct"], (int, float)) else 1.0
        logger.info("%-30s %12s %12s %10.4f %8s",
                     a["name"],
                     f"{a['flops']:,}",
                     f"{a['params']:,}",
                     ratio,
                     str(a["var_pct"]))

    logger.info("")
    logger.info("Wall-clock measurement (CPU, single token)...")

    from cube_memory_layer import CubeMemoryLayer

    x_single = torch.randn(1, 1, D)

    wallclock_results = {}

    linear_2048 = torch.nn.Sequential(
        torch.nn.Linear(D, 2048, bias=False),
        torch.nn.Linear(2048, D, bias=False),
    )
    wallclock_results["Rank-2048 linear"] = measure_wallclock(linear_2048, x_single)

    vsa = CubeMemoryLayer(d_in=D, d_codebook=256, d_value=D, m=64, p=2, n_slots=4096, top_k=4)
    wallclock_results["Cube Memory V2 (VSA)"] = measure_wallclock(vsa, x_single)

    class LinearMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.up = torch.nn.Linear(D, 2048, bias=False)
            self.down = torch.nn.Linear(2048, D, bias=False)
            self.mlp_up = torch.nn.Linear(D, 512, bias=False)
            self.mlp_act = torch.nn.SiLU()
            self.mlp_down = torch.nn.Linear(512, D, bias=False)

        def forward(self, x):
            linear_out = self.down(self.up(x))
            residual = x  # simplified; in real version this would be x - linear_out for MLP input
            return linear_out + self.mlp_down(self.mlp_act(self.mlp_up(x)))

    lm = LinearMLP()
    wallclock_results["Rank-2048 + MLP-512"] = measure_wallclock(lm, x_single)

    logger.info("")
    logger.info("%-30s %12s", "Architecture", "μs/token")
    for name, us in wallclock_results.items():
        logger.info("%-30s %12.1f", name, us)

    out_path = Path(__file__).resolve().parent.parent / "reviewer_results" / "exp3_flops.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "architectures": architectures,
            "wallclock_us": wallclock_results,
        }, f, indent=2, default=str)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
