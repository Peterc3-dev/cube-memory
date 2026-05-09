"""CubeMemoryLayerV3 — Hybrid low-rank linear + VSA memory on residual.

V2 diagnosis: the VSA cleanup→bind→retrieve pipeline is a rank-4
bottleneck (top_k=4). A 164K-param rank-16 linear beats the 35M-param
V2. But the FFN mapping isn't purely linear — there's non-linear
structure the linear can't capture. The hypothesis: VSA should shine
on the *residual* after the linear component is subtracted.

Architecture:
    output = W_down @ (W_up @ x) + gate * cube_memory(x)

The low-rank linear (W_up: d_in→rank, W_down: rank→d_in) captures
the dominant linear component. The cube memory only needs to learn
the non-linear residual. Training is staged:
    Stage 1: Train the low-rank linear alone (fast, ~minutes)
    Stage 2: Freeze linear, train cube memory on residual
    Stage 3: (optional) Fine-tune both jointly with small LR
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cube_memory_layer import CubeMemoryLayer, _unitize_complex, _to_phasor


class HybridCubeMemoryLayer(nn.Module):
    def __init__(
        self,
        d_in: int,
        rank: int = 64,
        d_codebook: int = 256,
        d_value: int = 512,
        m: int = 64,
        p: int = 2,
        n_slots: int = 4096,
        top_k: int = 4,
        seed: int = 0,
    ):
        super().__init__()
        self.d_in = d_in
        self.rank = rank

        self.linear_up = nn.Linear(d_in, rank, bias=False)
        self.linear_down = nn.Linear(rank, d_in, bias=False)

        self.cube = CubeMemoryLayer(
            d_in=d_in, d_codebook=d_codebook, d_value=d_value,
            m=m, p=p, n_slots=n_slots, top_k=top_k, seed=seed,
        )

        self.gate = nn.Parameter(torch.tensor(-2.0))

        nn.init.normal_(self.linear_up.weight, std=0.02)
        nn.init.zeros_(self.linear_down.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        in_dtype = h.dtype
        h = h.to(torch.float32) if in_dtype not in (torch.float32, torch.float64) else h

        linear_out = self.linear_down(self.linear_up(h))
        cube_out = self.cube(h)
        out = linear_out + torch.sigmoid(self.gate) * cube_out

        return out.to(in_dtype) if out.dtype != in_dtype else out

    def linear_params(self):
        return list(self.linear_up.parameters()) + list(self.linear_down.parameters())

    def cube_params(self):
        return list(self.cube.parameters()) + [self.gate]
