"""CubeMemoryLayerV2 — Revised VSA-keyed Memory Layer for FFN replacement.

Structural changes over v1:
  1. Learned codebooks (unfrozen, re-normalized to unit modulus each forward)
  2. Multi-head retrieval (independent top-k per head, concat values)
  3. Gumbel-softmax cleanup (replaces hard argmax + STE)
  4. Gated residual (learned scalar gate, initialized conservatively)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_phasor(real_vec: torch.Tensor) -> torch.Tensor:
    return torch.complex(real_vec.cos(), real_vec.sin())


def _unitize_complex(z: torch.Tensor) -> torch.Tensor:
    mag = z.abs().clamp(min=1e-8)
    return z / mag


class CubeMemoryLayerV2(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_codebook: int = 1024,
        d_value: int = None,
        m: int = 256,
        p: int = 3,
        n_slots: int = 65_536,
        top_k: int = 4,
        seed: int = 0,
        n_heads: int = 4,
        tau_init: float = 1.0,
    ):
        super().__init__()
        if d_value is None:
            d_value = d_in
        self.d_in = d_in
        self.d_codebook = d_codebook
        self.d_value = d_value
        self.m = m
        self.p = p
        self.n_slots = n_slots
        self.top_k = top_k
        self.n_heads = n_heads

        assert (2 * d_codebook) % n_heads == 0, "2*d_codebook must be divisible by n_heads"
        assert d_value % n_heads == 0, "d_value must be divisible by n_heads"
        assert n_slots % n_heads == 0, "n_slots must be divisible by n_heads"

        self.d_key_per_head = (2 * d_codebook) // n_heads
        self.d_value_per_head = d_value // n_heads
        self.n_slots_per_head = n_slots // n_heads

        # 1. Learned codebooks — initialized same as v1 but as Parameters
        g = torch.Generator().manual_seed(seed)
        for ax in range(p):
            phases = (torch.rand(m, d_codebook, generator=g) * 2 - 1) * torch.pi
            setattr(self, f"codebook_{ax}", nn.Parameter(_to_phasor(phases)))

        # Gumbel-softmax temperature (annealed externally, not learned)
        self.register_buffer("tau", torch.tensor(tau_init))

        self.role_proj = nn.Linear(d_in, p * d_codebook, bias=False)

        # 2. Multi-head slot store
        self.slot_keys = nn.Parameter(
            torch.empty(n_heads, self.n_slots_per_head, self.d_key_per_head)
        )
        self.slot_values = nn.Parameter(
            torch.empty(n_heads, self.n_slots_per_head, self.d_value_per_head)
        )

        self.out_proj = nn.Linear(d_value, d_in, bias=False)

        # 4. Gated residual — sigmoid(-2) ≈ 0.12
        self.gate = nn.Parameter(torch.tensor(-2.0))

        # Init
        nn.init.normal_(self.role_proj.weight, std=0.02)
        nn.init.normal_(self.slot_keys, std=0.02)
        nn.init.normal_(self.slot_values, std=0.02)
        nn.init.zeros_(self.out_proj.weight)

    def codebook(self, axis: int) -> torch.Tensor:
        return getattr(self, f"codebook_{axis}")

    def set_tau(self, new_tau: float) -> None:
        self.tau.fill_(new_tau)

    @torch.no_grad()
    def _project_codebooks(self) -> None:
        """Re-normalize each codebook entry to unit modulus in the complex domain."""
        for ax in range(self.p):
            cb = self.codebook(ax)
            cb.copy_(_unitize_complex(cb))

    def _query_to_phasor(self, h: torch.Tensor) -> torch.Tensor:
        x = self.role_proj(h)
        if x.dtype not in (torch.float32, torch.float64, torch.float16):
            x = x.to(torch.float32)
        x = x.view(*h.shape[:-1], self.p, self.d_codebook)
        return _to_phasor(x)

    def _cleanup_per_axis(self, q_phasor: torch.Tensor) -> torch.Tensor:
        """Gumbel-softmax cleanup: soft weighted sum during training, hard argmax at eval.

        q_phasor: (..., p, d) complex
        Returns: (..., p, d) complex
        """
        cleaned = []
        for ax in range(self.p):
            cb = self.codebook(ax)  # (m, d) complex
            q = q_phasor[..., ax, :]  # (..., d)

            # Hermitian similarity
            sims = (q.unsqueeze(-2) * cb.conj()).sum(-1).real / self.d_codebook  # (..., m)

            if self.training:
                # Gumbel noise
                eps = 1e-20
                u = torch.rand_like(sims.real if sims.is_complex() else sims)
                gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
                logits = sims + gumbel_noise
                weights = F.softmax(logits / self.tau, dim=-1)  # (..., m)
                # Soft weighted sum of codebook rows
                result = torch.einsum("...m, md -> ...d", weights.to(cb.dtype), cb)
            else:
                best_idx = sims.argmax(dim=-1)
                result = cb[best_idx]

            cleaned.append(result)
        return torch.stack(cleaned, dim=-2)

    def _bound_address(self, cleaned: torch.Tensor) -> torch.Tensor:
        addr = cleaned[..., 0, :]
        for ax in range(1, self.p):
            addr = addr * cleaned[..., ax, :]
        return addr

    def _addr_to_realq(self, addr: torch.Tensor) -> torch.Tensor:
        return torch.cat([addr.real, addr.imag], dim=-1)

    def _retrieve(self, q_real: torch.Tensor) -> torch.Tensor:
        """Multi-head top-k retrieval.

        q_real: (..., 2*d_codebook)
        Returns: (..., d_value)
        """
        batch_shape = q_real.shape[:-1]
        flat_q = q_real.reshape(-1, q_real.shape[-1])  # (N, 2d)
        N = flat_q.shape[0]

        # Split query into heads: (N, n_heads, d_key_per_head)
        q_heads = flat_q.view(N, self.n_heads, self.d_key_per_head)

        # Per-head similarities: (N, n_heads, n_slots_per_head)
        sims = torch.einsum("nhk, hsk -> nhs", q_heads, self.slot_keys)

        topk_sims, topk_idx = sims.topk(self.top_k, dim=-1)  # (N, n_heads, k)
        weights = topk_sims.softmax(dim=-1)  # (N, n_heads, k)

        # Gather values: expand topk_idx for value dim
        # topk_idx: (N, n_heads, k) → index into slot_values (n_heads, n_slots_per_head, d_v_per_head)
        topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_value_per_head)
        # Expand slot_values for batch: (1, n_heads, n_slots_per_head, d_v_per_head)
        sv_exp = self.slot_values.unsqueeze(0).expand(N, -1, -1, -1)
        gathered = torch.gather(sv_exp, 2, topk_idx_exp)  # (N, n_heads, k, d_v_per_head)

        out = (weights.unsqueeze(-1) * gathered).sum(dim=-2)  # (N, n_heads, d_v_per_head)
        out = out.reshape(N, self.d_value)  # concat heads
        return out.reshape(*batch_shape, self.d_value)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        in_dtype = h.dtype
        h = h.to(torch.float32) if in_dtype not in (torch.float32, torch.float64) else h

        # Project codebooks to unit modulus before use
        self._project_codebooks()

        q_phasor = self._query_to_phasor(h)
        cleaned = self._cleanup_per_axis(q_phasor)
        addr = self._bound_address(cleaned)
        addr = _unitize_complex(addr)
        q_real = self._addr_to_realq(addr)
        slot_val = self._retrieve(q_real)
        out = self.out_proj(slot_val)

        # Gated residual
        out = torch.sigmoid(self.gate) * out

        return out.to(in_dtype) if out.dtype != in_dtype else out
