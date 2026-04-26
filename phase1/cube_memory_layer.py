"""CubeMemoryLayer — VSA-keyed Memory Layer for FFN replacement.

Drop-in for a transformer FFN block. Same input/output contract:
hidden states in, residual update out. Internally:

    h ∈ ℝ^(B, T, D)
        ↓ role projection (D → p × d)
        ↓ split into p query-roles
        ↓ per-role nearest-codebook-vector cleanup
        ↓ FHRR bind across roles → composite address
        ↓ slot-store gather (top-1 by inner product against bound key)
        ↓ output projection (d_v → D)
    Δh ∈ ℝ^(B, T, D)

The codebook is frozen (random orthogonal init); the role-projection,
slot store, and output projection are learned. This matches the
Phase 0 verdict (frozen + learnable values gives a stable trainer).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_phasor(real_vec: torch.Tensor) -> torch.Tensor:
    """Map a real vector to unit-modulus complex phasors via cos+i·sin."""
    return torch.complex(real_vec.cos(), real_vec.sin())


def _from_phasor(z: torch.Tensor) -> torch.Tensor:
    """Recover a real vector from complex phasors as the angle (phase)."""
    return torch.atan2(z.imag, z.real)


def _unitize_complex(z: torch.Tensor) -> torch.Tensor:
    mag = z.abs().clamp(min=1e-8)
    return z / mag


class CubeMemoryLayer(nn.Module):
    """FFN drop-in.

    Args:
        d_in: hidden dim of the host transformer (e.g. 2048 for Qwen3.6-A3B)
        d_codebook: dim of the FHRR vectors (the algebra dim).
        d_value: dim of the slot value vectors before output projection.
        m: codebook size per role-axis.
        p: number of role-axes (bind depth).
        n_slots: number of value slots in the store.
        top_k: number of slots retrieved per token (soft attention over them).
        seed: deterministic init for the frozen codebook.
    """

    def __init__(
        self,
        d_in: int,
        d_codebook: int = 1024,
        d_value: int = 512,
        m: int = 256,
        p: int = 3,
        n_slots: int = 65_536,
        top_k: int = 4,
        seed: int = 0,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_codebook = d_codebook
        self.d_value = d_value
        self.m = m
        self.p = p
        self.n_slots = n_slots
        self.top_k = top_k

        # Frozen role-axis codebooks. Phases drawn uniformly from [-pi, pi];
        # registered as buffers so they don't get optimized.
        g = torch.Generator().manual_seed(seed)
        for ax in range(p):
            phases = (torch.rand(m, d_codebook, generator=g) * 2 - 1) * torch.pi
            self.register_buffer(f"codebook_{ax}", _to_phasor(phases))

        # Learnable: role-projection MLP, slot keys, slot values, output projection.
        # Role-projection produces p · d_codebook real values; we map to
        # phasors and unbind against the codebook entries.
        self.role_proj = nn.Linear(d_in, p * d_codebook, bias=False)
        # Slot keys: learnable real vectors of dim 2*d_codebook (real||imag of
        # the bound address signature). Compared against the bound query
        # via inner product to produce top-k soft routing.
        self.slot_keys = nn.Parameter(torch.empty(n_slots, 2 * d_codebook))
        # Slot values.
        self.slot_values = nn.Parameter(torch.empty(n_slots, d_value))
        # Output back to model dim.
        self.out_proj = nn.Linear(d_value, d_in, bias=False)

        # Init: small projections, slots near zero, out_proj zero so the
        # layer starts as the identity residual.
        nn.init.normal_(self.role_proj.weight, std=0.02)
        nn.init.normal_(self.slot_keys, std=0.02)
        nn.init.normal_(self.slot_values, std=0.02)
        nn.init.zeros_(self.out_proj.weight)

    def codebook(self, axis: int) -> torch.Tensor:
        return getattr(self, f"codebook_{axis}")

    def _query_to_phasor(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, D) -> phasor (B, T, p, d_codebook).

        Internally upcasts the projection output to fp32 because
        torch.complex requires Half/Float/Double inputs (no bf16).
        Whole-layer dtype handling lives in `forward` — the residual
        update is cast back to the caller's dtype on the way out.
        """
        x = self.role_proj(h)  # (B, T, p*d_codebook)
        if x.dtype not in (torch.float32, torch.float64, torch.float16):
            x = x.to(torch.float32)
        x = x.view(*h.shape[:-1], self.p, self.d_codebook)
        return _to_phasor(x)

    def _cleanup_per_axis(self, q_phasor: torch.Tensor) -> torch.Tensor:
        """For each role axis, find the nearest codebook entry and return
        its phasor (replacing the noisy query with a clean codebook entry).

        q_phasor: (..., p, d) complex
        Returns: (..., p, d) complex (same shape, snapped to codebook).
        """
        cleaned = []
        for ax in range(self.p):
            cb = self.codebook(ax)  # (m, d) complex
            q = q_phasor[..., ax, :]  # (..., d)
            # Cosine-like similarity in the complex plane: real part of
            # Hermitian inner product, normalized by d.
            sims = (q.unsqueeze(-2) * cb.conj()).sum(-1).real / self.d_codebook
            best_idx = sims.argmax(dim=-1)  # (...,)
            cleaned.append(cb[best_idx])  # (..., d)
        return torch.stack(cleaned, dim=-2)  # (..., p, d)

    def _bound_address(self, cleaned: torch.Tensor) -> torch.Tensor:
        """FHRR bind across the p role axes -> single complex vector."""
        # cleaned: (..., p, d) complex
        addr = cleaned[..., 0, :]
        for ax in range(1, self.p):
            addr = addr * cleaned[..., ax, :]
        return addr  # (..., d) complex

    def _addr_to_realq(self, addr: torch.Tensor) -> torch.Tensor:
        """Concatenate (real, imag) of the complex address to a 2*d real
        query vector for inner product against slot_keys."""
        return torch.cat([addr.real, addr.imag], dim=-1)

    def _retrieve(self, q_real: torch.Tensor) -> torch.Tensor:
        """Top-k soft retrieval against slot_keys/slot_values.

        q_real: (..., 2*d_codebook)
        Returns: (..., d_value)
        """
        flat_q = q_real.reshape(-1, q_real.shape[-1])  # (N, 2d)
        sims = flat_q @ self.slot_keys.t()  # (N, n_slots)
        topk_sims, topk_idx = sims.topk(self.top_k, dim=-1)  # (N, k)
        weights = topk_sims.softmax(dim=-1)  # (N, k)
        # Gather values for topk indices.
        gathered = self.slot_values[topk_idx]  # (N, k, d_value)
        out = (weights.unsqueeze(-1) * gathered).sum(dim=-2)  # (N, d_value)
        return out.reshape(*q_real.shape[:-1], self.d_value)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # torch.complex doesn't accept bf16; complex math runs in fp32
        # internally, then we cast the residual update back to the
        # caller's dtype before returning.
        in_dtype = h.dtype
        h = h.to(torch.float32) if in_dtype not in (torch.float32, torch.float64) else h

        # Cleanup uses argmax (non-differentiable) but we route the gradient
        # through the role_proj via a straight-through-style trick: the
        # cleaned phasor is `q + (cleaned - q).detach()` so forward sees
        # the snapped vector and backward sees the raw query.
        q_phasor = self._query_to_phasor(h)
        cleaned = self._cleanup_per_axis(q_phasor)
        cleaned_ste = q_phasor + (cleaned - q_phasor).detach()
        addr = self._bound_address(cleaned_ste)
        # Defensive unit-modulus projection. The product of unit-modulus
        # phasors is algebraically unit-modulus, so this is a no-op in
        # exact arithmetic; in fp32/fp16 it bounds float drift that
        # accumulates across many bind levels and through the STE path.
        addr = _unitize_complex(addr)
        q_real = self._addr_to_realq(addr)
        # Sparse top-k retrieval: only the selected slots receive
        # gradients on this forward pass. This is the standard Memory
        # Layer training pattern (Berges et al. 2024) — unselected
        # slots stay at their last-update values, which is fine because
        # the model learns to *route* hot tokens to slots rather than
        # update everything densely.
        slot_val = self._retrieve(q_real)
        out = self.out_proj(slot_val)
        return out.to(in_dtype) if out.dtype != in_dtype else out
