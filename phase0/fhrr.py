"""FHRR (Fourier Holographic Reduced Representation) primitives.

Vectors live in C^d as unit-modulus phasors. Bind = element-wise
complex multiplication (== adding phases). Unbind = bind with the
conjugate (== subtracting phases). Cleanup = nearest-codebook
cosine match in the real-valued representation.

Backprop-compatible. The unit-modulus projection in `unitize` is
mandatory between gradient steps to prevent magnitude drift
(Alam et al. 2021, arXiv 2109.02157).
"""
from __future__ import annotations

import torch
from torch import Tensor


def unitize(z: Tensor) -> Tensor:
    """Project complex vectors back onto the unit circle, element-wise.

    Mandatory after any operation that may have moved off the unit
    circle (notably learnable parameter updates). Stable: avoids
    division by zero by adding eps before normalizing.
    """
    mag = z.abs().clamp(min=1e-8)
    return z / mag


def random_codebook(m: int, d: int, *, device: torch.device | str = "cpu",
                    seed: int | None = None) -> Tensor:
    """A frozen random codebook of m unit-modulus phasors of dimension d.

    Phases are drawn uniformly from [-pi, pi]. The result is a complex
    tensor of shape (m, d) with all magnitudes == 1 by construction.
    """
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    phases = (torch.rand(m, d, generator=g, device=device) * 2 - 1) * torch.pi
    return torch.complex(phases.cos(), phases.sin())


def bind(*xs: Tensor) -> Tensor:
    """FHRR bind: element-wise complex multiplication.

    Associative and commutative. Result remains unit-modulus iff all
    inputs are unit-modulus.
    """
    out = xs[0]
    for x in xs[1:]:
        out = out * x
    return out


def unbind(z: Tensor, key: Tensor) -> Tensor:
    """FHRR unbind: bind with the complex conjugate of the key."""
    return z * key.conj()


def superpose(xs: Tensor) -> Tensor:
    """Bundle several phasors into one vector (sum, then re-unitize).

    Note: re-unitizing a sum is lossy — the sum encodes superposition
    information that must be recovered by unbinding with one of the
    constituent keys. The unitize step keeps magnitudes bounded for
    downstream ops; recall is via cleanup against a codebook.
    """
    return unitize(xs.sum(dim=0))


def cleanup(query: Tensor, codebook: Tensor) -> tuple[Tensor, Tensor]:
    """Nearest-codebook cosine match.

    Returns (best_idx, best_score). Score is the real part of the
    Hermitian inner product, normalized by d. Compares each row of
    `query` (shape (..., d)) against each row of `codebook` (m, d).
    """
    # Hermitian inner product: <a, b> = sum a * conj(b)
    # Real part is what FHRR similarity actually measures.
    sims = (query.unsqueeze(-2) * codebook.conj()).sum(-1).real
    sims = sims / query.shape[-1]
    best_score, best_idx = sims.max(dim=-1)
    return best_idx, best_score
