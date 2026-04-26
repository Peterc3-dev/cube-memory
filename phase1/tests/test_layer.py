"""Sanity checks for CubeMemoryLayer.

Verify:
1. Forward pass produces tensors of the right shape.
2. Backprop runs without NaN/inf and updates only the learnable params.
3. Frozen codebooks stay frozen.
4. Output is approximately zero on a zero-init out_proj (preserves
   residual on the first training step).
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, str(__file__.rsplit("/", 2)[0]))
from cube_memory_layer import CubeMemoryLayer


def main():
    torch.manual_seed(0)
    B, T, D = 2, 8, 128
    layer = CubeMemoryLayer(
        d_in=D, d_codebook=64, d_value=32, m=16, p=3, n_slots=512
    )
    x = torch.randn(B, T, D)

    # 1. Shape check.
    out = layer(x)
    assert out.shape == (B, T, D), f"shape mismatch: {out.shape}"

    # 2. Output on zero-init out_proj should be exactly zero.
    assert torch.allclose(out, torch.zeros_like(out)), \
        "out_proj is zero-init; output should be zero on first pass"

    # 3. Backprop. Loss = MSE against random target.
    target = torch.randn(B, T, D)
    # Detach the zeros and add a tiny perturbation so loss has gradient.
    layer.out_proj.weight.data.add_(torch.randn_like(layer.out_proj.weight) * 1e-2)
    out2 = layer(x)
    loss = (out2 - target).pow(2).mean()
    loss.backward()

    # All learnable params should have gradients; codebooks should not.
    learn = ["role_proj.weight", "slot_keys", "slot_values", "out_proj.weight"]
    frozen = [f"codebook_{i}" for i in range(layer.p)]
    for n, p in layer.named_parameters():
        assert n in learn, f"unexpected learnable param: {n}"
        assert p.grad is not None, f"no grad for {n}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {n}"
    for n, b in layer.named_buffers():
        assert n in frozen, f"unexpected buffer: {n}"
        assert not b.requires_grad, f"buffer {n} should be frozen"

    print(f"PASS  shape={out.shape}, learnable={learn}, frozen={frozen}")
    print(f"PASS  initial out norm = {out.norm().item():.4e}  (should be ~0)")
    print(f"PASS  loss after perturb = {loss.item():.4f}")


if __name__ == "__main__":
    main()
