"""Tests for swap_ffn_modules using a stub HF-style model.

The stub mimics the surface area of an AutoModelForCausalLM: a top-level
module with a `.model` attribute exposing a `.layers` ModuleList of
decoder blocks, each with a `.mlp` attribute. No transformers dep.
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(__file__.rsplit("/", 2)[0]))

from cube_memory_layer import CubeMemoryLayer
from swap_ffn import swap_ffn_modules


class StubBlock(nn.Module):
    """Tiny decoder block: just an mlp (Linear). Identity on everything else."""

    def __init__(self, d_in: int):
        super().__init__()
        self.mlp = nn.Linear(d_in, d_in, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class StubInner(nn.Module):
    def __init__(self, d_in: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([StubBlock(d_in) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class StubConfig:
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size


class StubModel(nn.Module):
    def __init__(self, d_in: int, n_layers: int):
        super().__init__()
        self.config = StubConfig(d_in)
        self.model = StubInner(d_in, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def test_basic_swap():
    torch.manual_seed(0)
    D, N = 64, 8
    model = StubModel(d_in=D, n_layers=N)

    cube_kwargs = {
        "d_codebook": 64,
        "d_value": 32,
        "m": 16,
        "p": 3,
        "n_slots": 256,
    }
    model, replaced = swap_ffn_modules(model, fraction=0.25, cube_kwargs=cube_kwargs)

    # 25% of 8 = 2 indices.
    assert len(replaced) == 2, f"expected 2 replaced indices, got {replaced}"

    # Forward pass shape unchanged.
    B, T = 2, 4
    x = torch.randn(B, T, D)
    out = model(x)
    assert out.shape == (B, T, D), f"shape mismatch: {out.shape}"

    # Swapped layers carry CubeMemoryLayer; others still have Linear.
    for i, layer in enumerate(model.model.layers):
        if i in replaced:
            assert isinstance(layer.mlp, CubeMemoryLayer), (
                f"layer {i} should be CubeMemoryLayer, got {type(layer.mlp).__name__}"
            )
        else:
            assert isinstance(layer.mlp, nn.Linear), (
                f"layer {i} should be unchanged Linear, got {type(layer.mlp).__name__}"
            )

    print(f"PASS  swapped indices = {replaced}")
    print(f"PASS  output shape preserved = {tuple(out.shape)}")


def test_explicit_indices():
    torch.manual_seed(0)
    D, N = 32, 6
    model = StubModel(d_in=D, n_layers=N)
    cube_kwargs = {"d_codebook": 32, "d_value": 16, "m": 8, "p": 3, "n_slots": 64}
    _, replaced = swap_ffn_modules(
        model, layer_indices=[1, 3, 5], cube_kwargs=cube_kwargs
    )
    assert replaced == [1, 3, 5], replaced
    print(f"PASS  explicit indices honored = {replaced}")


def test_moe_skip():
    """A stub MoE block named to trigger the MoE-skip heuristic should
    NOT be replaced even when its index is requested."""
    torch.manual_seed(0)
    D, N = 32, 4
    model = StubModel(d_in=D, n_layers=N)

    # Replace layer 2's mlp with a class whose name hits _MOE_NAME_HINTS.
    class Qwen3MoeSparseMoeBlock(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.gate = nn.Linear(d, d, bias=False)

        def forward(self, x):
            return self.gate(x)

    model.model.layers[2].mlp = Qwen3MoeSparseMoeBlock(D)

    cube_kwargs = {"d_codebook": 32, "d_value": 16, "m": 8, "p": 3, "n_slots": 64}
    _, replaced = swap_ffn_modules(
        model, layer_indices=[1, 2, 3], cube_kwargs=cube_kwargs
    )
    assert replaced == [1, 3], f"MoE block at idx 2 should be skipped, got {replaced}"
    assert isinstance(model.model.layers[2].mlp, Qwen3MoeSparseMoeBlock)
    print(f"PASS  MoE block skipped, replaced = {replaced}")


def main():
    test_basic_swap()
    test_explicit_indices()
    test_moe_skip()
    print("ALL PASS")


if __name__ == "__main__":
    main()
