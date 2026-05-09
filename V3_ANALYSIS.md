# Cube Memory V3 Analysis — Definitive Results

Date: 2026-05-09

## Summary

VSA-keyed memory layers do not work for FFN replacement. The finding is rigorous
and spans three experimental variants (V1, V2, V3 hybrid) across multiple
transformer layers.

## Experimental Matrix

### Per-architecture variance captured (layer 27, Qwen3.6-27B)

| Architecture | Var% | Params | Description |
|---|---|---|---|
| Zero baseline | 0.0% | 0 | Predict zero |
| Cube Memory V1 | ~5% | 35M | Frozen codebooks, STE |
| Cube Memory V2 | 4.8% | 35M | Learned codebooks, multi-head, Gumbel-softmax, gated residual |
| Rank-16 linear | 5.9% | 164K | Truncated SVD |
| Rank-64 linear | 8.2% | 655K | Truncated SVD |
| Rank-256 linear | 14.2% | 2.6M | Truncated SVD |
| Rank-1024 linear | 28.1% | 10.5M | Truncated SVD |
| **Rank-1024 + cube memory** | **28.5%** | **24M** | Staged: SVD init → train cube → joint |
| Rank-1024 + SGD fine-tune | 32.6% | 10.5M | Linear-only control |
| **Rank-1024 + cube + joint** | **32.6%** | **24M** | Identical to linear-only control |
| Rank-2048 linear | 36.5% | 21M | Truncated SVD |
| Rank-2048 + SGD | 36.6% | 21M | Marginal SGD gain over SVD |
| Full rank (5120) | 41.1% | 52M | Theoretical linear ceiling |

### Cross-layer rank sweep (SVD)

| Rank | Layer 3 | Layer 27 | Layer 43 |
|---|---|---|---|
| 256 | 13.6% | 14.2% | 13.0% |
| 512 | 17.4% | 20.1% | 18.3% |
| 1024 | 22.6% | 28.1% | 25.8% |
| 2048 | 28.0% | 36.5% | 34.1% |
| 5120 | 31.0% | 41.1% | 39.2% |

### Residual diagnostics (after rank-256 linear)

| Model on residual | L3 res% | L27 res% | L43 res% |
|---|---|---|---|
| MLP (5.2M params) | 0.0% | 6.0% | 6.0% |
| Memory Layer (23M, learned keys) | 3.7% | 6.3% | 5.4% |
| Cube Memory V1 (VSA) | ~0.8% | — | — |

## Root Cause

1. **FFN is nearly linear.** The full-rank linear ceiling is only 31-41%
   depending on layer depth, meaning 59-69% of FFN variance is genuinely
   non-linear or noise. But the linear component dominates what's learnable.

2. **VSA is a rank-4 bottleneck.** The cleanup→bind→retrieve pipeline with
   top_k=4 has an effective rank of 4. This is worse than any reasonable
   linear projection (rank-16 at 164K params already beats 35M-param VSA).

3. **The non-linear residual is small and hard.** After removing the linear
   component, only ~5-6% of the residual is learnable by ANY architecture
   (MLP, memory layer, or VSA). The rest is high-rank structure or noise.

4. **Memory layers don't outperform MLPs.** Meta-style learned-key memory
   layers (no VSA) perform identically to simple 2-layer MLPs on the residual,
   suggesting the non-linear structure doesn't benefit from sparse retrieval.

## What's Reusable

The cube memory infrastructure is solid and validated:
- ggml ops (CPU + Vulkan) — bit-identical to reference
- SPIR-V shaders — tiled multi-workgroup, 3.5x/1.9x speedups
- GGUF export pipeline — round-trip error 1.16e-10
- Parallelized CPU ops — thread-correct across 1/2/4 threads
- All on GitHub: Peterc3-dev/llama.cpp (cube-memory-op branch), Peterc3-dev/cube-memory

## Paths Forward

### A. Negative result paper (publishable as-is)
"VSA Memory Layers fail at FFN replacement: the FFN input→output mapping is
approximately linear (rank-1024 captures 28%), and the VSA pipeline is a
rank-4 bottleneck that destroys information. Standard linear projections
dominate at every parameter budget."

### B. Pivot: VSA for compositional generation (Paper 2)
The Rubik gen idea (~/projects/cube-memory/IDEAS_FUTURE_PAPER_2_RUBIK_GEN.md)
uses VSA where it shines: permutation-group-indexed compositional image
generation. This is a different design-space cell where discrete binding
matters.

### C. Pivot: VSA as attention routing
Use VSA for compositional key generation in attention layers rather than
FFN replacement. The discrete binding property matches attention's
key-query structure better than continuous FFN approximation.

### D. Low-rank linear FFN compression (not VSA, but uses our infra)
A rank-2048 linear captures 36.5% of layer 27 FFN at 21M params (vs 267M
for the full FFN). This is an 12.7x compression with 63.5% quality loss
per layer — potentially viable as a distillation target if the loss is
tolerable across the full model.

## DeepSeek v4 Pro Review

Confirmed independently:
- "VSA adds high distortion with no benefit for continuous approximation"
- "SVD-initialized linear handles 28% variance; fine-tuning adds 4%. VSA cube memory adds marginal 0.4%"
- "Rank-2048 linear at ~20M params already hits 36.5% — beating the hybrid with a simpler architecture"
- Recommended: if pursuing this further, try MLP (rank-256→512 via GELU) on top of linear for ~38-40% at same param count
