# Rubik Gen — Validation Experiment 0: Permutation Locality

**Date**: 2026-05-09
**Status**: CONFIRMED — permutation locality holds

## Setup

- **Tokenizer**: TiTok BL-64 VQ-8K (ByteDance, Apache 2.0)
- **Codebook**: 8192 entries × 64 dims
- **Tokens per image**: 64 (1D representation of 256×256 image)
- **Test image**: "grokgen orange claude.png" (oranges on black background)
- **Method**: Encode image → 64 tokens, apply N random position swaps, decode, measure MSE vs original reconstruction

## Results

| Swaps | MSE vs reconstruction |
|---|---|
| 0 | 0.000 |
| 1 | 0.001 |
| 4 | 0.022 |
| 16 | 0.073 |
| 64 | 0.153 |
| 256 | 0.149 |

**100x MSE increase from 1 swap to 64 swaps.**

Visual inspection:
- 4 swaps: Same semantic content (oranges on black), local distortions
- 64 swaps: Completely different image (shoes/clothing) — different semantic content
- The decoder's token positions carry spatial/semantic structure

## Implications

1. **Permutation distance maps to visual distance** — the core Rubik gen premise holds
2. **Frozen pre-trained codebook works** — no need to train a VQ-VAE from scratch (Gap 1 bypassed)
3. **Next step**: Learn the permutation→codebook-index mapping using VSA binding
4. **Permutation-locality loss** (per DeepSeek recommendation) is still needed for smooth interpolation

## Architecture for next experiment

Per DeepSeek v4 Pro review:
- Freeze TiTok encoder+decoder
- Learn only: tile hypervectors (150 × D=8192) + projection head (MLP, ~500K params)
- Loss: reconstruction + permutation-locality
- Dataset: Moving CLEVR at 128×128 (synthetic, known rotations)
- Target: <1M trainable params, <2 hours training on Radeon 890M
