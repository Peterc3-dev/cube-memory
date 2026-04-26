# Phase 1 — Distillation prep

Scope: build everything needed to distill **one** FFN-replaced Qwen3.6
variant on rented compute, then bench it on Strix Point. Local
hardware too small (16 GiB iGPU) to run the actual distill — local
work is sanity-only.

## What's done

- `cube_memory_layer.py` — `CubeMemoryLayer`, drop-in FFN replacement.
  Frozen orthogonal FHRR codebooks, learnable role-projection,
  slot-keys, slot-values, out-projection. Top-k soft retrieval (k=4
  default). Straight-through estimator on the codebook cleanup so
  gradients flow to `role_proj`. Zero-init `out_proj` so the layer
  starts as the identity residual.
- `tests/test_layer.py` — verifies shape, gradient flow to all
  learnable params, frozen buffers stay frozen, zero-init output.
  Passes.

## What's next (this Phase 1)

1. **`swap_ffn.py`** — utility to take an HF transformers Qwen3-class
   model, replace `model.layers[i].mlp` with `CubeMemoryLayer` for
   selected `i`. Mirror the FFN's `d_in`. Default: replace 25% of
   layers, evenly spaced. Returns the modified model + a list of
   replaced layer indices.

2. **`distill.py`** — knowledge distillation training loop.
   - Teacher: frozen original Qwen3.6-35B-A3B (no swap).
   - Student: swapped variant (with CubeMemoryLayer in 25% of layers).
   - Loss: KL divergence between teacher and student logits on a
     curated corpus, plus task-specific tool-call eval set.
   - Optimizer: AdamW with cosine schedule. LR 1e-4 for the new
     params, 1e-5 for any other unfrozen layers (we may want to
     unfreeze the layer norms around swapped layers).
   - Mandatory: unit-modulus projection step (already in layer fwd)
     and qk-style normalization on role queries.

3. **`eval.py`** — held-out perplexity on 10k tokens of curated
   text + the existing 20-case tool-call eval at `~/projects/llama.cpp/
   moe-bench/eval_tools.py`. Both run against the local server with
   the swapped model loaded.

4. **GGUF export** — for the Vulkan bench on Strix Point we need a
   GGUF that includes the new layer's weights and signals the new
   op (`CUBE_MEMORY` if we add one) or fits inside `MUL_MAT_ID`'s
   shape if we route the slot-keys/values gather through it. Phase 2
   work; flag as a dependency.

## Rented compute plan

**Goal:** Phase 1 produces a single distilled checkpoint of
Qwen3.6-35B-A3B with 12 of its 48 FFN blocks replaced by
CubeMemoryLayers, trained on ~2-4 B tokens of FineWeb. Measure:
PPL drift, BFCL drift, latency on a target dGPU.

### Minimum viable run

- **Hardware:** 1× H100 80GB (Lambda or RunPod) — fits the model in
  bf16 with batch size 1 + gradient accumulation. ~$2.50–3.00/hr.
- **Tokens:** 2 B at batch 8 × 4096 ctx ≈ 60K steps. ~12-20 hours.
- **Cost:** ~$30–60 for one variant.
- **Data:** FineWeb-Edu sample (open, on HF). Tokenize once with
  Qwen3.6 tokenizer, cache as packed `.bin`.

### Stretch run (if v1 is promising)

- 4 B tokens, full 25% replacement (12 of 48 layers), 100K steps
  with cosine LR. ~30 hours, ~$100.
- Same layer set distilled at multiple seeds for variance estimates.
- Adds: ablation runs swapping 12.5% (6 layers) and 50% (24 layers)
  to map the bandwidth-vs-quality curve.

### Pre-flight checklist before spending GPU time

- [ ] `swap_ffn.py` works on a tiny Qwen3-0.5B locally.
- [ ] `distill.py` runs 100 steps locally on tiny Qwen3-0.5B without
  NaN, with PPL trending down, on a CPU-only or Vulkan ROCm path.
- [ ] Tokenized data cached, sharded for multi-GPU if needed.
- [ ] Eval harness produces a score on the unmodified base for the
  reference number.
- [ ] Checkpoint serialization round-trips: train → save → load →
  same loss.

## Why Qwen3.6-35B-A3B as the base

From `~/projects/cube-memory/baselines/qwen36-vulkan-profile.md` we
know:

- Qwen3.6 is the forward shape (DeltaNet, 256 experts, MTP).
- FFN-block bandwidth share is ~60% of per-token active read.
- Vulkan inference will get a separate ~30% TG win when the
  `n_expert` dispatch fix lands. Cube Memory's bandwidth advantage
  has to be measured **after** that fix to avoid double-counting
  the scheduler win.

Distilling on Qwen3-30B-A3B would mean re-doing this on 3.6 in
3 months. Doing it once on 3.6 is the cheaper plan.

## Risks logged for Phase 2

- The Vulkan side of CubeMemoryLayer needs a custom op (or a clever
  reuse of `MUL_MAT_ID`). The slot-keys gather is the close cousin
  of a sparse top-k attention; the slot-values weighted sum is just
  an additional matmul. Both can fit in the existing op vocabulary
  with the right tensor type registration.
- FHRR bind = element-wise complex multiplication. On Vulkan compute,
  trivial. The complex storage (real || imag) doubles the codebook
  memory but we already accounted for that.
- The `n_expert` dispatch fix may land before Phase 1 results are
  in, in which case we re-bench against the corrected baseline
  before claiming any speedup attributable to Cube Memory itself.
