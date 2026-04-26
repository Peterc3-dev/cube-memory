# Cube Memory — VSA-keyed Memory Layers

A drop-in replacement for transformer FFN blocks where the lookup keys are
a structured Vector Symbolic Architecture (VSA) codebook instead of learnt
nearest-neighbor embeddings. Lookup by *orientation* (algebraic unbind),
not by similarity. Computation becomes navigation through a holographic
keyspace; storage is bandwidth-bounded but reads are sparse and predictable.

The "cube": a finite group of role vectors whose binding products span the
keyspace. Each retrieval is "rotate to face θ, read the snapshot." The
snapshot is a value vector returned by the unbind operation. The model
queries by composing role vectors instead of computing dot-products against
all keys.

## Problem

Dense transformer FFN blocks dominate per-token weight bandwidth.

Verified per-token active read on Qwen3-30B-A3B-Q4_K_M (HF config + arithmetic):

| Component | Active params | MiB / token |
|---|---|---|
| Attention QKVO × 48 layers | 906 M | 510 |
| FFN gate+up+down × 8 experts × 48 | 1 812 M | 1 019 |
| Router gating × 48 | 13 M | 7 |
| LM head (Q4_K_M) | 311 M | 175 |
| Norms / embedding lookup | 0.2 M | <1 |
| **Per-token active total** | **~3.0 B** | **~1 712 MiB ≈ 1.67 GiB** |

(LM head is often kept FP16 in real GGUFs — pushes total to ~2.13 GiB,
FFN share down to ~47%.)

**FFN share of active read = ~60%.**  At LPDDR5X-7500
(~120 GB/s, Strix Point) the bandwidth floor is ~1.67 GiB ÷ 120 GB/s ≈
70 t/s.  Real-world TG is ~38 t/s — half of that ceiling — and software
optimization within the same dense-FFN architecture has run out of room
(see `~/projects/llama.cpp/MOE_BENCH.md`).

Two known escapes that already work in production:

1. **MoE routing** — only fire k of N experts per token (Qwen3 MoE).
2. **Memory Layers at Scale** (Meta, 2024, arXiv 2412.09764) — replace
   FFN with a learnt key-value lookup, top-k gather only.  10× parameter
   efficiency for factual recall.  Same compute shape as MUL_MAT_ID.

Memory Layers are the right *shape* — sparse gather, top-k retrieval,
bandwidth-bounded read of only the selected slots.  Their keys are
unstructured but PEER (He, 2024, arXiv 2407.04153) already factors them
into product-key form to get sub-linear routing.  So **routing speed is
not the contribution gap** — that's been done.

## The contribution

Replace the unstructured key matrix `K ∈ ℝ^(N×d)` with a *structured
codebook*: `p` finite sets of role vectors `{r¹₁ … r¹_m}, …, {rᵖ₁ … rᵖ_m}`
and a VSA binding operation `⊗`, such that each key is

    k_{i₁, …, iₚ} = r¹_{i₁} ⊗ r²_{i₂} ⊗ … ⊗ rᵖ_{iₚ}

The codebook itself is `p · m · d` floats.  The *slot store* (the
values actually retrieved) is still `N · d_v`; only the *key* matrix
is compressed.

The novelty over PEER is **not** sub-linear routing (PEER already does
that with product keys).  It is the *algebra*:

- **Superposition.** A single hidden vector can encode multiple
  role-bound key-value pairs and `unbind(h, role_axis)` projects out
  the filler for that role — useful for representing multiple
  simultaneous "facts" in one residual update.
- **Compositionality.** Bind is associative and commutative (for HRR /
  FHRR), so `bind(a, b, c)` decomposes the same way regardless of
  query order.  Storage is order-invariant.
- **Cleanup-as-typing.** Each unbind output can be projected back onto
  the role-axis codebook (cleanup memory) before composition,
  *typing* the intermediate result.

PEER's product key gives factored *search*; VSA bind gives factored
*semantics*.  This matters only if downstream layers exploit
superposition or compositional retrieval — that's the live
question for Phase 0.

Bind/unbind options:

- **HRR (Plate 1995)** — circular convolution + correlation.  Real
  vectors.  Needs FFT for fast bind, awkward on Vulkan compute.
- **FHRR** — element-wise multiplication of unit-modulus complex
  phasors.  Bind is O(d), no FFT.  Backprop requires a
  unit-modulus projection step every forward pass (Alam et al.
  2021, arXiv 2109.02157) — without it, magnitude drift causes
  vanishing/exploding gradients.  **Default choice.**
- **MAP / TPR** — element-wise real multiplication.  Cheapest, but
  noisier cleanup at depth.

All three preserve the identity

    unbind(bind(a, b), a) ≈ b + noise

with bounded noise that grows with bind depth `p` and superposition
count.  Cleanup via cosine match against role-axis codebook entries.

## Why this is the cube

The "9-square faces, snapshot-from-angle" intuition maps cleanly
(as analogy, not literal group theory):

- **Faces** = role-axis codebooks.  A small codebook (m squares per
  axis) covers a manageable composition space.
- **Snapshot from an angle** = a query that picks one role per axis
  and looks up the bound product.  The "angle" is the index choice
  per role-axis; the "image" is the value vector returned by the
  slot lookup.
- **Composing faces** = the bind operation.  HRR/FHRR bind is
  commutative and associative — composition is order-invariant.
- **Rubik analogy** is decoration.  My construction is a free product
  over finite codebooks, not a group action.  The compactness
  intuition (small codebook generates a large reachable space)
  carries; the literal group structure does not.

Existing literature dances around this:

- HRR / Tensor Product Representations (Smolensky, Plate) — the algebra.
- Hrrformer (KDD 2022) — replaced self-attention with circular convolution,
  370× faster training, near-SOTA on Long Range Arena.
- *Attention as Binding: A Vector-Symbolic Perspective* (arXiv 2512.14709,
  Dec 2025) — formalized attention itself as VSA bind/unbind.
- Memory Layers at Scale (Meta 2024) — the engineering substrate, but
  unstructured keys.

**Nobody has wired a structured-VSA codebook as the keys of a Memory
Layer.**  That's the gap.

## Architecture

A `CubeMemoryLayer` is a drop-in replacement for a transformer FFN.

    Input: hidden state h ∈ ℝ^d
    Output: residual update Δh ∈ ℝ^d

    1.  Split h into p query-role vectors q₁, …, qₚ via a small
        learnt projection.
    2.  For each q_j, find the closest role vector r_{i_j} in the
        codebook (cosine similarity over m vectors — cheap, m « N).
    3.  Compose the address: addr = bind(r_{i₁}, …, r_{iₚ}).
    4.  Look up the value v_addr in the slot store (sparse gather,
        same shape as MUL_MAT_ID).
    5.  Return W_out · v_addr.

Storage: m·d (codebook) + N·d_v (slots).  Compute per token: p·m
similarity scores + p binds + 1 gather + 1 matmul.  Bandwidth per token:
m·d + d_v (one slot only).  Independent of N.

For Qwen3-30B-A3B-class scale (initial target — final dimensions set
empirically in Phase 0):

- p = 3 role axes
- m = 256 per axis  →  N = mᵖ = 16.7 M reachable slots
- d = 1024 (key/codebook), d_v = 512 (slot value), top-k = 1
- **Codebook is shared across all 48 layers.**  Per-layer-private
  codebooks would 48× the per-token bandwidth and kill the win.
- Codebook bytes (FP16, shared): 3·256·1024·2 = **1.5 MiB total**
- Slot store bytes per layer (Q4_K_M):
  16.7 M · 512 · 0.5625 ≈ **4.8 GiB per layer**.  At 48 layers this
  is 230 GiB and is **untenable** — see "Risks" for mitigation.
  The realistic config drops to `m = 64`, `N = 262 K`, slot store
  ~75 MiB / layer, per-layer-private allowed (~3.6 GiB total).
- Per-token active read at the cube layer:
  codebook 1.5 MiB (read once, cached after first token) +
  one slot per layer × 48 ≈ ~30 KiB total slot bytes.

**Whole-model bandwidth, shared codebook, all 48 FFN replaced:**

  before: 1019 MiB FFN + 510 attn + 175 LM head ≈ 1712 MiB / token
  after:   ~1.5 MiB cube codebook + ~30 KiB slots
           + 510 attn + 175 LM head ≈ 686 MiB / token
  speedup ceiling: 1712 / 686 ≈ **2.5× whole-model TG**

If only 50% of FFN blocks are replaced: ~1.4× whole-model TG.  If
LM head stays FP16 (real GGUFs do this), the denominator grows by
~420 MiB and the ceiling drops to ~1.9× / ~1.3×.  All ceilings,
not deliveries.

## Training mechanics (verified against literature)

What the architecture trains:

- **Frozen** — role-axis codebooks (random orthogonal init).  Plate's
  recall guarantees and LARS-VSA (arXiv 2405.14436) both show enforced
  orthogonality generalizes better.  Training the codebook tends to
  collapse the algebra.
- **Learned** — slot values (`N · d_v`), the role-projection MLP
  (hidden state → p query-roles), the output projection.

Mandatory training tricks:

1. **Unit-modulus projection every forward pass** (FHRR).  Without
   this, magnitude drift causes vanishing/exploding gradients
   (Alam et al. 2021, arXiv 2109.02157, "Learning with HRRs").  The
   same paper reports >100× retrieval improvement once this is added.
2. **qk-style normalization on role queries** before bind.  Memory
   Layers at Scale paper (arXiv 2412.09764) needs this for stability,
   especially small base models.
3. **Cleanup memory between bind levels** when p ≥ 3.  Each unbind
   amplifies noise variance ≈ p · N / d.  Cleanup = nearest-codebook
   projection re-snaps to the role-axis manifold and bounds drift.
4. **Sparse-memory finetuning** (Lin et al. 2025, arXiv 2510.15103)
   — the slot store is the right grain for selective updates if we
   later distill from a frozen base model.

What is *not* known (open questions Phase 0 must answer):

- Operating-point recall.  No published curve at (d=1024, m=256, p=3)
  for HRR/FHRR.  Schlegel et al. 2020 (arXiv 2001.11797) cover
  d=1000-class regimes for bundling but not exact bind-depth-3 with
  cleanup at every level.
- Backprop-through-bind at scale.  Hrrformer (Alam ICML 2023) is the
  largest published end-to-end-trained HRR model and tops out at
  Long-Range-Arena scale (tens of M params).  **No transformer-class
  (≥100 M) HRR/FHRR-keyed model has been published.**  This proposal
  is the first.

## Implementation plan

Phase 0 (week 1) — **prototype in PyTorch on a tiny model.**
 - 6-layer toy transformer, ~50M params, dense FFN baseline
 - Replace one FFN with a CubeMemoryLayer using HRR binding
 - Train on a synthetic associative-recall task
 - Verify unbind retrieves the right slot above noise floor

Phase 1 (weeks 2-4) — **distill from a small open model.**
 - Take a Qwen3-1.7B or similar base
 - Replace 25% of FFN blocks with CubeMemoryLayers
 - Distill from the original model on a curated corpus
 - Measure: perplexity, BFCL tool-calling, latency on Vulkan

Phase 2 (months 2-3) — **GGUF support + llama.cpp integration.**
 - Define a `CUBE_MEMORY` ggml op that wraps the unbind+gather
 - Add GGUF tensor type for codebook + slot store
 - Reuse `MUL_MAT_ID` Vulkan path for the gather (already optimized)
 - Bench on Strix Point Radeon 890M

Phase 3 — **publish.**
 - Paper: "Cube Memory: Structured VSA codebooks as keys for Memory
   Layers."  Compare against Memory Layers (unstructured), MoE FFN,
   dense FFN.  Headline metric: bandwidth/token vs perplexity.
 - PR to llama.cpp adding `CUBE_MEMORY` op support.

## Validation

The system is right if:

1. **Algebraic recall** — unbind retrieves the correct slot above a
   noise threshold ≥ 95% on synthetic data with N ≥ 10⁶ reachable keys.
2. **Bandwidth** — measured DRAM bytes/token at the cube layer is
   ≥ 100× lower than the FFN it replaces, on Strix Point Vulkan.
3. **Quality** — distilled model perplexity within **10%** of the
   source on held-out text (Memory Layers at Scale itself shows ~5%
   regressions in some configs at small scale, so 5% is unrealistic
   for v1).  BFCL tool-calling within 5 pp of source.
4. **Latency** — TG t/s on Strix Point **≥ 1.3× the source baseline**
   after replacing 50% of FFN, **≥ 2.0×** after replacing all 48 FFN
   blocks (corrected from the earlier `≥1.5×`/`≥2.5×` numbers — see
   the bandwidth math in *Architecture*; ceiling depends on whether
   LM head is FP16 or quantized in the GGUF).

If (1)+(2) hold but (3) fails, the algebraic structure is too rigid
for learnt content — relax to a hybrid where some slots are addressed
algebraically and others by similarity.  If (3)+(4) hold but the
ablation shows the structure didn't matter, it's just Memory Layers
with extra steps and we cite Meta and ship that.

## Risks

- **Slot-store storage explosion.**  Naïve `m=256, p=3, d_v=512` ≈
  4.8 GiB per layer × 48 layers = 230 GiB.  Untenable.  Mitigations
  (in order of preference): (a) shared slot store across all layers
  (Memory Layers at Scale does this), (b) reduce m or d_v, (c) only
  replace a fraction of FFN blocks, (d) Q2_K-class slot quantization.
  The 75 MiB / layer config in *Architecture* already applies (b).
- **Codebook collapse during training** if codebook is learnable.
  Mitigate by freezing as random orthogonal — the default config.
- **Binding noise at depth.**  Each unbind amplifies noise variance
  ~p · N / d.  Mitigate: cleanup at every level, hard cap p ≤ 3.
- **No published HRR/FHRR-keyed model at ≥100 M params.** This is
  legitimately uncharted; the existence proof is part of the
  contribution.  Phase 0 toy must validate the operating point
  before committing Phase 1 distillation budget.
- **Tool-calling gap.**  Structural representations historically lag
  on long-range structural tasks.  Track BFCL from Phase 1 onward;
  abandon if the gap exceeds 10 pp at fixed perplexity.
- **Vulkan primitive cost.**  FHRR bind = element-wise complex
  multiplication.  Not a stock Vulkan op but trivial as a custom
  compute shader (a few dozen lines).  HRR via FFT on Vulkan is
  doable but adds cost — prefer FHRR.
- **Distillation budget for Phase 1.**  Replacing 25% of FFN in a
  1.7B base requires hundreds of GPU-hours of distillation.  Strix
  Point is not the right hardware for this — Phase 1 runs on a
  rented dGPU node.  Strix Point only re-enters at Phase 2 (inference
  bench) where it is the actual deployment target.
- **KV-cache compatibility.**  Cube Memory replaces FFN, not
  attention; KV cache untouched.  Should compose cleanly but verify
  in Phase 2 by running a long-context eval.
- **Inference quantization break.**  Q4_K_M slot store may degrade
  unbind cleanup margin below the noise floor.  Phase 0 must include
  a quantization-error sweep on the toy model.

## References

- Plate (1995) — Holographic Reduced Representations
- Kanerva (2009) — Hyperdimensional computing introduction
- Smolensky (1990) — Tensor product representations
- Berges et al. (Meta, 2024) — Memory Layers at Scale, arXiv 2412.09764
- He (DeepMind, 2024) — Mixture of a Million Experts, arXiv 2407.04153
- Hrrformer (KDD 2022) — self-attention as HRR convolution
- *Attention as Binding* (arXiv 2512.14709, Dec 2025)
- LUT-NN (arXiv 2302.03213) — lookup-replaces-matmul prior art
- BitNet b1.58 — ternary weights, the production-grade lookup-shaped LLM

## What this is *not*

- A new attention mechanism.  Attention stays.
- A claim that VSA replaces transformers.  It replaces the FFN block.
- Promising 100× speedup on the whole model.  FFN is ~60% of the
  per-token bandwidth (verified from Qwen3-30B-A3B-Q4_K_M config);
  fully replacing it ceilings whole-model TG at ~2.5×, partial
  replacement is proportionally smaller.

## Peer-review notes (2026-04-25 revision)

This doc was reviewed by two independent research subagents.  Their
findings drove the corrections above:

- The original "17 GiB read per token" was wrong — that is the on-disk
  Q4_K_M model size.  Per-token *active* read is ~1.7 GiB and FFN is
  ~60% of that.  All bandwidth math has been recomputed.
- The original PEER differentiation (sub-linear routing) was a non-
  contribution — PEER already does that.  Reframed around superposition
  and compositionality, which PEER does not provide.
- Training tractability: backprop through HRR/FHRR works at ≤100M
  scale (Hrrformer 2023); no published ≥100M result.  Mandatory
  unit-modulus projection added to *Training mechanics*.
- Validation thresholds softened: perplexity ≤10% (was 5%, unrealistic),
  TG ≥1.3× at 50% replacement (was ≥1.5×, off by the corrected
  bandwidth math).
- Storage explosion (4.8 GiB / layer × 48 = 230 GiB) is now an
  explicit Risk with a primary mitigation (shared slot store, per the
  Memory Layers at Scale paper).

## Status

Drafted 2026-04-25.  Revision 2026-04-25 after sub-agent peer review.
Phase 0 prototype next.  Repository: `~/projects/cube-memory/`.
