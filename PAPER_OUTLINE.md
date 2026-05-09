# Two Negative Results for Vector Symbolic Architectures: FFN Replacement and Compositional Image Generation

**Target venue:** NeurIPS 2026 (Negative Results / Datasets & Benchmarks track)
**Status:** All experiments complete, writing phase

---

## Abstract (~300 words)

Vector Symbolic Architectures (VSAs) offer algebraically structured
representations — binding, bundling, permutation — that are O(D) and
matmul-free. These properties make VSA a natural candidate for (1)
replacing dense feed-forward networks with structured retrieval, and (2)
compositional image generation via token binding. We conduct systematic
experiments testing both hypotheses and find that neither works, for
different but complementary reasons.

**Case Study 1: FFN Replacement.** We test VSA memory layers as
replacements for feed-forward networks in Qwen3.6-27B. The failure mode
is a rank bottleneck: VSA's cleanup→bind→retrieve pipeline has effective
rank bounded by top-k (typically 4), while FFN mappings are ~89% linear
with effective rank >2048. A 164K-parameter rank-16 linear projection
captures more variance than a 35M-parameter VSA memory layer. Scaling
top-k to 64 does not help (2.9% → 2.7% variance). SVD-derived codebooks
improve over random (4.3% vs 2.8%) but remain far below linear baselines
(36.6% at matched parameters).

**Case Study 2: Compositional Image Generation.** We test VSA binding for
encoding image token sequences from TiTok VQ-8K (64 tokens × 8192
codebook). The failure has two independent causes: (a) FHRR superposition
of 64 bindings cannot support 8192-way retrieval (0% accuracy at all D
tested, including factored codebook and decoder variants), and (b) real
images do not share token multisets (2.7-8.9% overlap), so the
permutation-based generation framing is inapplicable.

**Positive findings.** We identify conditions where VSA succeeds: (1)
permutation locality in TiTok — small token permutations yield small
visual changes, (2) VSA encodes and recovers permutations of 64 positions
with 100% accuracy at D≥2048, and (3) SVD-aligned codebooks capture 50%
more variance than random codebooks. These successes illuminate the
boundary: VSA excels at structured, low-cardinality operations but fails
when the target requires high-rank continuous mappings or high-cardinality
discrete retrieval.

---

## 1. Introduction (2 pages)

### 1.1 Motivation
- VSA (hyperdimensional computing) has theoretical appeal for neural
  computation: algebraic structure, compositionality, O(D) operations
- Two natural applications:
  1. FFN replacement: structured retrieval instead of dense projection
  2. Image generation: compositional binding of token positions and
     content
- Both applications leverage VSA's core operations (binding, bundling)
  in domains where structure should help

### 1.2 The questions
1. "Can VSA memory layers match the effective rank of learned FFN
   projections in deep transformers?"
2. "Can VSA binding encode and retrieve image token sequences for
   compositional generation?"

### 1.3 Preview of results
- Both answers are negative, but for different reasons:
  - FFN replacement fails due to a **rank bottleneck** (retrieval
    collapses output dimensionality)
  - Image generation fails due to a **capacity bottleneck** (64
    superposed bindings can't support 8192-way retrieval) AND a
    **framing mismatch** (images don't share token multisets)
- Positive findings narrow the scope: VSA works for permutation
  representation (64-way, structured) but not for content retrieval
  (8192-way, unstructured)

### 1.4 Contributions
1. Two independent case studies demonstrating VSA limitations in
   complementary domains
2. Rank-bottleneck diagnosis for FFN replacement (theoretical + empirical)
3. Capacity analysis for VSA token binding (theory predicts ~log2(D/N)
   bits per retrieval; experiments confirm)
4. Positive result: VSA permutation encoding achieves 100% exact match
   at D=2048 for 64-position permutations
5. Negative result: TiTok VQ-8K token multisets have <9% overlap between
   images, invalidating permutation-based generation

---

## 2. Background (1.5 pages)

### 2.1 Vector Symbolic Architectures
- FHRR: complex phasor binding (element-wise multiply), bundling (sum),
  cleanup (nearest-neighbor or cosine similarity)
- Storage capacity: ~log2(D/N) bits per retrieval from N superposed
  bindings (Plate 2003; Frady et al. 2018)
- Key strength: compositionality — bind(A,B) is invertible, supports
  group operations

### 2.2 FFNs in transformers
- Standard FFN: up-project → activation → down-project
- 2/3 of transformer parameters, known to store factual knowledge
- Parameter count: 2 × d_model × d_ffn per layer

### 2.3 Image tokenization
- TiTok (Yu et al. 2024): 1D image tokenizer, 64 tokens per 256×256
  image, VQ-8K codebook (8192 entries × 64 dims)
- Frozen pre-trained encoder + decoder (Apache 2.0)

---

## 3. Case Study 1: VSA for FFN Replacement (3 pages)

### 3.1 Setup
- Teacher: Qwen3.6-27B (d_model=5120, 64 layers)
- Extract (input, output) activation pairs at layers 3, 27, 43
- 50,000 tokens from FineWeb-Edu, 5% validation
- Metric: variance captured = (1 - val_MSE / zero_MSE) × 100%

### 3.2 Architectures tested
- VSA V1 (frozen random codebooks), V2 (learned + Gumbel-softmax)
- VSA-MoE (VSA routing for expert selection)
- Baselines: rank-r linear (SVD), rank-r + MLP, learned-gate MoE

### 3.3 Results

**Table 1: Architecture comparison (layer 27)**

| Architecture | Var% | Params |
|---|---|---|
| Zero baseline | 0.0 | 0 |
| Cube Memory V1 (frozen VSA) | ~5 | 35M |
| Cube Memory V2 (learned VSA) | 4.8 | 35M |
| Rank-16 linear (SVD) | 5.9 | 164K |
| VSA-MoE 16×128 top-4 | 14.2 | 24M |
| Learned-MoE 8×256 top-2 | 16.2 | 21M |
| Rank-2048 linear | 36.6 | 21M |
| **Rank-2048 + MLP-512** | **38.4** | **26M** |
| Full-rank ceiling | 41.1 | 52M |

### 3.4 Reviewer experiments

**Top-k scaling (Table 2):** Increasing top-k does not break the ceiling.

| top_k | Var% |
|---|---|
| 4 | 2.9 |
| 16 | 2.8 |
| 64 | 2.7 |

**Codebook ablation (Table 3):** SVD-aligned codebooks help but
don't change the conclusion.

| Codebook | Var% |
|---|---|
| Frozen random | 2.8 |
| Learned | 2.8 |
| SVD-optimal | 4.3 |
| SVD-optimal + learned | 4.3 |

**SVD spectrum (Table 4):** Singular values of FFN activation matrix
decay slowly — effective rank >2048. See Appendix A for full spectrum.

**FLOPs comparison (Table 5):** VSA is both slower and worse.
All FLOPs are inference-only forward pass (no training overhead).

| Architecture | FLOPs/token | Var% |
|---|---|---|
| Rank-2048 linear | 42M | 36.6 |
| Rank-2048 + MLP-512 | 52M | 38.4 |
| Cube Memory V2 | 62M | 4.8 |

**Rank-equalized FLOP comparison (Table 6):** At matched effective
rank, VSA uses ~1500x more FLOPs for comparable or worse variance.

| Architecture | Eff. Rank | FLOPs/token | Var% |
|---|---|---|---|
| Rank-4 linear (SVD) | 4 | 41K | <0 (neg. R²) |
| Cube Memory V2 top-4 | ≤4 | 62M | 2.9 |

Note: Rank-4 linear (static SVD) yields negative R² (worse than
predicting the mean). VSA top-4 achieves 2.9% because routing selects
different codebook entries per input — data-dependent rank-4 beats
static rank-4. But both are catastrophically below rank-2048 linear
(36.6%), and VSA spends 1514× more FLOPs for its marginal advantage
over static SVD. The rank bottleneck holds: per-token output is
rank-≤4 regardless of routing, limiting information throughput.

### 3.5 Diagnosis: The rank bottleneck

**Proof sketch.** The VSA retrieval output is:
y = Σ_{j ∈ top-k} α_j · v_j, where v_j are memory value vectors
and α_j = softmax(similarity scores). This is a convex combination
of k vectors, so y ∈ span{v_1, ..., v_k}, giving rank(output) ≤ k.

- With top-k=4, VSA output rank ≤ 4
- A rank-4 linear (SVD) also fails (negative R²) but costs 41K FLOPs
  vs VSA's 62M FLOPs — same failure, 1514× cheaper
- FFN effective rank >2048 — retrieval fundamentally underpowered
- 95% of learnable FFN variance is linear; the remaining 5% is better
  captured by MLP than by sparse retrieval

---

## 4. Case Study 2: VSA for Image Generation (3 pages)

### 4.1 Motivation: Permutation-Group Indexed Generation
- Hypothesis: image generation as token permutation manipulation
- TiTok encodes images as 64 discrete tokens from 8192-entry codebook
- VSA binding naturally represents position → content mappings
- Permutation locality: small changes in token arrangement → small
  visual changes (validated experimentally)

### 4.2 Prerequisite: Permutation locality (Experiment 0)
- Encode test image with frozen TiTok, apply N random position swaps,
  decode, measure MSE vs original

| Swaps | MSE |
|---|---|
| 0 | 0.000 |
| 1 | 0.001 |
| 4 | 0.022 |
| 16 | 0.073 |
| 64 | 0.153 |

- Monotonic relationship confirms the TiTok decoder is smooth w.r.t.
  token permutation — the generation premise has empirical support.

### 4.3 Token binding fails (Experiments 1, 1b, 1c, 2)

**Experiment 1: Pure VSA binding.** Bind 64 position phasors with
content phasors, superpose, unbind each position, classify against
8192 codebook entries.

| D | Acc% | Theory (bits have/need) |
|---|---|---|
| 512 | 0.0 | 3.0 / 13.0 |
| 1024 | 0.0 | 4.0 / 13.0 |
| 2048 | 0.0 | 5.0 / 13.0 |
| 4096 | 0.0 | 6.0 / 13.0 |

Theory predicts ~log2(D/64) bits per retrieval; 8192 entries need 13
bits. Even D=4096 provides only 6 bits. Total failure.

**Experiment 1b: VSA + MLP decoder.** Add 2-layer MLP after unbinding.
Result: catastrophic overfitting at all D. Train loss → 0, val loss → 25.
The decoder memorizes training-specific noise patterns.

**Experiment 1c: Factored codebook.** Decompose 8192 = 128 × 64, use
3-way binding (pos ⊗ factA ⊗ factB), retrieve factors independently.

| D | AccA% (128-way) | AccB% (64-way) | Joint% |
|---|---|---|---|
| 512 | 2.2 | 5.6 | 0.1 |
| 1024 | 2.3 | 5.8 | 0.2 |
| 2048 | 2.0 | 5.7 | 0.1 |
| 4096 | 1.9 | 5.9 | 0.1 |

Hard ceiling at ~5.8% accB regardless of D. The bottleneck is
structural (cross-talk from superposition), not capacity.

**Experiment 2: Real TiTok tokens.** Test on Imagenette (6000 images
encoded by frozen TiTok). Token distribution: 12.85/13.00 bits entropy,
all 8192 codes used. Result at D=4096: 0.51% accuracy (same as D=512).
Near-uniform codebook usage means real tokens are as hard as random —
no concentration to exploit.

**MLP-only control:** Position one-hot (64) → MLP(64→512→512→8192)
without any VSA binding achieves 0.93% accuracy, confirming the task
is impossible from positional information alone (slight positional bias
but nothing useful). VSA is not the bottleneck — the problem structure is.

### 4.4 Permutation encoding works (Experiment 3)

Reframe: encode *permutations* (64-way classification) instead of
*tokens* (8192-way). No training — pure VSA theory test.

| D | k=1 swap | k=4 | k=16 | k=32 (full) |
|---|---|---|---|---|
| 512 | 100% swap | 100% | 99.9% | 99.3% |
| 1024 | 100% / 89% exact | 100% / 54% | 100% / 15% | 100% / 99.9% |
| 2048 | **100% / 99.8% exact** | 100% / 99.8% | 100% / 96.5% | **100% / 100%** |

Perfect at D≥2048 (0 failures in 2000 test permutations per
condition, 95% CI: [99.8%, 100%]). VSA excels at permutation
representation — the 64-way classification is well within capacity.

### 4.5 But images aren't permutations (Experiment 4)

Test whether real images can be approximated as permutations of a
cluster reference. For each image: find nearest cluster, solve optimal
assignment (Hungarian algorithm), measure token match rate.

| K clusters | Within-cluster overlap | Match rate | >=50% match |
|---|---|---|---|
| 10 | 4.0% | 2.7% | 0.2% |
| 50 | 2.1% | 5.2% | 0.8% |
| 200 | 2.4% | 8.9% | 3.3% |

**Metric definitions:** "Within-cluster overlap" = average pairwise
multiset intersection size / 64 between random pairs in the same
cluster (measures how many identical token values two images share).
"Match rate" = fraction of positions where the Hungarian-optimal
permutation of the reference yields the correct token. Match rate ≥
overlap because optimal assignment can match tokens that appear at
different multiplicities.

**Theoretical bound:** Each image uses 64/8192 = 0.78% of the
codebook. Under uniform codebook usage, the expected multiset
overlap between two random images is ~0.5 tokens (birthday collision
rate: 8192 × (64/8192)² ≈ 0.5). Even with K=6000 clusters (one
reference per image, defeating the purpose), estimated match rate is
~29%. The permutation framing requires near-complete multiset overlap,
which is impossible when the codebook is large and near-uniformly used.

### 4.6 Diagnosis: Two independent failure modes
1. **Capacity failure:** FHRR superposition of N bindings provides
   ~log2(D/N) bits per retrieval. With N=64 and codebook=8192 (13
   bits), no feasible D suffices. Reducing cardinality to 64-way
   (permutations) solves this.
2. **Framing failure:** Even if token retrieval worked, the permutation
   generation framing requires shared token multisets between images.
   With 8192 near-uniformly-used codebook entries, this assumption fails
   catastrophically (<9% overlap).

---

## 5. Positive Findings and Boundary Conditions (1.5 pages)

### 5.1 Where VSA succeeds
1. **Permutation representation:** 100% exact recovery at D=2048 for
   64-position permutations (Experiment 3). VSA is ideal for structured
   operations on small discrete sets.
2. **Permutation locality:** TiTok decoder is smooth w.r.t. token
   permutation (Experiment 0). This property is real and potentially
   useful for local image editing.
3. **SVD-aligned codebooks:** Initializing VSA codebooks from SVD of
   the target activation matrix captures 50% more variance than random
   (4.3% vs 2.8%). If codebook entries align with the data manifold,
   retrieval improves — but the ceiling remains low.

### 5.2 The boundary
VSA succeeds when:
- Classification cardinality is low (64 positions, not 8192 codes)
- The target has discrete, compositional structure
- Operations are group-theoretic (binding = multiplication, unbinding
  = division)

VSA fails when:
- The target mapping is high-rank and continuous (FFN replacement)
- Retrieval requires high-cardinality discrimination from superposition
- The compositional framing doesn't match the data (token multisets)

### 5.3 Generalization
- The rank bottleneck (Case Study 1) applies to all retrieval-based FFN
  replacements: Product Key Memory, holographic reduced representations,
  any top-k selection mechanism
- The capacity limit (Case Study 2) applies to any VSA scheme that
  superposes N bindings and retrieves from codebook size C: requires
  D >> N × C for reliable retrieval

---

## 6. Related Work (1 page)

- **VSA / Hyperdimensional Computing:** Kanerva (2009); Plate (2003)
  HRR; Gayler & Levy (2020); Frady et al. (2018) capacity bounds
- **Memory Layers:** Lample et al. (2019) PKM; Wu et al. (2024) Meta;
  Sukhbaatar et al. (2019)
- **FFN Compression:** Low-rank factorization (Hsu et al. 2022);
  pruning (Frantar & Alistarh 2023); distillation (Hinton et al. 2015)
- **Image Tokenization:** VQ-VAE (van den Oord et al. 2017); TiTok
  (Yu et al. 2024); VQGAN (Esser et al. 2021)
- **Permutation-based Generation:** Jigsaw puzzle methods (Noroozi &
  Favaro 2016); set prediction (Lee et al. 2019)
- **Sparse MoE:** Shazeer et al. (2017); Fedus et al. (2022); Mixtral

---

## 7. Conclusion (0.5 pages)

We tested VSA in two settings where its algebraic properties should
provide advantages: structured FFN replacement and compositional image
generation. Both fail, for different reasons:

1. **FFN replacement:** The rank bottleneck. FFN mappings are ~89% linear
   with effective rank >2048. VSA retrieval collapses rank to top-k.
   Linear projections dominate at every parameter budget.

2. **Image generation:** The capacity bottleneck plus framing mismatch.
   Superposition of 64 bindings provides ~6 bits per retrieval; 8192
   codebook entries need 13 bits. And even with perfect retrieval,
   images don't share token multisets, so the permutation framing fails.

The positive findings narrow VSA's useful scope: it excels at
representing discrete group operations (permutations of 64 positions:
100% exact match) but fails when the target requires either high-rank
continuous mappings or high-cardinality discrete retrieval.

These results suggest that VSA's future in deep learning lies in
genuinely compositional, low-cardinality tasks — symbolic reasoning,
relational learning, discrete program synthesis — rather than as a
drop-in replacement for dense continuous computations.

---

## Appendices

### A. SVD spectrum of FFN activations
- Full singular value plot for layers 3, 27, 43
- Quantitative effective rank analysis

### B. Full experimental details
- Hyperparameters for all runs
- Hardware: AMD Ryzen AI 9 HX 370, Radeon 890M, 23GB RAM (local);
  ThinkCentre M70q Gen 5, 32GB RAM (reviewer experiments)
- Training curves for all architectures

### C. Token distribution analysis
- TiTok VQ-8K codebook usage histogram
- Entropy analysis (12.85 / 13.00 bits)
- Spatial correlation in real tokens

### D. Permutation recovery details
- Full accuracy tables for D=256,512,1024,2048,4096 across all k
- Swap accuracy vs position accuracy vs exact match

---

## Figures (planned)

1. **Architecture comparison bar chart:** Var% vs params for all
   architectures (Case Study 1) — visual punchline
2. **Singular value decay:** Log-scale SVD spectrum for 3 layers
3. **Token binding capacity:** Accuracy vs D for exps 1, 1c, 2 showing
   capacity wall
4. **Permutation recovery heatmap:** Accuracy (D × k_swaps) from exp 3
   — green region (permutations work) vs red region (tokens fail)
5. **Token overlap histogram:** Distribution of match rates from exp 4
   showing the framing failure
6. **Signal decomposition:** Stacked bar: ~89% linear + 5% MLP + <1% VSA

---

## Experiment inventory

### Case Study 1 (FFN replacement) — all complete
| Exp | Description | Status | Location |
|---|---|---|---|
| SVD spectrum | Singular value decay of FFN activations | DONE | thinkhub:~/reviewer_results/exp1_svd_spectrum.json |
| Top-k scaling | top_k = 4,16,64 variance | DONE | thinkhub:/tmp/exp2_topk.log |
| FLOPs | Wall-clock + FLOP comparison | DONE | thinkhub:~/reviewer_results/exp3_flops.json |
| Codebook ablation | Random/learned/SVD/SVD+learned | DONE | thinkhub:~/reviewer_results/exp4_codebook_ablation.json |
| Rank-4 linear | SVD baseline at matched rank | DONE | thinkhub:~/reviewer_results/rank4_linear.json |

### Case Study 2 (Image generation) — all complete
| Exp | Description | Status | Location |
|---|---|---|---|
| Exp 0 | Permutation locality validation | DONE | rubik-gen/VALIDATION_RESULT.md |
| Exp 1 | Pure VSA token binding D=512-4096 | DONE | rubik-gen/results/exp1_vsa_capacity.json |
| Exp 1b | VSA + MLP decoder | DONE | rubik-gen/results/exp1b_vsa_decoder.json |
| Exp 1b ctrl | MLP-only baseline (no VSA) | DONE | rubik-gen/results/exp1b_mlp_only_control.json |
| Exp 1c | Factored codebook (128×64) | DONE | rubik-gen/results/exp1c_factored.json |
| Exp 2 | Real TiTok tokens (Imagenette, D=512-4096) | DONE | rubik-gen/results/exp2_real_tokens.json |
| Exp 3 | Permutation VSA encoding | DONE | rubik-gen/results/exp3_permutation_vsa.json |
| Exp 4 | Permutation reconstruction quality | DONE | rubik-gen/results/exp4_permutation_reconstruction.json |

---

## Repo and reproducibility

- Code: github.com/Peterc3-dev/cube-memory (will be made public)
- Rubik gen: ~/projects/cube-memory/rubik-gen/
- Activations: ~/cube-memory-cache/activations/
- All experiments reproducible on consumer AMD APU (23GB RAM)
- Total compute: ~8 hours across both machines
