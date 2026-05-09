# Local distillation plan — raz-gpd4 only, no rented compute

## Endpoint (refined 2026-04-26)

**An agent fully capable of tool calling, running 100% on raz-gpd4.**

- **Runtime**: `llama-server` on raz-gpd4 serving Qwen3.6-27B
  (eventually with cube-memory FFN replacements). No cloud APIs in
  the agent's hot path. Claude is the builder, not the runtime.
- **Tool-calling support**: qwen35 chat template + JSON-schema
  function definitions + structured `<tool_call>` emission
  parseable by an executor.
- **Eval bar**: BFCL (Berkeley Function Calling Leaderboard)
  - Simple ≥ 80% (single function, correct selection)
  - Multiple ≥ 60% (choose right function from a set)
  - Parallel + Multi-turn: nice-to-have
- **Wired**: openclaw (or direct curl harness) routes user queries
  → local llama-server → tool execution → response. Telegram already
  hooked into openclaw, so the agent can be reached from the user's
  phone without any cloud dependency.

The recursive loop does not stop until the above is demonstrated
end-to-end with measured BFCL numbers committed to the repo.

**Two viable paths, both run end-to-end on raz-gpd4.**

## Memory model — raz-gpd4 (corrected 2026-04-26)

The GPD Pocket 4 has **32 GB total RAM** physically, but the BIOS
UMA carve-out reserves **16 GB exclusively for VRAM** and leaves
**16 GB for system RAM**. The two pools are addressable separately
and cannot double-spend.

| Pool | Size | Used by |
|---|---|---|
| System RAM | 16 GB | OS, processes, FastFlowLM NPU model weights, llama.cpp CPU layers, HF Transformers offload buffers |
| iGPU VRAM (UMA carve) | 16 GB | Vulkan/ROCm tensors, llama.cpp GPU layers, PyTorch ROCm allocations |
| NVMe disk | hundreds of GB | Streamed weights (HF offload), activation caches, teacher logit dumps |
| NPU SRAM | small | Active during FastFlowLM inference, transparent to user |

**Concurrency rules**:
- Teacher inference (llama.cpp Qwen3.6-Q4_K_M, 17.28 GB) must
  CPU+GPU-split — won't fit in either pool alone.
- Student training (PyTorch ROCm) lands tensors in VRAM. Hard
  ceiling 16 GB. With disk offload + grad checkpointing, working
  set per step ~5–8 GB. Fits.
- Teacher and student CANNOT both occupy VRAM at once. Sequence
  them: teacher dumps logits to disk first, student trains from
  the cached logits.
- NPU (FastFlowLM Qwen3-4B int8, ~4 GB resident) lives in system
  RAM. Independent pool — runs concurrently with any iGPU work.

The original Phase 1 PLAN.md said "Local hardware too small (16 GiB
iGPU) to run the actual distill." That was right about the VRAM
cap but missed two routes around it:
1. The student can stream weights from disk per-layer, never
   resident in their entirety.
2. End-to-end distillation doesn't require teacher and student
   resident at the same time — sequence them.

Both paths below avoid rented compute. Path A (layer-wise) is
faster to ship; Path B (end-to-end) is the rigorous version. We
run A first as warmup, then B as polish.

## Path A — Layer-wise feature distillation (FitNets-style)

Standard distillation: student matches teacher *logits*; gradients
flow through every parameter of the student per training step.

Layer-wise distillation: student matches teacher *hidden states* at
the layer boundary; gradients flow only through the one module being
fit. Each replacement is an independent regression problem. The full
student is never instantiated for backprop.

This is exactly the right shape for FFN replacement: each swapped
position is a `(in_hidden, out_hidden)` pair, and each
`CubeMemoryLayer` is an autonomous mapping. We're not distilling the
model — we're distilling each FFN block separately.

## Path B — End-to-end logit distillation (locally)

Run the FULL student forward+backward on raz-gpd4 with three
techniques that together collapse the memory wall:

1. **Disk offload via HF Transformers**: `model.from_pretrained(...,
   device_map="auto", offload_folder="/path", offload_state_dict=
   True, torch_dtype=torch.bfloat16)`. Weights stream from NVMe
   per-layer during forward/backward. Working set per layer ~5 GB.
2. **Gradient checkpointing**: `model.gradient_checkpointing_enable()`.
   Activations recomputed on backward. Trades 30% wall time for ~5×
   activation memory savings.
3. **Selective `requires_grad=False`**: only the 12 swapped
   CubeMemoryLayer modules need gradients. Adam state for ~1B
   trainable params = ~8 GB; can be CPU-offloaded via
   `bitsandbytes.optim.AdamW8bit` or `torch.optim.Adam` with
   foreach=False + manual CPU placement.

Teacher logits do NOT need autograd — dump them once via llama.cpp
inference (`logits_all=true`), save to disk, replay during student
training. Teacher inference uses ~17 GB UMA, runs at ~38 t/s,
finishes 100K tokens of logits in ~45 minutes.

**Memory budget per step (student backward, batch=1, seq=512):**
- Layer working set (streamed from disk into VRAM): ~5 GB
- Activations (with checkpointing): ~600 MB
- Trainable param fp32 + grads (in VRAM): ~2 GB
- Adam state (8-bit, CPU-offloaded to system RAM): ~2 GB
- Teacher logits (memory-mapped from disk): negligible

VRAM resident: ~8 GB out of 16 GB ceiling. Headroom for activation
checkpointing buffer and CUDA caching allocator overhead. System
RAM use: ~6 GB Adam state + ~2 GB process baseline = ~8 GB out of
16 GB. Both pools fit with comfortable margin.

Concurrency note: NPU's FastFlowLM Qwen3-4B (~4 GB system RAM) can
run in parallel — pushes system RAM use to ~12 GB out of 16, still
fits. iGPU is fully owned by student training.

**Wall time estimate**: 30s–2min per step on iGPU+CPU offload.
100K tokens at batch=1, seq=512, accumulate=8 = ~12K steps.
Wall time: 1–4 days for one full distillation pass.

We pick Path B if the layer-wise warmup leaves >10% perplexity
regression on held-out text. We pick Path A alone if the warmup
already meets the spec target.

## Hardware fit (Path A — layer-wise)

- Teacher inference (sequenced first, then exit): llama.cpp on
  Q4_K_M GGUF, ~17 GB CPU+GPU-split.
- Per-layer trainer (after teacher exits): one CubeMemoryLayer in
  fp32 (~250 MB) + Adam state (2× params, ~500 MB) + activation
  batch (~256 MB). Total ~1 GB. Lives in VRAM via PyTorch ROCm.
  Trivial.
- Activation cache: stored on disk as bf16 chunks. For 100K FineWeb
  tokens × Qwen3 hidden=2048 × 2 bytes = 400 MB per layer position.
  12 swap positions × 400 MB = 4.8 GB. Fits NVMe.

## Phases

### Phase A — Activation cache extraction
1. ✓ Acquired **Qwen3.6-27B-Q4_K_M** at
   `/home/raz/models/Qwen3.6-27B-Q4_K_M/Qwen3.6-27B-Q4_K_M.gguf`
   (16.82 GB on disk, BPW 5.00, Unsloth-quantized with imatrix).
   Architecture is `LLM_ARCH_QWEN35` (NEW arch, recently added to
   mainline llama.cpp; the older Feb-built llama.cpp at
   `~/llama.cpp/build/` does NOT support it — must use
   `/tmp/llama-mainline/build/bin/llama-cli` only).

   Verified shape (from llama_model_loader print_info):
   - 64 layers, n_embd=5120, n_ff=17408
   - GQA: n_head=24, n_head_kv=4 (n_gqa=6)
   - **Hybrid attention/SSM** with `full_attention_interval=4`:
     **only every 4th layer is full attention + standard FFN** (16 of
     64). The other 48 layers are SSM/Mamba-like with
     `state_size=128, inner_size=6144, conv_kernel=4`.
   - n_vocab=248,320 (vs Qwen3's 152K); embedding ~2.4 GiB Q4
   - n_ctx_train=262,144 (256K)
   - Multimodal base (`Qwen3_5ForConditionalGeneration`,
     `image-text-to-text`) — text-only inference doesn't load
     `mmproj-*.gguf`, ignore.
   - BOS=`<|endoftext|>` (248044), EOS=`<|im_end|>` (248046)

   **Cube-memory swap implication**: only the 16 attention+FFN layers
   are swap candidates. Indices to swap (every 8th of the 16): layer
   positions {3, 11, 19, 27, 35, 43, 51, 59}. 8 swap positions, not
   8 of 64. Activation cache size unchanged: 8 × 100K × 5120 × 2 =
   **8 GB per side, 16 GB total**.

   **Baseline perf at -ngl 30**: PP 2.8 t/s, TG 0.3 t/s (terrible —
   only 8 GiB of 16 GB VRAM used; half layers ran on CPU through
   swap pressure). Per Phase 0 of the tok/s pivot: -ngl sweep is
   the immediate lever.
2. Patch llama.cpp with a new tool `tools/dump-activations` that
   registers a `cb()` callback at `llm_graph_context::cb()` and, for
   tensor name `ffn_inp` or `ffn_out` (or qwen35-equivalent — see
   note below) AND `il ∈ {chosen layer indices}`, schedules
   `ggml_backend_tensor_get` after compute and writes bf16 to
   `~/cube-memory-cache/activations/layer_{il}_{in|out}/chunk_{N}.bin`.

   **Architecture-specific name mapping** (verified 2026-04-26 by
   reading `src/models/qwen35.cpp` directly — Agent L's recon was
   wrong about FFN coverage):
   - **qwen35 has FFN on EVERY layer** (all 64), not just the 16
     full-attention ones. The hybrid is in the *attention* path
     (linear/SSM vs full-attn alternates per `full_attention_interval=4`),
     but `build_layer_ffn(attn_post_norm, il)` runs on every layer.
     This *increases* the swap candidate pool from 16 → 64.
   - **qwen35 emits NO `ffn_inp` callback.** Its FFN-input tensor is
     named `attn_post_norm` (line 57 of qwen35.cpp). The dump tool's
     filter must accept `attn_post_norm` as the qwen35 equivalent of
     `ffn_inp`, otherwise the input side of the (in, out) pair is
     silently dropped and the entire activation cache is unusable
     for distillation.
   - **dump-tool patch required** before Phase A can run on
     Qwen3.6-27B. Tracked as a Phase A blocker. The smoke test on
     Qwen3-1.7B passed because vanilla qwen3 *does* emit `ffn_inp`
     — different architecture, different tensor names.

   Updated swap indices for 64-layer Qwen3.6-27B: still every 8th =
   8 positions ({3, 11, 19, 27, 35, 43, 51, 59}), but now uniformly
   spaced across all layers since every layer is a candidate.
   Cache size: 8 positions × 100K tokens × 5120 hidden × 2 bytes ×
   2 sides = 16 GB total. Fits NVMe.
3. Run the tool over 100K-1M FineWeb-Edu sample tokens. Native
   llama.cpp speed on Strix Point ≈ 38 t/s, so 100K tokens ≈ 45 min,
   1M tokens ≈ 7.5 hours.

### Phase B — Per-layer fit
4. For each swap position (sequentially, in layer order):
   a. Load that position's `(in, out)` activation pairs.
   b. Initialize a fresh CubeMemoryLayer matching the FFN's d_in.
   c. Train with MSE (or distillation loss with temp) until
      validation MSE plateaus or 1–2 hours have passed.
   d. Save the trained layer.
   e. **Critical**: re-cache activations for the *next* swap position
      using the *partially swapped* student (not the original
      teacher), so each fit is against the real upstream signal it
      will see at inference. This prevents error compounding.

### Phase C — End-to-end calibration
5. Stitch all 12 trained CubeMemoryLayers into the full student.
6. Optional: low-LR end-to-end logit distillation pass (only swapped
   layers + adjacent norms unfrozen — small unfrozen set fits in
   gradient memory).
7. Eval perplexity vs teacher on held-out tokens. Target: ≤ 10%
   regression per Phase 2 spec.

### Phase D — GGUF export + Vulkan inference
8. Write CubeMemoryLayer weights into a GGUF, signaling the new
   `CUBE_MEMORY_CLEANUP` and `CUBE_MEMORY_RETRIEVE` ops. The
   ggml-vulkan path for these ops is the work happening in this
   session right now (`/tmp/llama-mainline` cube-memory-op branch).
9. Load in llama.cpp + bench TG on Strix Point.

### Phase E — Agent loop
10. Wire llama-server → the existing openclaw / CIN agent
    infrastructure. Validate against a small task set.

## Realistic time budget on raz-gpd4 only

- Phase A: 2–6 hours (one inference pass over 100K tokens, plus
  llama.cpp patching).
- Phase B: 12 layers × 1–2 hours each = 12–24 hours wall time. Can
  run overnight, in batches.
- Phase C: 1–4 hours.
- Phase D: 1–4 hours (already in flight).
- Phase E: 1–2 hours.

Total: 17–40 hours wall time, no rented compute.

## What invalidates this plan

- If layer-wise MSE doesn't generalize (validation MSE keeps
  dropping but downstream perplexity stays high) — would force a
  return to end-to-end distillation. Mitigation: Phase C's optional
  end-to-end pass acts as recovery.
- If the activation cache is too large (per-layer activations grow
  linearly with token count) — chunk and stream from disk rather
  than load full.
- If iGPU PyTorch (ROCm 7.2 + HSA_OVERRIDE) crashes on the long
  training runs — fall back to CPU PyTorch, accept 10× slowdown,
  still finishes in days not weeks.

## Reframed because

The earlier plan baked in "need an H100" as if it were a hardware
constraint. It was actually a *training-method* constraint. Picking
a different method removed the hardware requirement entirely.

This is the kind of mis-attribution the consortium plan's outer
loop (chess-strategist) is supposed to catch.

Drafted 2026-04-26.

## Phase B debug (added 2026-04-27)

Phase B trainer ran 8 layers × 5000 steps on real Qwen3.6-27B activations.
Ran a three-step diagnostic on layer 3 (cheapest) before doing more compute.

### Step 1 — Diagnostic (no training)

Script: `/tmp/phase_b_diag.py`. Loaded layer 3 in/out chunks (50K tokens,
n_embd=5120, fp32-from-bf16). All-data sanity:

| Tensor | mean | std | min | max | NaN/Inf |
|---|---|---|---|---|---|
| x_in  | -2.1e-4 | 1.6e-1  | -4.09 | +5.13 | 0/0 |
| x_out | +1.2e-4 | 2.79e-2 | -1.52 | +2.92 | 0/0 |

Var(x_in)=2.57e-2; **Var(x_out)=7.77e-4**. Critical observation: x_out has
~5.7× smaller std than x_in, so the layer 3 FFN delta is small in magnitude.

Loaded `layer_3.safetensors`. All learnable weights are non-zero post-training
(out_proj abs.mean=8e-3, slot_values abs.mean=1.5e-2, role_proj abs.mean=4.9e-2),
so the optimizer did move them — no "frozen" failure.

Loss decomposition on a 256-token batch of REAL data:

| | mse_loss | normalized = mse / Var(x_out) |
|---|---|---|
| pred(trained model) | 7.05e-4 | **0.908** |
| zeros (do-nothing baseline) | 7.41e-4 | 0.954 |
| mean(x_out) | 7.03e-4 | 0.905 |

Full 5% val split (same seed=42 as production):
- trained model val_mse=**7.67e-4**, normalized **0.953**
- zero baseline val_mse=8.05e-4, normalized 1.000

**Verdict on the user's confusion**: the "0.674 → 0.673" reading was from
layer 59 (deepest, residual stream is huge — Var grows ~1000× from layer 3
to layer 59). Layer 3's actual numbers were **0.000806 → 0.000767** —
exactly matching my diagnostic. The trainer was working correctly all along
in absolute terms. **The real problem is the relative number**: the model
captures only 4.7% of target variance after 5000 steps.

### Step 2 — Hypothesis A: normalized loss + 10× lr (FAILED)

Patched trainer at `/tmp/per_layer_trainer_v2.py`. Multiplied MSE by
1/Var(x_out) (= 1290 for layer 3), set lr=1e-2.

Result: catastrophic divergence. norm_val went 4.4 → 3.7 → **572** → **171**
between steps 0/1000/2000/3000. Adam with lr=1e-2 is far too large for this
loss landscape; constant rescaling of the loss is a near-no-op for Adam
(per-parameter scale invariance), so the only effective change is the lr
bump, and it overshoots.

Killed at step 3000 to save the time budget. **Loss-formulation fix does
not help — the original lr=1e-3 was already in the right ballpark.**

### Step 3 — Hypothesis B: 4× capacity (FAILED)

Re-ran on layer 3 with `n_slots=16384` (was 4096), keeping m=64, p=2,
d_value=2048, lr=1e-3 unchanged.

| step | v1 (n_slots=4096) val_mse | v3 (n_slots=16384) val_mse |
|---|---|---|
| 0    | 0.000806 | 0.000806 |
| 1000 | 0.000772 | 0.000779 |
| 2000 | 0.000771 | 0.000768 |
| 3000 | 0.000768 | 0.000767 |
| 4000 | 0.000767 | 0.000770 |
| 4999 | **0.000767** | **0.000766** |

**Bit-identical curves to noise.** 4× more reachable slots changed the final
val_mse by 0.1% (0.000767 → 0.000766) — within the run-to-run jitter visible
within either run. Going to p=3 / N=262K (the SPEC.md "realistic" config)
would give 64× more slots, but the v3 result tells us slot count is not the
bottleneck. The model converges to the same plateau independent of capacity.

### Verdict — architecture mismatch

Both hypothesis-A (loss formulation) and hypothesis-B (capacity) failed.
The remaining suspect is the **structural prior** itself:

- The query → role-projection → per-axis cleanup → FHRR bind → slot retrieval
  pipeline forces the model to address slots through a discrete, low-cardinality
  algebraic key (m=64 codewords per axis, 2 axes → 4096 distinct addresses, OR
  16384 with 4× more slots — but cleanup snaps to one of m^p discrete keys
  regardless of how many slots are physically present).
- The straight-through estimator on the cleanup step plus unit-modulus
  re-projection on the bound address discards most of the gradient signal
  about which slot to address.
- 4.7% of variance captured ≈ a heavily-bottlenecked linear codebook
  approximator, not a true FFN replacement.

Stitching these layers into the full model would tank perplexity badly, since
each layer's output is ~5% of the true delta (and 95% noise/zero).

**Decision: Paper 1 needs reframing. The current cube-memory architecture as
implemented in `cube_memory_layer.py` cannot fit Qwen3.6-27B FFN behavior at
this scale.** Options forward:

1. **Reframe Paper 1 as a negative result** — "VSA-keyed memory layers are
   insufficient for FFN replacement at scale, here's why and here's the
   capture-rate floor we measured (4.7% of variance, invariant to 4× capacity)."
2. **Retry with structural changes** — multi-head retrieval, learned (not
   frozen) codebooks, residual additive output instead of pure projection,
   gating on the query-to-cleanup nearest-neighbour confidence.
3. **Pivot to a different drop-in** — small dense FFN (e.g. d_in→d_ff/4→d_in
   with d_ff=4096) trained with the same MSE harness; uses ~80 MB and would
   likely capture >50% of variance.

**Do NOT proceed with Phase C stitching on the current 8 layers.** They are
known broken — they would make the model worse than the do-nothing baseline
(zero-init residual delta), since pred has 4× too small std AND nonzero
direction error.

### Files

- Diagnostic: `/tmp/phase_b_diag.py` (log: `/tmp/phase_b_diag.log`)
- v2 trainer (loss-norm + 10× lr): `/tmp/per_layer_trainer_v2.py` (log: `/tmp/phase_b_v2.log`)
- v3 run (4× capacity): same trainer, log `/tmp/phase_b_v3.log`,
  ckpt `/tmp/phase_b_v3/layer_3.safetensors`
- Original broken ckpts: `/home/raz/cube-memory-cache/trained-layers/layer_*.safetensors`
- Original run log: `/tmp/phase_b_real.log`

