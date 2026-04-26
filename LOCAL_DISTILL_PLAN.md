# Local distillation plan — raz-gpd4 only, no rented compute

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
1. Acquire Qwen3.6-35B-A3B-Q4_K_M (HF download, ~17 GB).
2. Patch llama.cpp to dump hidden states at chosen layer indices to
   disk. Chosen indices: distribute 12 evenly across 48 layers (every
   4th). Re-use the existing eval-prompt corpus from FineWeb-Edu
   sample.
3. Run inference over 100K–1M tokens, dumping `(in_hidden,
   out_hidden)` pairs at each swap position.

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
