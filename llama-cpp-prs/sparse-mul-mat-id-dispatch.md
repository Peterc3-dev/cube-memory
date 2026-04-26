# PR scope: sparse-aware MUL_MAT_ID workgroup dispatch (Vulkan)

## Problem

`ggml_vk_matmul_id()` in `ggml/src/ggml-vulkan/ggml-vulkan.cpp:7282`
dispatches with workgroup dimensions `{ m, nei1, n_as }` where `n_as`
is the **total expert count** of the MoE layer. For Qwen3.6-35B-A3B
that is 256; for Qwen3-30B-A3B it is 128. The shader early-exits on
unselected experts, but each workgroup still pays:

- Pipeline launch slot
- Initial push-constant + descriptor binding
- Dequant prologue / SMEM init
- Whatever subgroup synchronization the shader prologue does

Active experts are `top_k` (8 in both models). The other 248 (or 120)
workgroup slots do no useful work but consume launch overhead.

## Evidence

Profiled on Radeon 890M (gfx1150, RADV STRIX1) with
`GGML_VK_PERF_LOGGER=1`, llama.cpp `9725a31`, Q4_K_M GGUFs. Per-call
time of the *same kernel and shape* scales with `n_as`:

| Model | n_as | top_k | per-call MUL_MAT_ID q4_K (µs) |
|---|---|---|---|
| Qwen3-30B-A3B | 128 | 8 | 2 484 – 3 652 |
| Qwen3.6-35B-A3B | 256 | 8 | 3 528 – 4 648 |

Per-call time scales ~1.3–1.4× with the 2× expert count. Active
params per token unchanged. Output token rate drops 32% (38.17 →
25.93 t/s) entirely from dispatch overhead.

Kernel utilization is ~3% of peak compute on both models, so the
hardware is not the bottleneck — the launch path is.

## Fix sketch

Two viable approaches.

### Option A — host-side dispatch loop (smallest blast radius)

Replace the single dispatch over `{ m, nei1, n_as }` with one
dispatch per *selected* expert over `{ m, nei1, 1 }`. The host
already has the routing tensor (`ids`) before this call; iterate
over selected expert ids and emit `top_k` dispatches.

Pros:
- No shader changes.
- Trivially correct.
- Roughly halves dispatch overhead at top_k=8 / n_as=128, ~3.5×
  reduction at top_k=8 / n_as=256.

Cons:
- N small dispatches instead of 1 large one — Vulkan command-buffer
  build cost scales with dispatch count, may eat some of the win.
- Requires reading `ids` host-side, which on Vulkan UMA is cheap
  but on a real PCIe GPU is a roundtrip.

### Option B — vkCmdDispatchIndirect with expert_count_buf

Use `vkCmdDispatchIndirect` with `expert_count_buf` (already passed
to `ggml_vk_matmul_id` at line 7271 as a parameter). The buffer
contains the count of *active* experts; the GPU reads it at command
execution time and dispatches exactly that many workgroups in the
expert dimension.

Pros:
- Zero host roundtrip.
- Single dispatch call.
- Zero idle workgroups.
- The plumbing is partly there — `expert_count_buf` is already
  computed and passed.

Cons:
- Requires writing the indirect dispatch buffer in the right layout
  (`VkDispatchIndirectCommand` is 3 × uint32_t: x, y, z workgroup
  counts).
- Need to add a tiny compute shader (or a transfer) that fills the
  indirect buffer from the routing tensor. May already exist —
  there is an `expert_count_buf` parameter implying someone started.
- Shader still gets called with the full `nei0` value in push
  constants, but only enters the loop body once per active id —
  so the early-exit logic stays the same, the dispatch just doesn't
  launch idle workgroups.

### Recommendation

Start with **Option A** for the PR. It is a 30-line change with
predictable speedup. Option B is the right long-term fix and can
be a follow-up — the `expert_count_buf` plumbing suggests upstream
agreed in principle.

## Files to touch (Option A)

- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
  - `ggml_vk_matmul_id` (~line 7269-7283) — split the single dispatch
    into a loop over selected expert ids.
  - The caller (`ggml_vk_mul_mat_id_q_f16` at ~line 8500-8595) may
    need to read `ids` host-side once and pass an array of selected
    expert ids; on UMA this is a memcpy from the same DRAM.
- No shader changes.

## Test plan

1. **Correctness.** Run the existing llama.cpp test suite
   (`tests/test-backend-ops`) for `MUL_MAT_ID`. Verify per-quant-type
   parity with reference output. Run on Radeon 890M and on a
   discrete GPU (RTX/Arc) to catch UMA-vs-dGPU divergence.
2. **Bench.** `llama-bench` on Qwen3.6-35B-A3B Q4_K_M and Qwen3-30B-
   A3B Q4_K_M with and without the patch. Target: TG up by 25%+
   on Qwen3.6, no regression on Qwen3-30B-A3B.
3. **Profile.** Re-run the `GGML_VK_PERF_LOGGER` capture from
   `~/projects/cube-memory/baselines/qwen36-vulkan-profile.md` and
   confirm per-call MUL_MAT_ID time drops in proportion to top_k/n_as.
4. **Real-world.** Run the 20-case tool-call eval at
   `~/projects/llama.cpp/moe-bench/eval_tools.py` on both models;
   verify correctness preserved and latency drops.

## Risk and rollback

Low. The change is additive and falls back trivially if the loop
runs once with all experts active (matches current behavior). If
correctness breaks, revert one function. The shader is untouched,
so cross-vendor risk is minimal.

## Estimated effort

50–60 LOC total for Option A: ~30 lines in `ggml_vk_matmul_id` for
the per-expert dispatch loop, plus ~15–20 lines in
`ggml_vk_mul_mat_id_q_f16` to read the routing `ids` host-side and
pass selected expert ids to the loop. ~1–2 days of focused work.
Option B is a follow-up of similar size if A passes.

**Latency caveat for dGPU**: Option A requires a GPU→host readback of
the routing `ids` tensor on each MUL_MAT_ID call. On UMA (the user's
Strix Point) this is a same-DRAM memcpy and effectively free. On a
discrete GPU it is a PCIe roundtrip and may eat some of the win. The
PR should document this and recommend Option B (indirect dispatch,
no readback) as the dGPU-friendly path.

## Why this matters beyond Qwen3.6

Any future MoE with `n_as > top_k * 2` is currently leaving
performance on the floor in the Vulkan backend. As MoE designs
push expert counts up (Qwen3-Next, Qwen4 are likely 256+), this
becomes more acute. Fixing it once for the `mul_mat_id` path
benefits every Vulkan MoE deployment going forward.

## Bug-sweep audit (2026-04-25)

A read-only audit on clean context confirmed all technical claims:

- Third workgroup dim is indeed `n_as`. Verified at dispatch site
  (`ggml_vk_dispatch_pipeline`, line ~6604-6631 of `ggml-vulkan.cpp`).
- Shader **does** hard early-exit on unselected experts. Source:
  `ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp` line 144-145:

  ```glsl
  if (ic * BN >= data_expert_count[expert_idx]) {
      return;
  }
  ```

  The unselected workgroups still pay launch slot + descriptor
  bind + push constant broadcast + subgroup barrier setup before
  hitting the return.
- `expert_count_buf` is computed by a `count_experts` kernel
  (`ggml-vulkan.cpp` ~line 8523–8534) and *is* read by the shader
  (`mul_mm_id_funcs.glsl` line 16), but is **not** used to modify
  the host-side dispatch count. Option B's plumbing is therefore
  partially in place but the indirect-dispatch wiring still has
  to be written.
- Per-call time scaling is real and measured via GPU timestamps,
  not host wallclock — confound from profiler instrumentation
  ruled out.
- No in-flight PR or discussion in the issue tracker on this.
- CUDA backend uses a different dispatch path
  (`ggml_cuda_mul_mat_id`) and does not have this issue. Vulkan-only.

Adjustment from the original scope: line count was understated.
Honest estimate is 50–60 LOC for Option A (function + caller).
Reflected above.
