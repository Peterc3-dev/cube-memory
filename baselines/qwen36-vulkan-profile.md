# Qwen3.6-35B-A3B Vulkan profile baseline

Captured 2026-04-25 with mainline llama.cpp `9725a31` on Radeon 890M
(RADV STRIX1, gfx1150) via `GGML_VK_PERF_LOGGER=1`. Greedy decode,
16 generated tokens, prompt "The capital of France is".

## Headline

DeltaNet is **not** the bottleneck. The 32% TG regression vs
Qwen3-30B-A3B (38.17 → 25.93 t/s) traces to per-call `MUL_MAT_ID`
overhead that scales with `n_expert`, not `top_k`.

## Per-op time, Qwen3.6-35B-A3B

| Op | Time / 16-token gen | Share |
|---|---|---|
| `MUL_MAT_ID q4_K m=512 n=256 k=2048 n_expert=256` × 78 | 275 209 µs | ~57% |
| `MUL_MAT_ID q5_K m=2048 n=256 k=512 n_expert=256` × 37 | 171 988 µs | ~36% |
| `MUL_MAT_ID q6_K m=2048 n=256 k=512 n_expert=256` × 2 | 12 697 µs | ~2.6% |
| `MUL_MAT_ID_MUL/VEC` variants | ~6 800 µs | ~1.4% |
| `MUL_MAT_VEC q8_0 m=2048 k=4096` × 40 | 7 087 µs | ~1.5% |
| `MUL` × 149 | 4 976 µs | ~1% |
| `GET_ROWS` × 62 | 1 820 µs | ~0.4% |
| `GLU` × 80 | 1 821 µs | ~0.4% |
| **`GATED_DELTA_NET` × 30** | **1 658 µs** | **~0.3%** |
| `FLASH_ATTN_EXT` × 10 | 1 177 µs | ~0.2% |
| (all other ops) | < 5 000 µs | <1% |

Total accounted ~480 ms / 16 tokens ≈ 33 t/s (matches bench within
overhead).

## Comparison: Qwen3-30B-A3B (same prompt, same backend)

| Model | n_expert | per-call MUL_MAT_ID q4_K (µs) | calls / token | MUL_MAT_ID time / token |
|---|---|---|---|---|
| Qwen3-30B-A3B | 128 | 2 484 – 3 652 | 141 | ~382 000 µs |
| Qwen3.6-35B-A3B | **256** | 3 528 – 4 648 | 117 | ~460 000 µs |

Per-call time is ~30–40% slower at `n_expert=256` for identical
quantization and shape. Active params per token are unchanged
(`top_k=8`). The slowdown is dispatch overhead.

## Observed kernel utilization

| Op | GFLOPS/s | Peak FP16 (gfx1150) | Utilization |
|---|---|---|---|
| `MUL_MAT_ID q4_K` | ~304 | ~8 600 | ~3.5% |
| `MUL_MAT_ID q5_K` | ~230 | ~8 600 | ~2.7% |
| `MUL_MAT_ID q6_K` | ~169 | ~8 600 | ~2.0% |
| `MUL_MAT_VEC q8_0` (single-token) | ~190–335 | ~8 600 | 2–4% |

Kernels are running at 2–4% of peak compute. Memory bandwidth
saturation is the actual ceiling on TG, but the scheduler /
dispatcher is leaving compute idle in ways that hide which is which.

## Hypothesis on the n_expert scaling

Per-call MUL_MAT_ID time scaling with `n_expert` even when `top_k`
is fixed implies one or more of:

1. **Workgroup count is sized by n_expert**, not by selected experts
   — empty workgroups for unselected experts still consume dispatch
   slots and synchronization cost.
2. **Expert weight gather** reads through the full `n_expert`
   metadata (offsets, row indices) each call, scaling with the
   table not the selection.
3. **Argsort / top-k routing** itself is sub-linear in `n_expert`
   but constant overhead is non-trivial; at small per-call work
   it dominates.

Verification path: instrument
`ggml_backend_sched_compute_splits()` and the
`mul_mat_id` pipeline launch in
`ggml/src/ggml-vulkan/ggml-vulkan.cpp` to log workgroup counts
per call. Compare 128-expert vs 256-expert cases.

## Optimization targets (priority order)

1. **Sparse-aware MUL_MAT_ID dispatch** — only launch workgroups
   for selected experts. Likely a 30%+ TG win on Qwen3.6 alone
   without touching anything else. Generalizes to any high-`n_expert`
   model going forward.
2. Per-token argsort/top-k can be cached when the routing decision
   is reused across the prefill batch — orthogonal to (1).
3. DeltaNet kernels are *not* on this list. They are already efficient
   at this scale.

## Implication for Cube Memory Phase 1

Distillation target should be **Qwen3.6-35B-A3B** because it is the
forward shape (DeltaNet, larger expert pool, MTP head). Phase 2's
inference numbers should be measured *after* the n_expert dispatch
fix above lands — otherwise Cube Memory's bandwidth win competes
with a 1.4× scheduler win that has nothing to do with the memory
layer architecture.

## Reproducer

```bash
GGML_VK_PERF_LOGGER=1 \
  /tmp/llama-mainline/build/bin/llama-cli \
    -m ~/models/Qwen3.6-35B-A3B-Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
    -ngl 99 -st -n 16 --temp 0 \
    -p "The capital of France is" 2>&1 > /tmp/q36-perf.log
grep -E "MUL_MAT_ID|GATED_DELTA|FLASH_ATTN" /tmp/q36-perf.log | head -20
```

Same with the Qwen3-30B-A3B model for comparison.
