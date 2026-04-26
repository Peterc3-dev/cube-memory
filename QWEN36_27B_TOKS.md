# Qwen3.6-27B-Q4_K_M tok/s on raz-gpd4

Last updated: 2026-04-26
Hardware: AMD Ryzen AI 9 HX 370 + Radeon 890M (RADV STRIX1, gfx1150), 12 cores, 16 GiB RAM + 16 GiB UMA VRAM (24 GiB Vulkan-addressable)
Binary: `/tmp/llama-mainline/build/bin/llama-bench` (build 9725a31 (290), mainline)
Model: `~/models/Qwen3.6-27B-Q4_K_M/Qwen3.6-27B-Q4_K_M.gguf` (15.65 GiB on disk per loader, 26.90 B params; qwen35 hybrid arch, 64 layers — 16 attention + 48 SSM)

## Iteration 3 — baseline + ngl/threads sweep (2026-04-26)

### Pre-flight context
At session start: 6.4 GiB swap-in-use on zram, 6.8 GiB available RAM, 3.8 GiB tmpfs/shared. GPU idle, 614 MB VRAM baseline. Page cache dropped before each run via `sync && echo 3 > /proc/sys/vm/drop_caches` (sudo -n). `HSA_OVERRIDE_GFX_VERSION=11.0.0` exported.

### Phase 1: -ngl sweep (-t 12, --flash-attn 0, -p 256 -n 64)

| ngl | PP (t/s) | TG (t/s) | reps | notes |
|---|---|---|---|---|
| 30 | 37.15 ± 0.68 | 0.23 ± 0.00 (tg32, -r 1) | 2 | TG measured separately at n=32 r=1 (full -n 64 r 2 timed out at 600s); CPU-bound on swap-thrashed model halves |
| 50 | 77.68 ± 6.35 | 4.76 ± 0.00 | 2 | clean run, fits in VRAM |
| 64 | 91.60 ± 1.44 | 5.34 ± 0.01 | 2 | full offload, stable |

ngl=40 / 60 not run — curve is monotonic and time budget consumed by ngl=30 swap thrash. **ngl=64 is the optimum** (full-offload wins because partial offload pushes the CPU-side weights into already-saturated zram swap).

### Phase 2: -t sweep at ngl=64 (--flash-attn 0, -p 256 -n 64 -r 2)

| threads | PP (t/s) | TG (t/s) | notes |
|---|---|---|---|
| 6  | 90.66 ± 0.35 | 5.35 ± 0.01 | indistinguishable from higher t |
| 8  | 92.04 ± 0.95 | 5.36 ± 0.01 | safe practical pick (leaves cores for OS) |
| 10 | 93.18 ± 1.94 | 5.39 ± 0.00 | nominal best by ~1% |
| 12 | 91.60 ± 1.44 | 5.34 ± 0.01 | matches the ngl=64 default-threads run |

Threads barely affect throughput at full GPU offload — the Vulkan backend does the heavy lifting; CPU only orchestrates. All 4 runs are within ~3%, well inside VRAM-allocator/scheduler noise.

### Best config so far

```
-ngl 64 -t 10 --flash-attn 0
PP256 = 93.2 t/s
TG64  = 5.39 t/s
```

Practically: **`-ngl 64 -t 8`** is the recommended setting (within noise of t10, leaves 4 cores free for the OS / the ollama-coprocessor / other work). Avoid -t 12 if you want a responsive desktop during inference.

### VRAM observations
`rocm-smi` consistently reported ~530 MB used post-process; live runtime VRAM not captured (process exits before sample). Model + KV at ngl=64 fits comfortably under the 16 GiB UMA cap. The 24 GiB Vulkan-addressable headroom means we have room for KV-cache growth at longer contexts before hitting the wall.

### Next iteration candidates (ranked by expected payoff)

1. **KV cache quant (`-ctk q8_0 -ctv q8_0`)** — nearly halves KV memory; should boost both PP and TG slightly via reduced memory bandwidth, and free VRAM for longer context windows. Lowest-risk lever.
2. **Reduce zram swap usage** — system entered the test with 6.4 GiB swap-in-use; even with cache drops we never had >6.8 GiB available. Logging out of GUI / killing background processes before benching could close the swap-thrash window seen at ngl=30, and might bump ngl=64 TG by a hair.
3. **Flash attention (`-fa 1`)** — RADV STRIX1 advertises `KHR_coopmat` matrix cores, so FA may work. Test cautiously: if it segfaults, fall back. Could be 10-20% TG improvement on attention layers (16 of 64).
4. **Speculative decoding with Qwen3-1.7B as draft** — biggest single TG win available (often 2-3x for code/structured text), but requires draft model that shares vocab. Verify Qwen3-1.7B-GGUF tokenizer matches Qwen3.6-27B before investing setup time.
5. **Cube-memory FFN swap** — pending distillation pipeline (per LOCAL_DISTILL_PLAN.md). Would shrink active params and could push TG into the 8-10 t/s range, but blocked on Phase 2 distill artifacts.
6. **Batch/ubatch tuning (`-b 4096 -ub 1024`)** — only matters for PP (already 91 t/s, not the bottleneck for chat). Skip unless serving many concurrent prompts.

### Blockers / observations

- **No RADV bugs observed.** Vulkan backend stable across all 6 runs at ngl ≥ 50. Zero DEVICE_LOST events.
- **No llama-bench bugs observed.** The mainline build (9725a31) handles qwen35 hybrid arch correctly — confirms Agent J's rebuild succeeded.
- `-ot none` is **not** a valid argument (`-ot` requires `pattern=buffertype`). llama-bench silently dumps `--help` instead of erroring with a clear message — minor UX issue, just omit `-ot`.
- **System RAM pressure is the real ceiling on this hardware.** At any partial offload the system thrashes zram and TG collapses by ~20x. Recommendation: never run this model below ngl=64 on this box. If we ever need a 30B+ model that doesn't fit in 16 GiB VRAM, we need to either (a) free more system RAM first, (b) use a smaller quant (Q3_K_M is ~13 GiB), or (c) move that work to the M70q hub.
- **GPU was in low-power state at session start** ("AMD GPU device(s) is/are in a low-power state. Check power control/runtime_status" warning from rocm-smi). It woke up fine for the bench, but if energy/perf becomes a knob worth turning, look at `/sys/class/drm/card*/device/power_dpm_force_performance_level`.
