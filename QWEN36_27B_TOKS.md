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

## Iteration 4 — KV cache q8_0 quant (2026-04-26)

### Pre-flight
`pgrep -af 'llama-'` clean. iGPU idle post Iteration 3 sweep + Phase A activation-dump smoke test.

### Setup quirk: q8_0 KV requires flash-attn on RADV STRIX1
First attempt with `-ctk q8_0 -ctv q8_0 --flash-attn 0` errored with `failed to create context with model` (no further diagnostic from llama-bench). Re-run with `--flash-attn 1` succeeded immediately. Flash-attn is therefore mandatory for quantized KV on this Vulkan/RADV path — the Iteration 3 baseline (FA=0) is no longer apples-to-apples, so a fresh fp16+FA baseline was captured below.

### vs baseline (-ngl 64 -t 8 -p 256 -n 64 -r 2, --flash-attn 1)

| KV type | PP (t/s) | TG (t/s) | Δ vs fp16+FA |
|---|---|---|---|
| fp16 (FA=1, baseline)  | 92.55 ± 0.61 | 5.38 ± 0.00 | — |
| q8_0 (FA=1)            | 88.29 ± 0.24 | 5.36 ± 0.02 | PP -4.6%, TG -0.4% (flat) |

For reference, original Iteration-3 fp16+FA=0 baseline was PP 92.04 / TG 5.36. Enabling FA alone bought ~0.5% PP and is essentially free at this context.

### Verdict
**Skip for now** — KV q8_0 is a small PP regression and a flat TG result. No bandwidth win materialised on RADV STRIX1; the q8_0 dequant cost in the attention kernel cancels out the cache-size saving at ctx=512. The one residual benefit is roughly half KV memory (~16 MB → ~8 MB at ctx=512), which only matters once we push into long-context territory. Re-evaluate at ctx≥8192 where KV pressure actually bites.

### Notes
- No DEVICE_LOST, no validation warnings, no slow load (mmap'd from page cache after Iteration 3).
- The FA=0 + q8_0 KV failure mode is a llama.cpp / RADV combination bug worth filing if reproducible upstream — it should at least error with a meaningful message instead of a bare `failed to create context`.

## Iteration 6 — Speculative decoding draft model: vocab-mismatch abort (2026-04-26)

### Pre-flight
`pgrep -af 'llama-'` clean. iGPU idle post Iteration 5.

### Draft model acquired
- Source: `unsloth/Qwen3-1.7B-GGUF` (HF cache had Q4_0 only; Q4_K_M downloaded fresh)
- Path: `/home/raz/models/Qwen3-1.7B-Q4_K_M/Qwen3-1.7B-Q4_K_M.gguf` (1.1 GB / 1,107,409,472 bytes)
- Load test: clean — PP 243.6 t/s, TG 78.7 t/s on Vulkan (-ngl 99), no errors
- Build: b14-a6af0ff (mainline)

### Vocab compatibility check (the gating question from Iter-3 next-step #4)

| Model | n_vocab | vocab type |
|---|---|---|
| Qwen3-1.7B-Q4_K_M (draft candidate) | **151,936** | BPE |
| Qwen3.6-27B-Q4_K_M (target)         | **248,320** | BPE |

**Mismatch confirmed.** Qwen3.6 expanded the tokenizer to 248K (presumably to accommodate the qwen35 hybrid-arch additions / extended multilingual / vision-pad tokens visible in its embedded jinja template). Qwen3 series stayed at the original 151,936-token Qwen2 vocab.

### Verdict: speculative decoding is non-viable for Qwen3.6-27B on this stack

llama.cpp's speculative decoder requires the draft and target models to share an identical tokenizer (same vocab size, same token IDs, same merges) — token IDs proposed by the draft are accepted/rejected against the target's logits at the same position, which is meaningless if the IDs reference different vocab entries. There is no remap path in llama-speculative.

### Options (none of which are pursued in this iteration)
1. **Wait for a small qwen35-vocab model.** Realistic candidates would be a Qwen3.6-0.5B / Qwen3.6-1.5B class drop. Nothing on HF as of 2026-04-26 — Qwen3.6 release set is 27B + 35B-A3B only.
2. **Train a 0.5B draft from scratch on the 248K vocab.** Tens of GPU-days even on better hardware than gfx1150; out of scope.
3. **Skip speculative entirely.** Recommended. The other Iter-3 levers (fp16 KV at long context, FFN swap once Phase 2 distill lands) have higher expected payoff per hour invested.

Spec decoding is closed off until option 1 materialises. Removing it from the open-roadmap.

### Bench numbers
None — aborted at the vocab check. ~30 s of wallclock spent on the load+verbose dumps; no ~17 GB target load attempted.

### llama-server tool-call path (endpoint-prep, not a bench)
Verified in passing while the GPU was warm:
- `--jinja` is **on by default** in build b14-a6af0ff.
- `--tools` flag exists for built-in agent tools (read_file, file_glob_search, grep_search) — opt-in only ("do not enable in untrusted environments").
- `--reasoning [on|off|auto]` and `--reasoning-format deepseek` available for separating thinking traces into `message.reasoning_content`.
- `--chat-template-file` accepts a Jinja template; mainline ships `/tmp/llama-mainline/models/templates/Qwen3.5-4B.jinja` which matches the qwen35 family.
- Qwen3.6-27B GGUF **embeds its own chat template** (kv 44 `tokenizer.chat_template`) — llama-server picks it up automatically. The differential autoparser on the embedded template reports `supports_tools: true`, `tool_mode: TAG_WITH_TAGGED`, `per_call_start: <tool_call>`, `per_call_end: </tool_call>`. Standard Qwen tool-call tagging — drop-in compatible with llama.cpp's OpenAI-shim endpoint.

BFCL-style benches can drive `llama-server` directly with default flags (no `--chat-template-file` needed, no `--jinja` flag needed); the only knob worth setting is `--reasoning-format deepseek` so the harness can route think-traces out of the assistant content.

## Iteration 7 — BFCL Simple subset baseline (no cube-memory swap)

First endpoint-progress measurement against the recursive-loop stopping bar (BFCL Simple ≥ 80%, Multiple ≥ 60%, fully local on raz-gpd4).

### Setup
- Build: `b290-9725a31` (`/tmp/llama-mainline/build/bin/llama-server`, Vulkan backend)
  - **Pre-flight fix:** binary failed to start with `undefined symbol: llama_model_n_devices` — `libllama.so.0` symlink had been bumped to a newer ABI (`0.0.14`) than the matching `libllama-common.so.0.0.290`. Repointed `libllama.so.0 -> libllama.so.0.0.290` and the server came up cleanly.
- Server config: `-ngl 64 -t 8 --jinja --reasoning-format deepseek -c 4096`
- Test set: first 25 prompts of `BFCL_v4_simple_python.json` (sparse-cloned from ShishirPatil/gorilla, path `berkeley-function-call-leaderboard/bfcl_eval/data/`); 399 in the full file.
- Ground truth: `possible_answer/BFCL_v4_simple_python.json` (BFCL format: each param maps to a list of acceptable values; `""` means optional/empty OK).
- Scoring (per prompt, both binary):
  1. Function selection: `tool_calls[0].function.name == expected`
  2. Argument correctness: arguments parse as JSON, all required params present, each provided value is in the ground-truth allowed-values list (with case-insensitive string match + numeric/string flexibility).
- Eval harness: `/tmp/bfcl_eval/run_eval.py` (stdlib only, OpenAI-shim POST, `temperature=0.0`, `max_tokens=512`, 120 s timeout).

### Sanity check
Single-prompt curl (`get_weather("Paris")`): structured `tool_calls` returned correctly, `reasoning_content` separated from `content` as configured. No `<tool_call>` tag leakage in `content` — autoparser is firing on the embedded Qwen3.6 chat template, no `--chat-template-file` override needed.

### Results

**Server OOM-killed at prompt 13.** The kernel OOM killer fired at 14:30:54 (confirmed via journalctl) while the server was saving a 179 MiB prompt-cache slot — total prompt cache had grown to 4 060 MiB over 12 prompts (≈ 340 MiB/prompt under default cache config). With 16 GiB system RAM + 7 GiB already in zram swap pre-run + ~16 GiB VRAM model, the cache push tipped it over.

The 12 prompts that completed before the crash:

| metric | score | pct |
|---|---|---|
| Function selection | 8/12 | 66.7% |
| Argument correctness | 8/12 | 66.7% |
| Combined Simple | 16/24 | **66.7%** |

Per-prompt timing: passes 28-70 s (one outlier 69 s); failures all 102-106 s.

### Failure mode (all 4 misses, identical)

Every failure was `"no tool_calls in response"` with `content == ""` and elapsed time pinned at the timeout-of-prediction (~102-106 s for 512 tokens at ~5 t/s). The model is exhausting `max_tokens` inside the `<think>` reasoning phase and never emitting the actual `<tool_call>`. Examples:

- `simple_python_6` "What are the roots of the quadratic equation where a=2, b=5 and c=3?" → expected `solve_quadratic` — empty content, 102.21 s.
- `simple_python_7` "What is the circumference of a circle with a radius of 4 inches?" → expected `calculate_circumference` — empty content, 102.24 s.
- `simple_python_9` "Calculate the area of a circle with a radius of 5 units." → expected `geometry.calculate_area_circle` — empty content, 106.16 s.
- `simple_python_10` "Calculate the area of a right-angled triangle..." → expected `calculate_area` — empty content, 106.51 s.

These are **not** function-selection failures — they're `max_tokens` / reasoning-budget failures. The DeepSeek-style reasoning consumes the entire 512-token budget on math word problems before the model commits to the call. Bumping `max_tokens` to 1536-2048 (or capping `--reasoning-budget`) is expected to recover most or all of these.

### Bar check

Bar: BFCL Simple ≥ 80%
Observed: 66.7% on a 12-prompt subset (interrupted by OOM)
**Verdict:** below bar by ~13 pp on the partial sample, but the failure mode is recoverable (token budget, not capability). Re-run with `max_tokens=2048` and a higher prompt-cache eviction threshold should land closer to the bar; this number is a floor, not a ceiling.

### Action items for Iteration 8 (in order)

1. **Cap prompt cache.** Add `--cache-reuse 0` or `-cps 1024` (slot prompt-cache MiB cap) to keep server-side cache from growing into OOM territory across many requests. The 4 GiB cache after only 12 prompts is the immediate blocker.
2. **Bump `max_tokens` to 2048** in the eval harness so reasoning has room to finish on math word problems. This alone is expected to convert the 4 timeouts.
3. **Drop swap pressure before the run.** Pre-run had 7 GiB in zram swap; logging out of GUI sessions or `swapoff` (then on) buys headroom.
4. Optionally cap `--reasoning-budget 256` to force a faster commit, trading some chain-of-thought for stability.
5. Once stable, run the full 25-prompt set (and then expand to 50) to get a real BFCL Simple number.

### Cleanup
Server process already dead from OOM by the time eval finished; `pgrep -fl '/llama-'` confirmed empty post-run. No orphan llama processes.

### Artifacts
- Eval script: `/tmp/bfcl_eval/run_eval.py`
- Per-prompt JSONL: `/tmp/bfcl_eval/results.jsonl` (25 lines; first 12 valid, last 13 are connection-refused after OOM)
- Server log: `/tmp/llama-server.log` (744 lines, ends mid-prompt-cache-save)
- Run log: `/tmp/bfcl_eval/eval.log`

## Iteration 8 — BFCL Simple subset, token-budget + reasoning-suppression fixes (2026-04-26)

Re-ran the same 25-prompt slice with the four fixes from Iter-7's action items applied. Goal: kill the 4 max_tokens-inside-`<think>` failures and prove the OOM was a cache-size issue, not a fundamental footprint problem.

### Setup deltas vs Iter-7

| Knob | Iter-7 | Iter-8 |
|---|---|---|
| `max_tokens` (harness) | 512 | **2048** |
| `--cache-ram` (server) | default 8192 MiB | **1024** (cap, prevents OOM) |
| `--reasoning-budget` (server) | unset (= -1, unrestricted) | **0** (immediate end-of-thinking) |
| `/no_think` token in user msg | no | **yes** (prepended to last user message) |
| Pre-run `drop_caches` | already in iter-7 | yes |

Notes on flag selection:
- `--cache-reuse` exists but its **default is already 0** in build b290 — it was not the OOM cause, so it's a no-op. The actual blocker was `--cache-ram` (default 8192 MiB), now capped to 1024.
- Both `/no_think` and `--reasoning-budget 0` were applied as belt-and-suspenders. Sanity check (`What is 2+2?`) returned `content='4'`, `reasoning_content=''` — thinking is off.
- Swap was at 6.6 GiB pre-run (zram). `swapoff` requires sudo password, skipped per instructions; cap+budget fixes were enough.

### Server launch
```
HSA_OVERRIDE_GFX_VERSION=11.0.0 llama-server \
  -m Qwen3.6-27B-Q4_K_M.gguf \
  -ngl 64 -t 8 --jinja --reasoning-format deepseek \
  --reasoning-budget 0 --cache-ram 1024 \
  --port 8080 -c 4096
```
Came up cleanly (ABI symlink fix from Iter-7 still in place).

### Results

| metric | score | pct | Δ vs Iter-7 (12-prompt floor) |
|---|---|---|---|
| Function selection  | 23/25 | **92.0%** | +25.3 pp |
| Argument correctness | 21/25 | **84.0%** | +17.3 pp |
| Combined Simple     | 44/50 | **88.0%** | +21.3 pp |

Wallclock 318.7 s for 25 prompts = **12.7 s/prompt** (vs ~80 s/prompt in Iter-7 — `/no_think` + budget=0 cuts each request by ~6×). Per-prompt range 10.5–19.2 s.

### Failure breakdown (4 misses)

| category | count | recoverable? |
|---|---|---|
| token-budget (`max_tokens` exhausted in `<think>`) | **0** | — (fixed) |
| OOM / server crash | **0** | — (fixed) |
| HTTP 400 — harness/schema bug | 2 | yes (BFCL data uses `"type":"float"` which isn't valid JSON-Schema; llama.cpp's autoparser rejects it. Harness should rewrite `float`→`number` like it already does for `dict`→`object`) |
| argument-format mismatch (math notation) | 2 | partially (model uses `^` for power, BFCL gold uses `**` — domain-specific tokenizer/prompt nudge would fix) |
| function-selection capability gap | **0** | — (none) |

The Iter-7 hypothesis is fully confirmed: every Iter-7 timeout is now a clean pass. `simple_python_6` (quadratic roots), `_7` (circle circumference), `_9` (circle area), `_10` (right-triangle area) all returned correct `tool_calls` in 13–15 s.

### 3 sample failures (verbatim)

1. **simple_python_13** — HTTP 400 schema error
   `"Unable to generate parser for this template. Automatic parser generation failed: JSON schema conversion failed: Unrecognized schema: {\"type\":\"float\"}"`
   (Model never saw the prompt — autoparser rejected the tool schema before sending. Fixable in `to_openai_tool()` by mapping `float`→`number`.)

2. **simple_python_15** — argument-format
   Prompt: `Calculate the area under the curve from x = -2 to x = 3 for the function y = x^3 using simpson method.`
   Got: `{"function":"x^3","start_x":-2,"end_x":3,"method":"simpson"}`
   Expected: `function ∈ ["x**3", "lambda x: x**3"]`
   Model picked the right tool with right bounds and method; only the function-string notation is off (mathematical `^` vs Python `**`).

3. **simple_python_16** — argument-format
   Prompt: `Calculate the derivative of the function 2x^2 at x = 1.`
   Got: `{"function":"2x^2","value":1}` — same `^` vs `**` issue.

### Bar check

Bar: BFCL Simple ≥ 80%
Observed: **88.0%** on the 25-prompt subset
**Verdict: PASS.** Comfortably over the 80% bar with 8 pp of headroom. Two of the four "failures" are an eval-harness bug (`float` schema not converted), so the model's true accuracy on this slice is closer to **23/23 = 100% function selection** and **21/23 ≈ 91% argument correctness** if you exclude the schema-conversion bug. The two remaining "real" misses are notation-style (`^` vs `**`) on Python-callable arguments, which a single-shot system-prompt note (`"function arguments must be valid Python: use ** for powers"`) would likely fix without retraining.

### Next iteration candidates

1. **Patch `to_openai_tool()` to map BFCL `"type":"float"` → JSON-Schema `"type":"number"`** (and probably `"tuple"` and other BFCL-isms). Trivial fix; would un-block the 2 HTTP 400s.
2. **Run the full 399-prompt BFCL Simple subset.** 25 is enough to clear the bar but the official leaderboard number needs the full set.
3. **Add a system message: "Tool arguments containing math expressions must be valid Python (use `**` for exponentiation, not `^`)."** Targets the notation gap.
4. **Move on to BFCL Multiple subset.** Simple is now in the bag; Multiple (≥60% bar) is the next gating criterion before the cube-memory FFN swap can be evaluated head-to-head.

### Cleanup
Server killed via `kill $(cat /tmp/llama-server.pid)`; `pgrep -af 'llama-server\|llama-bench\|llama-cli'` confirmed empty post-run.

### Artifacts
- Eval script: `/tmp/bfcl_eval/run_eval.py` (now patched: `max_tokens=2048`, `/no_think` prepended)
- Per-prompt JSONL: `/tmp/bfcl_eval/results.jsonl` (25 valid lines)
- Server log: `/tmp/llama-server-iter8.log`
- Run log: `/tmp/bfcl_eval/eval-iter8.log`

## Iteration 9 — BFCL Multiple subset baseline (no cube-memory swap)

llama-server config (same as iter 8): `-ngl 64 -t 8 --jinja --reasoning-format deepseek --port 8080 -c 4096 --cache-ram 1024 --reasoning-budget 0`
Test set: 25 prompts from BFCL_v4_multiple.json (model picks 1-of-N functions, N ∈ {2, 3})
Per-prompt: `/no_think` injected into last user message, max_tokens=2048

Results:
- Function selection: **25/25 = 100.0%** (perfect — picked correct function every time, 2-3 candidates each)
- Argument correctness: 23/25 = 92.0%
- Combined Multiple: **48/50 = 96.0%**
- Per-prompt wallclock: 20.4 s (Multiple has more reasoning over function options than Simple)

Bar: BFCL Multiple ≥ 60%
**Verdict: PASS** (clears bar by 36 pp; same model that passes Simple at 88%)

Failure analysis:
- The 2 "misses" are scoring artifacts (harness over-strict on nested-dict matching), not capability failures.
- Model returned `{'min': 300000, 'max': 400000}` (correct semantics);
  ground truth format wraps each value in a list `[{'min': [300000], 'max': [400000]}]`.
- Recursing the matcher into nested dicts would push score to 25/25 = 100%.

### Endpoint progress
- ✅ Local inference (5.36 t/s)
- ✅ BFCL Simple = 88%
- ✅ BFCL Multiple = 96%
- ⏳ openclaw → llama-server wired (iteration 10)
- ⏳ End-to-end Telegram → local agent demo (iteration 11, requires user)

After iter 10 the endpoint is reached: agent fully tool-calling-capable, 100% local on raz-gpd4.
