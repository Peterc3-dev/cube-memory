# Cube Memory Distillation Status

As of 2026-05-08.

## What exists

### Infrastructure (fully built)

| Component | Path | Status |
|---|---|---|
| CubeMemoryLayer V1 | `phase1/cube_memory_layer.py` | Working, tested |
| CubeMemoryLayer V2 | `phase1/cube_memory_layer_v2.py` | Working, untested on real activations |
| Per-layer trainer | `phase1/per_layer_trainer.py` | Working, supports V1 and V2 |
| End-to-end distiller | `phase1/distill.py` | Working, tested on Qwen3-1.7B |
| FFN swapper | `phase1/swap_ffn.py` | Working |
| FineWeb sampler | `phase1/fineweb_sample.py` | Working |
| BFCL eval harness | `bfcl_eval.py` | Working, Qwen3-8B scores 95.8%/93.5% |
| ggml ops (CPU) | `/tmp/llama-mainline` branch `cube-memory-op` | Working, parallelized |
| ggml ops (Vulkan) | `/tmp/llama-mainline` branch `cube-memory-op` | Working, tiled |
| GGUF roundtrip | `phase1/export_to_gguf.py` + `test_cube_memory_roundtrip.cpp` | Working, max err 1.16e-10 |
| Bootstrap script | `bootstrap_distill.py` | New (this session) |

### Activation cache (from Qwen3.6-27B teacher)

Location: `~/cube-memory-cache/activations/`

| Detail | Value |
|---|---|
| Source model | Qwen3.6-27B-Q4_K_M (16.82 GB, 64 layers, n_embd=5120) |
| Tokens cached | 50,000 (FineWeb-Edu) |
| Layers cached | 8 positions: {3, 11, 19, 27, 35, 43, 51, 59} |
| Format | CACT (bf16 chunks, 25 chunks per side per layer) |
| Total size | 7.7 GB |
| Each layer | ~978 MB (489 MB input + 489 MB output) |

The activations are valid and reusable. No re-extraction needed for
the bootstrap; more tokens (100K-1M) should be extracted for
production training.

### Trained layers (V1, known broken)

Location: `~/cube-memory-cache/trained-layers/`

8 safetensors files, ~90 MB each. Config: d_codebook=256, m=64, p=2,
n_slots=4096, d_value=2048, top_k=4. All trained 5000 steps at lr=1e-3.

**These are V1 checkpoints and are known to fail.** Phase B debug
(2026-04-27) measured:

- Layer 3: 4.7% variance captured (normalized val_mse = 0.953)
- Hypothesis A (loss rescaling + 10x lr): catastrophic divergence
- Hypothesis B (4x capacity, n_slots=16384): no improvement (0.1% delta)

**Root cause**: the VSA structural prior (STE cleanup, frozen codebooks,
single-head retrieval) forces addressing through a discrete
low-cardinality algebraic key. The straight-through estimator discards
most gradient signal about which slot to address. Result: the model
converges to a heavily-bottlenecked linear codebook approximator.

### V2 architecture (untested on real data)

`phase1/cube_memory_layer_v2.py` addresses the three structural failures:

1. **Learned codebooks** (unfrozen, re-normalized to unit modulus each forward)
   -- lets the codebook adapt to the data distribution
2. **Multi-head retrieval** (independent top-k per head, concat values)
   -- n_heads=4 means 4 independent addressing paths, each with its own
   slot partition
3. **Gumbel-softmax cleanup** (soft weighted sum during training, hard argmax
   at eval) -- replaces the STE which killed gradient flow
4. **Gated residual** (learned scalar gate, initialized conservatively at
   sigmoid(-2) ~ 0.12) -- prevents the layer from trying to match the full
   output magnitude immediately

V2 has passed unit tests (`phase1/tests/test_layer.py`-style) but has
never been trained against real Qwen3.6-27B activations.

## What's missing

### 1. Teacher model GGUF (BLOCKER for new activations, NOT for bootstrap)

Qwen3.6-27B-Q4_K_M was deleted from `~/models/` on 2026-04-29 to free
disk for Laguna XS.2. The existing 50K-token activation cache is
sufficient for bootstrap training, but extracting more tokens requires
the GGUF.

**To re-pull:**
```bash
# ~17 GB download
huggingface-cli download unsloth/Qwen3.6-27B-Q4_K_M-GGUF \
  --local-dir ~/models/Qwen3.6-27B-Q4_K_M \
  --include "Qwen3.6-27B-Q4_K_M.gguf"
```

Note: Qwen3.6-27B uses the `LLM_ARCH_QWEN35` architecture which
requires a recent llama.cpp build. The standalone Vulkan build at
`~/builds/llama-cpp-vulkan` (version 5, commit 58e68df) should
support it, but verify with:
```bash
~/bin/llama-cli -m ~/models/Qwen3.6-27B-Q4_K_M/Qwen3.6-27B-Q4_K_M.gguf \
  -p "test" -n 1 --no-display-prompt 2>&1 | head -5
```

### 2. V2 training on real activations (NEXT STEP)

V2 has never been tested against the existing activation cache.
The bootstrap script provides the training loop:

```bash
source ~/rocm-gpu-test/venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0
cd ~/projects/cube-memory

# First: diagnose layer 3 to see baseline stats
python bootstrap_distill.py diagnose --layer 3

# Train V2 on layer 3 (proof-of-concept, ~20 min on iGPU)
python bootstrap_distill.py train --layer 3 --steps 10000

# A/B compare V1 vs V2
python bootstrap_distill.py compare --layer 3 --steps 5000
```

### 3. llama-dump-activations rebuild

The activation extraction tool was built from the `cube-memory-op` branch
at `/tmp/llama-mainline`. That directory may not survive reboots (it's in
/tmp). To rebuild:

```bash
cd /tmp/llama-mainline
git checkout cube-memory-op
cmake --build build --target llama-dump-activations -j$(nproc)
```

If the directory is gone, the branch needs to be recreated from the
commits described in PHASE2_STATUS.md.

### 4. More training data

50K tokens is a small cache. The plan calls for 100K-1M tokens.
After re-pulling the teacher GGUF:

```bash
# Prepare corpus (one-time)
cd ~/projects/cube-memory
python phase1/fineweb_sample.py --target-tokens 100000

# Extract activations (2-6 hours)
/tmp/llama-mainline/build/bin/llama-dump-activations \
  -m ~/models/Qwen3.6-27B-Q4_K_M/Qwen3.6-27B-Q4_K_M.gguf \
  -f ~/cube-memory-cache/corpus/fineweb_100k.txt \
  --layers 3,11,19,27,35,43,51,59 \
  --out-dir ~/cube-memory-cache/activations/ \
  --chunk-tokens 2000 \
  -ngl 30
```

### 5. Sequential re-caching (Phase A step 4e)

The plan calls for re-caching activations through the partially-swapped
model after each layer is fit, so downstream layers train against the
real upstream signal. This requires:

- GGUF export of trained CubeMemoryLayers (Phase D)
- Loading hybrid model in llama-server
- Neither is ready yet

For now, all 8 layers train independently against the original teacher's
activations. Error compounding is deferred to Phase C (end-to-end
calibration).

## Decision point: V2 outcome determines path

If V2 captures **>30% of target variance** on layer 3 at 10K steps:
  - Proceed with full 8-layer pipeline
  - Expected wall time: 8 layers x ~30 min = ~4 hours
  - Move to Phase C (end-to-end calibration)

If V2 captures **10-30%**:
  - Try expanded configs: n_heads=16, n_slots=32768, d_codebook=1024
  - Consider hybrid approach: CubeMemoryLayer for early layers (where
    FFN deltas are small) + dense mini-FFN for deep layers

If V2 captures **<10%** (same as V1):
  - The VSA approach is fundamentally capacity-limited for FFN replacement
  - Pivot to Option 3 from Phase B debug: small dense FFN replacement
    (d_in -> d_ff/4 -> d_in with d_ff=4096, ~80 MB, should capture >50%)
  - Or: reframe as negative result paper

## Available models for teacher inference

| Model | Location | Size | Suitable |
|---|---|---|---|
| Qwen3-8B-Q4_K_M | ~/models/ | 4.7 GB | Yes, smaller teacher for quick iteration |
| Qwen3-1.7B | ~/models/ | ~3.4 GB (HF) | Yes, for smoke tests |
| Qwen3.5-0.8B-Q4_K_M | ~/models/ | ~0.5 GB | Toy only |
| Gemma-4-E4B | ~/models/ + ollama | ~9.6 GB | Alternative teacher |
| Qwen3.6-27B-Q4_K_M | DELETED | 16.82 GB | Primary teacher, needs re-pull |

The existing activation cache was generated from Qwen3.6-27B and should
be used for V2 training. Qwen3-8B could serve as a lighter teacher for
hyperparameter sweeps (extract new activations from it first).

## ROCm venv status

```
Path: ~/rocm-gpu-test/venv
Python: 3.11.14
PyTorch: 2.5.1+rocm6.2
CUDA available: True (gfx1100 via HSA_OVERRIDE)
safetensors: installed
numpy: 2.3.5
```

Grandfathered ROCm exception for training. Activate with:
```bash
source ~/rocm-gpu-test/venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

## Commands to start the pipeline

```bash
# 1. Activate ROCm venv
source ~/rocm-gpu-test/venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# 2. Navigate to project
cd ~/projects/cube-memory

# 3. Run diagnostics
python bootstrap_distill.py diagnose --layer 3

# 4. Train V2 proof-of-concept on layer 3
python bootstrap_distill.py train --layer 3 --steps 10000

# 5. If capture > 30%, run full pipeline
python bootstrap_distill.py pipeline --steps 10000

# 6. After all layers trained, evaluate BFCL
# (requires llama-server with hybrid model -- not yet wired)
```
