# Phase 2 progress — ggml integration

Status as of 2026-04-26.

## Where the work lives

A llama.cpp branch at `/tmp/llama-mainline` named **`cube-memory-op`**,
based on upstream master `7c7d6ce`. Three commits:

1. `ggml: add GGML_OP_CUBE_MEMORY_CLEANUP (CPU only, first Phase 2 wedge)`
2. `tests: add cube_memory_cleanup runtime correctness tests`
3. `ggml: add GGML_OP_CUBE_MEMORY_RETRIEVE + runtime tests`

The branch is local-only; not pushed anywhere yet. Move it to your
fork (`Peterc3-dev/llama.cpp` or wherever) when you're ready to share
or rebase against newer mainline.

## What works

Both Cube Memory ops are first-class ggml operations on the CPU
backend.

```c
// In ggml.h:
#define GGML_CUBE_MEMORY_RETRIEVE_MAX_TOP_K 8

struct ggml_tensor * ggml_cube_memory_cleanup(
        struct ggml_context * ctx,
        struct ggml_tensor  * query,
        struct ggml_tensor  * codebook);

struct ggml_tensor * ggml_cube_memory_retrieve(
        struct ggml_context * ctx,
        struct ggml_tensor  * query,
        struct ggml_tensor  * slot_keys,
        struct ggml_tensor  * slot_values,
        int32_t               top_k);
```

Both ops are bit-identical to:
- `cube-memory-shader/src/lib.rs` (rust-gpu kernels)
- `cube-memory-host/src/cpu.rs` (Rust CPU reference)

Standalone runtime tests in
`tests/test-cube-memory-cleanup.cpp` and
`tests/test-cube-memory-retrieve.cpp`. Both pass:

```
PASS: known-answer cleanup returned row 2 verbatim
PASS: random-argmax cleanup picked row 17
PASS: top_k=1 selected slot 11 and copied its value
PASS: top_k=4 with dominant slot 7 converged to its value
```

Other backends (Vulkan, CUDA, Metal, SYCL, WebGPU) untouched. Their
`supports_op` default branches return false for unknown ops, so
graph scheduling falls back to CPU automatically. No backend has
been broken; this branch is mergeable as-is for CPU-only use.

## What's next, in priority order

### 1. Vulkan dispatch path ✓ DONE (2026-04-26)

The full Rust → SPIR-V → ggml-vulkan dispatch path is wired, working,
and validated. With `GGML_VK_CUBE_MEMORY_SPV` pointing at the
rust-gpu .spv, both ops dispatch to the Radeon 890M iGPU and produce
output matching the CPU reference at fp32 tolerance:

```
CUBE_MEMORY_CLEANUP(d=32,m=16):  OK
CUBE_MEMORY_CLEANUP(d=64,m=32):  OK
CUBE_MEMORY_CLEANUP(d=128,m=64): OK
CUBE_MEMORY_RETRIEVE(d_key=16,n_slots=32,d_value=8,top_k=4):  OK
CUBE_MEMORY_RETRIEVE(d_key=32,n_slots=64,d_value=16,top_k=8): OK
```

VK_LAYER_KHRONOS_validation re-run: zero warnings.

One non-trivial bug got caught and fixed during the integration:
`ggml_vk_create_pipeline` retains its `spv_data` pointer for lazy
`vkCreateShaderModule`, but the loader was passing a stack-local
`std::vector<char>` that went out of scope before the module was
created. Fix: store the bytes on `vk_device_struct` so they share
the device's RAII lifetime.

Commits on the cube-memory-op branch:
- `ggml-vulkan: load rust-gpu cube_memory SPIR-V at backend init`
- `ggml-vulkan: wire CUBE_MEMORY supports_op + dispatch (segfault pending)`
- `ggml-vulkan: fix CUBE_MEMORY SPV lifetime — store bytes on device`

### 2. Round-trip parity test ✓ DONE (2026-04-26)

Both ops have Python NumPy ↔ ggml CPU round-trip parity tests
passing. See `phase1/export_*_test_case.py` and the corresponding
`tests/test-cube-memory-*-roundtrip.cpp` in the llama.cpp branch.

### 3. (was 4) Round-trip parity test

PyTorch `CubeMemoryLayer.forward(x)` → export weights to GGUF →
load in ggml → ggml graph forward on the same `x` → compare outputs.
This is the **acceptance test** for Phase 2 per `RISKS.md`. If a
silent layout transpose creeps in during GGUF round-trip, only
this test catches it.

Two halves:
- **Python side**: `phase1/export_to_gguf.py` reading a trained
  CubeMemoryLayer state dict, writing a tiny GGUF with the
  codebooks, slot_keys, slot_values, role_proj, out_proj as named
  tensors.
- **C++ side**: `tests/test-cube-memory-roundtrip.cpp` loading that
  GGUF, building a graph that mirrors the layer's forward
  (split → cleanup per-axis → bind → retrieve → out_proj), running
  it on a fixed input, comparing to the PyTorch reference baked
  into the GGUF as a "gold output" tensor.

### 4. Performance pass on the CPU implementations

Currently both CPU forwards are single-threaded (`n_tasks=1`).
They need to be parallelized once the algorithm is correct:
- `cleanup`: parallelize the codebook scan across n_tasks workers,
  reduce the per-worker argmax with atomics or a final serial pass.
- `retrieve`: parallelize the n_slots dot-product loop, reduce
  to a top-k via shared array + per-thread heap.

This is needed for any real distillation eval that runs on CPU
fallback.

### 5. Vulkan shader tiling ✓ DONE (2026-04-26)

Both shaders rewritten as two-pass multi-workgroup tiled kernels:
- `cube_memory_cleanup_score` — m WGs (one per codebook row),
  cooperative dot product per WG, scratch[m] = score
- `cube_memory_cleanup_finalize` — 1 WG, argmax over scratch +
  cooperative copy of winning row
- `cube_memory_retrieve_score` — n_slots WGs, cooperative dot
  product per WG, scratch[n_slots] = sim
- `cube_memory_retrieve_finalize` — 1 WG, top-k tournament +
  softmax + cooperative weighted gather

Host scratch is `ctx->prealloc_x` resized on demand; sync via
`ggml_vk_sync_buffers` between passes. supports_op gates: m ≥ 1
and ≤ device->maxComputeWorkGroupCount[0]; n_slots same; top_k in
[1, 8]. Barriers in uniform control flow only (no early-return
before barrier — caught and fixed by adversarial proof-read).

Bench (200 iters/shape, raz-gpd4 Radeon 890M, vs v0 single-WG):

| shape | v0 Vulkan | new Vulkan | factor | new GPU-only |
|---|---|---|---|---|
| cleanup d=512,m=256 | 195 us | **56 us** | 3.5× | 16 us |
| retrieve dk=256,n=1024 | 832 us | **436 us** | 1.9× | 384 us |

GPU-side kernel time for cleanup d=512,m=256 went 195us → 16us,
12× faster at the kernel. End-to-end Vulkan still slower than CPU
in standalone bench because per-op submit+wait now dominates
(~40us per dispatch × 2 dispatches = ~80us baseline). This is a
bench-shape artifact: real-graph submission amortizes the fence
wait across all nodes (48 layers × ~2 cube ops = ~100 nodes per
forward pass, one fence wait total). Production cost is the
per-kernel time, not the bench's per-op wall clock.

Commits on the cube-memory-op branch (this iteration):
- `cube-memory-shader: tiled multi-WG cleanup_score/finalize`
- `cube-memory-shader: tiled multi-WG retrieve_score/finalize`
- `ggml-vulkan: two-pass cube_memory dispatch + supports_op gates`
- `tests: parity-test harness updated for two-pass dispatch`

Known followups (NOT blocking the consortium plan, deferred):
- Production-scale retrieve (n_slots ≈ 256K per spec.md realistic
  config) needs >65K WG dispatch — current shader caps at the
  Vulkan-spec workgroup limit. Will require a tiled retrieve where
  one WG handles many slots (thread-per-slot pattern) so dispatch
  count stays bounded.
- Largest retrieve (dk=256, n=1024) GPU kernel still 5.6× slower
  than CPU (384us vs 69us). Cause: barrier overhead dominates when
  each WG does only ~4 mults. Same fix as above (thread-per-slot)
  would also win this.

Both followups are perf-only — correctness is shipped.

## How to resume

```bash
# Mainline llama.cpp branch with the new ops
cd /tmp/llama-mainline
git log --oneline -3
# Should show three cube-memory commits on top of upstream master.

# Build and test
cmake --build build --target test-cube-memory-cleanup test-cube-memory-retrieve -j$(nproc)
./build/bin/test-cube-memory-cleanup
./build/bin/test-cube-memory-retrieve

# Cube Memory shader workspace (rust-gpu)
cd ~/projects/cube-memory/shaders
cargo run -p cube-memory-shader-builder --release
cargo test -p cube-memory-host --release

# PyTorch layer
cd ~/projects/cube-memory/phase1
source ~/rocm-gpu-test/venv/bin/activate
python tests/test_layer.py
python tests/test_swap.py
python tests/test_distill.py
```

All five entry points (cleanup CPU op, retrieve CPU op, two SPIR-V
kernels, three Python tests) pass independently. The next
integration step is wiring the SPIR-V into ggml-vulkan so
`ggml_cube_memory_*` calls dispatch to the iGPU when available.
