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

### 1. Vulkan dispatch path (the big one)

Wire the rust-gpu SPIR-V at
`shaders/target/spirv-builder/spirv-unknown-vulkan1.2/release/deps/cube_memory_shader.spv`
into ggml-vulkan. Tasks:

- At backend init, read the .spv (or embed via `xxd -i` at build time).
- For each entry point name (`cube_memory_cleanup`,
  `cube_memory_retrieve`), call `ggml_vk_create_pipeline` with the
  push-constant size and descriptor layout.
- Add `case GGML_OP_CUBE_MEMORY_CLEANUP:` and `case
  GGML_OP_CUBE_MEMORY_RETRIEVE:` to the Vulkan
  `supports_op` switch (return true when shapes match the kernel's
  expectations).
- Add corresponding cases to the dispatch site (somewhere around
  `ggml_vk_build_graph`); set push constants from `op_params`,
  bind buffers, dispatch the kernel.
- The host harness in `shaders/cube-memory-host/src/gpu.rs` is the
  reference for the descriptor-set layout and dispatch shape.

Estimated effort: 2-4 days. Sweep policy says: clean-context
bug-sweep before each commit; integration sweep when the path is
end-to-end working.

### 2. Round-trip parity test

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

### 3. Performance pass on the CPU implementations

Currently both CPU forwards are single-threaded (`n_tasks=1`).
They need to be parallelized once the algorithm is correct:
- `cleanup`: parallelize the codebook scan across n_tasks workers,
  reduce the per-worker argmax with atomics or a final serial pass.
- `retrieve`: parallelize the n_slots dot-product loop, reduce
  to a top-k via shared array + per-thread heap.

This is needed for any real distillation eval that runs on CPU
fallback.

### 4. Vulkan shader hardening

The rust-gpu shaders have `cube_memory_cleanup` and
`cube_memory_retrieve` as single-thread workgroups (workgroup size
64, 63 idle). Real perf needs:
- `cube_memory_cleanup`: parallel scan with subgroup reductions.
- `cube_memory_retrieve`: parallel dot-products + parallel top-k.

Shadow-tracking against the host-harness parity tests at
`shaders/cube-memory-host/tests/parity.rs` catches regressions.

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
