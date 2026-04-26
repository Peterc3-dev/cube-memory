# Cube Memory shaders (rust-gpu)

GPU compute kernels for the Cube Memory FFN-replacement layer,
written in Rust and compiled to SPIR-V via
[rust-gpu](https://github.com/Rust-GPU/rust-gpu). Designed to be
loaded by llama.cpp's Vulkan backend (or any Vulkan host) at the
Phase 2 integration stage described in
`../phase1/PLAN.md`.

## Why Rust shaders here

GLSL is the default in the llama.cpp Vulkan backend. We're using
rust-gpu for *new* Cube Memory kernels because:

- Type safety on push-constant and buffer layouts catches mismatches
  at compile time instead of "wrong output, no error" at runtime.
- The bind/unbind/superpose primitives have clean Rust expressions;
  GLSL with manual complex math is harder to keep correct.
- `cargo test` can run the same kernels on the CPU via Rust's
  built-in execution path — useful for unit-testing FHRR algebra.

The host-side Vulkan plumbing (descriptor sets, pipeline creation,
dispatch) stays in C++ where llama.cpp lives. This crate produces
SPIR-V binaries that the C++ side loads via `vkCreateShaderModule`.

## Building

Requires a specific Rust nightly toolchain pinned in
`rust-toolchain.toml`. Install once:

```bash
rustup install nightly-2024-11-22
rustup component add rust-src rustc-dev llvm-tools-preview \
    --toolchain nightly-2024-11-22
```

Then:

```bash
cd ~/projects/cube-memory/shaders
cargo run -p cube-memory-shader-builder --release
```

On success the path to the produced SPIR-V file is printed; you can
also find it under
`target/spirv-unknown-vulkan1.2/release/cube-memory-shader.spv`.

If the build fails with a panic inside the SPIR-V codegen backend,
check that the toolchain in `rust-toolchain.toml` matches the one
expected by the `spirv-builder` crate version pinned in `Cargo.toml`.
Mismatches are the most common cause of obscure failures.

## Current shaders

### `fhrr_bind`

Element-wise complex multiplication. Two input phasor vectors of
length N, one output. Workgroup size 64.

Dispatch:
```text
groupCountX = ceil(N / 64)
groupCountY = 1
groupCountZ = 1
```

Buffers (descriptor_set=0):
```text
binding 0: in_a    — readonly  [Vec2; N]
binding 1: in_b    — readonly  [Vec2; N]
binding 2: out     — readwrite [Vec2; N]
```

Push constants:
```text
struct { uint32_t n; }
```

## Roadmap

In rough order of when each kernel is needed:

1. `fhrr_bind` — done (this skeleton)
2. `fhrr_unbind` — same shape, conjugate the second operand
3. `fhrr_unitize` — element-wise normalize to unit modulus
4. `fhrr_superpose` — sum + unitize (the bundle operation)
5. `cube_memory_cleanup` — argmax cosine match against a frozen
   codebook of M phasors, returning the snapped phasor + index
6. `cube_memory_retrieve` — top-k slot-key dot product +
   softmax-weighted slot-value gather. The full forward path.

The early kernels (1–4) validate the toolchain and let us write
unit tests against the Phase 0 CPU implementation
(`../phase0/fhrr.py`). Numerical parity between rust-gpu output
and the PyTorch reference is the gate before plugging into the
real ggml-vulkan integration.

## Integration with llama.cpp ggml-vulkan

(Phase 2 work, not done yet.) The expected path:

1. Add a new ggml op `GGML_OP_CUBE_MEMORY` in `ggml/include/ggml.h`.
2. In `ggml/src/ggml-vulkan/ggml-vulkan.cpp`, register a pipeline
   that loads the SPIR-V produced here. Use `ggml_vk_create_pipeline`
   with the bytes loaded from `cube-memory-shader.spv` at startup,
   or vendored at build time by a small CMake step that runs
   `cargo run -p cube-memory-shader-builder` before configure.
3. The op's `compute_forward` walks src tensors and dispatches.

Two practical wrinkles:
- ggml's tensor types must round-trip the slot-store and codebook;
  we'll likely need a new `GGML_TYPE_FHRR` (complex64) or store
  phasors as (re, im) pairs in `GGML_TYPE_F16` and reinterpret.
- The shader output dimensions need to match what the host computes
  for workgroup count; mismatches are silent and produce wrong
  output, not crashes.

## License

MIT, matching the rest of the project.
