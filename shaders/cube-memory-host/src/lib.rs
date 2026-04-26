//! Host-side parity harness for the Cube Memory shaders.
//!
//! Two halves:
//!
//! - `cpu` module — reference implementations of FHRR / Cube Memory
//!   primitives in pure Rust. Used as ground truth in tests.
//! - `gpu` module — wgpu setup, SPIR-V loading, and a `Kernel` helper
//!   that wraps "load entry point → bind buffers → dispatch → readback".
//!
//! Tests in `tests/parity.rs` generate random inputs, run both paths,
//! and assert the outputs match within a numerical tolerance. The harness
//! catches GPU/CPU divergence before the kernels move to ggml-vulkan in
//! Phase 2.

pub mod cpu;
pub mod gpu;

/// Path to the SPIR-V module produced by `cube-memory-shader-builder`.
/// Resolved relative to the workspace target dir at test time.
pub const SHADER_RELATIVE_PATH: &str =
    "spirv-builder/spirv-unknown-vulkan1.2/release/deps/cube_memory_shader.spv";
