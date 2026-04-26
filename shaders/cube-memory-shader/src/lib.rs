//! Cube Memory compute shaders, written in Rust → SPIR-V via rust-gpu.
//!
//! This is the GPU side of the FHRR Memory Layer described in
//! `~/projects/cube-memory/SPEC.md`. The host-side Vulkan plumbing
//! (descriptor sets, pipeline creation, dispatch) lives in the
//! llama.cpp ggml-vulkan backend; this crate only provides the
//! kernel.
//!
//! First kernel: `fhrr_bind` — element-wise complex multiplication
//! of two unit-modulus phasor vectors. This is the simplest VSA
//! primitive and is the building block for the full Cube Memory
//! retrieval path.
//!
//! Each complex phasor is stored as a `Vec2` (re, im). The shader
//! reads two input buffers of length N and writes one output buffer
//! of length N. Workgroup size is 64; dispatch with
//! `ceil_div(N, 64)` workgroups in x.

#![no_std]
#![cfg_attr(target_arch = "spirv", deny(warnings))]

use spirv_std::glam::{UVec3, Vec2};
use spirv_std::spirv;

/// Push-constant block. Kept minimal; expand when later kernels
/// need shape/stride.
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct FhrrBindPushConsts {
    /// Number of complex elements in each input buffer.
    pub n: u32,
}

/// Element-wise complex multiplication.
///
/// `(a.re + i a.im) * (b.re + i b.im) =
///  (a.re*b.re - a.im*b.im) + i (a.re*b.im + a.im*b.re)`
#[inline]
fn cmul(a: Vec2, b: Vec2) -> Vec2 {
    Vec2::new(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x,
    )
}

/// Complex conjugate. `(re, im) -> (re, -im)`.
#[inline]
fn cconj(z: Vec2) -> Vec2 {
    Vec2::new(z.x, -z.y)
}

/// FHRR bind kernel.
///
/// Layout:
///   binding=0  in_a:  &[Vec2; n]   first phasor vector
///   binding=1  in_b:  &[Vec2; n]   second phasor vector
///   binding=2  out:   &mut [Vec2; n]
///   push       FhrrBindPushConsts
///
/// Dispatch with workgroups = ceil(n / 64) in x; y=z=1.
#[spirv(compute(threads(64)))]
pub fn fhrr_bind(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(push_constant)] pc: &FhrrBindPushConsts,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] in_a: &[Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] in_b: &[Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] out: &mut [Vec2],
) {
    let i = gid.x;
    if i >= pc.n {
        return;
    }
    out[i as usize] = cmul(in_a[i as usize], in_b[i as usize]);
}

/// FHRR unbind kernel.
///
/// `unbind(z, key) = z * conj(key)`. For unit-modulus keys this is the
/// inverse of `bind`: `unbind(bind(a, b), b) ≈ a + noise`. Reuses
/// `FhrrBindPushConsts` since the layout is identical (n complex pairs
/// in, n complex out).
///
/// Layout:
///   binding=0  in_z:    &[Vec2; n]    bound vector
///   binding=1  in_key:  &[Vec2; n]    role key to unbind by
///   binding=2  out:     &mut [Vec2; n]
///   push       FhrrBindPushConsts
///
/// Dispatch the same as fhrr_bind: ceil(n / 64) workgroups in x.
#[spirv(compute(threads(64)))]
pub fn fhrr_unbind(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(push_constant)] pc: &FhrrBindPushConsts,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] in_z: &[Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] in_key: &[Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] out: &mut [Vec2],
) {
    let i = gid.x;
    if i >= pc.n {
        return;
    }
    out[i as usize] = cmul(in_z[i as usize], cconj(in_key[i as usize]));
}
