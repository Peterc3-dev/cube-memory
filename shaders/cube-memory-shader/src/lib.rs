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
use spirv_std::num_traits::Float;
use spirv_std::spirv;

/// Push-constant block. Kept minimal; expand when later kernels
/// need shape/stride.
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct FhrrBindPushConsts {
    /// Number of complex elements in each input buffer.
    pub n: u32,
}

/// Push constants for `fhrr_superpose`.
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct FhrrSuperposePushConsts {
    /// Number of complex elements per bundled vector.
    pub n: u32,
    /// Number of vectors being bundled together.
    pub k: u32,
}

/// Push constants for `cube_memory_cleanup`.
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct CubeCleanupPushConsts {
    /// Codebook size (number of role vectors per axis).
    pub m: u32,
    /// Vector dimensionality.
    pub d: u32,
}

/// Push constants for `cube_memory_retrieve`.
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct CubeRetrievePushConsts {
    /// Number of slots in the slot store.
    pub n_slots: u32,
    /// Dimensionality of slot keys (= 2 * d_codebook for re||im concat).
    pub d_key: u32,
    /// Dimensionality of slot values.
    pub d_value: u32,
    /// Number of top slots to gather.
    pub top_k: u32,
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

/// FHRR unitize kernel.
///
/// Element-wise normalize each phasor to unit modulus. After many
/// bind/unbind compositions the magnitudes drift in fp arithmetic;
/// this kernel projects them back onto the unit circle. Required
/// every forward pass per Alam et al. 2021 (arXiv 2109.02157).
///
/// Layout:
///   binding=0  in:  &[Vec2; n]
///   binding=1  out: &mut [Vec2; n]
///   push       FhrrBindPushConsts
///
/// Dispatch ceil(n / 64) workgroups in x.
#[spirv(compute(threads(64)))]
pub fn fhrr_unitize(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(push_constant)] pc: &FhrrBindPushConsts,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] r#in: &[Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] out: &mut [Vec2],
) {
    let i = gid.x;
    if i >= pc.n {
        return;
    }
    let z = r#in[i as usize];
    // glam's Vec2::length() lowers to the SPIR-V Length GLSL.std.450
    // intrinsic, which is the right primitive on the GPU. The 1e-8
    // floor prevents division by zero on degenerate (0, 0) inputs
    // without perturbing already-unit vectors.
    let mag = z.length().max(1e-8);
    out[i as usize] = z / mag;
}

/// FHRR superpose (bundle) kernel.
///
/// Sum K complex vectors element-wise, then unitize. Inputs are laid
/// out as a single contiguous K*N buffer in row-major order: vector
/// k starts at offset k*n. This is the bundle operation in VSA.
///
/// Layout:
///   binding=0  in:  &[Vec2; k * n]   stacked input vectors
///   binding=1  out: &mut [Vec2; n]   bundled + unitized output
///   push       FhrrSuperposePushConsts
///
/// Dispatch ceil(n / 64) workgroups in x. Each thread handles one
/// element of the output, summing across K bundle slots serially.
#[spirv(compute(threads(64)))]
pub fn fhrr_superpose(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(push_constant)] pc: &FhrrSuperposePushConsts,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] r#in: &[Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] out: &mut [Vec2],
) {
    let i = gid.x;
    if i >= pc.n {
        return;
    }
    let mut acc = Vec2::ZERO;
    let mut k = 0u32;
    while k < pc.k {
        acc += r#in[(k * pc.n + i) as usize];
        k += 1;
    }
    let mag = acc.length().max(1e-8);
    out[i as usize] = acc / mag;
}

/// Cube Memory cleanup kernel — argmax cosine match against a
/// frozen codebook of m phasor vectors.
///
/// For each codebook entry j the similarity is the real part of
/// `<query, codebook[j]>` (Hermitian inner product) normalized by
/// the dimensionality. The kernel writes the codebook entry with
/// the highest similarity to `out_cleaned`. The downstream layer
/// uses the *snapped phasor* (not the index) for subsequent bind
/// operations, so we avoid a separate index output here.
///
/// Layout:
///   binding=0  in_query:    &[Vec2; d]      query vector
///   binding=1  in_codebook: &[Vec2; m * d]  m codebook entries
///   binding=2  out_cleaned: &mut [Vec2; d]  copied winning entry
///   push       CubeCleanupPushConsts
///
/// Dispatch: 1 workgroup, 1 thread. This is intentionally
/// single-threaded — m and d are small (m ~ 256, d ~ 1024) and
/// running serially keeps the algorithm verifiable. A parallel
/// version with subgroup reductions is a Phase 2 optimization
/// gated on a perf benchmark.
#[spirv(compute(threads(64)))]
pub fn cube_memory_cleanup(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(push_constant)] pc: &CubeCleanupPushConsts,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] in_query: &[Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] in_codebook: &[Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] out_cleaned: &mut [Vec2],
) {
    if gid.x != 0 {
        return;
    }
    let d = pc.d as usize;
    let m = pc.m as usize;
    let mut best_idx: usize = 0;
    let mut best_score: f32 = -1e30;
    for j in 0..m {
        let mut s: f32 = 0.0;
        let row_off = j * d;
        for i in 0..d {
            let q = in_query[i];
            let c = in_codebook[row_off + i];
            s += q.x * c.x + q.y * c.y;
        }
        if s > best_score {
            best_score = s;
            best_idx = j;
        }
    }
    let row_off = best_idx * d;
    for i in 0..d {
        out_cleaned[i] = in_codebook[row_off + i];
    }
}

/// Maximum top_k supported by `cube_memory_retrieve`. Bounded so we
/// can use stack-allocated arrays for the running top-k tournament.
/// 8 covers the typical Memory Layer setting (PEER uses k=4 or 8).
const MAX_TOP_K: usize = 8;

/// Cube Memory retrieve kernel — top-k slot-key dot product +
/// softmax-weighted slot-value gather.
///
/// Algorithm:
///   sims[j]      = <query, slot_keys[j]>          for j in 0..n_slots
///   topk_idx, topk_sims = topk(sims, k)
///   weights      = softmax(topk_sims)
///   out          = sum_k weights[k] * slot_values[topk_idx[k]]
///
/// Layout:
///   binding=0  in_query:       &[f32; d_key]
///   binding=1  in_slot_keys:   &[f32; n_slots * d_key]
///   binding=2  in_slot_values: &[f32; n_slots * d_value]
///   binding=3  out:            &mut [f32; d_value]
///   push       CubeRetrievePushConsts
///
/// Single-thread workgroup. `top_k` capped at MAX_TOP_K=8 for the
/// stack-allocated tournament. Like cleanup this is a v1 kernel —
/// the parallel + subgroup-reduce version is a Phase 2 task.
#[spirv(compute(threads(64)))]
pub fn cube_memory_retrieve(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(push_constant)] pc: &CubeRetrievePushConsts,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] in_query: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] in_slot_keys: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] in_slot_values: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] out: &mut [f32],
) {
    if gid.x != 0 {
        return;
    }
    let n_slots = pc.n_slots as usize;
    let d_key = pc.d_key as usize;
    let d_value = pc.d_value as usize;
    let k = (pc.top_k as usize).min(MAX_TOP_K);

    // Running top-k tournament. Initialised with -infinity sentinel.
    let mut topk_idx: [u32; MAX_TOP_K] = [0; MAX_TOP_K];
    let mut topk_sim: [f32; MAX_TOP_K] = [-1e30; MAX_TOP_K];

    for j in 0..n_slots {
        let mut s: f32 = 0.0;
        let row_off = j * d_key;
        for i in 0..d_key {
            s += in_query[i] * in_slot_keys[row_off + i];
        }
        // Insert into top-k if it beats the current minimum.
        // Sliding minimum scan: find the smallest entry, replace
        // if s > it. O(k) per insert; k <= 8.
        let mut min_pos: usize = 0;
        let mut min_val: f32 = topk_sim[0];
        for t in 1..k {
            if topk_sim[t] < min_val {
                min_val = topk_sim[t];
                min_pos = t;
            }
        }
        if s > min_val {
            topk_sim[min_pos] = s;
            topk_idx[min_pos] = j as u32;
        }
    }

    // Softmax over top-k similarities. Numerical stability: subtract
    // max before exp.
    let mut sim_max: f32 = topk_sim[0];
    for t in 1..k {
        if topk_sim[t] > sim_max {
            sim_max = topk_sim[t];
        }
    }
    let mut sum_exp: f32 = 0.0;
    let mut weights: [f32; MAX_TOP_K] = [0.0; MAX_TOP_K];
    for t in 0..k {
        let w = (topk_sim[t] - sim_max).exp();
        weights[t] = w;
        sum_exp += w;
    }
    let inv_sum = 1.0 / sum_exp.max(1e-8);
    for t in 0..k {
        weights[t] *= inv_sum;
    }

    // Weighted gather of slot values into output.
    for i in 0..d_value {
        let mut acc: f32 = 0.0;
        for t in 0..k {
            let row_off = (topk_idx[t] as usize) * d_value;
            acc += weights[t] * in_slot_values[row_off + i];
        }
        out[i] = acc;
    }
}
