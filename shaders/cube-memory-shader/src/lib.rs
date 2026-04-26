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
/// Workgroup size for the parallel cleanup/retrieve kernels. Chosen
/// to match the warp/subgroup width on RDNA (64) — also a multiple of
/// 32 so NVIDIA subgroups don't fragment.
const WG_SIZE: usize = 64;

/// Parallel cube memory cleanup. Each thread in the workgroup
/// scans a strided slice of the codebook (j = tid, tid + WG, …)
/// and keeps a local (idx, score) best. A shared-memory tree
/// reduction picks the global argmax. Thread 0 writes the winning
/// codebook row to the output, with the d-element copy itself
/// parallelized across the workgroup.
///
/// Tie-break: preserved as first-wins (lower codebook index) to
/// match the serial CPU reference exactly. The reduction's
/// comparison favors strictly-greater scores, then for ties the
/// smaller index. The local within-thread scan also favors lower
/// indices on ties (`if s > local_best`).
#[spirv(compute(threads(64)))]
pub fn cube_memory_cleanup(
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] pc: &CubeCleanupPushConsts,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] in_query: &[Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] in_codebook: &[Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] out_cleaned: &mut [Vec2],
    #[spirv(workgroup)] shared_idx: &mut [u32; WG_SIZE],
    #[spirv(workgroup)] shared_score: &mut [f32; WG_SIZE],
) {
    let tid = lid.x as usize;
    let d = pc.d as usize;
    let m = pc.m as usize;

    // Each thread scans codebook entries j = tid, tid + WG, ...
    let mut local_best_idx: u32 = 0;
    let mut local_best_score: f32 = -1e30;
    {
        let mut j = tid;
        while j < m {
            let mut s: f32 = 0.0;
            let row_off = j * d;
            for i in 0..d {
                let q = in_query[i];
                let c = in_codebook[row_off + i];
                s += q.x * c.x + q.y * c.y;
            }
            if s > local_best_score {
                local_best_score = s;
                local_best_idx = j as u32;
            }
            j += WG_SIZE;
        }
    }

    shared_idx[tid] = local_best_idx;
    shared_score[tid] = local_best_score;
    spirv_std::arch::workgroup_memory_barrier_with_group_sync();

    // Tree reduction over the workgroup. Tie-break: prefer
    // smaller codebook index when scores are exactly equal.
    let mut stride = WG_SIZE / 2;
    while stride > 0 {
        if tid < stride {
            let other_score = shared_score[tid + stride];
            let other_idx = shared_idx[tid + stride];
            // Tie-break: at exact-equal scores prefer the smaller
            // index. Using strict equality is ULP-fragile under fp32
            // sums-in-different-order, but at our scale (d ≤ 128 in
            // current tests, m ≤ 64) the rounding error envelope is
            // far below any deliberate tie. Documented as a known
            // soft spot for very-large-d future configs.
            let take = other_score > shared_score[tid]
                || (other_score == shared_score[tid] && other_idx < shared_idx[tid]);
            if take {
                shared_score[tid] = other_score;
                shared_idx[tid] = other_idx;
            }
        }
        spirv_std::arch::workgroup_memory_barrier_with_group_sync();
        stride /= 2;
    }

    // Cooperative copy of the winning row.
    let best_idx = shared_idx[0] as usize;
    let row_off = best_idx * d;
    let mut i = tid;
    while i < d {
        out_cleaned[i] = in_codebook[row_off + i];
        i += WG_SIZE;
    }
}

/// Maximum top_k supported by `cube_memory_retrieve`. Bounded so we
/// can use stack-allocated arrays for the running top-k tournament.
/// 8 covers the typical Memory Layer setting (PEER uses k=4 or 8).
const MAX_TOP_K: usize = 8;

/// Cube Memory retrieve kernel — top-k slot-key dot product +
/// softmax-weighted slot-value gather.
///
/// Phase A (parallel): each thread in the workgroup computes a
///   strided slice of the n_slots similarities into shared memory.
/// Phase B (serial, thread 0): full sliding-min top-k tournament
///   over the populated sims array, then softmax.
/// Phase C (parallel): each thread computes one or more elements
///   of the weighted-sum output.
///
/// Phase B is single-thread because k <= 8 makes the tournament
/// negligible compared to phase A's d_key * n_slots multiply-adds;
/// parallelizing the tournament adds barriers and reduction
/// machinery for an op that already runs in microseconds.
///
/// Layout:
///   binding=0  in_query:       &[f32; d_key]
///   binding=1  in_slot_keys:   &[f32; n_slots * d_key]
///   binding=2  in_slot_values: &[f32; n_slots * d_value]
///   binding=3  out:            &mut [f32; d_value]
///   push       CubeRetrievePushConsts
///
/// Maximum n_slots in this v0 parallel kernel is `MAX_SHARED_SLOTS`,
/// bounded by the workgroup-shared sims array size. Future versions
/// will tile across multiple workgroups; the host clamps n_slots
/// to ≤ MAX_SHARED_SLOTS via supports_op.
const MAX_SHARED_SLOTS: usize = 1024;

#[spirv(compute(threads(64)))]
pub fn cube_memory_retrieve(
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] pc: &CubeRetrievePushConsts,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] in_query: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] in_slot_keys: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] in_slot_values: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] out: &mut [f32],
    #[spirv(workgroup)] sims: &mut [f32; MAX_SHARED_SLOTS],
) {
    let tid = lid.x as usize;
    let n_slots = pc.n_slots as usize;
    let d_key = pc.d_key as usize;
    let d_value = pc.d_value as usize;
    let k = (pc.top_k as usize).min(MAX_TOP_K);

    // Phase A: strided dot products into shared sims.
    {
        let mut j = tid;
        while j < n_slots {
            let mut s: f32 = 0.0;
            let row_off = j * d_key;
            for i in 0..d_key {
                s += in_query[i] * in_slot_keys[row_off + i];
            }
            sims[j] = s;
            j += WG_SIZE;
        }
    }
    spirv_std::arch::workgroup_memory_barrier_with_group_sync();

    // Phase B (thread 0): top-k tournament + numerically-stable
    // softmax over the populated sims. Same algorithm as the v0
    // serial kernel; output bit-equal up to fp32 noise.
    let mut topk_idx: [u32; MAX_TOP_K] = [0; MAX_TOP_K];
    let mut topk_sim: [f32; MAX_TOP_K] = [-1e30; MAX_TOP_K];
    let mut weights: [f32; MAX_TOP_K] = [0.0; MAX_TOP_K];

    if tid == 0 {
        for j in 0..n_slots {
            let s = sims[j];
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

        let mut sim_max: f32 = topk_sim[0];
        for t in 1..k {
            if topk_sim[t] > sim_max {
                sim_max = topk_sim[t];
            }
        }
        let mut sum_exp: f32 = 0.0;
        for t in 0..k {
            let w = (topk_sim[t] - sim_max).exp();
            weights[t] = w;
            sum_exp += w;
        }
        let inv_sum = 1.0 / sum_exp.max(1e-8);
        for t in 0..k {
            weights[t] *= inv_sum;
        }

        // Stash the top-k indices and weights in the first 2*k slots
        // of `sims` (no longer needed for raw similarities) so the
        // other threads can read them after the next barrier.
        for t in 0..k {
            sims[t] = f32::from_bits(topk_idx[t]);
            sims[MAX_TOP_K + t] = weights[t];
        }
    }
    spirv_std::arch::workgroup_memory_barrier_with_group_sync();

    // Phase C: parallel weighted gather. Each thread computes a
    // slice of the d_value-dim output using the topk_idx and
    // weights stashed in shared memory by thread 0.
    let mut i = tid;
    while i < d_value {
        let mut acc: f32 = 0.0;
        for t in 0..k {
            let idx = sims[t].to_bits() as usize;
            let w = sims[MAX_TOP_K + t];
            let row_off = idx * d_value;
            acc += w * in_slot_values[row_off + i];
        }
        out[i] = acc;
        i += WG_SIZE;
    }
}
