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

/// Push constants for the cleanup score + finalize passes. Both
/// passes share the struct (m, d) so the host only writes one push
/// block per dispatch pair.
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct CubeCleanupPushConsts {
    /// Codebook size (number of role vectors per axis).
    pub m: u32,
    /// Vector dimensionality (in Vec2 phasors).
    pub d: u32,
}

/// Push constants for the retrieve score + finalize passes. d_key is
/// only consumed by score; d_value and top_k only by finalize. n_slots
/// is read by both. Single struct keeps the dispatch site simple.
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

/// Workgroup size for the parallel cleanup/retrieve kernels. Chosen
/// to match the warp/subgroup width on RDNA (64) — also a multiple of
/// 32 so NVIDIA subgroups don't fragment.
const WG_SIZE: usize = 64;

/// Cube memory cleanup — pass 1 of 2. One workgroup per codebook
/// row; the WG cooperatively computes the dot product
/// `<query, codebook[wg_id]>` (real Hermitian inner product) and
/// writes the score to `scratch[wg_id]`. Index is implicit in the
/// workgroup id, so we don't need to store it.
///
/// Dispatch: m workgroups in x, 1 in y/z. Each WG is 64 threads.
/// `m` covers the entire codebook in one launch — the host gates
/// supports_op on m fitting under the device's per-dimension WG-count
/// limit.
///
/// Reduction: each thread accumulates a strided slice of the dot
/// product, then a single `subgroup_f_add` (`OpGroupNonUniformFAdd`,
/// `Reduce` group operation) collapses the 64 thread-local sums into
/// one wave-wide value. On RDNA wave64 the subgroup IS the workgroup,
/// so this is one HW shuffle — no LDS partials, no tree-reduction
/// barriers. Requires the `GroupNonUniformArithmetic` SPIR-V
/// capability (rust-gpu emits it automatically when the intrinsic is
/// used).
#[spirv(compute(threads(64)))]
pub fn cube_memory_cleanup_score(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] pc: &CubeCleanupPushConsts,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] in_query: &[Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] in_codebook: &[Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] scratch: &mut [f32],
) {
    let tid = lid.x as usize;
    let row = wid.x as usize;
    let d = pc.d as usize;
    let m = pc.m as usize;

    // Each thread strides through the d-element row, accumulating its
    // local share of the dot product. Out-of-range rows still execute
    // the subgroup op below (the op must be in convergent control
    // flow), but contribute zero to the reduction; we guard the
    // memory loads by clamping the loop bound.
    let row_off = row * d;
    let mut acc: f32 = 0.0;
    if row < m {
        let mut i = tid;
        while i < d {
            let q = in_query[i];
            let c = in_codebook[row_off + i];
            acc += q.x * c.x + q.y * c.y;
            i += WG_SIZE;
        }
    }

    // Wave64 reduce: one HW instruction, no LDS partials, no barriers.
    let sum = spirv_std::arch::subgroup_f_add::<f32>(acc);

    if tid == 0 && row < m {
        scratch[row] = sum;
    }
}

/// Cube memory cleanup — pass 2 of 2. Single workgroup of 64 threads
/// reduces `scratch[0..m]` to a global argmax (tie-break: smaller
/// index wins, matching the CPU reference) and then cooperatively
/// copies `codebook[best_idx]` to `out_cleaned`.
///
/// Dispatch: 1 workgroup in x/y/z.
#[spirv(compute(threads(64)))]
pub fn cube_memory_cleanup_finalize(
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] pc: &CubeCleanupPushConsts,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] in_codebook: &[Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] scratch: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] out_cleaned: &mut [Vec2],
    #[spirv(workgroup)] shared_idx: &mut [u32; WG_SIZE],
    #[spirv(workgroup)] shared_score: &mut [f32; WG_SIZE],
) {
    let tid = lid.x as usize;
    let d = pc.d as usize;
    let m = pc.m as usize;

    // Each thread scans scratch entries j = tid, tid+WG, ... keeping
    // a local best. Tie-break on score equality: prefer the smaller
    // codebook index (matches CPU `>` comparison: first wins).
    let mut local_best_idx: u32 = 0;
    let mut local_best_score: f32 = -1e30;
    {
        let mut j = tid;
        while j < m {
            let s = scratch[j];
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

    let mut stride = WG_SIZE / 2;
    while stride > 0 {
        if tid < stride {
            let other_score = shared_score[tid + stride];
            let other_idx = shared_idx[tid + stride];
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

    let best_idx = shared_idx[0] as usize;
    let row_off = best_idx * d;
    let mut i = tid;
    while i < d {
        out_cleaned[i] = in_codebook[row_off + i];
        i += WG_SIZE;
    }
}

/// Maximum top_k supported by retrieve. Bounded so we can use
/// stack-allocated arrays for the running top-k tournament. 8 covers
/// the typical Memory Layer setting (PEER uses k=4 or 8).
const MAX_TOP_K: usize = 8;

/// Maximum d_key (in f32 elements) supported by retrieve_score's LDS
/// staging buffer. 4096 floats = 16 KB per WG, well under the RDNA3
/// 64 KB/CU shared LDS budget. The host's supports_op gates d_key on
/// this; values larger than MAX_D_KEY_LDS fall back to CPU.
pub const MAX_D_KEY_LDS: usize = 4096;

/// Cube memory retrieve — pass A of B. Cooperative wave64 reduction:
/// one workgroup per slot, 64 threads per WG. Each thread accumulates
/// a strided slice of the slot's `<query, slot_keys[wg]>` dot product,
/// then a single `subgroup_f_add` (`OpGroupNonUniformFAdd`, `Reduce`
/// group operation) collapses the 64 thread-local sums into one
/// wave-wide value in a single HW instruction. Thread 0 writes
/// `scratch[wg]`.
///
/// Why this layout (vs the thread-per-slot variant): cooperative wave
/// reads consecutive elements of the same slot row → 64-lane coalesced
/// DRAM loads. Why subgroup_f_add (vs LDS tree reduction): drops the
/// 6 barriers + 6 LDS round-trips of the log2(64) tree, replacing
/// them with one HW shuffle. On RDNA wave64 the subgroup IS the
/// workgroup, so the reduction is exact.
///
/// The query is read directly from global memory by every thread
/// (cache-resident after the first stride), no LDS staging needed —
/// removing the staging buffer also frees up LDS budget so the
/// d_key ≤ MAX_D_KEY_LDS gate is no longer required by this kernel.
/// We keep MAX_D_KEY_LDS and the host gate anyway as a sanity bound.
///
/// Dispatch: n_slots workgroups in x, 1 in y/z. Host gates
/// supports_op on n_slots ≤ device per-dimension WG-count limit.
#[spirv(compute(threads(64)))]
pub fn cube_memory_retrieve_score(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] pc: &CubeRetrievePushConsts,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] in_query: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] in_slot_keys: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] scratch: &mut [f32],
) {
    let tid = lid.x as usize;
    let wg  = wid.x as usize;
    let n_slots = pc.n_slots as usize;
    let d_key = pc.d_key as usize;

    // Strided cooperative read of slot row `wg`. Consecutive lanes
    // touch consecutive elements — the load is 64-wide coalesced.
    // Out-of-range workgroups still execute the subgroup op below
    // (it must be in convergent control flow), but skip the loads.
    let mut acc: f32 = 0.0;
    if wg < n_slots {
        let row_off = wg * d_key;
        let mut i = tid;
        while i < d_key {
            acc += in_query[i] * in_slot_keys[row_off + i];
            i += WG_SIZE;
        }
    }

    // Wave64 reduce: one HW instruction.
    let sum = spirv_std::arch::subgroup_f_add::<f32>(acc);

    if tid == 0 && wg < n_slots {
        scratch[wg] = sum;
    }
}

/// Maximum n_slots gated through the host. Picked one below the
/// Vulkan-spec minimum guarantee for `maxComputeWorkGroupCount[0]`
/// (65535) so any conformant device accepts the dispatch. Hosts on
/// devices that report a higher limit may raise this; hosts must
/// also clamp to the device's actual limit.
pub const MAX_RETRIEVE_SLOTS: usize = 65535;

/// Cube memory retrieve — pass B of B. Single workgroup of 64
/// threads. Reads `scratch[0..n_slots]` (similarities), runs a
/// sliding-min top-k tournament + numerically-stable softmax in
/// thread 0, then cooperatively gathers the weighted slot_values
/// into `out` (parallel over d_value).
///
/// Dispatch: 1 workgroup in x/y/z.
#[spirv(compute(threads(64)))]
pub fn cube_memory_retrieve_finalize(
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] pc: &CubeRetrievePushConsts,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] scratch: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] in_slot_values: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] out: &mut [f32],
    #[spirv(workgroup)] shared_idx: &mut [u32; MAX_TOP_K],
    #[spirv(workgroup)] shared_w: &mut [f32; MAX_TOP_K],
) {
    let tid = lid.x as usize;
    let n_slots = pc.n_slots as usize;
    let d_value = pc.d_value as usize;
    let k = (pc.top_k as usize).min(MAX_TOP_K);

    if tid == 0 {
        let mut topk_idx: [u32; MAX_TOP_K] = [0; MAX_TOP_K];
        let mut topk_sim: [f32; MAX_TOP_K] = [-1e30; MAX_TOP_K];

        let mut j = 0usize;
        while j < n_slots {
            let s = scratch[j];
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
            j += 1;
        }

        let mut sim_max: f32 = topk_sim[0];
        for t in 1..k {
            if topk_sim[t] > sim_max {
                sim_max = topk_sim[t];
            }
        }
        let mut weights: [f32; MAX_TOP_K] = [0.0; MAX_TOP_K];
        let mut sum_exp: f32 = 0.0;
        for t in 0..k {
            let w = (topk_sim[t] - sim_max).exp();
            weights[t] = w;
            sum_exp += w;
        }
        let inv_sum = 1.0 / sum_exp.max(1e-8);
        for t in 0..k {
            shared_idx[t] = topk_idx[t];
            shared_w[t]   = weights[t] * inv_sum;
        }
    }
    spirv_std::arch::workgroup_memory_barrier_with_group_sync();

    // Parallel weighted gather. Each thread strides through the
    // d_value-dim output, summing the contribution from each of the
    // top-k slots.
    let mut i = tid;
    while i < d_value {
        let mut acc: f32 = 0.0;
        for t in 0..k {
            let idx = shared_idx[t] as usize;
            let w   = shared_w[t];
            acc += w * in_slot_values[idx * d_value + i];
        }
        out[i] = acc;
        i += WG_SIZE;
    }
}
