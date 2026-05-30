//! GPU/CPU parity tests for the Cube Memory shaders.
//!
//! Each kernel is exercised on a small random input. The CPU
//! reference and GPU passthrough output should agree within a small
//! float tolerance. Run with:
//!
//!   cd ~/projects/cube-memory/shaders
//!   cargo test -p cube-memory-host --release
//!
//! The shader binary must already exist — build it first with
//! `cargo run -p cube-memory-shader-builder --release`. The test
//! resolves the path relative to the workspace target dir.

use std::env;
use std::path::PathBuf;

use bytemuck::{Pod, Zeroable};
use cube_memory_host::cpu;
use cube_memory_host::gpu::GpuCtx;
use cube_memory_host::SHADER_RELATIVE_PATH;
use glam::Vec2;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FhrrBindPushConsts {
    n: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FhrrSuperposePushConsts {
    n: u32,
    k: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CubeCleanupPushConsts {
    m: u32,
    d: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CubeRetrievePushConsts {
    n_slots: u32,
    d_key: u32,
    d_value: u32,
    top_k: u32,
}

fn random_real_vec(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn shader_path() -> PathBuf {
    // Workspace root is two dirs up from CARGO_MANIFEST_DIR (the host crate).
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let mut p = PathBuf::from(manifest_dir);
    p.pop(); // shaders/
    p.push("target");
    p.push(SHADER_RELATIVE_PATH);
    assert!(p.exists(), "shader not built: {}\n  run `cargo run -p cube-memory-shader-builder --release` first", p.display());
    p
}

fn random_phasor_vec(n: usize, seed: u32) -> Vec<Vec2> {
    // Deterministic linear congruential pseudo-random phases for
    // reproducible tests. Magnitude exactly 1.0 for unit-modulus.
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let phase = (state as f32 / u32::MAX as f32) * std::f32::consts::TAU - std::f32::consts::PI;
            Vec2::new(phase.cos(), phase.sin())
        })
        .collect()
}

fn assert_close(a: &[Vec2], b: &[Vec2], eps: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (xa, xb)) in a.iter().zip(b.iter()).enumerate() {
        let dx = (xa.x - xb.x).abs();
        let dy = (xa.y - xb.y).abs();
        assert!(
            dx < eps && dy < eps,
            "mismatch at index {i}: cpu={xa:?} gpu={xb:?} (eps={eps})"
        );
    }
}

#[test]
fn fhrr_bind_parity() {
    let n: usize = 128;
    let a = random_phasor_vec(n, 1);
    let b = random_phasor_vec(n, 2);

    let cpu_out = cpu::fhrr_bind(&a, &b);

    let ctx = GpuCtx::new(&shader_path());
    let push = FhrrBindPushConsts { n: n as u32 };
    let groups = ((n as u32).div_ceil(64), 1, 1);
    let gpu_out: Vec<Vec2> = ctx.run(
        "fhrr_bind",
        push,
        &[bytemuck::cast_slice(&a), bytemuck::cast_slice(&b)],
        n * std::mem::size_of::<Vec2>(),
        groups,
    );

    assert_close(&cpu_out, &gpu_out, 1e-5);
}

#[test]
fn fhrr_unbind_parity() {
    let n: usize = 128;
    let z = random_phasor_vec(n, 3);
    let key = random_phasor_vec(n, 4);

    let cpu_out = cpu::fhrr_unbind(&z, &key);

    let ctx = GpuCtx::new(&shader_path());
    let push = FhrrBindPushConsts { n: n as u32 };
    let groups = ((n as u32).div_ceil(64), 1, 1);
    let gpu_out: Vec<Vec2> = ctx.run(
        "fhrr_unbind",
        push,
        &[bytemuck::cast_slice(&z), bytemuck::cast_slice(&key)],
        n * std::mem::size_of::<Vec2>(),
        groups,
    );

    assert_close(&cpu_out, &gpu_out, 1e-5);
}

#[test]
fn fhrr_unitize_parity() {
    let n: usize = 128;
    // Use non-unit-modulus inputs to actually exercise the projection.
    let v: Vec<Vec2> = random_real_vec(2 * n, 5)
        .chunks_exact(2)
        .map(|c| Vec2::new(c[0] * 3.0, c[1] * 3.0))
        .collect();

    let cpu_out = cpu::fhrr_unitize(&v);

    let ctx = GpuCtx::new(&shader_path());
    let push = FhrrBindPushConsts { n: n as u32 };
    let groups = ((n as u32).div_ceil(64), 1, 1);
    let gpu_out: Vec<Vec2> = ctx.run(
        "fhrr_unitize",
        push,
        &[bytemuck::cast_slice(&v)],
        n * std::mem::size_of::<Vec2>(),
        groups,
    );

    assert_close(&cpu_out, &gpu_out, 1e-5);
}

#[test]
fn fhrr_superpose_parity() {
    let n: usize = 64;
    let k: usize = 8;
    let inputs: Vec<Vec2> = (0..k)
        .flat_map(|i| random_phasor_vec(n, 10 + i as u32))
        .collect();

    let cpu_out = cpu::fhrr_superpose(&inputs, n, k);

    let ctx = GpuCtx::new(&shader_path());
    let push = FhrrSuperposePushConsts {
        n: n as u32,
        k: k as u32,
    };
    let groups = ((n as u32).div_ceil(64), 1, 1);
    let gpu_out: Vec<Vec2> = ctx.run(
        "fhrr_superpose",
        push,
        &[bytemuck::cast_slice(&inputs)],
        n * std::mem::size_of::<Vec2>(),
        groups,
    );

    assert_close(&cpu_out, &gpu_out, 1e-5);
}

#[test]
fn cube_memory_cleanup_parity() {
    let m: usize = 16;
    let d: usize = 32;
    let query = random_phasor_vec(d, 20);
    let codebook: Vec<Vec2> = (0..m)
        .flat_map(|i| random_phasor_vec(d, 30 + i as u32))
        .collect();

    let cpu_out = cpu::cube_memory_cleanup(&query, &codebook, m, d);

    let ctx = GpuCtx::new(&shader_path());
    let push = CubeCleanupPushConsts {
        m: m as u32,
        d: d as u32,
    };
    let gpu_out: Vec<Vec2> = ctx.run_pair(
        "cube_memory_cleanup_score",
        &[bytemuck::cast_slice(&query), bytemuck::cast_slice(&codebook)],
        (m as u32, 1, 1),
        "cube_memory_cleanup_finalize",
        &[bytemuck::cast_slice(&codebook)],
        1,
        (1, 1, 1),
        m * std::mem::size_of::<f32>(),
        d * std::mem::size_of::<Vec2>(),
        push,
    );

    assert_close(&cpu_out, &gpu_out, 1e-5);
}

#[test]
fn cube_memory_retrieve_parity() {
    let n_slots: usize = 32;
    let d_key: usize = 16;
    let d_value: usize = 8;
    let top_k: usize = 4;
    let query = random_real_vec(d_key, 100);
    let slot_keys = random_real_vec(n_slots * d_key, 101);
    let slot_values = random_real_vec(n_slots * d_value, 102);

    let cpu_out = cpu::cube_memory_retrieve(
        &query,
        &slot_keys,
        &slot_values,
        n_slots,
        d_key,
        d_value,
        top_k,
    );

    let ctx = GpuCtx::new(&shader_path());
    let push = CubeRetrievePushConsts {
        n_slots: n_slots as u32,
        d_key: d_key as u32,
        d_value: d_value as u32,
        top_k: top_k as u32,
    };
    let gpu_out: Vec<f32> = ctx.run_pair(
        "cube_memory_retrieve_score",
        &[
            bytemuck::cast_slice(&query),
            bytemuck::cast_slice(&slot_keys),
        ],
        (n_slots as u32, 1, 1),
        "cube_memory_retrieve_finalize",
        &[bytemuck::cast_slice(&slot_values)],
        0,
        (1, 1, 1),
        n_slots * std::mem::size_of::<f32>(),
        d_value * std::mem::size_of::<f32>(),
        push,
    );

    // Slightly looser tolerance — softmax/exp introduces tiny fp drift
    // between CPU sort-based and GPU tournament-based top-k when
    // similarities are very close.
    assert_eq!(cpu_out.len(), gpu_out.len());
    for (i, (a, b)) in cpu_out.iter().zip(gpu_out.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "retrieve mismatch at {i}: cpu={a} gpu={b}"
        );
    }
}
