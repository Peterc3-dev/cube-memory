//! CPU reference implementations of FHRR and Cube Memory primitives.
//!
//! These mirror `phase0/fhrr.py` exactly. Each takes flat `[f32]`
//! buffers (re/im interleaved as `Vec2(re, im)`) so the layouts match
//! the GPU side without conversion.

use glam::Vec2;

/// Element-wise complex multiply.
pub fn cmul(a: Vec2, b: Vec2) -> Vec2 {
    Vec2::new(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x)
}

/// Complex conjugate.
pub fn cconj(z: Vec2) -> Vec2 {
    Vec2::new(z.x, -z.y)
}

/// FHRR bind: element-wise complex multiplication of two phasor vectors.
pub fn fhrr_bind(a: &[Vec2], b: &[Vec2]) -> Vec<Vec2> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| cmul(x, y)).collect()
}

/// FHRR unbind: bind by the conjugate.
pub fn fhrr_unbind(z: &[Vec2], key: &[Vec2]) -> Vec<Vec2> {
    assert_eq!(z.len(), key.len());
    z.iter()
        .zip(key.iter())
        .map(|(&zi, &ki)| cmul(zi, cconj(ki)))
        .collect()
}

/// Element-wise project to unit modulus, with eps floor to avoid
/// division by zero.
pub fn fhrr_unitize(v: &[Vec2]) -> Vec<Vec2> {
    v.iter()
        .map(|&z| {
            let mag = z.length().max(1e-8);
            z / mag
        })
        .collect()
}

/// Bundle K vectors of length n: sum element-wise then unitize.
/// Input layout: contiguous `K * n` block, vector k starts at `k*n`.
pub fn fhrr_superpose(input: &[Vec2], n: usize, k: usize) -> Vec<Vec2> {
    assert_eq!(input.len(), k * n);
    let mut out = vec![Vec2::ZERO; n];
    for i in 0..n {
        let mut acc = Vec2::ZERO;
        for kk in 0..k {
            acc += input[kk * n + i];
        }
        let mag = acc.length().max(1e-8);
        out[i] = acc / mag;
    }
    out
}

/// Cube Memory cleanup: argmax-cosine codebook lookup.
/// Returns the *winning codebook entry* (snapped phasor).
pub fn cube_memory_cleanup(query: &[Vec2], codebook: &[Vec2], m: usize, d: usize) -> Vec<Vec2> {
    assert_eq!(query.len(), d);
    assert_eq!(codebook.len(), m * d);
    let mut best_idx = 0usize;
    // -1e30 (not NEG_INFINITY) to match the GPU shader sentinel exactly.
    // Algebraically NEG_INFINITY would also work, but parity tests are
    // tightest when both paths use the same constant.
    let mut best_score: f32 = -1e30;
    for j in 0..m {
        let mut s = 0.0_f32;
        let row = j * d;
        for i in 0..d {
            let q = query[i];
            let c = codebook[row + i];
            s += q.x * c.x + q.y * c.y;
        }
        if s > best_score {
            best_score = s;
            best_idx = j;
        }
    }
    let row = best_idx * d;
    codebook[row..row + d].to_vec()
}

/// Cube Memory retrieve: top-k slot-key dot product, softmax-weighted
/// slot-value gather. Real-valued throughout (FHRR addresses are
/// converted to real `(re, im)` pairs by the caller before reaching
/// this function).
pub fn cube_memory_retrieve(
    query: &[f32],
    slot_keys: &[f32],
    slot_values: &[f32],
    n_slots: usize,
    d_key: usize,
    d_value: usize,
    top_k: usize,
) -> Vec<f32> {
    assert_eq!(query.len(), d_key);
    assert_eq!(slot_keys.len(), n_slots * d_key);
    assert_eq!(slot_values.len(), n_slots * d_value);

    // Compute all sims.
    let mut sims = vec![0.0_f32; n_slots];
    for (j, sim) in sims.iter_mut().enumerate() {
        let row = j * d_key;
        let mut s = 0.0_f32;
        for i in 0..d_key {
            s += query[i] * slot_keys[row + i];
        }
        *sim = s;
    }

    // Top-k by descending sim. NOTE on tie-break semantics: CPU uses
    // a stable sort (preserves input order on ties); GPU uses a sliding-
    // min tournament that takes the first slot to displace the running
    // minimum. The two diverge only on exact float ties, which random
    // unit-phasor or uniform real inputs essentially never produce.
    // For deterministic test inputs with repeated similarity values,
    // expect the parity test to fail and adjust the test rather than
    // the algorithms (each side is internally consistent).
    let mut idxs: Vec<usize> = (0..n_slots).collect();
    idxs.sort_by(|&a, &b| sims[b].partial_cmp(&sims[a]).unwrap());
    let topk: Vec<usize> = idxs.into_iter().take(top_k).collect();

    // Softmax over top-k sims (numerically stable: subtract max).
    let mut topk_sims: Vec<f32> = topk.iter().map(|&j| sims[j]).collect();
    let smax = topk_sims.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0_f32;
    for s in topk_sims.iter_mut() {
        *s = (*s - smax).exp();
        sum_exp += *s;
    }
    let inv = 1.0_f32 / sum_exp.max(1e-8);
    for s in topk_sims.iter_mut() {
        *s *= inv;
    }

    // Weighted gather.
    let mut out = vec![0.0_f32; d_value];
    for (t, &j) in topk.iter().enumerate() {
        let row = j * d_value;
        let w = topk_sims[t];
        for i in 0..d_value {
            out[i] += w * slot_values[row + i];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    //! Unit tests for the CPU reference primitives.
    //!
    //! These do not touch the GPU (unlike `tests/parity.rs`) — they pin
    //! the algebraic behaviour of the reference path itself, so a future
    //! refactor of `cpu.rs` cannot silently change the ground truth the
    //! parity tests compare against. They build and run on stable Rust.

    use super::*;

    /// Deterministic unit-modulus phasors for reproducible assertions.
    fn phasors(n: usize, seed: u32) -> Vec<Vec2> {
        let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let phase =
                    (state as f32 / u32::MAX as f32) * std::f32::consts::TAU - std::f32::consts::PI;
                Vec2::new(phase.cos(), phase.sin())
            })
            .collect()
    }

    fn assert_vec2_close(a: &[Vec2], b: &[Vec2], eps: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x.x - y.x).abs() < eps && (x.y - y.y).abs() < eps,
                "mismatch at {i}: {x:?} vs {y:?}"
            );
        }
    }

    #[test]
    fn cmul_matches_complex_multiply() {
        // (1+2i)(3+4i) = (3-8) + (4+6)i = -5 + 10i
        let r = cmul(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        assert!((r.x - (-5.0)).abs() < 1e-6);
        assert!((r.y - 10.0).abs() < 1e-6);
    }

    #[test]
    fn cconj_negates_imaginary() {
        let r = cconj(Vec2::new(3.0, -4.0));
        assert_eq!(r, Vec2::new(3.0, 4.0));
    }

    #[test]
    fn bind_then_unbind_is_identity_for_unit_phasors() {
        // For unit-modulus phasors, unbind(bind(z, k), k) == z because
        // k * conj(k) == |k|^2 == 1. This is the core FHRR property the
        // whole "rotate to face θ, read the snapshot" idea rests on.
        let z = phasors(64, 7);
        let k = phasors(64, 11);
        let bound = fhrr_bind(&z, &k);
        let recovered = fhrr_unbind(&bound, &k);
        assert_vec2_close(&z, &recovered, 1e-5);
    }

    #[test]
    fn bind_is_commutative() {
        let a = phasors(32, 1);
        let b = phasors(32, 2);
        assert_vec2_close(&fhrr_bind(&a, &b), &fhrr_bind(&b, &a), 1e-6);
    }

    #[test]
    fn unitize_produces_unit_modulus() {
        let v = vec![
            Vec2::new(3.0, 4.0),
            Vec2::new(-6.0, 8.0),
            Vec2::new(0.0, 0.0),
        ];
        let u = fhrr_unitize(&v);
        // First two have modulus 5 and 10 -> normalize to 1.
        assert!((u[0].length() - 1.0).abs() < 1e-5);
        assert!((u[1].length() - 1.0).abs() < 1e-5);
        // Zero vector: eps floor keeps the magnitude tiny, not NaN.
        assert!(u[2].x.is_finite() && u[2].y.is_finite());
    }

    #[test]
    fn superpose_outputs_unit_modulus() {
        let n = 16;
        let k = 4;
        let input: Vec<Vec2> = (0..k).flat_map(|i| phasors(n, 100 + i as u32)).collect();
        let out = fhrr_superpose(&input, n, k);
        assert_eq!(out.len(), n);
        for z in &out {
            assert!((z.length() - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn cleanup_returns_exact_codebook_entry_for_self_query() {
        // Querying with a codebook entry must snap back to that same
        // entry (cosine self-similarity is maximal).
        let m = 8;
        let d = 16;
        let codebook: Vec<Vec2> = (0..m).flat_map(|i| phasors(d, 200 + i as u32)).collect();
        let target = 5usize;
        let query = codebook[target * d..(target + 1) * d].to_vec();
        let snapped = cube_memory_cleanup(&query, &codebook, m, d);
        assert_vec2_close(&snapped, &query, 1e-6);
    }

    #[test]
    fn retrieve_softmax_weights_sum_to_one() {
        // With a single dominant slot, retrieve should approach that
        // slot's value vector; more generally the weighted gather is a
        // convex combination of the top-k value rows, so every output
        // coordinate lies within the min/max of the gathered values.
        let n_slots = 8;
        let d_key = 4;
        let d_value = 3;
        let top_k = 4;
        let query = vec![1.0, 0.0, 0.0, 0.0];
        // Slot 0 keyed to align perfectly with the query.
        let mut slot_keys = vec![0.0_f32; n_slots * d_key];
        for j in 0..n_slots {
            slot_keys[j * d_key] = j as f32 / n_slots as f32;
        }
        slot_keys[0] = 10.0; // dominant
        let slot_values: Vec<f32> = (0..n_slots * d_value).map(|x| x as f32).collect();
        let out = cube_memory_retrieve(
            &query,
            &slot_keys,
            &slot_values,
            n_slots,
            d_key,
            d_value,
            top_k,
        );
        assert_eq!(out.len(), d_value);
        // Dominant slot 0 -> output should be close to slot 0's value row.
        for i in 0..d_value {
            assert!(
                (out[i] - slot_values[i]).abs() < 1e-2,
                "coord {i}: {} vs {}",
                out[i],
                slot_values[i]
            );
        }
    }

    #[test]
    fn retrieve_is_convex_combination() {
        // Each output coordinate must lie within [min, max] of the
        // corresponding coordinate across all slot value rows, since the
        // softmax weights are non-negative and sum to one.
        let n_slots = 6;
        let d_key = 3;
        let d_value = 2;
        let top_k = 3;
        let query = vec![0.3, -0.7, 0.5];
        let slot_keys: Vec<f32> = (0..n_slots * d_key)
            .map(|x| ((x as f32) * 0.137).sin())
            .collect();
        let slot_values: Vec<f32> = (0..n_slots * d_value)
            .map(|x| ((x as f32) * 0.91).cos() * 4.0)
            .collect();
        let out = cube_memory_retrieve(
            &query,
            &slot_keys,
            &slot_values,
            n_slots,
            d_key,
            d_value,
            top_k,
        );
        for c in 0..d_value {
            let col: Vec<f32> = (0..n_slots).map(|j| slot_values[j * d_value + c]).collect();
            let lo = col.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = col.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            assert!(
                out[c] >= lo - 1e-4 && out[c] <= hi + 1e-4,
                "coord {c}: {} not in [{lo}, {hi}]",
                out[c]
            );
        }
    }
}
