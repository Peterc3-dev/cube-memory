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
    for j in 0..n_slots {
        let row = j * d_key;
        let mut s = 0.0_f32;
        for i in 0..d_key {
            s += query[i] * slot_keys[row + i];
        }
        sims[j] = s;
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
