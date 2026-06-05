//! NEON kernels for aarch64.
//!
//! Every function in this module is `unsafe fn` and carries
//! `#[target_feature(enable = "neon")]`. Calling any of them on a host
//! that does not advertise NEON is undefined behavior. The runtime guard
//! in [`crate::features`] is the contract: the per-metric dispatch in
//! [`crate::metrics`] consults [`crate::detect_features`] and only enters
//! one of these functions when `CpuFeatures::neon` is `true`.
//!
//! Most NEON arithmetic and reinterpret intrinsics are safe to call when
//! the host advertises NEON, so they are invoked directly. Only the
//! pointer-taking load `vld1q_f32` carries an explicit `unsafe {}` block
//! tying its safety justification to the `chunks_exact(LANES)` source.

use std::arch::aarch64::{
    vabsq_f32, vaddq_f32, vaddvq_f32, vaddvq_u32, vceqq_u32, vdupq_n_f32, vld1q_f32, vmulq_f32,
    vmvnq_u32, vreinterpretq_u32_f32, vshrq_n_u32, vsubq_f32,
};

const LANES: usize = 4;

/// NEON dot-product. Caller MUST have validated `a.len() == b.len() != 0`
/// and confirmed NEON is available.
///
/// # Safety
///
/// Calling this on a host without NEON is undefined behavior. The caller
/// MUST gate the call on [`crate::features::detect_features`]
/// returning `neon = true`.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = vdupq_n_f32(0.0);
    let chunks_a = a.chunks_exact(LANES);
    let chunks_b = b.chunks_exact(LANES);
    let tail_a = chunks_a.remainder();
    let tail_b = chunks_b.remainder();
    for (ca, cb) in chunks_a.zip(chunks_b) {
        // SAFETY: `ca`/`cb` are 4 contiguous f32 (16 bytes) from
        // `chunks_exact(4)`; `vld1q_f32` reads exactly 4 f32 from the
        // pointer. NEON is enabled by the enclosing `target_feature`.
        let va = unsafe { vld1q_f32(ca.as_ptr()) };
        // SAFETY: see above.
        let vb = unsafe { vld1q_f32(cb.as_ptr()) };
        acc = vaddq_f32(acc, vmulq_f32(va, vb));
    }
    let mut sum = vaddvq_f32(acc);
    for (x, y) in tail_a.iter().zip(tail_b.iter()) {
        sum += x * y;
    }
    sum
}

/// NEON cosine distance. Caller MUST have validated inputs and NEON.
///
/// Uses the same `na.sqrt() * nb.sqrt()` denominator strategy and
/// `denom <= f32::MIN_POSITIVE` guard as the scalar kernel — see
/// [`crate::scalar`] for the full rationale, the underflow floor,
/// and the NaN passthrough note.
///
/// # Safety
///
/// See [`dot`].
#[target_feature(enable = "neon")]
pub(crate) unsafe fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot_acc = vdupq_n_f32(0.0);
    let mut na_acc = vdupq_n_f32(0.0);
    let mut nb_acc = vdupq_n_f32(0.0);

    let chunks_a = a.chunks_exact(LANES);
    let chunks_b = b.chunks_exact(LANES);
    let tail_a = chunks_a.remainder();
    let tail_b = chunks_b.remainder();
    for (ca, cb) in chunks_a.zip(chunks_b) {
        // SAFETY: 4 contiguous f32 from `chunks_exact(4)`. NEON enabled.
        let va = unsafe { vld1q_f32(ca.as_ptr()) };
        // SAFETY: see above.
        let vb = unsafe { vld1q_f32(cb.as_ptr()) };
        dot_acc = vaddq_f32(dot_acc, vmulq_f32(va, vb));
        na_acc = vaddq_f32(na_acc, vmulq_f32(va, va));
        nb_acc = vaddq_f32(nb_acc, vmulq_f32(vb, vb));
    }
    let mut dot = vaddvq_f32(dot_acc);
    let mut na = vaddvq_f32(na_acc);
    let mut nb = vaddvq_f32(nb_acc);
    for (x, y) in tail_a.iter().zip(tail_b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    // Each square root runs independently so the `na * nb` product
    // cannot underflow for low-magnitude inputs; the guard catches
    // true-zero norms and subnormal-product underflows uniformly.
    // See the scalar cosine for the full rationale (audit finding
    // M11). Identical strategy across scalar, AVX2, and NEON.
    let denom = na.sqrt() * nb.sqrt();
    if denom <= f32::MIN_POSITIVE {
        return 1.0;
    }
    1.0 - dot / denom
}

/// NEON Euclidean (L2) distance. Caller MUST have validated inputs and NEON.
///
/// # Safety
///
/// See [`dot`].
#[target_feature(enable = "neon")]
pub(crate) unsafe fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = vdupq_n_f32(0.0);
    let chunks_a = a.chunks_exact(LANES);
    let chunks_b = b.chunks_exact(LANES);
    let tail_a = chunks_a.remainder();
    let tail_b = chunks_b.remainder();
    for (ca, cb) in chunks_a.zip(chunks_b) {
        // SAFETY: 4 contiguous f32 from `chunks_exact(4)`. NEON enabled.
        let va = unsafe { vld1q_f32(ca.as_ptr()) };
        // SAFETY: see above.
        let vb = unsafe { vld1q_f32(cb.as_ptr()) };
        let diff = vsubq_f32(va, vb);
        acc = vaddq_f32(acc, vmulq_f32(diff, diff));
    }
    let mut sum = vaddvq_f32(acc);
    for (x, y) in tail_a.iter().zip(tail_b.iter()) {
        let d = x - y;
        sum += d * d;
    }
    sum.sqrt()
}

/// NEON Manhattan (L1) distance. Caller MUST have validated inputs and NEON.
///
/// # Safety
///
/// See [`dot`].
#[target_feature(enable = "neon")]
pub(crate) unsafe fn manhattan(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = vdupq_n_f32(0.0);
    let chunks_a = a.chunks_exact(LANES);
    let chunks_b = b.chunks_exact(LANES);
    let tail_a = chunks_a.remainder();
    let tail_b = chunks_b.remainder();
    for (ca, cb) in chunks_a.zip(chunks_b) {
        // SAFETY: 4 contiguous f32 from `chunks_exact(4)`. NEON enabled.
        let va = unsafe { vld1q_f32(ca.as_ptr()) };
        // SAFETY: see above.
        let vb = unsafe { vld1q_f32(cb.as_ptr()) };
        let diff = vsubq_f32(va, vb);
        // `vabsq_f32` clears the sign bit lane-wise: NaN-preserving
        // (still NaN, payload retained) and avoids the prior
        // `vmaxq_f32(diff, -diff)` formulation, which is correct here
        // only because both operands are NaN-or-both-finite — relying
        // on FMAX tie-break semantics is unnecessary when a one-
        // instruction `vabsq_f32` says exactly what we mean.
        acc = vaddq_f32(acc, vabsq_f32(diff));
    }
    let mut sum = vaddvq_f32(acc);
    for (x, y) in tail_a.iter().zip(tail_b.iter()) {
        sum += (x - y).abs();
    }
    sum
}

/// NEON bit-exact Hamming on `&[f32]`. Caller MUST have validated inputs
/// and NEON.
///
/// # Safety
///
/// See [`dot`].
#[target_feature(enable = "neon")]
pub(crate) unsafe fn hamming(a: &[f32], b: &[f32]) -> f32 {
    let mut diff_count: u64 = 0;
    let chunks_a = a.chunks_exact(LANES);
    let chunks_b = b.chunks_exact(LANES);
    let tail_a = chunks_a.remainder();
    let tail_b = chunks_b.remainder();
    for (ca, cb) in chunks_a.zip(chunks_b) {
        // SAFETY: 4 contiguous f32 from `chunks_exact(4)`. NEON enabled.
        let va = unsafe { vld1q_f32(ca.as_ptr()) };
        // SAFETY: see above.
        let vb = unsafe { vld1q_f32(cb.as_ptr()) };
        // Bit-pattern compare: reinterpret as 4 × u32, then `vceqq_u32`
        // gives all-1s where equal and all-0s where differing.
        let ia = vreinterpretq_u32_f32(va);
        let ib = vreinterpretq_u32_f32(vb);
        let eq = vceqq_u32(ia, ib);
        let neq = vmvnq_u32(eq);
        // Each differing lane becomes 1, each equal lane becomes 0.
        let bits = vshrq_n_u32::<31>(neq);
        diff_count += vaddvq_u32(bits) as u64;
    }
    for (x, y) in tail_a.iter().zip(tail_b.iter()) {
        if x.to_bits() != y.to_bits() {
            diff_count += 1;
        }
    }
    diff_count as f32
}
