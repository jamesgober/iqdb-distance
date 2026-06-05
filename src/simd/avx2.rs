//! AVX2 kernels for x86_64.
//!
//! Every function in this module is `unsafe fn` and carries
//! `#[target_feature(enable = "avx2")]`. Calling any of them on a host
//! that does not advertise AVX2 is undefined behavior. The runtime guard
//! in [`crate::features`] is the contract: the per-metric dispatch in
//! [`crate::metrics`] consults [`crate::detect_features`] and only enters
//! one of these functions when `CpuFeatures::avx2` is `true`.
//!
//! Most AVX2 arithmetic intrinsics are safe to call inside a
//! `#[target_feature(enable = "avx2")]` function and are invoked
//! directly. Only the pointer-taking load `_mm256_loadu_ps` and store
//! `_mm256_storeu_ps` carry explicit `unsafe {}` blocks tying their
//! safety justification to the `chunks_exact(LANES)` source.

use std::arch::x86_64::{
    __m256, _mm256_add_ps, _mm256_andnot_ps, _mm256_castps_si256, _mm256_castsi256_ps,
    _mm256_cmpeq_epi32, _mm256_loadu_ps, _mm256_movemask_ps, _mm256_mul_ps, _mm256_set1_ps,
    _mm256_setzero_ps, _mm256_storeu_ps, _mm256_sub_ps,
};

const LANES: usize = 8;

#[inline]
#[target_feature(enable = "avx2")]
fn horizontal_sum(vec: __m256) -> f32 {
    let mut buf = [0.0_f32; LANES];
    // SAFETY: `buf` is 8 f32 = 32 bytes of contiguous, properly aligned
    // stack memory; `_mm256_storeu_ps` writes 32 unaligned bytes. AVX2 is
    // enabled by the enclosing `target_feature`.
    unsafe { _mm256_storeu_ps(buf.as_mut_ptr(), vec) };
    buf.iter().sum()
}

/// AVX2 dot-product. Caller MUST have validated `a.len() == b.len() != 0`
/// and confirmed AVX2 is available.
///
/// # Safety
///
/// Calling this on a host without AVX2 is undefined behavior. The caller
/// MUST gate the call on [`crate::features::detect_features`]
/// returning `avx2 = true`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let chunks_a = a.chunks_exact(LANES);
    let chunks_b = b.chunks_exact(LANES);
    let tail_a = chunks_a.remainder();
    let tail_b = chunks_b.remainder();
    for (ca, cb) in chunks_a.zip(chunks_b) {
        // SAFETY: `ca`/`cb` are 8 contiguous f32 (32 bytes) from
        // `chunks_exact(8)`; `_mm256_loadu_ps` reads 32 unaligned bytes.
        // AVX2 enabled by the enclosing `target_feature`.
        let va = unsafe { _mm256_loadu_ps(ca.as_ptr()) };
        // SAFETY: see above.
        let vb = unsafe { _mm256_loadu_ps(cb.as_ptr()) };
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
    }
    let mut sum = horizontal_sum(acc);
    for (x, y) in tail_a.iter().zip(tail_b.iter()) {
        sum += x * y;
    }
    sum
}

/// AVX2 cosine distance. Caller MUST have validated inputs and AVX2.
///
/// Uses the same `na.sqrt() * nb.sqrt()` denominator strategy and
/// `denom <= f32::MIN_POSITIVE` guard as the scalar kernel — see
/// [`crate::scalar`] for the full rationale, the underflow floor,
/// and the NaN passthrough note.
///
/// # Safety
///
/// See [`dot`].
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot_acc = _mm256_setzero_ps();
    let mut na_acc = _mm256_setzero_ps();
    let mut nb_acc = _mm256_setzero_ps();

    let chunks_a = a.chunks_exact(LANES);
    let chunks_b = b.chunks_exact(LANES);
    let tail_a = chunks_a.remainder();
    let tail_b = chunks_b.remainder();
    for (ca, cb) in chunks_a.zip(chunks_b) {
        // SAFETY: 32 contiguous bytes from `chunks_exact(8)`; AVX2 enabled.
        let va = unsafe { _mm256_loadu_ps(ca.as_ptr()) };
        // SAFETY: see above.
        let vb = unsafe { _mm256_loadu_ps(cb.as_ptr()) };
        dot_acc = _mm256_add_ps(dot_acc, _mm256_mul_ps(va, vb));
        na_acc = _mm256_add_ps(na_acc, _mm256_mul_ps(va, va));
        nb_acc = _mm256_add_ps(nb_acc, _mm256_mul_ps(vb, vb));
    }
    let mut dot = horizontal_sum(dot_acc);
    let mut na = horizontal_sum(na_acc);
    let mut nb = horizontal_sum(nb_acc);
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

/// AVX2 Euclidean (L2) distance. Caller MUST have validated inputs and AVX2.
///
/// # Safety
///
/// See [`dot`].
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let chunks_a = a.chunks_exact(LANES);
    let chunks_b = b.chunks_exact(LANES);
    let tail_a = chunks_a.remainder();
    let tail_b = chunks_b.remainder();
    for (ca, cb) in chunks_a.zip(chunks_b) {
        // SAFETY: 32 contiguous bytes from `chunks_exact(8)`; AVX2 enabled.
        let va = unsafe { _mm256_loadu_ps(ca.as_ptr()) };
        // SAFETY: see above.
        let vb = unsafe { _mm256_loadu_ps(cb.as_ptr()) };
        let diff = _mm256_sub_ps(va, vb);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
    }
    let mut sum = horizontal_sum(acc);
    for (x, y) in tail_a.iter().zip(tail_b.iter()) {
        let d = x - y;
        sum += d * d;
    }
    sum.sqrt()
}

/// AVX2 Manhattan (L1) distance. Caller MUST have validated inputs and AVX2.
///
/// # Safety
///
/// See [`dot`].
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn manhattan(a: &[f32], b: &[f32]) -> f32 {
    // `-0.0_f32` has only the sign bit set; `andnot(sign_mask, x)`
    // computes `!sign_mask & x`, lane-wise clearing each f32 sign bit.
    // This is the canonical SIMD abs: NaN-preserving (still NaN with
    // sign bit cleared, payload intact) and avoids the prior
    // `_mm256_max_ps(diff, -diff)` formulation, which is correct only
    // because both operands are NaN-or-both-finite — relying on MAXPS
    // tie-break semantics is unnecessary when one `andnot` says
    // exactly what we mean.
    let sign_mask = _mm256_set1_ps(-0.0_f32);
    let mut acc = _mm256_setzero_ps();
    let chunks_a = a.chunks_exact(LANES);
    let chunks_b = b.chunks_exact(LANES);
    let tail_a = chunks_a.remainder();
    let tail_b = chunks_b.remainder();
    for (ca, cb) in chunks_a.zip(chunks_b) {
        // SAFETY: 32 contiguous bytes from `chunks_exact(8)`; AVX2 enabled.
        let va = unsafe { _mm256_loadu_ps(ca.as_ptr()) };
        // SAFETY: see above.
        let vb = unsafe { _mm256_loadu_ps(cb.as_ptr()) };
        let diff = _mm256_sub_ps(va, vb);
        let abs = _mm256_andnot_ps(sign_mask, diff);
        acc = _mm256_add_ps(acc, abs);
    }
    let mut sum = horizontal_sum(acc);
    for (x, y) in tail_a.iter().zip(tail_b.iter()) {
        sum += (x - y).abs();
    }
    sum
}

/// AVX2 bit-exact Hamming on `&[f32]`. Caller MUST have validated inputs
/// and AVX2.
///
/// # Safety
///
/// See [`dot`].
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn hamming(a: &[f32], b: &[f32]) -> f32 {
    let mut diff_count: u64 = 0;
    let chunks_a = a.chunks_exact(LANES);
    let chunks_b = b.chunks_exact(LANES);
    let tail_a = chunks_a.remainder();
    let tail_b = chunks_b.remainder();
    for (ca, cb) in chunks_a.zip(chunks_b) {
        // SAFETY: 32 contiguous bytes from `chunks_exact(8)`; AVX2 enabled.
        let va = unsafe { _mm256_loadu_ps(ca.as_ptr()) };
        // SAFETY: see above.
        let vb = unsafe { _mm256_loadu_ps(cb.as_ptr()) };
        let ia = _mm256_castps_si256(va);
        let ib = _mm256_castps_si256(vb);
        let eq = _mm256_cmpeq_epi32(ia, ib);
        let eq_mask = _mm256_movemask_ps(_mm256_castsi256_ps(eq)) as u32;
        let equal = eq_mask.count_ones() as u64;
        diff_count += LANES as u64 - equal;
    }
    for (x, y) in tail_a.iter().zip(tail_b.iter()) {
        if x.to_bits() != y.to_bits() {
            diff_count += 1;
        }
    }
    diff_count as f32
}
