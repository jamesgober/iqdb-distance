//! `f64` fallbacks for results that overflowed `f32`.
//!
//! The fast kernels accumulate in `f32`. Finite components above roughly
//! `1e19` square past `f32::MAX`, so a sum can reach `inf`, and `inf - inf`
//! or `inf / inf` gives NaN. On x86 that NaN has its sign bit set, which
//! [`f32::total_cmp`] orders below every number, so it would rank as the
//! best match. The metric entry points only call these when the fast
//! result is not finite, which never happens for ordinary embeddings, so
//! the hot path is unchanged.
//!
//! Every product of two finite `f32` values fits in `f64` with a wide
//! margin, so these sums stay finite for any realistic dimension and the
//! result is cast back to `f32` once at the end.

/// Cosine distance accumulated in `f64`. Same zero-norm rule as the fast
/// kernels: a zero-magnitude input gives `1.0`.
pub(crate) fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let x = f64::from(x);
        let y = f64::from(y);
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom <= f64::MIN_POSITIVE {
        return 1.0;
    }
    (1.0 - dot / denom) as f32
}

/// Dot product accumulated in `f64`. A true result beyond the `f32` range
/// comes back as `inf` or `-inf`, never NaN.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        acc += f64::from(x) * f64::from(y);
    }
    acc as f32
}

/// Euclidean distance accumulated in `f64`. Recovers a finite distance
/// whose squared sum overflowed `f32` even though the distance itself fits.
pub(crate) fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = f64::from(x) - f64::from(y);
        acc += d * d;
    }
    acc.sqrt() as f32
}
