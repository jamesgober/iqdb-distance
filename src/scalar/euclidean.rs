//! Scalar Euclidean (L2) distance.

/// Compute `sqrt(sum((a[i] - b[i]) ^ 2))`.
///
/// The caller MUST have validated that `a.len() == b.len() != 0`. The
/// result is non-negative and exactly zero when `a == b` componentwise.
pub(crate) fn compute(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc.sqrt()
}
