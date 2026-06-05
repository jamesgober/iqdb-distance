//! Scalar Manhattan (L1) distance.

/// Compute `sum(|a[i] - b[i]|)`.
///
/// The caller MUST have validated that `a.len() == b.len() != 0`. The
/// result is non-negative and exactly zero when `a == b` componentwise.
pub(crate) fn compute(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    for i in 0..a.len() {
        acc += (a[i] - b[i]).abs();
    }
    acc
}
