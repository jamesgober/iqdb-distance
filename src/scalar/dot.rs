//! Scalar dot-product similarity.

/// Compute the inner product `sum(a[i] * b[i])`.
///
/// The caller MUST have validated that `a.len() == b.len() != 0`. The
/// result is the raw dot product — it is *not* converted to a distance,
/// and it carries the sign of the underlying inner product.
pub(crate) fn compute(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}
