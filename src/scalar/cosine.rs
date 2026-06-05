//! Scalar cosine distance.

/// Compute `1 - cos(angle(a, b))`.
///
/// The caller MUST have validated that `a.len() == b.len() != 0`.
///
/// # Denominator strategy (audit M11)
///
/// The denominator is computed as `norm_a.sqrt() * norm_b.sqrt()`, not
/// `(norm_a * norm_b).sqrt()`. The two are mathematically equivalent
/// but the former is numerically robust: the latter can underflow the
/// f32 product even when both norms are individually finite (a pair of
/// vectors at magnitude `1e-15` has `norm_a, norm_b ≈ 8e-30`, but
/// `norm_a * norm_b ≈ 6e-59` underflows to 0 and the original
/// `denom == 0` guard would mis-report orthogonality).
///
/// The guard tests `denom <= f32::MIN_POSITIVE`, which covers both
/// the legitimate zero-norm case and the residual case where both
/// square roots are subnormal. The guard reports `1.0` (no
/// similarity), matching common embedding-store conventions.
///
/// # Underflow floor
///
/// There is still a floor: when `norm_a` (or `norm_b`) itself
/// underflows below `f32::MIN_POSITIVE`, the direction is unrecoverable
/// and the result is `1.0`. Inputs whose components are `~1e-22` reach
/// this floor on `dim=8` (`norm ≈ 8e-44 < f32::MIN_POSITIVE ≈ 1.18e-38`).
/// Embedding pipelines should normalise out of this regime before
/// reaching the cosine path.
///
/// # NaN passthrough
///
/// The guard does not intercept NaN: a NaN component in either input
/// makes `norm_a` (or `norm_b`) NaN, and `NaN <= f32::MIN_POSITIVE` is
/// `false` per IEEE-754. The function then returns `1.0 - dot / NaN`,
/// which is also NaN. This is consistent with the documented caller
/// contract: `iqdb_types::Vector::new` rejects non-finite components
/// at the type boundary, so the kernels never see a NaN under normal
/// use. Callers reaching the kernels via `Vector::new_unchecked` (a
/// `testing`-gated escape hatch) accept the NaN-propagation
/// consequence.
pub(crate) fn compute(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;
    for i in 0..a.len() {
        let x = a[i];
        let y = b[i];
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom <= f32::MIN_POSITIVE {
        return 1.0;
    }
    1.0 - dot / denom
}
