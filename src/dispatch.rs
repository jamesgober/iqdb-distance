//! Top-level dispatch over [`iqdb_types::DistanceMetric`].
//!
//! These functions are the runtime equivalent of the type-level
//! [`crate::Distance`] trait: the caller chooses the metric at runtime
//! through the enum tag, the dispatcher routes to the matching metric
//! type. They are the entry points used by the engine and query crates,
//! which do not know the metric at compile time.

use iqdb_types::{DistanceMetric, IqdbError, Result};

use crate::{Cosine, Distance, DotProduct, Euclidean, Hamming, Manhattan};

/// Compute the distance for `metric` between `a` and `b`.
///
/// Mismatched or empty inputs surface as [`iqdb_types::IqdbError`].
///
/// [`DistanceMetric`] is `#[non_exhaustive]`. A metric this crate does not
/// implement yet surfaces as [`iqdb_types::IqdbError::InvalidMetric`] rather
/// than a panic, so a newer `iqdb-types` that adds a variant never breaks the
/// dispatcher at runtime.
///
/// # Examples
///
/// ```
/// use iqdb_distance::compute;
/// use iqdb_types::DistanceMetric;
///
/// let a = [0.0_f32, 0.0, 0.0];
/// let b = [3.0_f32, 4.0, 0.0];
/// let d = compute(DistanceMetric::Euclidean, &a, &b)
///     .expect("valid pair");
/// assert!((d - 5.0).abs() < 1e-6);
/// ```
pub fn compute(metric: DistanceMetric, a: &[f32], b: &[f32]) -> Result<f32> {
    match metric {
        DistanceMetric::Cosine => Cosine::compute(a, b),
        DistanceMetric::DotProduct => DotProduct::compute(a, b),
        DistanceMetric::Euclidean => Euclidean::compute(a, b),
        DistanceMetric::Manhattan => Manhattan::compute(a, b),
        DistanceMetric::Hamming => Hamming::compute(a, b),
        // `DistanceMetric` is `#[non_exhaustive]`; a future variant this crate
        // has not implemented is a configuration error, not a panic.
        _ => Err(IqdbError::InvalidMetric),
    }
}

/// Compute distances for `metric` between `query` and each entry in
/// `candidates`, writing into `out`.
///
/// `out.len()` MUST equal `candidates.len()`. The function is
/// allocation-free; the caller owns the output buffer.
///
/// As with [`compute`], a metric outside the implemented set surfaces as
/// [`iqdb_types::IqdbError::InvalidMetric`] (`DistanceMetric` is
/// `#[non_exhaustive]`).
///
/// # Examples
///
/// ```
/// use iqdb_distance::compute_batch;
/// use iqdb_types::DistanceMetric;
///
/// let q = [0.0_f32, 0.0];
/// let cs: [&[f32]; 2] = [&[1.0, 0.0], &[0.0, 2.0]];
/// let mut out = [0.0_f32; 2];
///
/// compute_batch(DistanceMetric::Manhattan, &q, &cs, &mut out)
///     .expect("matching lengths");
/// assert!((out[0] - 1.0).abs() < 1e-6);
/// assert!((out[1] - 2.0).abs() < 1e-6);
/// ```
pub fn compute_batch(
    metric: DistanceMetric,
    query: &[f32],
    candidates: &[&[f32]],
    out: &mut [f32],
) -> Result<()> {
    match metric {
        DistanceMetric::Cosine => Cosine::compute_batch(query, candidates, out),
        DistanceMetric::DotProduct => DotProduct::compute_batch(query, candidates, out),
        DistanceMetric::Euclidean => Euclidean::compute_batch(query, candidates, out),
        DistanceMetric::Manhattan => Manhattan::compute_batch(query, candidates, out),
        DistanceMetric::Hamming => Hamming::compute_batch(query, candidates, out),
        // See `compute`: a future, unimplemented metric is a configuration
        // error rather than a panic.
        _ => Err(IqdbError::InvalidMetric),
    }
}
