//! Fast paths for pre-normalized (unit-length) vectors.
//!
//! When embeddings are L2-normalized up front — a common ingest-time step —
//! cosine distance collapses to `1 - (a · b)`: the per-call norm, square root,
//! and division the general [`crate::Cosine`] kernel performs are all
//! unnecessary. [`cosine_normalized`] takes that fast path, reusing the same
//! runtime-dispatched dot kernel (scalar / AVX2 / NEON) the rest of the crate
//! uses. [`normalize`] produces the unit vectors it expects.
//!
//! These are the only allocation note in the crate: [`normalize`] returns an
//! owned `Vec<f32>` because it produces a new vector. [`cosine_normalized`],
//! like every other distance call, is allocation-free.

use iqdb_types::{IqdbError, Result};

use crate::{Distance, DotProduct};

/// Cosine distance for two **already unit-length** vectors: `1 - (a · b)`.
///
/// This is the fast path for pre-normalized embeddings. It skips the norm,
/// square root, and division that [`crate::Cosine`] computes, evaluating only
/// the dot product through the runtime-dispatched SIMD kernel. For genuinely
/// unit-length inputs the result equals [`crate::Cosine::compute`] within
/// floating-point tolerance and lies in `[0, 2]`.
///
/// **Contract.** The caller guarantees `a` and `b` are unit length (use
/// [`normalize`] to produce them). If they are not, the return value is still
/// `1 - (a · b)` but is no longer a cosine distance and may fall outside
/// `[0, 2]` — there is no normalization to rescue it. When magnitudes are
/// unknown, use [`crate::Cosine`] instead, which normalizes internally.
///
/// # Errors
///
/// Returns [`IqdbError::InvalidVector`] if either slice is empty, and
/// [`IqdbError::DimensionMismatch`] if the lengths differ — the same input
/// contract as every other distance call.
///
/// # Examples
///
/// ```
/// use iqdb_distance::cosine_normalized;
///
/// // Identical unit vectors → distance 0.
/// let a = [1.0_f32, 0.0, 0.0];
/// let d = cosine_normalized(&a, &a).expect("valid pair");
/// assert!(d.abs() < 1e-6);
///
/// // Perpendicular unit vectors → distance 1.0.
/// let b = [0.0_f32, 1.0, 0.0];
/// let d = cosine_normalized(&a, &b).expect("valid pair");
/// assert!((d - 1.0).abs() < 1e-6);
/// ```
///
/// Matching the general cosine kernel once the inputs are normalized:
///
/// ```
/// use iqdb_distance::{Cosine, Distance, cosine_normalized, normalize};
///
/// let a = normalize(&[1.0_f32, 2.0, 3.0]).expect("non-zero");
/// let b = normalize(&[-2.0_f32, 0.5, 4.0]).expect("non-zero");
///
/// let fast = cosine_normalized(&a, &b).expect("valid pair");
/// let full = Cosine::compute(&a, &b).expect("valid pair");
/// assert!((fast - full).abs() < 1e-6);
/// ```
pub fn cosine_normalized(a: &[f32], b: &[f32]) -> Result<f32> {
    // `DotProduct::compute` validates the pair and runs the dispatched SIMD
    // dot kernel; the normalized cosine distance is one subtraction on top.
    let dot = DotProduct::compute(a, b)?;
    Ok(1.0 - dot)
}

/// Return the L2-normalized (unit-length) copy of `v`: `v / ‖v‖`.
///
/// Use it once at ingest to turn raw embeddings into the unit vectors
/// [`cosine_normalized`] expects, then store the result. The squared norm is
/// computed through the same SIMD dot kernel (`‖v‖² = v · v`).
///
/// Unlike the distance functions this **allocates** — it returns a new
/// `Vec<f32>` — because it produces a new vector rather than reducing two to a
/// scalar.
///
/// # Errors
///
/// Returns [`IqdbError::InvalidVector`] if `v` is empty, or if its magnitude is
/// not a usable positive, finite value — a zero vector, a subnormal-magnitude
/// vector, or one whose norm is non-finite (a `NaN`/`∞` component, or an
/// overflowing sum of squares) cannot be turned into a unit vector. This
/// mirrors the family's validate-at-construction stance: a vector you cannot
/// normalize is rejected rather than silently returned as `NaN`s.
///
/// # Examples
///
/// ```
/// use iqdb_distance::normalize;
///
/// // 3-4-5 triangle → unit vector [0.6, 0.8].
/// let unit = normalize(&[3.0_f32, 4.0]).expect("non-zero magnitude");
/// assert!((unit[0] - 0.6).abs() < 1e-6);
/// assert!((unit[1] - 0.8).abs() < 1e-6);
///
/// // The result has unit length.
/// let norm: f32 = unit.iter().map(|x| x * x).sum::<f32>().sqrt();
/// assert!((norm - 1.0).abs() < 1e-6);
///
/// // A zero-magnitude vector cannot be normalized.
/// assert!(normalize(&[0.0_f32, 0.0, 0.0]).is_err());
/// ```
pub fn normalize(v: &[f32]) -> Result<Vec<f32>> {
    // `v · v` validates non-empty and yields the squared norm via SIMD.
    let norm_sq = DotProduct::compute(v, v)?;
    let norm = norm_sq.sqrt();
    if !norm.is_finite() || norm <= f32::MIN_POSITIVE {
        return Err(IqdbError::InvalidVector);
    }
    let inv = 1.0 / norm;
    Ok(v.iter().map(|x| x * inv).collect())
}
