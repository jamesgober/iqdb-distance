//! The [`Distance`] trait — one entry point per metric.
//!
//! Each metric in the crate ([`crate::Cosine`], [`crate::DotProduct`],
//! [`crate::Euclidean`], [`crate::Manhattan`], [`crate::Hamming`]) is a
//! zero-sized type that implements this trait. The associated functions
//! (`compute`, `compute_batch`) are dispatched at the type level — there is
//! no virtual call, no `dyn` indirection, and no allocation on the hot
//! path.

use iqdb_types::Result;

/// Compute a distance between two `&[f32]` slices.
///
/// The associated functions take no receiver — every metric type in this
/// crate is zero-sized and is used as a tag, not a value. Mismatched or
/// empty inputs are returned as
/// [`iqdb_types::IqdbError`](iqdb_types::IqdbError) and the library never
/// panics.
///
/// # Examples
///
/// ```
/// use iqdb_distance::{Distance, Euclidean};
///
/// let a = [0.0_f32, 0.0, 0.0];
/// let b = [3.0_f32, 4.0, 0.0];
///
/// let d = Euclidean::compute(&a, &b).expect("non-empty, same length");
/// assert!((d - 5.0).abs() < 1e-6);
/// ```
pub trait Distance {
    /// Compute the distance between `a` and `b`.
    ///
    /// Returns [`iqdb_types::IqdbError::InvalidVector`] if either slice is
    /// empty, and [`iqdb_types::IqdbError::DimensionMismatch`] if the two
    /// slices have different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use iqdb_distance::{Distance, DotProduct};
    ///
    /// let a = [1.0_f32, 2.0, 3.0];
    /// let b = [4.0_f32, -5.0, 6.0];
    ///
    /// let d = DotProduct::compute(&a, &b).expect("non-empty, same length");
    /// // 1*4 + 2*(-5) + 3*6 = 12.
    /// assert!((d - 12.0).abs() < 1e-6);
    /// ```
    fn compute(a: &[f32], b: &[f32]) -> Result<f32>;

    /// Compute distances between `query` and each entry in `candidates`,
    /// writing the results in order into `out`.
    ///
    /// The function is allocation-free: `out` is caller-supplied. Returns
    /// [`iqdb_types::IqdbError::InvalidConfig`] if `out.len()` does not
    /// equal `candidates.len()`, and propagates per-pair errors via
    /// [`Distance::compute`].
    ///
    /// # Examples
    ///
    /// ```
    /// use iqdb_distance::{Distance, Manhattan};
    ///
    /// let q = [0.0_f32, 0.0];
    /// let cs: [&[f32]; 2] = [&[1.0, 0.0], &[0.0, 2.0]];
    /// let mut out = [0.0_f32; 2];
    ///
    /// Manhattan::compute_batch(&q, &cs, &mut out)
    ///     .expect("matching lengths");
    /// assert!((out[0] - 1.0).abs() < 1e-6);
    /// assert!((out[1] - 2.0).abs() < 1e-6);
    /// ```
    fn compute_batch(query: &[f32], candidates: &[&[f32]], out: &mut [f32]) -> Result<()>;
}
