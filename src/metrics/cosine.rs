//! [`Cosine`] — cosine distance.

use iqdb_types::Result;

use crate::Distance;

/// Cosine distance: `1 - cos(angle(a, b))`.
///
/// Magnitude-independent — well-suited to normalized embeddings. Zero
/// vectors are defined to have distance `1.0` (no similarity).
///
/// # Examples
///
/// ```
/// use iqdb_distance::{Cosine, Distance};
///
/// let a = [1.0_f32, 0.0];
/// let b = [0.0_f32, 1.0];
/// let d = Cosine::compute(&a, &b).expect("valid pair");
/// assert!((d - 1.0).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Cosine;

impl Distance for Cosine {
    fn compute(a: &[f32], b: &[f32]) -> Result<f32> {
        crate::validate::pair(a, b)?;
        Ok(dispatch(a, b))
    }

    fn compute_batch(query: &[f32], candidates: &[&[f32]], out: &mut [f32]) -> Result<()> {
        crate::metrics::batch_with(query, candidates, out, Cosine::compute)
    }
}

fn dispatch(a: &[f32], b: &[f32]) -> f32 {
    let features = crate::features::detect_features();
    match crate::features::select_kernel(features) {
        crate::features::Kernel::Scalar => crate::scalar::cosine::compute(a, b),
        #[cfg(target_arch = "x86_64")]
        crate::features::Kernel::Avx2 => {
            // SAFETY: `Kernel::Avx2` is only returned by `select_kernel` when
            // `features.avx2` is true, which itself requires
            // `is_x86_feature_detected!("avx2")` at startup — exactly the
            // contract `#[target_feature(enable = "avx2")]` requires.
            unsafe { crate::simd::avx2::cosine(a, b) }
        }
        #[cfg(target_arch = "aarch64")]
        crate::features::Kernel::Neon => {
            // SAFETY: `Kernel::Neon` is only returned by `select_kernel` when
            // `features.neon` is true, which itself requires
            // `is_aarch64_feature_detected!("neon")` at startup —
            // `#[target_feature(enable = "neon")]` is sound.
            unsafe { crate::simd::neon::cosine(a, b) }
        }
    }
}
