//! [`Euclidean`] — L2 distance.

use iqdb_types::Result;

use crate::Distance;

/// Euclidean (L2) distance: `sqrt(sum((a[i] - b[i])^2))`.
///
/// Straight-line distance between coordinates. Non-negative and exactly
/// zero when the vectors agree componentwise.
///
/// # Examples
///
/// ```
/// use iqdb_distance::{Distance, Euclidean};
///
/// let a = [0.0_f32, 0.0, 0.0];
/// let b = [3.0_f32, 4.0, 0.0];
/// let d = Euclidean::compute(&a, &b).expect("valid pair");
/// assert!((d - 5.0).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Euclidean;

impl Distance for Euclidean {
    fn compute(a: &[f32], b: &[f32]) -> Result<f32> {
        crate::validate::pair(a, b)?;
        Ok(dispatch(a, b))
    }

    fn compute_batch(query: &[f32], candidates: &[&[f32]], out: &mut [f32]) -> Result<()> {
        crate::metrics::batch_with(query, candidates, out, Euclidean::compute)
    }
}

fn dispatch(a: &[f32], b: &[f32]) -> f32 {
    let features = crate::features::detect_features();
    match crate::features::select_kernel(features) {
        crate::features::Kernel::Scalar => crate::scalar::euclidean::compute(a, b),
        #[cfg(target_arch = "x86_64")]
        crate::features::Kernel::Avx2 => {
            // SAFETY: `Kernel::Avx2` is only selected when AVX2 detection
            // succeeded at startup; `#[target_feature(enable = "avx2")]`
            // is sound here.
            unsafe { crate::simd::avx2::euclidean(a, b) }
        }
        #[cfg(target_arch = "aarch64")]
        crate::features::Kernel::Neon => {
            // SAFETY: `Kernel::Neon` is only selected when NEON detection
            // succeeded at startup; `#[target_feature(enable = "neon")]`
            // is sound.
            unsafe { crate::simd::neon::euclidean(a, b) }
        }
    }
}
