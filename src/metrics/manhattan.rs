//! [`Manhattan`] — L1 distance.

use iqdb_types::Result;

use crate::Distance;

/// Manhattan (L1) distance: `sum(|a[i] - b[i]|)`.
///
/// Sum of absolute per-component differences. Non-negative and exactly
/// zero when the vectors agree componentwise.
///
/// # Examples
///
/// ```
/// use iqdb_distance::{Distance, Manhattan};
///
/// let a = [0.0_f32, 0.0];
/// let b = [3.0_f32, 4.0];
/// let d = Manhattan::compute(&a, &b).expect("valid pair");
/// assert!((d - 7.0).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Manhattan;

impl Distance for Manhattan {
    fn compute(a: &[f32], b: &[f32]) -> Result<f32> {
        crate::validate::pair(a, b)?;
        Ok(dispatch(a, b))
    }

    fn compute_batch(query: &[f32], candidates: &[&[f32]], out: &mut [f32]) -> Result<()> {
        crate::metrics::batch_with(query, candidates, out, Manhattan::compute)
    }
}

fn dispatch(a: &[f32], b: &[f32]) -> f32 {
    let features = crate::features::detect_features();
    match crate::features::select_kernel(features) {
        crate::features::Kernel::Scalar => crate::scalar::manhattan::compute(a, b),
        #[cfg(target_arch = "x86_64")]
        crate::features::Kernel::Avx2 => {
            // SAFETY: `Kernel::Avx2` is only selected when AVX2 detection
            // succeeded at startup; `#[target_feature(enable = "avx2")]`
            // is sound here.
            unsafe { crate::simd::avx2::manhattan(a, b) }
        }
        #[cfg(target_arch = "aarch64")]
        crate::features::Kernel::Neon => {
            // SAFETY: `Kernel::Neon` is only selected when NEON detection
            // succeeded at startup; `#[target_feature(enable = "neon")]`
            // is sound.
            unsafe { crate::simd::neon::manhattan(a, b) }
        }
    }
}
