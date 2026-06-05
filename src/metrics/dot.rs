//! [`DotProduct`] — dot-product similarity.

use iqdb_types::Result;

use crate::Distance;

/// Dot-product similarity: the inner product `sum(a[i] * b[i])`.
///
/// This is *not* a distance — the result carries the sign of the
/// underlying inner product. Use it when magnitude carries signal (the
/// embedding store is not normalized) and you want a similarity score.
///
/// # Examples
///
/// ```
/// use iqdb_distance::{Distance, DotProduct};
///
/// let a = [1.0_f32, 2.0, 3.0];
/// let b = [4.0_f32, -5.0, 6.0];
/// let s = DotProduct::compute(&a, &b).expect("valid pair");
/// assert!((s - 12.0).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct DotProduct;

impl Distance for DotProduct {
    fn compute(a: &[f32], b: &[f32]) -> Result<f32> {
        crate::validate::pair(a, b)?;
        Ok(dispatch(a, b))
    }

    fn compute_batch(query: &[f32], candidates: &[&[f32]], out: &mut [f32]) -> Result<()> {
        crate::metrics::batch_with(query, candidates, out, DotProduct::compute)
    }
}

fn dispatch(a: &[f32], b: &[f32]) -> f32 {
    let features = crate::features::detect_features();
    match crate::features::select_kernel(features) {
        crate::features::Kernel::Scalar => crate::scalar::dot::compute(a, b),
        #[cfg(target_arch = "x86_64")]
        crate::features::Kernel::Avx2 => {
            // SAFETY: `Kernel::Avx2` reflects the cached AVX2 runtime probe
            // through `select_kernel`; `#[target_feature(enable = "avx2")]`
            // is sound when that detection succeeded.
            unsafe { crate::simd::avx2::dot(a, b) }
        }
        #[cfg(target_arch = "aarch64")]
        crate::features::Kernel::Neon => {
            // SAFETY: `Kernel::Neon` reflects the cached NEON runtime probe
            // through `select_kernel`; `#[target_feature(enable = "neon")]`
            // is sound.
            unsafe { crate::simd::neon::dot(a, b) }
        }
    }
}
