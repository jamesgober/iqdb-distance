//! [`Hamming`] — count of differing positions.

use iqdb_types::Result;

use crate::Distance;

/// Hamming distance on `&[f32]`: the count of positions where the two
/// values differ at the bit level, returned as `f32`.
///
/// The comparison is `a[i].to_bits() != b[i].to_bits()` — `-0.0` and
/// `+0.0` differ in bit pattern and therefore differ here. Suits binary
/// codes encoded as 0.0/1.0 f32 vectors.
///
/// # Examples
///
/// ```
/// use iqdb_distance::{Distance, Hamming};
///
/// let a = [0.0_f32, 1.0, 0.0, 1.0];
/// let b = [0.0_f32, 0.0, 0.0, 1.0];
/// let d = Hamming::compute(&a, &b).expect("valid pair");
/// // One position differs.
/// assert!((d - 1.0).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Hamming;

impl Distance for Hamming {
    fn compute(a: &[f32], b: &[f32]) -> Result<f32> {
        crate::validate::pair(a, b)?;
        Ok(dispatch(a, b))
    }

    fn compute_batch(query: &[f32], candidates: &[&[f32]], out: &mut [f32]) -> Result<()> {
        crate::metrics::batch_with(query, candidates, out, Hamming::compute)
    }
}

fn dispatch(a: &[f32], b: &[f32]) -> f32 {
    let features = crate::features::detect_features();
    match crate::features::select_kernel(features) {
        crate::features::Kernel::Scalar => crate::scalar::hamming::compute(a, b),
        #[cfg(target_arch = "x86_64")]
        crate::features::Kernel::Avx2 => {
            // SAFETY: `Kernel::Avx2` is only selected when AVX2 detection
            // succeeded at startup; `#[target_feature(enable = "avx2")]`
            // is sound here.
            unsafe { crate::simd::avx2::hamming(a, b) }
        }
        #[cfg(target_arch = "aarch64")]
        crate::features::Kernel::Neon => {
            // SAFETY: `Kernel::Neon` is only selected when NEON detection
            // succeeded at startup; `#[target_feature(enable = "neon")]`
            // is sound.
            unsafe { crate::simd::neon::hamming(a, b) }
        }
    }
}
