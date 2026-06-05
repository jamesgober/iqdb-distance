//! Per-metric implementations of the [`crate::Distance`] trait.
//!
//! Each module exposes a public zero-sized type — [`Cosine`],
//! [`DotProduct`], [`Euclidean`], [`Manhattan`], [`Hamming`] — used as a
//! tag for the type-level dispatch. The implementations validate the input
//! pair via [`crate::validate::pair`], then route the math through the
//! scalar reference today and the SIMD kernels in a later release.

mod cosine;
mod dot;
mod euclidean;
mod hamming;
mod manhattan;

pub use cosine::Cosine;
pub use dot::DotProduct;
pub use euclidean::Euclidean;
pub use hamming::Hamming;
pub use manhattan::Manhattan;

use iqdb_types::Result;

/// Shared `compute_batch` implementation. Each metric's
/// [`crate::Distance::compute_batch`] delegates here, passing the pair
/// kernel as a function pointer so no virtual dispatch is needed.
pub(crate) fn batch_with(
    query: &[f32],
    candidates: &[&[f32]],
    out: &mut [f32],
    pair: fn(&[f32], &[f32]) -> Result<f32>,
) -> Result<()> {
    crate::validate::batch(candidates, out)?;
    for (slot, candidate) in out.iter_mut().zip(candidates.iter()) {
        *slot = pair(query, candidate)?;
    }
    Ok(())
}
