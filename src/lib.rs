//! # iqdb-distance
//!
//! Distance and similarity functions for the **iQDB** vector-database
//! spine. The crate owns the math: given two `&[f32]` slices and a
//! [`iqdb_types::DistanceMetric`] it computes the requested distance and
//! returns it as `f32`. The five metrics — Cosine, DotProduct, Euclidean
//! (L2), Manhattan (L1), and Hamming — sit behind the single [`Distance`]
//! trait, with one zero-sized type per metric ([`Cosine`], [`DotProduct`],
//! [`Euclidean`], [`Manhattan`], [`Hamming`]). The top-level [`compute`] and
//! [`compute_batch`] functions take the metric tag and route to the right
//! implementation.
//!
//! For pre-normalized embeddings, [`cosine_normalized`] is a fast path that
//! skips the norm and division (`1 - a · b`), and [`normalize`] produces the
//! unit vectors it expects.
//!
//! Performance: every distance call is allocation-free ([`normalize`], which
//! returns a new vector, is the sole documented exception). SIMD kernels (AVX2
//! on x86_64, NEON on aarch64) are picked at runtime from [`detect_features`]
//! and short-circuit to the scalar reference when the host lacks the feature
//! or when `force_scalar` has been called.
//!
//! ## Example
//!
//! ```
//! use iqdb_distance::{Cosine, Distance};
//!
//! let a = [1.0_f32, 0.0, 0.0];
//! let b = [0.0_f32, 1.0, 0.0];
//!
//! // Perpendicular unit vectors -> cosine distance ~ 1.0.
//! let d = Cosine::compute(&a, &b).expect("non-empty, same length");
//! assert!((d - 1.0).abs() < 1e-6);
//! ```
//!
//! ## Errors
//!
//! Every fallible call returns [`iqdb_types::Result`]. Empty inputs and
//! length mismatches surface as [`iqdb_types::IqdbError::InvalidVector`] and
//! [`iqdb_types::IqdbError::DimensionMismatch`] respectively; the library
//! never panics on bad input.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(warnings)]
#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(unused_must_use)]
#![deny(unused_results)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![deny(clippy::print_stdout)]
#![deny(clippy::print_stderr)]
#![deny(clippy::dbg_macro)]
#![deny(clippy::unreachable)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod dispatch;
mod features;
mod metrics;
mod normalized;
mod scalar;
mod simd;
mod traits;
mod validate;

pub use crate::dispatch::{compute, compute_batch};
pub use crate::features::{CpuFeatures, detect_features, forced_scalar};
#[cfg(any(test, feature = "testing"))]
pub use crate::features::{force_scalar, which_kernel};
pub use crate::metrics::{Cosine, DotProduct, Euclidean, Hamming, Manhattan};
pub use crate::normalized::{cosine_normalized, normalize};
pub use crate::traits::Distance;

/// The version of this crate, taken from `Cargo.toml` at compile time.
///
/// Exposed so a consumer can report the exact `iqdb-distance` build it links
/// against — useful in diagnostics and version-skew checks across the iqdb
/// crate family.
///
/// # Examples
///
/// ```
/// // Carries a `major.minor.patch` SemVer core.
/// let version = iqdb_distance::VERSION;
/// assert_eq!(version.split('.').count(), 3);
/// assert!(version.split('.').all(|part| !part.is_empty()));
/// ```
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
