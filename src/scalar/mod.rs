//! Scalar reference implementations of the five distance metrics.
//!
//! Each function in this module assumes its inputs have already been
//! validated by [`crate::validate::pair`] — same non-zero length. The
//! scalar paths are always compiled and act as the correctness reference
//! against which the SIMD kernels are differentially tested.

pub(crate) mod cosine;
pub(crate) mod dot;
pub(crate) mod euclidean;
pub(crate) mod hamming;
pub(crate) mod manhattan;
