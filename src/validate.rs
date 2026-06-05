//! Shared input validation for the metric implementations.
//!
//! Empty and mismatched slices are surfaced as [`iqdb_types::IqdbError`]
//! variants so callers receive a typed error instead of a panic.

use iqdb_types::{IqdbError, Result};

/// Reject empty or mismatched-length inputs.
///
/// Returns [`IqdbError::InvalidVector`] if either slice is empty, and
/// [`IqdbError::DimensionMismatch`] when the two slices have different
/// lengths. On success the call is effectively free — no allocation, two
/// length reads.
pub(crate) fn pair(a: &[f32], b: &[f32]) -> Result<()> {
    if a.is_empty() || b.is_empty() {
        return Err(IqdbError::InvalidVector);
    }
    if a.len() != b.len() {
        return Err(IqdbError::DimensionMismatch {
            expected: a.len(),
            found: b.len(),
        });
    }
    Ok(())
}

/// Reject a batch where the output buffer cannot hold one distance per
/// candidate.
pub(crate) fn batch(candidates: &[&[f32]], out: &[f32]) -> Result<()> {
    if out.len() != candidates.len() {
        return Err(IqdbError::InvalidConfig {
            reason: "batch output length does not match candidate count",
        });
    }
    Ok(())
}
