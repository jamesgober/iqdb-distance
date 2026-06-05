//! Scalar Hamming distance on `&[f32]`.
//!
//! Hamming on floats is defined as the count of positions where the two
//! values differ at the bit level (`to_bits()` comparison). This matches
//! the convention of "binary codes encoded as a 0.0/1.0 f32 vector" and
//! keeps the input shape uniform with the other metrics in the crate.

/// Count the positions where `a[i].to_bits() != b[i].to_bits()`, returned
/// as `f32`.
///
/// The caller MUST have validated that `a.len() == b.len() != 0`. Note
/// that `-0.0` and `+0.0` carry different bit patterns and therefore
/// contribute `1` to the count if a position has one of each; NaN bit
/// patterns compare bit-equal to themselves.
pub(crate) fn compute(a: &[f32], b: &[f32]) -> f32 {
    let mut count = 0u64;
    for i in 0..a.len() {
        if a[i].to_bits() != b[i].to_bits() {
            count += 1;
        }
    }
    count as f32
}
