//! Cosine correctness on low-magnitude inputs (audit finding M11).
//!
//! The original `(na * nb).sqrt()` formulation underflows the f32
//! product when both norms are small but each is well above
//! `f32::MIN_POSITIVE`. A pair of parallel vectors at magnitude `1e-15`
//! has `na = nb ≈ 8e-30` — each individually finite and well-formed —
//! but `na * nb ≈ 6e-59` underflows to zero, the original guard fires,
//! and the call returns `1.0` (orthogonal) for vectors that are
//! geometrically parallel.
//!
//! The fix computes `denom = na.sqrt() * nb.sqrt()` so each square root
//! runs independently, then keeps a `denom <= f32::MIN_POSITIVE` guard
//! to catch both the legitimate zero-norm case and the residual case
//! where both square roots are subnormal.
//!
//! There is still a floor: when `na` (or `nb`) itself underflows below
//! `f32::MIN_POSITIVE`, no later arithmetic can recover the direction.
//! `1e-22` magnitude gives `na ≈ 8e-44 < f32::MIN_POSITIVE` — the
//! kernel treats both vectors as zero-magnitude and returns `1.0`.
//! That floor is documented in the public cosine rustdoc.

#![allow(clippy::unwrap_used)]

use iqdb_distance::{Cosine, Distance};

const DIM: usize = 8;

/// Build a vector by scaling a unit-shape pattern. The shape is chosen
/// so two vectors built with the same scale are colinear (one is a
/// positive multiple of the other → cosine distance ≈ 0).
fn scaled(scale: f32) -> Vec<f32> {
    (0..DIM).map(|i| (i as f32 + 1.0) * scale).collect()
}

/// Build an orthogonal pair at the given scale: `a` has support on the
/// first half, `b` on the second half. Cosine distance ≈ 1.
fn orthogonal_pair(scale: f32) -> (Vec<f32>, Vec<f32>) {
    let mut a = vec![0.0_f32; DIM];
    let mut b = vec![0.0_f32; DIM];
    for i in 0..DIM / 2 {
        a[i] = (i as f32 + 1.0) * scale;
        b[DIM / 2 + i] = (i as f32 + 1.0) * scale;
    }
    (a, b)
}

#[test]
fn parallel_low_magnitude_vectors_are_not_orthogonal() {
    // 1e-15 lives well above `f32::MIN_POSITIVE` (~1.18e-38) on each
    // component AND on each squared-norm, but the *product* of the two
    // squared norms underflows. The fix routes around the product
    // underflow, so the parallel pair must report distance ≈ 0.
    let a = scaled(1.0e-15);
    let b = scaled(1.0e-15);
    let d = Cosine::compute(&a, &b).unwrap();
    assert!(
        d.abs() < 1.0e-3,
        "parallel vectors at 1e-15 should report distance ≈ 0, got {d}",
    );
}

#[test]
fn orthogonal_low_magnitude_vectors_are_still_orthogonal() {
    // The fix MUST NOT collapse orthogonal pairs to distance 0. At the
    // same low magnitude, an orthogonal pair still reports ≈ 1.0.
    let (a, b) = orthogonal_pair(1.0e-15);
    let d = Cosine::compute(&a, &b).unwrap();
    assert!(
        (d - 1.0).abs() < 1.0e-3,
        "orthogonal vectors at 1e-15 should report distance ≈ 1, got {d}",
    );
}

#[test]
fn vectors_below_underflow_floor_are_treated_as_zero_magnitude() {
    // At 1e-22 the per-component squared sum itself underflows below
    // `f32::MIN_POSITIVE` and there is no arithmetic recovery. The
    // contract is: such vectors hit the guard and report 1.0, matching
    // the documented "zero-magnitude vectors are not similar"
    // convention. This pins the floor so a future change cannot
    // accidentally extend the fix into a region where the answer would
    // be a fabrication.
    let a = scaled(1.0e-22);
    let b = scaled(1.0e-22);
    let d = Cosine::compute(&a, &b).unwrap();
    assert!(
        (d - 1.0).abs() < 1.0e-6,
        "sub-underflow vectors must be treated as zero-magnitude (distance 1.0), got {d}",
    );
}

#[test]
fn moderate_magnitude_unchanged() {
    // Sanity: the fix MUST NOT shift the answer for the bread-and-butter
    // case. A parallel pair at unit magnitude still reports ≈ 0.
    let a = scaled(1.0);
    let b = scaled(1.0);
    let d = Cosine::compute(&a, &b).unwrap();
    assert!(
        d.abs() < 1.0e-6,
        "parallel unit-scale vectors should report distance ≈ 0, got {d}",
    );
}
