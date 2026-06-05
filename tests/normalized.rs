//! Coverage for the pre-normalized fast path: `cosine_normalized` and
//! `normalize`.
//!
//! The central contract is that, once inputs are unit length,
//! `cosine_normalized` agrees with the general `Cosine` kernel within
//! floating-point tolerance — proven here both on hand-picked pairs and over
//! generated vectors. The error paths (empty, mismatched, zero-magnitude,
//! non-finite) are pinned alongside.

#![allow(clippy::unwrap_used)]

use iqdb_distance::{Cosine, Distance, cosine_normalized, normalize};
use iqdb_types::IqdbError;
use proptest::prelude::*;

const EPS: f32 = 1e-4;

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// --- cosine_normalized: hand-picked ----------------------------------------

#[test]
fn identical_unit_vectors_distance_is_zero() {
    let a = [1.0_f32, 0.0, 0.0];
    assert!(cosine_normalized(&a, &a).unwrap().abs() < 1e-6);
}

#[test]
fn perpendicular_unit_vectors_distance_is_one() {
    let a = [1.0_f32, 0.0, 0.0];
    let b = [0.0_f32, 1.0, 0.0];
    assert!((cosine_normalized(&a, &b).unwrap() - 1.0).abs() < 1e-6);
}

#[test]
fn opposite_unit_vectors_distance_is_two() {
    let a = [1.0_f32, 0.0];
    let b = [-1.0_f32, 0.0];
    assert!((cosine_normalized(&a, &b).unwrap() - 2.0).abs() < 1e-6);
}

#[test]
fn empty_inputs_return_invalid_vector() {
    let empty: [f32; 0] = [];
    assert_eq!(
        cosine_normalized(&empty, &[1.0]).unwrap_err(),
        IqdbError::InvalidVector,
    );
}

#[test]
fn mismatched_lengths_report_dimension_mismatch() {
    let err = cosine_normalized(&[1.0, 0.0, 0.0], &[1.0, 0.0]).unwrap_err();
    assert_eq!(
        err,
        IqdbError::DimensionMismatch {
            expected: 3,
            found: 2
        }
    );
}

// --- normalize: hand-picked ------------------------------------------------

#[test]
fn normalize_three_four_is_point_six_point_eight() {
    let unit = normalize(&[3.0_f32, 4.0]).unwrap();
    assert!((unit[0] - 0.6).abs() < 1e-6);
    assert!((unit[1] - 0.8).abs() < 1e-6);
    assert!((l2_norm(&unit) - 1.0).abs() < 1e-6);
}

#[test]
fn normalize_rejects_empty() {
    let empty: [f32; 0] = [];
    assert_eq!(normalize(&empty).unwrap_err(), IqdbError::InvalidVector);
}

#[test]
fn normalize_rejects_zero_vector() {
    assert_eq!(
        normalize(&[0.0_f32, 0.0, 0.0]).unwrap_err(),
        IqdbError::InvalidVector,
    );
}

#[test]
fn normalize_rejects_non_finite() {
    assert!(normalize(&[1.0_f32, f32::NAN]).is_err());
    assert!(normalize(&[f32::INFINITY, 1.0]).is_err());
    // A finite vector whose squared-norm sum overflows to +inf is also rejected.
    assert!(normalize(&[f32::MAX, f32::MAX]).is_err());
}

// --- equivalence: cosine_normalized(normalize(a), normalize(b)) == Cosine(a, b)

#[test]
fn equivalence_on_hand_picked_pairs() {
    let pairs: &[(&[f32], &[f32])] = &[
        (&[1.0, 2.0, 3.0], &[-2.0, 0.5, 4.0]),
        (&[0.1, 0.2, 0.3, 0.4], &[0.4, 0.3, 0.2, 0.1]),
        (&[5.0, -5.0], &[1.0, 1.0]),
    ];
    for (a, b) in pairs {
        let ua = normalize(a).unwrap();
        let ub = normalize(b).unwrap();
        let fast = cosine_normalized(&ua, &ub).unwrap();
        let full = Cosine::compute(&ua, &ub).unwrap();
        assert!((fast - full).abs() < EPS, "fast={fast} full={full}");
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// For any non-degenerate pair, normalizing then taking the fast cosine
    /// matches the general cosine kernel on the same normalized inputs.
    #[test]
    fn fast_matches_full_on_normalized(
        (a, b) in (1usize..=64).prop_flat_map(|len| {
            let comp = -1e3_f32..1e3_f32;
            (prop::collection::vec(comp.clone(), len), prop::collection::vec(comp, len))
        })
    ) {
        // Skip pairs that cannot be normalized (near-zero magnitude).
        if let (Ok(ua), Ok(ub)) = (normalize(&a), normalize(&b)) {
            let fast = cosine_normalized(&ua, &ub).unwrap();
            let full = Cosine::compute(&ua, &ub).unwrap();
            prop_assert!((fast - full).abs() < EPS, "fast={fast} full={full}");
            // And the fast result stays in the cosine range.
            prop_assert!((-EPS..=(2.0 + EPS)).contains(&fast));
        }
    }

    /// `normalize` always yields a unit-length vector.
    #[test]
    fn normalize_yields_unit_length(
        v in (1usize..=64).prop_flat_map(|len| prop::collection::vec(-1e3_f32..1e3_f32, len))
    ) {
        if let Ok(unit) = normalize(&v) {
            prop_assert!((l2_norm(&unit) - 1.0).abs() < EPS);
        }
    }
}
