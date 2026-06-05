//! Edge-case coverage for the public surface.
//!
//! These are the boundary inputs every metric MUST tolerate without
//! panicking: empty slices, length-1 vectors, mismatched lengths,
//! NaN/inf inputs, and very large dims.

#![allow(clippy::unwrap_used)]

use iqdb_distance::{Cosine, Distance, DotProduct, Euclidean, Hamming, Manhattan, compute};
use iqdb_types::{DistanceMetric, IqdbError};

const ALL_METRICS: [DistanceMetric; 5] = [
    DistanceMetric::Cosine,
    DistanceMetric::DotProduct,
    DistanceMetric::Euclidean,
    DistanceMetric::Manhattan,
    DistanceMetric::Hamming,
];

#[test]
fn empty_inputs_return_invalid_vector() {
    let empty: [f32; 0] = [];
    let one = [1.0_f32];
    for metric in ALL_METRICS {
        let err_a = compute(metric, &empty, &one).unwrap_err();
        let err_b = compute(metric, &one, &empty).unwrap_err();
        assert_eq!(err_a, IqdbError::InvalidVector, "metric {metric:?} a-empty");
        assert_eq!(err_b, IqdbError::InvalidVector, "metric {metric:?} b-empty");
    }
}

#[test]
fn mismatched_lengths_report_expected_and_found() {
    let a = [1.0_f32, 2.0, 3.0];
    let b = [1.0_f32, 2.0];
    for metric in ALL_METRICS {
        let err = compute(metric, &a, &b).unwrap_err();
        assert_eq!(
            err,
            IqdbError::DimensionMismatch {
                expected: 3,
                found: 2,
            },
            "metric {metric:?}",
        );
    }
}

#[test]
fn length_one_vectors_produce_finite_output() {
    let a = [3.0_f32];
    let b = [4.0_f32];
    let expected: [(DistanceMetric, f32); 5] = [
        // 1 - (3*4)/(3*4) = 0
        (DistanceMetric::Cosine, 0.0),
        // 3 * 4 = 12
        (DistanceMetric::DotProduct, 12.0),
        // |3 - 4| = 1
        (DistanceMetric::Euclidean, 1.0),
        // |3 - 4| = 1
        (DistanceMetric::Manhattan, 1.0),
        // bit-different
        (DistanceMetric::Hamming, 1.0),
    ];
    for (metric, want) in expected {
        let got = compute(metric, &a, &b).unwrap();
        assert!(
            (got - want).abs() < 1e-6,
            "metric {metric:?}: {got} vs {want}"
        );
    }
}

#[test]
fn nan_inputs_do_not_panic() {
    let a = [f32::NAN, 1.0, 2.0];
    let b = [0.0_f32, f32::INFINITY, 2.0];
    for metric in ALL_METRICS {
        // Result may be NaN/Inf for some metrics, but the call must
        // succeed without panicking.
        let _ = compute(metric, &a, &b).unwrap();
    }
}

#[test]
fn large_dim_does_not_panic_or_misreport() {
    const DIM: usize = 4096;
    let a: Vec<f32> = (0..DIM).map(|i| (i as f32).sin()).collect();
    let b: Vec<f32> = (0..DIM).map(|i| (i as f32).cos()).collect();

    let d = Euclidean::compute(&a, &b).unwrap();
    assert!(d.is_finite());
    assert!(d >= 0.0);

    let m = Manhattan::compute(&a, &b).unwrap();
    assert!(m.is_finite());
    assert!(m >= 0.0);

    let c = Cosine::compute(&a, &b).unwrap();
    assert!(c.is_finite());

    let dot = DotProduct::compute(&a, &b).unwrap();
    assert!(dot.is_finite());

    let h = Hamming::compute(&a, &b).unwrap();
    assert!(h >= 0.0);
    assert!(h <= DIM as f32);
}

#[test]
fn cosine_zero_vector_distance_is_one() {
    let zeros = [0.0_f32; 4];
    let nonzero = [1.0_f32, 2.0, 3.0, 4.0];

    let d = Cosine::compute(&zeros, &nonzero).unwrap();
    assert!((d - 1.0).abs() < 1e-6);

    let d2 = Cosine::compute(&zeros, &zeros).unwrap();
    assert!((d2 - 1.0).abs() < 1e-6);
}

#[test]
fn batch_rejects_wrong_output_length() {
    let q = [0.0_f32, 0.0];
    let cs: [&[f32]; 2] = [&[1.0, 0.0], &[0.0, 2.0]];
    let mut out = [0.0_f32; 1];

    let err = Manhattan::compute_batch(&q, &cs, &mut out).unwrap_err();
    assert!(
        matches!(err, IqdbError::InvalidConfig { .. }),
        "expected InvalidConfig, got {err:?}",
    );
}
