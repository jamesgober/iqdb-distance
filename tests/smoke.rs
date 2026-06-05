//! End-to-end smoke coverage for the `iqdb-distance` public surface.
//!
//! Exercises every metric through both the [`Distance`] trait and the
//! runtime [`compute`]/[`compute_batch`] entry points, plus the
//! feature-detection seam.

#![allow(clippy::unwrap_used)]

use iqdb_distance::{
    Cosine, CpuFeatures, Distance, DotProduct, Euclidean, Hamming, Manhattan, VERSION, compute,
    compute_batch, detect_features, forced_scalar,
};
use iqdb_types::DistanceMetric;

#[test]
fn version_is_semver_triplet() {
    assert_eq!(VERSION.split('.').count(), 3);
    assert!(VERSION.split('.').all(|part| !part.is_empty()));
}

#[test]
fn cosine_perpendicular_unit_vectors_is_one() {
    let a = [1.0_f32, 0.0, 0.0];
    let b = [0.0_f32, 1.0, 0.0];

    let d = Cosine::compute(&a, &b).unwrap();
    assert!((d - 1.0).abs() < 1e-6);
}

#[test]
fn cosine_identical_unit_vectors_is_zero() {
    let a = [1.0_f32, 0.0, 0.0];
    let d = Cosine::compute(&a, &a).unwrap();
    assert!(d.abs() < 1e-6);
}

#[test]
fn dot_product_matches_hand_calculation() {
    let a = [1.0_f32, 2.0, 3.0];
    let b = [4.0_f32, -5.0, 6.0];

    let s = DotProduct::compute(&a, &b).unwrap();
    assert!((s - 12.0).abs() < 1e-6);
}

#[test]
fn euclidean_three_four_five() {
    let a = [0.0_f32, 0.0, 0.0];
    let b = [3.0_f32, 4.0, 0.0];

    let d = Euclidean::compute(&a, &b).unwrap();
    assert!((d - 5.0).abs() < 1e-6);
}

#[test]
fn manhattan_three_four_seven() {
    let a = [0.0_f32, 0.0];
    let b = [3.0_f32, 4.0];

    let d = Manhattan::compute(&a, &b).unwrap();
    assert!((d - 7.0).abs() < 1e-6);
}

#[test]
fn hamming_counts_bit_distinct_positions() {
    let a = [0.0_f32, 1.0, 0.0, 1.0, 1.0];
    let b = [0.0_f32, 0.0, 0.0, 1.0, 0.0];

    let d = Hamming::compute(&a, &b).unwrap();
    assert!((d - 2.0).abs() < 1e-6);
}

#[test]
fn runtime_dispatch_matches_trait_dispatch() {
    let a = [0.5_f32, 0.5, 0.5, 0.5];
    let b = [0.1_f32, 0.2, 0.3, 0.4];

    let metrics = [
        DistanceMetric::Cosine,
        DistanceMetric::DotProduct,
        DistanceMetric::Euclidean,
        DistanceMetric::Manhattan,
        DistanceMetric::Hamming,
    ];

    for metric in metrics {
        let via_enum = compute(metric, &a, &b).unwrap();
        let via_trait = match metric {
            DistanceMetric::Cosine => Cosine::compute(&a, &b).unwrap(),
            DistanceMetric::DotProduct => DotProduct::compute(&a, &b).unwrap(),
            DistanceMetric::Euclidean => Euclidean::compute(&a, &b).unwrap(),
            DistanceMetric::Manhattan => Manhattan::compute(&a, &b).unwrap(),
            DistanceMetric::Hamming => Hamming::compute(&a, &b).unwrap(),
            // `DistanceMetric` is `#[non_exhaustive]`; this local array only
            // holds the five implemented variants.
            other => panic!("unexpected metric {other:?}"),
        };
        assert_eq!(via_enum.to_bits(), via_trait.to_bits(), "metric {metric:?}");
    }
}

#[test]
fn batch_writes_one_distance_per_candidate() {
    let q = [0.0_f32, 0.0];
    let cs: [&[f32]; 3] = [&[1.0, 0.0], &[0.0, 2.0], &[3.0, 4.0]];
    let mut out = [0.0_f32; 3];

    compute_batch(DistanceMetric::Euclidean, &q, &cs, &mut out).unwrap();
    assert!((out[0] - 1.0).abs() < 1e-6);
    assert!((out[1] - 2.0).abs() < 1e-6);
    assert!((out[2] - 5.0).abs() < 1e-6);
}

#[test]
fn batch_with_zero_candidates_is_noop() {
    let q = [0.0_f32, 0.0];
    let cs: [&[f32]; 0] = [];
    let mut out: [f32; 0] = [];

    compute_batch(DistanceMetric::Cosine, &q, &cs, &mut out).unwrap();
}

#[test]
fn detect_features_is_stable() {
    let first = detect_features();
    let second = detect_features();
    assert_eq!(first, second);
}

#[test]
fn cpu_features_default_layout_is_observable() {
    let f: CpuFeatures = detect_features();
    let _both = (f.avx2, f.neon, f.forced_scalar);
    // `forced_scalar` follows the global flag rather than the cached probe.
    assert_eq!(f.forced_scalar, forced_scalar());
}
