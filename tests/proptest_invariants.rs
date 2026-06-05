//! Property tests for the math invariants of each metric.
//!
//! Inputs are bounded f32 vectors (no NaN/Inf, magnitudes <= 1e3) so the
//! arithmetic stays within representable range and the comparisons remain
//! meaningful. NaN/Inf paths are covered separately in `edge_cases.rs`.

#![allow(clippy::unwrap_used)]

use iqdb_distance::{Cosine, Distance, DotProduct, Euclidean, Hamming, Manhattan};
use proptest::prelude::*;

const EPS_ABS: f32 = 1e-3;
const EPS_REL: f32 = 1e-4;

fn close(x: f32, y: f32) -> bool {
    let diff = (x - y).abs();
    diff <= EPS_ABS || diff <= EPS_REL * x.abs().max(y.abs())
}

fn bounded_vec(len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-1e3_f32..1e3_f32, len)
}

fn paired_vecs() -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
    (1usize..=128).prop_flat_map(|len| (bounded_vec(len), bounded_vec(len)))
}

fn one_vec() -> impl Strategy<Value = Vec<f32>> {
    (1usize..=128).prop_flat_map(bounded_vec)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn euclidean_is_symmetric((a, b) in paired_vecs()) {
        let ab = Euclidean::compute(&a, &b).unwrap();
        let ba = Euclidean::compute(&b, &a).unwrap();
        prop_assert!(close(ab, ba));
    }

    #[test]
    fn euclidean_non_negative((a, b) in paired_vecs()) {
        let d = Euclidean::compute(&a, &b).unwrap();
        prop_assert!(d >= 0.0);
    }

    #[test]
    fn euclidean_identity_is_zero(a in one_vec()) {
        let d = Euclidean::compute(&a, &a).unwrap();
        prop_assert!(d.abs() <= EPS_ABS);
    }

    #[test]
    fn manhattan_is_symmetric((a, b) in paired_vecs()) {
        let ab = Manhattan::compute(&a, &b).unwrap();
        let ba = Manhattan::compute(&b, &a).unwrap();
        prop_assert!(close(ab, ba));
    }

    #[test]
    fn manhattan_non_negative((a, b) in paired_vecs()) {
        let d = Manhattan::compute(&a, &b).unwrap();
        prop_assert!(d >= 0.0);
    }

    #[test]
    fn manhattan_identity_is_zero(a in one_vec()) {
        let d = Manhattan::compute(&a, &a).unwrap();
        prop_assert!(d.abs() <= EPS_ABS);
    }

    #[test]
    fn dot_product_is_symmetric((a, b) in paired_vecs()) {
        let ab = DotProduct::compute(&a, &b).unwrap();
        let ba = DotProduct::compute(&b, &a).unwrap();
        prop_assert!(close(ab, ba));
    }

    #[test]
    fn dot_product_self_equals_squared_norm(a in one_vec()) {
        let dot = DotProduct::compute(&a, &a).unwrap();
        let norm_sq: f32 = a.iter().map(|x| x * x).sum();
        prop_assert!(close(dot, norm_sq));
    }

    #[test]
    fn cosine_is_symmetric((a, b) in paired_vecs()) {
        let ab = Cosine::compute(&a, &b).unwrap();
        let ba = Cosine::compute(&b, &a).unwrap();
        prop_assert!(close(ab, ba));
    }

    #[test]
    fn cosine_in_zero_two_range((a, b) in paired_vecs()) {
        let d = Cosine::compute(&a, &b).unwrap();
        prop_assert!((-EPS_ABS..=(2.0 + EPS_ABS)).contains(&d));
    }

    #[test]
    fn hamming_is_symmetric((a, b) in paired_vecs()) {
        let ab = Hamming::compute(&a, &b).unwrap();
        let ba = Hamming::compute(&b, &a).unwrap();
        prop_assert!((ab - ba).abs() < EPS_ABS);
    }

    #[test]
    fn hamming_bounded_by_length((a, b) in paired_vecs()) {
        let d = Hamming::compute(&a, &b).unwrap();
        prop_assert!(d >= 0.0);
        prop_assert!(d <= a.len() as f32);
    }

    #[test]
    fn hamming_identity_is_zero(a in one_vec()) {
        let d = Hamming::compute(&a, &a).unwrap();
        prop_assert!(d.abs() <= EPS_ABS);
    }
}
