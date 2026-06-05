//! No-panic robustness fuzz target for [`iqdb_distance::compute`].
//!
//! Drives the dispatched distance function with arbitrary
//! `(metric, a, perturb)` inputs. Contract: the call MUST return a typed
//! `Result<f32, IqdbError>` and MUST NOT panic, regardless of length or
//! `NaN`/`Inf` content. Unlike the `equivalence` target, inputs here are
//! **not** bounded — feeding hostile values (non-finite, extreme magnitude) is
//! exactly the point.
//!
//! ## Length-match strategy
//!
//! `b` is derived from `a` by bit-perturbing each component, so most calls have
//! matching lengths and exercise the kernel paths rather than landing
//! overwhelmingly on the `DimensionMismatch` validation path. A truncated `b`
//! probes that validation path once per iteration so it stays covered.

#![no_main]

use libfuzzer_sys::fuzz_target;

use iqdb_types::DistanceMetric;

const METRICS: [DistanceMetric; 5] = [
    DistanceMetric::Cosine,
    DistanceMetric::DotProduct,
    DistanceMetric::Euclidean,
    DistanceMetric::Manhattan,
    DistanceMetric::Hamming,
];

fuzz_target!(|input: (u8, Vec<f32>, u32)| {
    let (metric_idx, a, perturb_bits) = input;
    let metric = METRICS[(metric_idx as usize) % METRICS.len()];

    // Same length as `a`; NaN/Inf shapes stay reachable.
    let b: Vec<f32> = a
        .iter()
        .map(|x| f32::from_bits(x.to_bits() ^ perturb_bits))
        .collect();
    let _ = std::hint::black_box(iqdb_distance::compute(metric, &a, &b));

    // Keep an explicit `DimensionMismatch` probe so the validation path is not
    // orphaned from the corpus.
    if a.len() >= 2 {
        let short = &b[..a.len() / 2];
        let _ = std::hint::black_box(iqdb_distance::compute(metric, &a, short));
    }
});
