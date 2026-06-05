//! SIMD-vs-scalar equivalence fuzz target.
//!
//! For arbitrary `(metric, components, perturb)` inputs this asserts the
//! dispatched kernel — `compute`, which routes to AVX2 / NEON on a SIMD host —
//! agrees with the scalar reference, `compute_scalar`. This is the fuzzed
//! counterpart to the fixed-corpus differential test (`tests/differential.rs`):
//! the fixed corpus pins known-hard cases, the fuzzer explores the structural
//! space (dimensions, SIMD-tail boundaries, sign and magnitude patterns) the
//! corpus cannot enumerate.
//!
//! ## Bounded inputs (why, and what it does not cover)
//!
//! Components are bounded to `[-1e3, 1e3]` and non-finite values are mapped to
//! `0.0`. This is deliberate: SIMD and scalar accumulate in different orders,
//! so for unconstrained `f32` they legitimately differ — one path can overflow
//! to `inf` where the other stays finite purely from summation order, which is
//! a floating-point fact, not a kernel bug. Bounding keeps every comparison in
//! the regime where the two paths *must* agree, so a divergence here is a real
//! defect. Non-finite **equivalence** (both `NaN`, signed `inf` agreement) is
//! covered by the differential test's adversarial corpus; non-finite **no-panic**
//! robustness is covered by the `robustness` target.
//!
//! ## Length-match strategy
//!
//! `b` is derived from `a` by bit-perturbing each component (then bounding), so
//! every call has matching lengths and the corpus density lands on the kernel
//! paths rather than the `DimensionMismatch` validation path.

#![no_main]

use libfuzzer_sys::fuzz_target;

use iqdb_distance::{compute, compute_scalar};
use iqdb_types::DistanceMetric;

const METRICS: [DistanceMetric; 5] = [
    DistanceMetric::Cosine,
    DistanceMetric::DotProduct,
    DistanceMetric::Euclidean,
    DistanceMetric::Manhattan,
    DistanceMetric::Hamming,
];

const EPS_ABS: f32 = 1e-3;
const EPS_REL: f32 = 1e-4;
const BOUND: f32 = 1.0e3;

/// Map an arbitrary component into the bounded, finite comparison regime.
fn bound(x: f32) -> f32 {
    if x.is_finite() { x.clamp(-BOUND, BOUND) } else { 0.0 }
}

/// The same finite-tolerance / signed-non-finite contract the differential
/// test uses. With bounded inputs the non-finite arms should never fire, but
/// they keep the comparison honest if a kernel ever produces one.
fn close(x: f32, y: f32) -> bool {
    if x.is_nan() && y.is_nan() {
        return true;
    }
    if x.is_nan() != y.is_nan() {
        return false;
    }
    if x.is_infinite() || y.is_infinite() {
        return x.is_infinite() && y.is_infinite() && x.is_sign_positive() == y.is_sign_positive();
    }
    let diff = (x - y).abs();
    diff <= EPS_ABS || diff <= EPS_REL * x.abs().max(y.abs())
}

fuzz_target!(|input: (u8, Vec<f32>, u32)| {
    let (metric_idx, raw, perturb_bits) = input;
    let metric = METRICS[(metric_idx as usize) % METRICS.len()];

    let a: Vec<f32> = raw.iter().map(|x| bound(*x)).collect();
    // Derive `b` from `a`'s bit patterns, then bound back into range.
    let b: Vec<f32> = a
        .iter()
        .map(|x| bound(f32::from_bits(x.to_bits() ^ perturb_bits)))
        .collect();

    if let (Ok(simd), Ok(scalar)) = (compute(metric, &a, &b), compute_scalar(metric, &a, &b)) {
        assert!(
            close(simd, scalar),
            "metric {metric:?} divergence: simd={simd} scalar={scalar}\n  a={a:?}\n  b={b:?}",
        );
    }
});
