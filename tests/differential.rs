//! Differential tests: every SIMD path must agree with its scalar twin
//! across a finite corpus AND an adversarial non-finite corpus. This is
//! the SIMD correctness contract.
//!
//! Two correctness gates the older design lacked:
//!
//! 1. **"SIMD actually ran" assertion.** Without this, a runtime detection
//!    regression that silently routed to scalar would let the differential
//!    pass vacuously (scalar-vs-scalar). The gate reads
//!    [`iqdb_distance::which_kernel`] — which calls the same internal
//!    `select_kernel` the real per-metric dispatch uses, so the assertion
//!    and the dispatch path cannot drift.
//!
//! 2. **Race-free SIMD-vs-scalar collection.** `force_scalar()` is sticky
//!    for the lifetime of the process. The older test used a `Once` and
//!    a two-phase per-test gather, which is racy when Cargo runs tests in
//!    parallel inside one binary: one test's `force_scalar()` can land
//!    between another test's phase-1 collection and its dispatch, turning
//!    that test's "SIMD" samples into scalar samples. The new design
//!    collects ALL SIMD samples for ALL metrics inside a single
//!    `OnceLock::get_or_init` *before* any test flips the flag, then every
//!    `diff_*` test reads the cached pre-flip samples and compares against
//!    post-flip scalar.
//!
//! ## Non-finite contract
//!
//! Finite inputs: SIMD must agree with scalar within a tight relative+
//! absolute epsilon. Non-finite inputs: both sides must be NaN (payload
//! may differ), or both Inf with the same sign, or both finite-and-close.
//! This is the explicit contract enforced by [`close_or_same_nonfinite`].
//!
//! Compiled only under the `testing` feature, which is what gates
//! `force_scalar` on the crate. CI runs this via `cargo test --all-features`.

#![cfg(feature = "testing")]
#![allow(clippy::unwrap_used)]

use std::collections::HashMap;
use std::sync::{Once, OnceLock};

use iqdb_distance::{
    Cosine, Distance, DotProduct, Euclidean, Hamming, Manhattan, force_scalar, which_kernel,
};
use iqdb_types::DistanceMetric;

const FINITE_DIMS: &[usize] = &[1, 3, 7, 8, 15, 16, 31, 32, 63, 64, 255, 256, 4096];
const ADVERSARIAL_DIMS: &[usize] = &[1, 4, 7, 8, 15, 16, 32];

const EPS_ABS: f32 = 1e-3;
const EPS_REL: f32 = 1e-4;

const ALL_METRICS: [DistanceMetric; 5] = [
    DistanceMetric::Cosine,
    DistanceMetric::DotProduct,
    DistanceMetric::Euclidean,
    DistanceMetric::Manhattan,
    DistanceMetric::Hamming,
];

// --- Snapshot infrastructure -------------------------------------------------

struct SimdSnapshot {
    /// Kernel observed before any `force_scalar()` was called. On a
    /// SIMD-capable host this must be `"avx2"` or `"neon"`.
    kernel: &'static str,
    /// Pre-flip SIMD samples, keyed by metric. Each `Vec<f32>` is the
    /// per-pair result vector in the corpus's iteration order.
    samples: HashMap<DistanceMetric, Vec<f32>>,
}

static SIMD_SNAPSHOT: OnceLock<SimdSnapshot> = OnceLock::new();
static SCALAR_LOCKED: Once = Once::new();

/// Return the (cached) pre-flip SIMD snapshot, building it on first call.
///
/// `OnceLock::get_or_init` is internally synchronized — concurrent test
/// threads block until init finishes, then receive the same cached
/// snapshot. This guarantees ALL samples were collected before any test
/// flipped the `force_scalar` flag.
fn simd_snapshot() -> &'static SimdSnapshot {
    SIMD_SNAPSHOT.get_or_init(|| {
        let kernel = which_kernel();
        let pairs = full_corpus();
        let mut samples: HashMap<DistanceMetric, Vec<f32>> = HashMap::new();
        for metric in ALL_METRICS {
            let row: Vec<f32> = pairs
                .iter()
                .map(|(a, b)| compute_for(metric, a, b))
                .collect();
            let _ = samples.insert(metric, row);
        }
        SimdSnapshot { kernel, samples }
    })
}

fn lock_scalar() {
    SCALAR_LOCKED.call_once(force_scalar);
}

// --- Corpus -----------------------------------------------------------------

/// Combine the finite waveform corpus with the adversarial special-value
/// corpus. Order is stable so cached SIMD samples align with the on-the-
/// fly scalar samples.
fn full_corpus() -> Vec<(Vec<f32>, Vec<f32>)> {
    let mut pairs = finite_corpus();
    pairs.extend(adversarial_corpus());
    pairs
}

fn finite_corpus() -> Vec<(Vec<f32>, Vec<f32>)> {
    FINITE_DIMS.iter().copied().map(finite_vectors).collect()
}

fn finite_vectors(dim: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
    let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1 + 0.7).cos()).collect();
    (a, b)
}

/// Adversarial non-finite / extreme-magnitude corpus.
///
/// Builds three shapes per (dim, special) combination:
/// 1. `[special; dim]` vs `[1.0; dim]`
/// 2. `[1.0; dim]` vs `[special; dim]`
/// 3. `[special; dim]` vs `[special; dim]`
///
/// Plus a "mixed" case per (dim, special): a 1.0 background with the
/// special value at index 0 of `a` and index `dim-1` of `b`. This
/// exercises the SIMD tail path against the main-loop path on the same
/// input.
///
/// Plus a small **low-magnitude** finite corpus targeting audit M11:
/// a colinear pair at `~1e-15` (where per-norm `na`/`nb` are well above
/// `f32::MIN_POSITIVE` but `na * nb` underflows the product) and a
/// colinear pair at `~1e-22` (where `na`/`nb` themselves underflow to
/// subnormal and the cosine guard correctly returns 1.0). Both scalar
/// and SIMD kernels MUST agree on each pair.
fn adversarial_corpus() -> Vec<(Vec<f32>, Vec<f32>)> {
    const SPECIALS: &[f32] = &[
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0_f32,
        -0.0_f32,
        f32::MIN_POSITIVE / 2.0, // subnormal
        1.0e30_f32,
        -1.0e30_f32,
        1.0e-30_f32,
        -1.0e-30_f32,
        1.0_f32,
        -1.0_f32,
    ];

    let mut pairs = Vec::new();
    for &dim in ADVERSARIAL_DIMS {
        for &s in SPECIALS {
            pairs.push((vec![s; dim], vec![1.0_f32; dim]));
            pairs.push((vec![1.0_f32; dim], vec![s; dim]));
            pairs.push((vec![s; dim], vec![s; dim]));

            // Mixed: special at endpoints on a 1.0 background.
            let mut a = vec![1.0_f32; dim];
            let mut b = vec![1.0_f32; dim];
            a[0] = s;
            b[dim - 1] = s;
            pairs.push((a, b));
        }
        // Zero vectors of this dim — cosine's denom-zero path, others
        // give exact zeros.
        pairs.push((vec![0.0_f32; dim], vec![0.0_f32; dim]));

        // M11 low-magnitude pairs. Built from a varying-slope shape so
        // every kernel exercises real multiply-add work, not a constant
        // fold. `1e-15` exposes the product-underflow path the cosine
        // fix routes around; `1e-22` pins the documented zero-magnitude
        // floor. Two shapes each: parallel (a == b) and orthogonal
        // (disjoint support across the two halves).
        let shape: Vec<f32> = (0..dim).map(|i| (i as f32) + 1.0).collect();
        for &scale in &[1.0e-15_f32, 1.0e-22_f32] {
            let scaled: Vec<f32> = shape.iter().map(|x| x * scale).collect();
            pairs.push((scaled.clone(), scaled.clone()));

            let half = dim / 2;
            if half > 0 {
                let mut left = vec![0.0_f32; dim];
                let mut right = vec![0.0_f32; dim];
                for i in 0..half {
                    left[i] = ((i as f32) + 1.0) * scale;
                    right[half + i] = ((i as f32) + 1.0) * scale;
                }
                pairs.push((left, right));
            }
        }
    }
    pairs
}

// --- Compute + compare ------------------------------------------------------

fn compute_for(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        DistanceMetric::Cosine => Cosine::compute(a, b).unwrap(),
        DistanceMetric::DotProduct => DotProduct::compute(a, b).unwrap(),
        DistanceMetric::Euclidean => Euclidean::compute(a, b).unwrap(),
        DistanceMetric::Manhattan => Manhattan::compute(a, b).unwrap(),
        DistanceMetric::Hamming => Hamming::compute(a, b).unwrap(),
        // `DistanceMetric` is `#[non_exhaustive]`; `ALL_METRICS` only holds the
        // five implemented variants.
        other => panic!("unexpected metric {other:?}"),
    }
}

/// Compare two distance results under the documented non-finite contract.
///
/// - Both NaN → ok (payload may differ).
/// - Exactly one NaN → divergence.
/// - Inf on either side → both must be Inf with matching sign.
/// - Both finite → within `EPS_ABS` or within `EPS_REL` relative.
fn close_or_same_nonfinite(x: f32, y: f32) -> bool {
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

fn simd_vs_scalar(metric: DistanceMetric) {
    // Populate the SIMD snapshot before flipping. OnceLock guarantees
    // every parallel test thread sees the same pre-flip cache.
    let snap = simd_snapshot();
    let simd = snap.samples.get(&metric).unwrap();

    // Flip the global override and gather scalar samples on-the-fly.
    lock_scalar();
    let pairs = full_corpus();
    let scalar: Vec<f32> = pairs
        .iter()
        .map(|(a, b)| compute_for(metric, a, b))
        .collect();

    assert_eq!(
        simd.len(),
        scalar.len(),
        "metric {metric:?}: corpus length drift simd={} scalar={}",
        simd.len(),
        scalar.len(),
    );

    for (i, (&s, &sc)) in simd.iter().zip(scalar.iter()).enumerate() {
        let (a, b) = &pairs[i];
        assert!(
            close_or_same_nonfinite(s, sc),
            "metric {metric:?} pair[{i}] dim={} divergence: simd={s} scalar={sc}\n  a={a:?}\n  b={b:?}",
            a.len(),
        );
    }
}

// --- Tests ------------------------------------------------------------------

/// The dispatched kernel on this host MUST be the SIMD one the audit
/// expects. Hard-fails (rather than silently passing) if runtime
/// detection regressed and the differential is about to compare
/// scalar-vs-scalar.
#[test]
fn simd_kernel_dispatched_on_simd_host() {
    // Take the kernel from the SAME pre-flip snapshot the SIMD samples
    // were collected under. After this test runs, other tests may flip
    // the override, but the snapshot captured the pre-flip value.
    let snap = simd_snapshot();
    let observed = snap.kernel;

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            assert_eq!(
                observed, "avx2",
                "AVX2 host: expected dispatcher to route to AVX2, got {observed:?}",
            );
            return;
        }
        assert_eq!(
            observed, "scalar",
            "non-AVX2 x86_64 host: expected scalar fallback, got {observed:?}",
        );
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            assert_eq!(
                observed, "neon",
                "aarch64 host: expected dispatcher to route to NEON, got {observed:?}",
            );
            return;
        }
        assert_eq!(
            observed, "scalar",
            "non-NEON aarch64 host: expected scalar fallback, got {observed:?}",
        );
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        assert_eq!(
            observed, "scalar",
            "unsupported-arch host: expected scalar fallback, got {observed:?}",
        );
    }
}

/// `force_scalar()` MUST flip dispatch to the scalar kernel. Without
/// this, the rest of the differential's phase-2 (scalar) gather has no
/// proof it is actually exercising the scalar path.
#[test]
fn force_scalar_changes_dispatch_to_scalar() {
    // Ensure the pre-flip snapshot is captured before this test flips.
    let _ = simd_snapshot();
    lock_scalar();
    assert_eq!(
        which_kernel(),
        "scalar",
        "force_scalar() did not route dispatch to scalar",
    );
}

#[test]
fn diff_cosine() {
    simd_vs_scalar(DistanceMetric::Cosine);
}

#[test]
fn diff_dot_product() {
    simd_vs_scalar(DistanceMetric::DotProduct);
}

#[test]
fn diff_euclidean() {
    simd_vs_scalar(DistanceMetric::Euclidean);
}

#[test]
fn diff_manhattan() {
    simd_vs_scalar(DistanceMetric::Manhattan);
}

#[test]
fn diff_hamming() {
    simd_vs_scalar(DistanceMetric::Hamming);
}
