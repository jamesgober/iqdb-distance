//! Runtime CPU-feature detection and the test-only scalar override.
//!
//! [`detect_features`] probes the host once via `std::sync::OnceLock` and
//! returns a [`CpuFeatures`] snapshot. The per-metric dispatch in
//! [`crate::metrics`] consults it to pick AVX2 (x86_64), NEON (aarch64), or
//! the scalar reference. `force_scalar` is a sticky global override used
//! by tests to exercise the scalar path on a host that would otherwise pick
//! a SIMD kernel.

use core::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

/// Snapshot of the host CPU features [`detect_features`] cares about.
///
/// The struct is intentionally small and `Copy`: it is read on the hot
/// path of every distance call. New fields will be added in additive
/// releases — match on it exhaustively at your own risk.
///
/// The `forced_scalar` field reflects the value of the override at the
/// moment [`detect_features`] returned. Do not cache a [`CpuFeatures`]
/// across a `force_scalar` call: call [`detect_features`] each time
/// you need a fresh view.
///
/// # Examples
///
/// ```
/// let features = iqdb_distance::detect_features();
/// // Repeated calls return the same value (snapshot is cached).
/// assert_eq!(features, iqdb_distance::detect_features());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuFeatures {
    /// True if the host advertises AVX2 (x86_64 only). Always `false` on
    /// other architectures.
    pub avx2: bool,
    /// True if the host advertises NEON (aarch64 only). Always `false` on
    /// other architectures.
    pub neon: bool,
    /// True if `force_scalar` has been called in this process. Once set,
    /// the flag is read on every dispatch — see the `force_scalar` docs.
    pub forced_scalar: bool,
}

static CPU_FEATURES: OnceLock<CpuFeatures> = OnceLock::new();
static FORCED_SCALAR: AtomicBool = AtomicBool::new(false);

/// Return the host CPU-feature snapshot, computing it on first call.
///
/// The probe runs at most once per process; subsequent calls return the
/// cached value. The `forced_scalar` field reflects the *current* state of
/// the `force_scalar` override, so the snapshot remains accurate even if
/// the override is set after the probe ran.
///
/// # Examples
///
/// ```
/// let features = iqdb_distance::detect_features();
/// // On a host without AVX2 the flag is false; on a host without NEON
/// // the flag is false. Both fields are always observable.
/// let _ = (features.avx2, features.neon, features.forced_scalar);
/// ```
#[must_use]
pub fn detect_features() -> CpuFeatures {
    let probed = *CPU_FEATURES.get_or_init(probe);
    CpuFeatures {
        forced_scalar: forced_scalar(),
        ..probed
    }
}

/// Return `true` if `force_scalar` has been called in this process.
///
/// Reads an atomic flag — cheap, allocation-free, monotonic once set.
/// `Relaxed` is sufficient: the flag is set-once `false → true` and the
/// test harness coordinates the set/observe boundary through
/// `std::sync::Once`, whose `call_once` provides happens-before for
/// observers.
///
/// # Examples
///
/// ```
/// // This crate never calls `force_scalar` itself, so the flag is normally
/// // false unless a test has set it.
/// let _ = iqdb_distance::forced_scalar();
/// ```
#[must_use]
pub fn forced_scalar() -> bool {
    FORCED_SCALAR.load(Ordering::Relaxed)
}

/// Force every dispatched distance call in this process onto the scalar
/// reference path.
///
/// The flag is **sticky**: once set, it remains set for the lifetime of
/// the process. There is intentionally no `unforce_scalar` — the override
/// exists so test suites can exercise the scalar path on hardware that
/// would otherwise pick a SIMD kernel, and a sticky flag keeps the test
/// state visible.
///
/// Available only when the crate is built with the `testing` feature. A
/// production build cannot reach the override, so SIMD cannot be disabled
/// at runtime by accident.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "testing")]
/// # {
/// use iqdb_distance::{Cosine, Distance};
///
/// let a = [1.0_f32, 0.0];
/// let b = [0.0_f32, 1.0];
/// let before = Cosine::compute(&a, &b).expect("valid pair");
///
/// // Calling `force_scalar` makes every subsequent call go scalar.
/// // iqdb_distance::force_scalar();
///
/// let after = Cosine::compute(&a, &b).expect("valid pair");
/// assert!((before - after).abs() < 1e-6);
/// # }
/// ```
#[cfg(any(test, feature = "testing"))]
pub fn force_scalar() {
    FORCED_SCALAR.store(true, Ordering::Relaxed);
}

/// The kernel a distance call would dispatch to right now.
///
/// Held internally and consumed by the per-metric dispatch in
/// [`crate::metrics`] via [`select_kernel`]. Variants exist only on
/// architectures where they are reachable, so `match` arms in each
/// metric's `dispatch` stay exhaustive without an `_` fallback that
/// could mask a routing mistake.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kernel {
    /// Scalar reference path. Always available.
    Scalar,
    /// AVX2 kernel; only constructible on x86_64.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// NEON kernel; only constructible on aarch64.
    #[cfg(target_arch = "aarch64")]
    Neon,
}

/// Decide which kernel a distance call should route to, given a snapshot
/// of the host CPU features.
///
/// This is the **single source of truth** for the dispatch decision. The
/// per-metric `dispatch` fns in [`crate::metrics`] and the testing-only
/// [`which_kernel`] accessor both call this function — they cannot drift,
/// so the differential test's "SIMD actually ran" assertion is asserting
/// the real path, not a copy of it.
pub(crate) fn select_kernel(features: CpuFeatures) -> Kernel {
    if features.forced_scalar {
        return Kernel::Scalar;
    }
    #[cfg(target_arch = "x86_64")]
    if features.avx2 {
        return Kernel::Avx2;
    }
    #[cfg(target_arch = "aarch64")]
    if features.neon {
        return Kernel::Neon;
    }
    Kernel::Scalar
}

/// Return the kernel a distance call would dispatch to right now, as a
/// short identifier: `"scalar"`, `"avx2"`, or `"neon"`.
///
/// This accessor exists so the differential SIMD-vs-scalar test can prove
/// the dispatcher actually routed to the host's SIMD kernel before
/// gathering "SIMD" samples — without this, a runtime detection
/// regression that silently fell back to scalar would let the test pass
/// vacuously (scalar-vs-scalar comparison).
///
/// Built only under `cfg(any(test, feature = "testing"))`. **Not part of
/// the stable public surface** — the return type and strings are
/// testing-internals and may change.
///
/// Internally delegates to the crate-private `select_kernel`, the same
/// function the real dispatch path uses, so the test cannot disagree
/// with reality.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "testing")]
/// # {
/// // On any host the accessor returns one of "scalar", "avx2", "neon".
/// let kernel = iqdb_distance::which_kernel();
/// assert!(matches!(kernel, "scalar" | "avx2" | "neon"));
/// # }
/// ```
#[cfg(any(test, feature = "testing"))]
#[must_use]
pub fn which_kernel() -> &'static str {
    match select_kernel(detect_features()) {
        Kernel::Scalar => "scalar",
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx2 => "avx2",
        #[cfg(target_arch = "aarch64")]
        Kernel::Neon => "neon",
    }
}

fn probe() -> CpuFeatures {
    CpuFeatures {
        avx2: probe_avx2(),
        neon: probe_neon(),
        forced_scalar: false,
    }
}

#[cfg(target_arch = "x86_64")]
fn probe_avx2() -> bool {
    std::is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
fn probe_avx2() -> bool {
    false
}

#[cfg(target_arch = "aarch64")]
fn probe_neon() -> bool {
    std::arch::is_aarch64_feature_detected!("neon")
}

#[cfg(not(target_arch = "aarch64"))]
fn probe_neon() -> bool {
    false
}
