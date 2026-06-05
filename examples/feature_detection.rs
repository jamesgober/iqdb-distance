//! Inspecting the SIMD kernel the host will use, via [`detect_features`].
//!
//! Dispatch is automatic — you never have to call this to get the fast path —
//! but the snapshot is useful in diagnostics and version-skew checks.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example feature_detection
//! ```

use iqdb_distance::detect_features;

fn main() {
    let features = detect_features();

    println!("iqdb-distance {}", iqdb_distance::VERSION);
    println!("  AVX2 (x86_64): {}", features.avx2);
    println!("  NEON (aarch64): {}", features.neon);
    println!("  forced scalar: {}", features.forced_scalar);

    let kernel = if features.forced_scalar {
        "scalar (forced)"
    } else if features.avx2 {
        "AVX2"
    } else if features.neon {
        "NEON"
    } else {
        "scalar"
    };
    println!("=> distance calls run on the {kernel} kernel");

    // The snapshot is stable across calls (the probe runs once per process).
    assert_eq!(features, detect_features());
}
