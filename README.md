<h1 align="center">
    <img width="99" alt="Rust logo" src="https://raw.githubusercontent.com/jamesgober/rust-collection/72baabd71f00e14aa9184efcb16fa3deddda3a0a/assets/rust-logo.svg">
    <br>
    <b>iqdb-distance</b>
    <br>
    <sub><sup>iQDB DISTANCE FUNCTIONS</sup></sub>
</h1>

<div align="center">
    <a href="https://crates.io/crates/iqdb-distance"><img alt="Crates.io" src="https://img.shields.io/crates/v/iqdb-distance"></a>
    <a href="https://crates.io/crates/iqdb-distance"><img alt="Downloads" src="https://img.shields.io/crates/d/iqdb-distance?color=%230099ff"></a>
    <a href="https://docs.rs/iqdb-distance"><img alt="docs.rs" src="https://img.shields.io/docsrs/iqdb-distance"></a>
    <a href="https://github.com/jamesgober/iqdb-distance/actions"><img alt="CI" src="https://github.com/jamesgober/iqdb-distance/actions/workflows/ci.yml/badge.svg"></a>
    <a href="https://github.com/rust-lang/rfcs/blob/master/text/2495-min-rust-version.md"><img alt="MSRV" src="https://img.shields.io/badge/MSRV-1.87%2B-blue"></a>
</div>

<br>

<div align="left">
    <p>
        <strong>iqdb-distance</strong> is the innermost loop of the database: every search computes thousands of distances, so this crate is optimized aggressively while keeping a readable scalar reference.
    </p>
    <p>
        It provides every metric vector search needs &mdash; cosine, dot product, Euclidean, Manhattan, and Hamming &mdash; with SIMD implementations that are property-tested to match the scalar ground truth within floating-point tolerance.
    </p>
    <br>
    <hr>
    <p>
        <strong>MSRV is 1.87+</strong> (Rust 2024 edition). SIMD-accelerated. Scalar fallback. Property-tested equivalence.
    </p>
    <blockquote>
        <strong>Status: pre-1.0, in active development.</strong> The public API is being designed across the 0.x series and frozen at <code>1.0.0</code>. See <a href="./CHANGELOG.md"><code>CHANGELOG.md</code></a>.
    </blockquote>
</div>

<hr>
<br>

<h2>What it does</h2>

- **Every metric** &mdash; cosine, dot product, Euclidean (L2), Manhattan (L1), Hamming
- **SIMD + scalar** &mdash; AVX2 on x86_64, NEON on aarch64, with a readable scalar fallback
- **Runtime dispatch** &mdash; detect CPU features once; route each call to the fastest available kernel
- **Two entry points** &mdash; a type-level `Distance` trait when the metric is known at compile time, and `compute`/`compute_batch` over `DistanceMetric` when it is chosen at runtime
- **Normalized fast path** &mdash; `cosine_normalized` (and a `normalize` helper) skip the norm work for pre-normalized embeddings
- **Allocation-free** &mdash; every distance call borrows; batch evaluation writes into a caller-owned buffer
- **Never panics on bad input** &mdash; empty, mismatched, and non-finite inputs return a typed `IqdbError`
- **Standalone** &mdash; usable by anyone doing vector similarity in Rust, iQDB or not

<br>

## Installation

```toml
[dependencies]
iqdb-distance = "0.4"
```

<br>

## Quick start

Pick the metric at compile time through the `Distance` trait:

```rust
use iqdb_distance::{Cosine, Distance, Euclidean};

let a = [1.0_f32, 0.0, 0.0];
let b = [0.0_f32, 1.0, 0.0];

// Cosine distance of perpendicular unit vectors is 1.0.
let cos = Cosine::compute(&a, &b).expect("non-empty, same length");
assert!((cos - 1.0).abs() < 1e-6);

// Euclidean (L2): a 3-4-5 right triangle.
let l2 = Euclidean::compute(&[0.0, 0.0, 0.0], &[3.0, 4.0, 0.0]).expect("valid pair");
assert!((l2 - 5.0).abs() < 1e-6);
```

Or pick it at runtime with the `DistanceMetric` tag from `iqdb-types`:

```rust
use iqdb_distance::compute;
use iqdb_types::DistanceMetric;

let a = [1.0_f32, 2.0, 3.0];
let b = [4.0_f32, -5.0, 6.0];

// Dot-product similarity: 1*4 + 2*(-5) + 3*6 = 12.
let s = compute(DistanceMetric::DotProduct, &a, &b).expect("valid pair");
assert!((s - 12.0).abs() < 1e-6);
```

Score one query against many candidates into a caller-owned buffer (no allocation):

```rust
use iqdb_distance::compute_batch;
use iqdb_types::DistanceMetric;

let query = [0.0_f32, 0.0];
let candidates: [&[f32]; 3] = [&[1.0, 0.0], &[0.0, 2.0], &[3.0, 4.0]];
let mut out = [0.0_f32; 3];

compute_batch(DistanceMetric::Euclidean, &query, &candidates, &mut out)
    .expect("output length matches candidate count");
assert_eq!(out, [1.0, 2.0, 5.0]);
```

When embeddings are already unit length, take the fast cosine path — it skips the norm, square root, and division:

```rust
use iqdb_distance::{cosine_normalized, normalize};

// Normalize once at ingest, then compare with the fast path.
let a = normalize(&[1.0_f32, 2.0, 3.0]).expect("non-zero magnitude");
let b = normalize(&[-2.0_f32, 0.5, 4.0]).expect("non-zero magnitude");

let d = cosine_normalized(&a, &b).expect("valid pair"); // 1 - dot(a, b)
assert!((0.0..=2.0).contains(&d));
```

Inspect the kernel the host will use:

```rust
let features = iqdb_distance::detect_features();
// On an AVX2 x86_64 host `features.avx2` is true; on aarch64 `features.neon` is.
let _ = (features.avx2, features.neon);
```

<br>

## Errors

Every fallible call returns `iqdb_types::Result`. Empty inputs surface as
`IqdbError::InvalidVector`; length mismatches as
`IqdbError::DimensionMismatch { expected, found }`; a batch whose output buffer
does not match the candidate count returns `IqdbError::InvalidConfig`. The
library never panics on hostile input.

<br>

## Status

This is the <code>v0.4.0</code> feature-freeze release: all five metrics ship with a scalar reference path plus runtime-dispatched AVX2 and NEON kernels (each property-tested and differentially verified against the scalar twin), the `cosine_normalized` fast path for pre-normalized embeddings, and allocation-free batch evaluation. **The public surface is now frozen for 1.x** &mdash; additive changes only until 2.0 (the frozen surface is recorded in the <a href="./dev/ROADMAP.md"><code>ROADMAP</code></a>). The remaining 0.x work &mdash; equivalence fuzzing and the RC soak against real index consumers &mdash; lands per the ROADMAP, with the full surface documented in <a href="./docs/API.md"><code>docs/API.md</code></a>.

<hr>
<br>

## Where It Fits

`iqdb-distance` sits just above the types crate. It powers:

- `iqdb-types` &mdash; the `DistanceMetric` enum and vector types it operates on
- `iqdb-quantize` &mdash; quantized distance reuses this SIMD infrastructure
- `iqdb-flat` / `iqdb-hnsw` / `iqdb-ivf` &mdash; every index computes distances here

Its only first-party dependency is `iqdb-types`, so it is unblocked today.

<br>

## Standards

Built to the iQDB Rust standard. See <a href="./REPS.md"><code>REPS.md</code></a> (Rust Efficiency &amp; Performance Standards) and <a href="./dev/DIRECTIVES.md"><code>dev/DIRECTIVES.md</code></a> for the engineering law and the definition of done. Before a PR: `cargo fmt --all`, `cargo clippy --all-targets --all-features -- -D warnings`, and `cargo test --all-features` must be clean.

<br>

<div id="license">
    <h2>License</h2>
    <p>Licensed under either of</p>
    <ul>
        <li><b>Apache License, Version 2.0</b> &mdash; <a href="./LICENSE-APACHE">LICENSE-APACHE</a></li>
        <li><b>MIT License</b> &mdash; <a href="./LICENSE-MIT">LICENSE-MIT</a></li>
    </ul>
    <p>at your option.</p>
</div>

<div align="center">
  <h2></h2>
  <sup>COPYRIGHT <small>&copy;</small> 2026 <strong>JAMES GOBER.</strong></sup>
</div>
