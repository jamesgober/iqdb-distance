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
    <a href="https://github.com/rust-lang/rfcs/blob/master/text/2495-min-rust-version.md"><img alt="MSRV" src="https://img.shields.io/badge/MSRV-1.85%2B-blue"></a>
</div>

<br>

<div align="left">
    <p>
        <strong>iqdb-distance</strong> is the innermost loop of the database: every search computes thousands of distances, so this crate is optimized aggressively while keeping a readable scalar reference.
    </p>
    <p>
        It provides every metric vector search needs, with SIMD implementations that are property-tested to match the scalar ground truth within floating-point tolerance.
    </p>
    <br>
    <hr>
    <p>
        <strong>MSRV is 1.85+</strong> (Rust 2024 edition). SIMD-accelerated. Scalar fallback. Property-tested equivalence.
    </p>
    <blockquote>
        <strong>Status: pre-1.0, in active development.</strong> The public API is being designed across the 0.x series and frozen at <code>1.0.0</code>. See <a href="./CHANGELOG.md"><code>CHANGELOG.md</code></a>.
    </blockquote>
</div>

<hr>
<br>

<h2>What it does</h2>

- **Every metric** &mdash; cosine, dot product, Euclidean (L2), Manhattan (L1), Hamming
- **SIMD + scalar** &mdash; AVX2/AVX-512 on x86_64, NEON on aarch64, with a readable scalar fallback
- **Runtime dispatch** &mdash; detect CPU features once; force-scalar mode for testing
- **Batch ops** &mdash; compute a query against many candidates with cache-friendly batching
- **Standalone** &mdash; usable by anyone doing vector similarity in Rust, iQDB or not


<br>

## Installation

```toml
[dependencies]
iqdb-distance = "0.1"
```

<br>

## Status

This is the <code>v0.1.0</code> scaffold: structure, tooling, and quality gates are in place; the implementation lands across the 0.x series per the <a href="./dev/ROADMAP.md"><code>ROADMAP</code></a> and <a href="./docs/API.md"><code>docs/API.md</code></a>.

<hr>
<br>

## Where It Fits

`iqdb-distance` sits just above the types crate. It powers:

- `iqdb-types` &mdash; the `DistanceMetric` enum and vector types
- `iqdb-quantize` &mdash; quantized distance reuses this SIMD infrastructure
- `iqdb-flat` / `iqdb-hnsw` / `iqdb-ivf` &mdash; every index computes distances here

It has no first-party deps beyond `iqdb-types`, so it is unblocked today.

<br>

## Contributing

See <a href="./dev/DIRECTIVES.md"><code>dev/DIRECTIVES.md</code></a> for engineering standards and the definition of done. Before a PR: `cargo fmt --all`, `cargo clippy --all-targets --all-features -- -D warnings`, and `cargo test --all-features` must be clean.

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
