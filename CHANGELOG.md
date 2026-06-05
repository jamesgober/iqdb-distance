<h1 align="center">
    <img width="90px" height="auto" src="https://raw.githubusercontent.com/jamesgober/jamesgober/main/media/icons/hexagon-3.svg" alt="Triple Hexagon">
    <br><b>CHANGELOG</b>
</h1>
<p>
  All notable changes to <code>iqdb-distance</code> will be documented in this file. The format is based on <a href="https://keepachangelog.com/en/1.1.0/">Keep a Changelog</a>,
  and this project adheres to <a href="https://semver.org/spec/v2.0.0.html/">Semantic Versioning</a>.
</p>

---

## [Unreleased]

### Added

### Changed

### Fixed

### Security

---

## [0.5.0] - 2026-06-05

Equivalence fuzzing and **API freeze**. The SIMD-vs-scalar contract is now
fuzzed, not just covered by a fixed corpus, and the public surface is locked for
the 1.x series.

### Added

- `fuzz/` cargo-fuzz crate (nightly) with two targets: `equivalence`, which
  asserts the dispatched SIMD kernel agrees with the scalar reference over
  bounded finite inputs across all five metrics (the fuzzed counterpart to
  `tests/differential.rs`), and `robustness`, which asserts `compute` never
  panics on arbitrary, including non-finite and length-mismatched, input. Local
  soak: 13.6M+ / 10.3M+ executions with zero findings.
- `compute_scalar(metric, a, b)`: a testing-only accessor (gated on
  `feature = "testing"`) that computes a metric on the scalar reference path for
  a single call, bypassing the process-sticky `force_scalar`. It gives the
  equivalence fuzzer a per-input scalar oracle. **Not** part of the stable
  surface — same status as `force_scalar` / `which_kernel`.
- CI `fuzz` job: builds both targets on nightly and smoke-runs each.

### Changed

- **Public API frozen for 1.x.** The stable surface recorded under the 0.4.0
  feature freeze is now locked; only additive, non-breaking changes land before
  2.0. `cargo audit` and `cargo deny check` are clean.

---

## [0.4.0] - 2026-06-05

Normalized fast path and **feature freeze**. The public surface is now complete
and declared frozen — no new public items before 1.0. This release adds the
pre-normalized cosine path for embeddings that are already unit length.

### Added

- `cosine_normalized(a, b) -> Result<f32>`: cosine distance for already
  unit-length vectors, computed as `1 - (a · b)` through the same
  runtime-dispatched SIMD dot kernel. It skips the per-call norm, square root,
  and division of the general `Cosine` kernel; for unit inputs the result
  matches `Cosine::compute` within tolerance and lies in `[0, 2]`. The
  equivalence is property-tested against the full cosine kernel.
- `normalize(v) -> Result<Vec<f32>>`: the L2-normalized (`v / ‖v‖`) copy of a
  vector, for producing the unit inputs `cosine_normalized` expects. Rejects
  empty, zero-magnitude, subnormal-magnitude, and non-finite-norm vectors with
  `IqdbError::InvalidVector`. This is the crate's only allocating call, by
  necessity (it returns a new vector).

### Changed

- **Public API declared frozen for 1.x.** The frozen surface is recorded in
  `dev/ROADMAP.md`; additive, non-breaking changes remain allowed, anything else
  waits for 2.0. No `todo!`/`unimplemented!` anywhere in shipping code.

---

## [0.3.0] - 2026-06-05

SIMD release. The scalar reference is joined by runtime-dispatched AVX2 (x86_64) and NEON (aarch64) kernels for all five metrics, with every SIMD path property-and-differentially tested to agree with its scalar twin within floating-point tolerance.

### Added

- AVX2 kernels (`src/simd/avx2.rs`) for `dot`, `cosine`, `euclidean`,
  `manhattan`, and `hamming`, each `#[target_feature(enable = "avx2")]` and
  reached only behind the runtime guard. Eight-lane main loop with a scalar
  tail; `// SAFETY:` notes tie every intrinsic to the `chunks_exact(8)` source.
- NEON kernels (`src/simd/neon.rs`) with the same metric set, four-lane main
  loop plus scalar tail, gated identically on `aarch64` feature detection.
- Runtime feature detection: `detect_features() -> CpuFeatures` snapshots the
  host once through `OnceLock`, and a single crate-private
  `select_kernel(CpuFeatures) -> Kernel` is the one source of truth the
  per-metric dispatch and the test-only accessor both consult, so they cannot
  drift.
- `testing` Cargo feature exposing `force_scalar()` (a sticky, process-wide
  override that pins dispatch onto the scalar reference) and
  `which_kernel() -> &'static str` (the kernel dispatch would route to right
  now). A production build cannot reach either — SIMD cannot be disabled at
  runtime by accident, and the testing accessors are not part of the stable
  surface.
- Differential SIMD-vs-scalar suite (`tests/differential.rs`, `--features
  testing`): asserts the host actually dispatched to its SIMD kernel before
  gathering "SIMD" samples (no vacuous scalar-vs-scalar pass), caches the
  pre-flip SIMD snapshot in a `OnceLock` so Cargo's parallel test execution
  cannot interleave a `force_scalar()` flip with another test's gather, and
  compares finite **and** adversarial (NaN, ±∞, ±0.0, subnormal, ±1e30,
  ±1e-30, zero-vector) corpora under an explicit non-finite contract.
- Criterion benches (`benches/distance.rs`, `--features testing`) reporting both
  the SIMD and the forced-scalar path per metric at dim 768.

### Changed

- AVX2 and NEON Manhattan switched to the canonical lane-wise sign-bit-clear
  form (`_mm256_andnot_ps` / `vabsq_f32`) instead of `max(d, -d)`: unambiguously
  NaN-correct without relying on MAXPS/FMAX tie-break semantics, and NEON saves
  one instruction per chunk.
- `forced_scalar()` / `force_scalar()` atomic ordering tightened from `SeqCst`
  to `Relaxed`; the flag is set-once monotonic and the test harness coordinates
  the set/observe boundary through `std::sync::Once`.
- The runtime dispatchers (`compute`, `compute_batch`) now carry a wildcard arm
  returning `IqdbError::InvalidMetric` for any metric outside the implemented
  set — `iqdb-types` 1.0 made `DistanceMetric` `#[non_exhaustive]`, and this is
  the documented forward-compatible integration step.

---

## [0.2.0] - 2026-06-05

First implementation release. The five distance metrics land behind the
`Distance` trait, with always-compiled scalar reference paths, runtime
dispatch over `iqdb_types::DistanceMetric`, allocation-free batch evaluation,
and typed-error input validation.

### Added

- `Distance` trait with associated `compute(a, b) -> Result<f32>` and
  `compute_batch(query, candidates, out) -> Result<()>`. No receiver, no `dyn`,
  no allocation — every metric is a zero-sized tag dispatched at the type level.
- Per-metric zero-sized types: `Cosine`, `DotProduct`, `Euclidean`,
  `Manhattan`, `Hamming`.
- Scalar reference implementations (`src/scalar/`) for all five metrics, always
  compiled and used as the correctness contract the SIMD kernels are tested
  against.
- Runtime dispatch over `iqdb_types::DistanceMetric` via top-level
  `compute(metric, a, b)` and `compute_batch(metric, query, candidates, out)` —
  the entry points for consumers that pick the metric at runtime.
- Input validation surfacing typed `iqdb_types::IqdbError`: empty inputs as
  `InvalidVector`, length mismatches as `DimensionMismatch { expected, found }`,
  and a mis-sized batch output buffer as `InvalidConfig`. The library never
  panics on bad input.
- `VERSION` constant exposing the crate's compile-time `CARGO_PKG_VERSION`.
- Property tests (`proptest`) for the math invariants per metric: symmetry,
  non-negativity, identity-is-zero, cosine range, Hamming bounds, and
  `dot(a, a) == ‖a‖²`.
- Edge-case and smoke suites: empty/length-1/mismatched inputs, NaN/∞
  non-panic, large dims, zero-vector cosine, and trait-vs-runtime-dispatch
  agreement.
- Cosine low-magnitude robustness (`tests/cosine_low_magnitude.rs`): the
  denominator is computed as `‖a‖·‖b‖` via independent square roots
  (`na.sqrt() * nb.sqrt()`) so the squared-norm product cannot underflow for
  small-magnitude inputs, with the zero-magnitude floor pinned and documented.

### Changed

- Now depends on `iqdb-types` 1.0 for the shared `DistanceMetric`, `IqdbError`,
  and `Result` vocabulary.

---

## [0.1.0] - 2026-05-30

Initial scaffold and repository bootstrap. No domain logic yet &mdash; this release establishes the structure, tooling, and quality gates the implementation is built on.

### Added

- `Cargo.toml` with crate metadata, Rust 2024 edition, MSRV 1.87.
- Dual `Apache-2.0 OR MIT` license files.
- `README.md`, `CHANGELOG.md`, and a documentation skeleton.
- `REPS.md` compliance baseline.
- `.github/workflows/ci.yml` CI matrix; `deny.toml`, `clippy.toml`, `rustfmt.toml`.
- `dev/DIRECTIVES.md` and `dev/ROADMAP.md` (committed engineering standards + plan).

[Unreleased]: https://github.com/jamesgober/iqdb-distance/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/jamesgober/iqdb-distance/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/jamesgober/iqdb-distance/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/jamesgober/iqdb-distance/releases/tag/v0.3.0
