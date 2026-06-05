# iqdb-distance &mdash; API Reference

> Complete reference for **every** public item in `iqdb-distance` as of
> **v1.0.0**: what it is, its parameters and return shape, the traits it
> implements, and worked examples for each use case.
>
> **Status: stable (1.0).** The public API is committed under SemVer for the 1.x
> series — no breaking changes until 2.0 (the frozen surface is recorded in
> `dev/ROADMAP.md`). Only additive, non-breaking changes are made within 1.x.
> The testing-only accessors (`force_scalar`, `which_kernel`, `compute_scalar`)
> are explicitly *not* part of the stable API.

## Table of Contents

- [Overview](#overview)
- [Crate constants](#crate-constants)
  - [`VERSION`](#version)
- [Computing a distance](#computing-a-distance)
  - [`Distance` (trait)](#distance-trait)
  - [The metric types](#the-metric-types)
  - [`compute` (runtime dispatch)](#compute-runtime-dispatch)
  - [`compute_batch` (runtime dispatch)](#compute_batch-runtime-dispatch)
- [The metrics](#the-metrics)
- [Normalized fast path](#normalized-fast-path)
  - [`cosine_normalized`](#cosine_normalized)
  - [`normalize`](#normalize)
- [CPU features &amp; dispatch](#cpu-features--dispatch)
  - [`CpuFeatures`](#cpufeatures)
  - [`detect_features`](#detect_features)
  - [`forced_scalar`](#forced_scalar)
- [Testing surface (feature `testing`)](#testing-surface-feature-testing)
  - [`force_scalar`](#force_scalar)
  - [`which_kernel`](#which_kernel)
  - [`compute_scalar`](#compute_scalar)
- [Errors](#errors)
- [Feature flags](#feature-flags)
- [Trait implementation matrix](#trait-implementation-matrix)

---

## Overview

`iqdb-distance` is the innermost loop of the iQDB vector database: every search computes thousands of distances, so the crate is optimized aggressively while keeping a readable scalar reference. Given two `&[f32]` slices and a metric, it returns the distance (or similarity) as an `f32`.

There are two ways in, for the two ways a caller knows the metric:

```rust
use iqdb_distance::{Cosine, Distance, compute};
use iqdb_types::DistanceMetric;

let a = [1.0_f32, 0.0, 0.0];
let b = [0.0_f32, 1.0, 0.0];

// 1. Metric known at compile time — the type-level `Distance` trait.
let d1 = Cosine::compute(&a, &b).expect("valid pair");

// 2. Metric chosen at runtime — the `compute` dispatcher over the enum tag.
let d2 = compute(DistanceMetric::Cosine, &a, &b).expect("valid pair");

assert_eq!(d1.to_bits(), d2.to_bits()); // identical result, same kernel
```

**Performance.** Every public path is allocation-free. SIMD kernels (AVX2 on x86_64, NEON on aarch64) are selected at runtime from [`detect_features`](#detect_features) and short-circuit to the scalar reference when the host lacks the feature. The scalar path is always compiled and serves as the auditable definition of each metric and the ground truth the SIMD kernels are differentially tested against.

**No panics.** Empty, length-mismatched, and (for the dispatchers) unknown-metric inputs return a typed [`IqdbError`](#errors). Non-finite components (`NaN`, `±∞`) do not panic; they propagate through the arithmetic per IEEE-754.

---

## Crate constants

### `VERSION`

```rust
pub const VERSION: &str;
```

The crate's compile-time version (`CARGO_PKG_VERSION`), a `major.minor.patch` SemVer core. Use it to report the exact `iqdb-distance` build a binary links against — useful in diagnostics and version-skew checks across the iQDB crate family.

```rust
let v = iqdb_distance::VERSION;
assert_eq!(v.split('.').count(), 3);
assert!(v.split('.').all(|part| !part.is_empty()));
```

---

## Computing a distance

### `Distance` (trait)

```rust
pub trait Distance {
    fn compute(a: &[f32], b: &[f32]) -> Result<f32>;
    fn compute_batch(query: &[f32], candidates: &[&[f32]], out: &mut [f32]) -> Result<()>;
}
```

The single entry point per metric. Each metric type in the crate ([`Cosine`](#the-metric-types), [`DotProduct`](#the-metric-types), [`Euclidean`](#the-metric-types), [`Manhattan`](#the-metric-types), [`Hamming`](#the-metric-types)) is a zero-sized type that implements `Distance`. The associated functions take **no receiver** — the type is used as a tag, not a value — so dispatch is resolved at compile time with no `dyn` indirection and no allocation on the hot path.

#### `Distance::compute`

```rust
fn compute(a: &[f32], b: &[f32]) -> Result<f32>;
```

Compute the distance between `a` and `b`.

- **`a`, `b`** — the two vectors, as `f32` slices. Must be non-empty and equal length.
- **Returns** `Ok(f32)`, or:
  - [`Err(IqdbError::InvalidVector)`](#errors) if either slice is empty.
  - [`Err(IqdbError::DimensionMismatch { expected, found })`](#errors) if the lengths differ (`expected = a.len()`, `found = b.len()`).

```rust
use iqdb_distance::{Distance, Euclidean};

let a = [0.0_f32, 0.0, 0.0];
let b = [3.0_f32, 4.0, 0.0];
let d = Euclidean::compute(&a, &b).expect("non-empty, same length");
assert!((d - 5.0).abs() < 1e-6);

// Length mismatch is a typed error, not a panic.
use iqdb_types::IqdbError;
let err = Euclidean::compute(&[1.0, 2.0, 3.0], &[1.0, 2.0]).unwrap_err();
assert_eq!(err, IqdbError::DimensionMismatch { expected: 3, found: 2 });
```

#### `Distance::compute_batch`

```rust
fn compute_batch(query: &[f32], candidates: &[&[f32]], out: &mut [f32]) -> Result<()>;
```

Compute the distance from `query` to each entry in `candidates`, writing the results in order into `out`. Allocation-free: the output buffer is caller-supplied.

- **`query`** — the query vector.
- **`candidates`** — the vectors to score against `query`.
- **`out`** — the output buffer; **`out.len()` must equal `candidates.len()`**.
- **Returns** `Ok(())`, or:
  - [`Err(IqdbError::InvalidConfig { reason })`](#errors) if `out.len() != candidates.len()`.
  - any per-pair error from [`compute`](#distancecompute) (e.g. a candidate of the wrong length).

```rust
use iqdb_distance::{Distance, Manhattan};

let q = [0.0_f32, 0.0];
let cs: [&[f32]; 2] = [&[1.0, 0.0], &[0.0, 2.0]];
let mut out = [0.0_f32; 2];

Manhattan::compute_batch(&q, &cs, &mut out).expect("matching lengths");
assert_eq!(out, [1.0, 2.0]);
```

### The metric types

```rust
pub struct Cosine;
pub struct DotProduct;
pub struct Euclidean;
pub struct Manhattan;
pub struct Hamming;
```

Five zero-sized tag types, one per metric, each implementing [`Distance`](#distance-trait). They carry no data — construct one only if you want a value to pass around; the trait functions never need an instance.

**Derives / traits:** `Debug`, `Clone`, `Copy`, `Default`, `PartialEq`, `Eq`, `Hash` on every metric type, plus `Distance`.

See [The metrics](#the-metrics) for the formula and semantics of each.

```rust
use iqdb_distance::{Cosine, DotProduct, Distance};

// Used purely as a tag — no instance required.
let d = Cosine::compute(&[1.0, 0.0], &[1.0, 0.0]).expect("valid pair");
assert!(d.abs() < 1e-6); // identical direction → distance 0

let s = DotProduct::compute(&[1.0, 2.0, 3.0], &[4.0, -5.0, 6.0]).expect("valid pair");
assert!((s - 12.0).abs() < 1e-6);
```

### `compute` (runtime dispatch)

```rust
pub fn compute(metric: DistanceMetric, a: &[f32], b: &[f32]) -> Result<f32>;
```

The runtime equivalent of the [`Distance`](#distance-trait) trait: the caller picks the metric through the [`DistanceMetric`](#errors) tag and the dispatcher routes to the matching kernel. This is the entry point for the index and query crates, which do not know the metric at compile time.

- **`metric`** — which metric to compute ([`iqdb_types::DistanceMetric`]).
- **`a`, `b`** — the two vectors (same contract as [`Distance::compute`](#distancecompute)).
- **Returns** `Ok(f32)`, the same typed errors as [`Distance::compute`](#distancecompute), or [`Err(IqdbError::InvalidMetric)`](#errors) for a metric this crate does not implement.

> **`DistanceMetric` is `#[non_exhaustive]`.** A future `iqdb-types` may add a metric this crate has not yet implemented; rather than panic, `compute` returns `IqdbError::InvalidMetric`. Existing callers keep compiling and fail gracefully at runtime.

```rust
use iqdb_distance::compute;
use iqdb_types::DistanceMetric;

let a = [0.0_f32, 0.0, 0.0];
let b = [3.0_f32, 4.0, 0.0];
let d = compute(DistanceMetric::Euclidean, &a, &b).expect("valid pair");
assert!((d - 5.0).abs() < 1e-6);
```

### `compute_batch` (runtime dispatch)

```rust
pub fn compute_batch(
    metric: DistanceMetric,
    query: &[f32],
    candidates: &[&[f32]],
    out: &mut [f32],
) -> Result<()>;
```

The runtime counterpart of [`Distance::compute_batch`](#distancecompute_batch): score `query` against every candidate under `metric`, writing into the caller-owned `out`. Same length contract (`out.len() == candidates.len()`) and the same `InvalidMetric` behaviour as [`compute`](#compute-runtime-dispatch).

```rust
use iqdb_distance::compute_batch;
use iqdb_types::DistanceMetric;

let q = [0.0_f32, 0.0];
let cs: [&[f32]; 3] = [&[1.0, 0.0], &[0.0, 2.0], &[3.0, 4.0]];
let mut out = [0.0_f32; 3];

compute_batch(DistanceMetric::Euclidean, &q, &cs, &mut out).expect("matching lengths");
assert_eq!(out, [1.0, 2.0, 5.0]);
```

---

## The metrics

All five take two equal-length `f32` slices. Smaller is nearer for the true distances; `DotProduct` is a similarity (larger is more similar) and is the one signed result.

| Metric | Formula | Range | Notes |
|---|---|---|---|
| `Cosine` | `1 − (a·b) / (‖a‖·‖b‖)` | `[0, 2]` | Magnitude-independent; suits normalized embeddings. A zero-magnitude vector is defined to have distance `1.0` (no similarity). |
| `DotProduct` | `a·b` | unbounded, signed | A **similarity**, not a distance — carries the sign of the inner product. Use when magnitude carries signal. |
| `Euclidean` | `√(Σ (aᵢ − bᵢ)²)` | `[0, ∞)` | Straight-line (L2) distance. Exactly `0` when `a == b` componentwise. |
| `Manhattan` | `Σ │aᵢ − bᵢ│` | `[0, ∞)` | Sum of absolute differences (L1). Exactly `0` when `a == b`. |
| `Hamming` | `#{ i : aᵢ.to_bits() ≠ bᵢ.to_bits() }` | `[0, len]` | Count of bit-distinct positions, returned as `f32`. Suits binary codes encoded as `0.0`/`1.0`. `−0.0` and `+0.0` differ. |

**Cosine numerical note.** The denominator is computed as `‖a‖·‖b‖` via two independent square roots (`na.sqrt() * nb.sqrt()`), not `(na·nb).sqrt()`. The former is numerically robust: the squared-norm product `na·nb` can underflow to `0` for low-magnitude inputs even when each norm is individually finite. The guard `denom <= f32::MIN_POSITIVE` catches both genuine zero-norm vectors and residual subnormal underflow, returning `1.0`. Inputs whose per-component squared norm itself underflows below `f32::MIN_POSITIVE` (magnitudes around `1e-22` at small dims) hit a documented floor and also report `1.0`. The scalar, AVX2, and NEON kernels use the identical strategy, so they agree.

```rust
use iqdb_distance::{Cosine, DotProduct, Euclidean, Hamming, Manhattan, Distance};

// Cosine: perpendicular unit vectors → 1.0.
assert!((Cosine::compute(&[1.0, 0.0], &[0.0, 1.0]).unwrap() - 1.0).abs() < 1e-6);

// DotProduct: signed similarity.
assert!((DotProduct::compute(&[1.0, 2.0], &[3.0, 4.0]).unwrap() - 11.0).abs() < 1e-6);

// Euclidean and Manhattan on the same pair.
assert!((Euclidean::compute(&[0.0, 0.0], &[3.0, 4.0]).unwrap() - 5.0).abs() < 1e-6);
assert!((Manhattan::compute(&[0.0, 0.0], &[3.0, 4.0]).unwrap() - 7.0).abs() < 1e-6);

// Hamming: one bit-distinct position.
assert!((Hamming::compute(&[0.0, 1.0, 0.0], &[0.0, 0.0, 0.0]).unwrap() - 1.0).abs() < 1e-6);
```

---

## Normalized fast path

When embeddings are L2-normalized at ingest — a common preprocessing step — cosine distance collapses to `1 − (a · b)`, so the per-call norm, square root, and division the general [`Cosine`](#the-metrics) kernel performs are unnecessary. These two functions take that path; both reuse the same runtime-dispatched SIMD dot kernel as the rest of the crate.

### `cosine_normalized`

```rust
pub fn cosine_normalized(a: &[f32], b: &[f32]) -> Result<f32>;
```

Cosine distance for two **already unit-length** vectors: `1 − (a · b)`.

- **`a`, `b`** — the two vectors, **assumed unit length**. Same length contract as [`Distance::compute`](#distancecompute).
- **Returns** `Ok(f32)`. For genuinely unit-length inputs the result equals [`Cosine::compute`](#the-metrics) within floating-point tolerance and lies in `[0, 2]`.
- **Errors** — [`InvalidVector`](#errors) for an empty slice, [`DimensionMismatch`](#errors) for unequal lengths.

> **Contract.** The caller guarantees `a` and `b` are unit length (use [`normalize`](#normalize)). If they are not, the return value is still `1 − (a · b)` but is no longer a cosine distance and may fall outside `[0, 2]` — there is no internal normalization to rescue it. When magnitudes are unknown, use [`Cosine`](#the-metrics), which normalizes internally.

```rust
use iqdb_distance::cosine_normalized;

// Identical unit vectors → 0; perpendicular → 1.
let a = [1.0_f32, 0.0, 0.0];
assert!(cosine_normalized(&a, &a).expect("valid pair").abs() < 1e-6);
assert!((cosine_normalized(&a, &[0.0, 1.0, 0.0]).expect("valid pair") - 1.0).abs() < 1e-6);
```

Once inputs are normalized, it matches the general kernel:

```rust
use iqdb_distance::{Cosine, Distance, cosine_normalized, normalize};

let a = normalize(&[1.0_f32, 2.0, 3.0]).expect("non-zero");
let b = normalize(&[-2.0_f32, 0.5, 4.0]).expect("non-zero");
let fast = cosine_normalized(&a, &b).expect("valid pair");
let full = Cosine::compute(&a, &b).expect("valid pair");
assert!((fast - full).abs() < 1e-6);
```

### `normalize`

```rust
pub fn normalize(v: &[f32]) -> Result<Vec<f32>>;
```

Return the L2-normalized (unit-length) copy of `v`: `v / ‖v‖`. Use it once at ingest to produce the unit vectors [`cosine_normalized`](#cosine_normalized) expects, then store the result. The squared norm is computed through the SIMD dot kernel (`‖v‖² = v · v`).

- **`v`** — the vector to normalize.
- **Returns** `Ok(Vec<f32>)` — a **new** vector (this is the crate's one allocating call, by necessity).
- **Errors** — [`InvalidVector`](#errors) if `v` is empty, or if its magnitude is not a usable positive, finite value: a zero vector, a subnormal-magnitude vector, or one whose norm is non-finite (a `NaN`/`∞` component, or an overflowing sum of squares). A vector you cannot normalize is rejected rather than returned as `NaN`s.

```rust
use iqdb_distance::normalize;

// 3-4-5 triangle → unit vector [0.6, 0.8].
let unit = normalize(&[3.0_f32, 4.0]).expect("non-zero magnitude");
assert!((unit[0] - 0.6).abs() < 1e-6 && (unit[1] - 0.8).abs() < 1e-6);

// A zero-magnitude vector cannot be normalized.
assert!(normalize(&[0.0_f32, 0.0, 0.0]).is_err());
```

---

## CPU features &amp; dispatch

### `CpuFeatures`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuFeatures {
    pub avx2: bool,
    pub neon: bool,
    pub forced_scalar: bool,
}
```

A small, `Copy` snapshot of the host features the dispatcher cares about, read on the hot path of every distance call.

- **`avx2`** — host advertises AVX2 (x86_64 only; always `false` elsewhere).
- **`neon`** — host advertises NEON (aarch64 only; always `false` elsewhere).
- **`forced_scalar`** — `true` if [`force_scalar`](#force_scalar) has been called in this process.

Do not cache a `CpuFeatures` across a `force_scalar` call — call [`detect_features`](#detect_features) again for a fresh view.

### `detect_features`

```rust
pub fn detect_features() -> CpuFeatures;
```

Return the host feature snapshot, probing the CPU once per process (cached in a `OnceLock`) and reading the current `force_scalar` override each call. The `avx2`/`neon` fields are stable for the process; `forced_scalar` reflects the live override.

```rust
let f = iqdb_distance::detect_features();
// Repeated calls return the same probe result.
assert_eq!(f.avx2, iqdb_distance::detect_features().avx2);
// Every field is always observable.
let _ = (f.avx2, f.neon, f.forced_scalar);
```

### `forced_scalar`

```rust
pub fn forced_scalar() -> bool;
```

Return `true` if [`force_scalar`](#force_scalar) has been called in this process. Reads a single relaxed atomic — cheap and allocation-free. This crate never calls `force_scalar` itself, so it is normally `false`.

```rust
// Normally false in a production build.
let _ = iqdb_distance::forced_scalar();
```

---

## Testing surface (feature `testing`)

These items exist only when the crate is built with the `testing` feature (or under `cfg(test)`). A production build cannot reach them, so SIMD cannot be disabled at runtime by accident. **They are not part of the stable public API** — their shapes may change without a major bump.

### `force_scalar`

```rust
#[cfg(any(test, feature = "testing"))]
pub fn force_scalar();
```

Pin every dispatched distance call in this process onto the scalar reference path. The flag is **sticky**: once set it stays set for the process lifetime (there is intentionally no `unforce_scalar`). It exists so a test or benchmark can exercise the scalar path on hardware that would otherwise pick a SIMD kernel.

### `which_kernel`

```rust
#[cfg(any(test, feature = "testing"))]
pub fn which_kernel() -> &'static str;
```

Return the kernel dispatch would route to right now: `"scalar"`, `"avx2"`, or `"neon"`. It delegates to the same crate-private `select_kernel` the real dispatch uses, so the differential test's "SIMD actually ran" assertion cannot disagree with the production path.

```rust
# #[cfg(feature = "testing")]
# {
let kernel = iqdb_distance::which_kernel();
assert!(matches!(kernel, "scalar" | "avx2" | "neon"));
# }
```

### `compute_scalar`

```rust
#[cfg(any(test, feature = "testing"))]
pub fn compute_scalar(metric: DistanceMetric, a: &[f32], b: &[f32]) -> Result<f32>;
```

Compute `metric` on the **scalar reference path**, bypassing runtime dispatch — a per-call scalar oracle. Unlike [`force_scalar`](#force_scalar), which is a sticky, process-wide override, this computes scalar for a single call without touching global state, so a caller can obtain SIMD and scalar results for the *same* input in the same process. The equivalence fuzz target uses it to assert `compute(metric, a, b) ≈ compute_scalar(metric, a, b)` per input. Same input contract and errors as [`compute`](#compute-runtime-dispatch).

```rust
# #[cfg(feature = "testing")]
# {
use iqdb_distance::compute_scalar;
use iqdb_types::DistanceMetric;

let d = compute_scalar(DistanceMetric::Euclidean, &[0.0, 0.0], &[3.0, 4.0])
    .expect("valid pair");
assert!((d - 5.0).abs() < 1e-6);
# }
```

---

## Errors

`iqdb-distance` returns the shared [`iqdb_types::IqdbError`] / [`iqdb_types::Result`] vocabulary — it adds no error type of its own. The variants it can produce:

| Variant | When |
|---|---|
| `InvalidVector` | Either input slice is empty. |
| `DimensionMismatch { expected, found }` | The two slices differ in length (`expected = a.len()`, `found = b.len()`). |
| `InvalidConfig { reason }` | A batch call where `out.len() != candidates.len()`. |
| `InvalidMetric` | A runtime [`compute`](#compute-runtime-dispatch)/[`compute_batch`](#compute_batch-runtime-dispatch) call with a `DistanceMetric` this crate does not implement (forward-compatibility for the `#[non_exhaustive]` enum). |

`IqdbError` is `Copy` and `#[non_exhaustive]`; match it with a wildcard arm. See the `iqdb-types` API reference for `Display`, `kind()`, and `caption()`.

```rust
use iqdb_distance::{Distance, Cosine};
use iqdb_types::IqdbError;

// Empty input.
assert_eq!(Cosine::compute(&[], &[1.0]).unwrap_err(), IqdbError::InvalidVector);

// Length mismatch.
assert_eq!(
    Cosine::compute(&[1.0, 2.0], &[1.0]).unwrap_err(),
    IqdbError::DimensionMismatch { expected: 2, found: 1 },
);
```

---

## Feature flags

| Feature | Default | Effect |
|---|---|---|
| `testing` | off | Exposes [`force_scalar`](#force_scalar), [`which_kernel`](#which_kernel), and [`compute_scalar`](#compute_scalar), used by the differential SIMD test, the equivalence fuzz target, and the criterion benches to exercise the scalar path on a SIMD-capable host. A production build cannot reach them. |

The crate has no optional runtime dependencies. Its one dependency, `iqdb-types`, is always pulled.

---

## Trait implementation matrix

| Type | `Copy` | `Eq` / `Hash` | `Default` | `Distance` |
|---|:---:|:---:|:---:|:---:|
| `Cosine` | ✅ | ✅ | ✅ | ✅ |
| `DotProduct` | ✅ | ✅ | ✅ | ✅ |
| `Euclidean` | ✅ | ✅ | ✅ | ✅ |
| `Manhattan` | ✅ | ✅ | ✅ | ✅ |
| `Hamming` | ✅ | ✅ | ✅ | ✅ |
| `CpuFeatures` | ✅ | `Eq` only | — | — |

All metric types and `CpuFeatures` implement `Debug`, `Clone`, and `PartialEq`.

---

<sub>Copyright &copy; 2026 <strong>James Gober</strong>.</sub>
