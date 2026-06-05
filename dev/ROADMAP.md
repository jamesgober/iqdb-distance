# iqdb-distance -- Roadmap

> Path from scaffold to a stable 1.0. Hard parts are front-loaded; each phase has hard exit criteria.
>
> **Anti-deferral rule:** no listed hard task moves to a later phase unless this file records the move and the reason.

---

## v0.1.0 -- Scaffold (DONE)

Compiles, CI green, structure correct, no domain logic.

- [x] Manifest, README, CHANGELOG, REPS, license, CI, lints in place.
- [x] API surface sketched in `docs/API.md`.

---

## v0.2.0 -- scalar reference metrics + the `Distance` trait (THE HARD PART, NOT DEFERRED) (DONE)

Exit criteria:
- [x] Every public item has rustdoc + a runnable example.
- [x] Core invariants property-tested.

All five metrics shipped behind the `Distance` trait with always-compiled
scalar reference paths, runtime `compute`/`compute_batch` dispatch, allocation-
free batch evaluation, and typed-error validation. Cosine low-magnitude
robustness (independent-sqrt denominator) landed and is pinned by test.

---

## v0.3.0 -- SIMD (AVX2/NEON) + runtime dispatch (DONE)

Exit criteria:
- [x] New surface tested and benchmarked where it is a hot path.

AVX2 (x86_64) and NEON (aarch64) kernels for every metric, runtime-dispatched
via `detect_features`/`select_kernel`, differentially tested against the scalar
reference over finite and adversarial corpora (with a "SIMD actually ran" gate),
and benchmarked SIMD-vs-scalar per metric. **AVX-512 is deferred** (MSRV 1.87 +
CI coverage gap) — recorded here per the anti-deferral rule; revisit when the
MSRV and CI runners support it.

---

## v0.4.0 -- batch ops + normalized variants + feature freeze (DONE)

Exit criteria:
- [x] No `todo!`/`unimplemented!`. Feature freeze declared.

Batch ops (`compute_batch`, `Distance::compute_batch`) shipped in 0.2.0. The
normalized fast path landed as two free functions, kept deliberately minimal
per the "keep the API simple" mandate: `cosine_normalized` (`1 - a · b` over the
SIMD dot kernel) and `normalize` (`v / ‖v‖`) — **not** a parallel metric type
and **not** a new `DistanceMetric` variant (that enum lives in frozen
`iqdb-types`).

**Feature freeze — the public surface is now complete and frozen for 1.x.**
Additive, non-breaking changes remain allowed; anything else waits for 2.0. The
frozen surface:

- Metric tags: `Cosine`, `DotProduct`, `Euclidean`, `Manhattan`, `Hamming`.
- Trait: `Distance` (`compute`, `compute_batch`).
- Runtime dispatch: `compute`, `compute_batch` over `DistanceMetric`
  (`InvalidMetric` for unimplemented `#[non_exhaustive]` variants).
- Normalized: `cosine_normalized`, `normalize`.
- Feature detection: `CpuFeatures`, `detect_features`, `forced_scalar`.
- Constant: `VERSION`.
- Not part of the stable surface (gated on `feature = "testing"`):
  `force_scalar`, `which_kernel`.

---

## v0.5.0 -- SIMD-vs-scalar equivalence fuzzing + API freeze (DONE)

Exit criteria:
- [x] Public API frozen (recorded here). `cargo audit` + `cargo deny` clean.

Equivalence fuzzing landed as a standalone `fuzz/` crate (cargo-fuzz, nightly):
- `equivalence` — asserts the dispatched (SIMD) kernel agrees with the scalar
  reference (`compute_scalar`) over bounded finite inputs across all metrics.
  The fuzzed counterpart to the fixed-corpus `tests/differential.rs`.
- `robustness` — asserts `compute` never panics on arbitrary (incl. non-finite,
  length-mismatched) input.

Both are built and smoke-run in CI (the `fuzz` job). Local soak: 13.6M+
executions on `equivalence` and 10.3M+ on `robustness` with zero findings. A
testing-only `compute_scalar` accessor was added to give the fuzzer a per-call
scalar oracle (gated on `feature = "testing"`, **not** part of the stable
surface — same status as `force_scalar` / `which_kernel`).

**API freeze.** The stable public surface recorded under v0.4.0 is locked for
the 1.x series. Only additive, non-breaking changes land before 2.0; the
testing-gated accessors remain outside the guarantee. `cargo audit` and
`cargo deny check` are clean.

---

## v0.6.0 -> v0.9.x -- Alpha / Beta -> RC (folded into 1.0.0)

- 0.6.x-0.7.x: integrate against real consumers; MINOR-compatible additions only.
- 0.8.x (beta): bug fixes; broader testing; final benchmarks.
- 0.9.x (rc): critical fixes + doc polish.

The RC track's *intent* — prove the surface serves real consumers, settle final
benchmarks, polish docs — was met without separate tags. Under the spine-first
ordering the real index crates are not yet published against this surface, so
the soak was carried out by the `tests/consumer_simulation.rs` suite, built at
the **exact** signatures `iqdb-flat` / `iqdb-hnsw` / `iqdb-ivf` expose (each
routes every metric through `compute_batch` and never reimplements one) and
cross-checked against their implementations. Final benchmarks and doc/example
polish shipped with 1.0.0. This mirrors how `iqdb-types` reached 1.0 on a
satisfied checklist rather than a calendar.

---

## v1.0.0 -- Stable (DONE)

- [x] Definition of Done (DIRECTIVES section 7) satisfied.
- [x] Public API frozen until 2.0.
- [x] Release note written. (Publish to crates.io + tag push: owner action.)

**On the `loom` Definition-of-Done item (§7.6).** The crate's only shared
mutable state is `FORCED_SCALAR` — a set-once, monotonic, `Relaxed` `AtomicBool`
— plus a `std::sync::OnceLock` that performs its own synchronization. There is
no lock-free data structure or multi-step concurrent protocol to model: the flag
goes `false -> true` once and is only ever read. A `loom` test would require
making the `static` injectable purely to model a primitive whose correctness is
self-evident, which trades real complexity for no assurance (KISS/YAGNI).
Concurrent behaviour is instead exercised directly: `tests/differential.rs`
runs `force_scalar()` against Cargo's parallel test execution and is hardened
(via a pre-flip `OnceLock` snapshot) precisely against the set/observe race.
This is recorded as a deliberate, settled decision per the anti-deferral rule.

---

## Out of scope for 1.0

- Index or storage logic -- distance math only.
- External SIMD crates -- `std::arch` is sufficient.
