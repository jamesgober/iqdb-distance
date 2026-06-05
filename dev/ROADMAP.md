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

## v0.5.0 -- SIMD-vs-scalar equivalence fuzzing + API freeze

Exit criteria:
- [ ] Public API frozen (recorded here). `cargo audit` + `cargo deny` clean.

---

## v0.6.0 -> v0.9.x -- Alpha / Beta -> RC

- 0.6.x-0.7.x: integrate against real consumers; MINOR-compatible additions only.
- 0.8.x (beta): bug fixes; broader testing; final benchmarks.
- 0.9.x (rc): critical fixes + doc polish.

---

## v1.0.0 -- Stable

- [ ] Definition of Done (DIRECTIVES section 7) satisfied.
- [ ] Public API frozen until 2.0.
- [ ] Release note written; published to crates.io; tag pushed.

---

## Out of scope for 1.0

- Index or storage logic -- distance math only.
- External SIMD crates -- `std::arch` is sufficient.
