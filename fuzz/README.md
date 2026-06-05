# iqdb-distance fuzz targets

Cargo-fuzz targets for `iqdb-distance`. This is a standalone crate (its own
workspace), excluded from the normal build; it requires a **nightly** toolchain
and [`cargo-fuzz`](https://github.com/rust-fuzz/cargo-fuzz).

```sh
cargo install cargo-fuzz
```

## Targets

| Target | What it checks |
|---|---|
| `equivalence` | The dispatched kernel (`compute`, SIMD on a SIMD host) agrees with the scalar reference (`compute_scalar`) within floating-point tolerance, over bounded finite inputs across all five metrics. The fuzzed counterpart to `tests/differential.rs`. |
| `robustness` | `compute` returns a typed `Result` and never panics on arbitrary input, including non-finite and length-mismatched vectors. |

`equivalence` bounds components to `[-1e3, 1e3]` (non-finite → `0.0`): SIMD and
scalar accumulate in different orders, so unconstrained `f32` diverges by
summation order alone — a floating-point fact, not a bug. Non-finite
*equivalence* is covered by the differential test's adversarial corpus;
non-finite *no-panic* is covered by `robustness`.

## Run

From the crate root (the parent of this directory):

```sh
# Run until the first failure (or Ctrl-C).
cargo +nightly fuzz run equivalence
cargo +nightly fuzz run robustness

# Time-boxed run (CI-friendly).
cargo +nightly fuzz run equivalence -- -max_total_time=60

# Just build the targets (no fuzzing) — what CI does to prevent bitrot.
cargo +nightly fuzz build
```

Findings are written to `fuzz/artifacts/<target>/`; the evolving corpus lives in
`fuzz/corpus/<target>/`. Both are git-ignored.
