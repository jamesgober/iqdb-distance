//! Criterion benches for the five distance metrics.
//!
//! Each group reports two samples per metric: the SIMD path (the default
//! dispatch on a NEON or AVX2 host) and the scalar reference reached via
//! `force_scalar`. `force_scalar` is sticky for the process, so the SIMD
//! samples are taken first and the scalar samples second.
//!
//! The bench bodies are gated behind the `testing` feature because
//! `force_scalar` is only public under that feature. Without it, the file
//! still compiles to an empty `main` so `cargo build --benches` succeeds
//! under default features. Run the real bench with
//! `cargo bench --features testing`.

#[cfg(not(feature = "testing"))]
fn main() {}

#[cfg(feature = "testing")]
use std::hint::black_box;

#[cfg(feature = "testing")]
use criterion::{Criterion, criterion_group, criterion_main};
#[cfg(feature = "testing")]
use iqdb_distance::{Cosine, Distance, DotProduct, Euclidean, Hamming, Manhattan, force_scalar};

#[cfg(feature = "testing")]
const DIM: usize = 768;

#[cfg(feature = "testing")]
fn vectors() -> (Vec<f32>, Vec<f32>) {
    let mut a = vec![0.0_f32; DIM];
    let mut b = vec![0.0_f32; DIM];
    for i in 0..DIM {
        a[i] = (i as f32).sin();
        b[i] = (i as f32).cos();
    }
    (a, b)
}

#[cfg(feature = "testing")]
fn bench_simd_paths(c: &mut Criterion) {
    let (a, b) = vectors();
    let _ = c.bench_function("simd/cosine/768", |bencher| {
        bencher.iter(|| Cosine::compute(black_box(&a), black_box(&b)));
    });
    let _ = c.bench_function("simd/dot/768", |bencher| {
        bencher.iter(|| DotProduct::compute(black_box(&a), black_box(&b)));
    });
    let _ = c.bench_function("simd/euclidean/768", |bencher| {
        bencher.iter(|| Euclidean::compute(black_box(&a), black_box(&b)));
    });
    let _ = c.bench_function("simd/manhattan/768", |bencher| {
        bencher.iter(|| Manhattan::compute(black_box(&a), black_box(&b)));
    });
    let _ = c.bench_function("simd/hamming/768", |bencher| {
        bencher.iter(|| Hamming::compute(black_box(&a), black_box(&b)));
    });
}

#[cfg(feature = "testing")]
fn bench_scalar_paths(c: &mut Criterion) {
    // Sticky for the rest of the process.
    force_scalar();
    let (a, b) = vectors();
    let _ = c.bench_function("scalar/cosine/768", |bencher| {
        bencher.iter(|| Cosine::compute(black_box(&a), black_box(&b)));
    });
    let _ = c.bench_function("scalar/dot/768", |bencher| {
        bencher.iter(|| DotProduct::compute(black_box(&a), black_box(&b)));
    });
    let _ = c.bench_function("scalar/euclidean/768", |bencher| {
        bencher.iter(|| Euclidean::compute(black_box(&a), black_box(&b)));
    });
    let _ = c.bench_function("scalar/manhattan/768", |bencher| {
        bencher.iter(|| Manhattan::compute(black_box(&a), black_box(&b)));
    });
    let _ = c.bench_function("scalar/hamming/768", |bencher| {
        bencher.iter(|| Hamming::compute(black_box(&a), black_box(&b)));
    });
}

#[cfg(feature = "testing")]
criterion_group!(benches, bench_simd_paths, bench_scalar_paths);
#[cfg(feature = "testing")]
criterion_main!(benches);
