//! SIMD kernels for the supported target architectures.
//!
//! At v0.1 the SIMD module is a scaffold: it owns the per-arch namespaces
//! and the runtime-dispatch seam that the per-metric implementations call
//! into, but the actual AVX2 and NEON kernels are stitched in during the
//! SIMD pass (see `.dev/ROADMAP.md`). Until that lands the dispatch in
//! [`crate::metrics`] always falls through to the scalar reference, and the
//! `CpuFeatures::avx2` / `CpuFeatures::neon` flags are read but not yet
//! acted on.

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2;

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
