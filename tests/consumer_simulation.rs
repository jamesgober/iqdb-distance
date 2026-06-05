//! Consumer-simulation suite — the RC soak gate.
//!
//! A stand-in index built **only** on the public `iqdb-distance` surface,
//! exercising it exactly the way the real index crates do. Verified against
//! their implementations: `iqdb-flat`, `iqdb-hnsw`, and `iqdb-ivf` each route
//! every metric through [`iqdb_distance::compute_batch`] (they never reimplement
//! a metric), store candidates as `Arc<[f32]>`, negate `DotProduct` at the
//! boundary to honour the "smaller is nearer" ordering, then select top-`k` by
//! ascending distance. This suite reproduces that pipeline and asserts the
//! surface is sufficient and ergonomic for it.

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use iqdb_distance::{compute, compute_batch, cosine_normalized, normalize};
use iqdb_types::{DistanceMetric, Hit, IqdbError, VectorId};

/// A brute-force index over `Arc<[f32]>` rows — the shape `iqdb-flat` exposes.
struct SimIndex {
    dim: usize,
    metric: DistanceMetric,
    ids: Vec<VectorId>,
    vectors: Vec<Arc<[f32]>>,
}

impl SimIndex {
    fn new(dim: usize, metric: DistanceMetric) -> Self {
        Self {
            dim,
            metric,
            ids: Vec::new(),
            vectors: Vec::new(),
        }
    }

    fn insert(&mut self, id: u64, vector: Arc<[f32]>) -> Result<(), IqdbError> {
        if vector.len() != self.dim {
            return Err(IqdbError::DimensionMismatch {
                expected: self.dim,
                found: vector.len(),
            });
        }
        self.ids.push(VectorId::from(id));
        self.vectors.push(vector);
        Ok(())
    }

    /// Top-`k` nearest, best-first — the real search pipeline.
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<Hit>, IqdbError> {
        // Borrow each stored `Arc<[f32]>` as a slice (no payload copy).
        let candidates: Vec<&[f32]> = self.vectors.iter().map(|a| &a[..]).collect();
        let mut distances = vec![0.0_f32; candidates.len()];
        compute_batch(self.metric, query, &candidates, &mut distances)?;

        // DotProduct is a similarity (larger is closer); negate so every metric
        // follows the smaller-is-nearer ordering, exactly as `iqdb-flat` does.
        if matches!(self.metric, DistanceMetric::DotProduct) {
            for d in distances.iter_mut() {
                *d = -*d;
            }
        }

        // Stable top-k by (distance, insertion order).
        let mut order: Vec<usize> = (0..candidates.len()).collect();
        order.sort_by(|&i, &j| {
            distances[i]
                .partial_cmp(&distances[j])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(i.cmp(&j))
        });
        Ok(order
            .into_iter()
            .take(k)
            .map(|i| Hit::new(self.ids[i].clone(), distances[i]))
            .collect())
    }
}

fn ids(hits: &[Hit]) -> Vec<VectorId> {
    hits.iter().map(|h| h.id.clone()).collect()
}

// --- Per-metric ranking ------------------------------------------------------

#[test]
fn euclidean_returns_nearest_first() {
    let mut index = SimIndex::new(2, DistanceMetric::Euclidean);
    index
        .insert(1, Arc::from([0.0_f32, 0.0].as_slice()))
        .unwrap();
    index
        .insert(2, Arc::from([1.0_f32, 1.0].as_slice()))
        .unwrap();
    index
        .insert(3, Arc::from([5.0_f32, 5.0].as_slice()))
        .unwrap();

    let hits = index.search(&[0.1, 0.1], 3).unwrap();
    assert_eq!(
        ids(&hits),
        [VectorId::U64(1), VectorId::U64(2), VectorId::U64(3)],
    );
    // Distances are non-decreasing.
    assert!(hits[0].distance <= hits[1].distance);
    assert!(hits[1].distance <= hits[2].distance);
}

#[test]
fn dot_product_returns_most_similar_first() {
    // Larger inner product is "closer"; the index negates so the nearest Hit
    // has the smallest (most-negative) distance.
    let mut index = SimIndex::new(2, DistanceMetric::DotProduct);
    index
        .insert(1, Arc::from([1.0_f32, 0.0].as_slice()))
        .unwrap();
    index
        .insert(2, Arc::from([10.0_f32, 0.0].as_slice()))
        .unwrap();
    index
        .insert(3, Arc::from([-5.0_f32, 0.0].as_slice()))
        .unwrap();

    let hits = index.search(&[1.0, 0.0], 3).unwrap();
    // Dots: id1=1, id2=10, id3=-5 → most similar is id2, then id1, then id3.
    assert_eq!(
        ids(&hits),
        [VectorId::U64(2), VectorId::U64(1), VectorId::U64(3)],
    );
    assert!(hits[0].distance <= hits[1].distance);
}

#[test]
fn cosine_ranks_by_angle_not_magnitude() {
    let mut index = SimIndex::new(2, DistanceMetric::Cosine);
    index
        .insert(1, Arc::from([1.0_f32, 0.0].as_slice()))
        .unwrap(); // same dir as query
    index
        .insert(2, Arc::from([100.0_f32, 0.0].as_slice()))
        .unwrap(); // same dir, large mag
    index
        .insert(3, Arc::from([0.0_f32, 1.0].as_slice()))
        .unwrap(); // perpendicular

    let hits = index.search(&[2.0, 0.0], 3).unwrap();
    // Cosine ignores magnitude: ids 1 and 2 tie at distance ~0, id 3 is ~1.
    assert_eq!(hits[2].id, VectorId::U64(3));
    assert!(hits[0].distance < hits[2].distance);
}

#[test]
fn every_implemented_metric_runs_through_the_index() {
    for metric in [
        DistanceMetric::Cosine,
        DistanceMetric::DotProduct,
        DistanceMetric::Euclidean,
        DistanceMetric::Manhattan,
        DistanceMetric::Hamming,
    ] {
        let mut index = SimIndex::new(3, metric);
        index
            .insert(1, Arc::from([1.0_f32, 0.0, 0.0].as_slice()))
            .unwrap();
        index
            .insert(2, Arc::from([0.0_f32, 1.0, 0.0].as_slice()))
            .unwrap();
        let hits = index.search(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(hits.len(), 2, "metric {metric:?}");
    }
}

// --- Contracts the consumers rely on ----------------------------------------

#[test]
fn batch_scoring_matches_per_pair_compute() {
    let metric = DistanceMetric::Euclidean;
    let query = [0.2_f32, 0.4, 0.6];
    let rows: [&[f32]; 3] = [&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.5, 0.5, 0.5]];

    let mut batched = [0.0_f32; 3];
    compute_batch(metric, &query, &rows, &mut batched).unwrap();
    for (row, got) in rows.iter().zip(batched.iter()) {
        let single = compute(metric, &query, row).unwrap();
        assert_eq!(single.to_bits(), got.to_bits());
    }
}

#[test]
fn dimension_mismatch_propagates_from_search() {
    let mut index = SimIndex::new(3, DistanceMetric::Euclidean);
    index
        .insert(1, Arc::from([1.0_f32, 0.0, 0.0].as_slice()))
        .unwrap();
    // Query of the wrong dimension surfaces the typed error from the kernel.
    let err = index.search(&[1.0, 0.0], 1).unwrap_err();
    assert!(matches!(err, IqdbError::DimensionMismatch { .. }));
}

#[test]
fn insert_rejects_wrong_dimension() {
    let mut index = SimIndex::new(3, DistanceMetric::Cosine);
    let err = index
        .insert(1, Arc::from([1.0_f32, 0.0].as_slice()))
        .unwrap_err();
    assert_eq!(
        err,
        IqdbError::DimensionMismatch {
            expected: 3,
            found: 2
        }
    );
}

// --- Normalized-index variant (the cosine fast path) ------------------------

#[test]
fn normalized_index_matches_cosine_ranking() {
    // An index that normalizes at ingest and queries with `cosine_normalized`
    // must rank identically to a plain `Cosine` index over the raw vectors.
    let raw: [&[f32]; 3] = [&[1.0, 2.0, 3.0], &[-3.0, 1.0, 0.5], &[2.0, 2.0, 2.0]];
    let query = [1.0_f32, 2.0, 2.5];

    // Plain cosine index.
    let mut cosine = SimIndex::new(3, DistanceMetric::Cosine);
    for (i, r) in raw.iter().enumerate() {
        cosine.insert(i as u64, Arc::from(*r)).unwrap();
    }
    let cosine_order = ids(&cosine.search(&query, 3).unwrap());

    // Normalized index: store unit vectors, query with the unit query.
    let units: Vec<Vec<f32>> = raw.iter().map(|r| normalize(r).unwrap()).collect();
    let uq = normalize(&query).unwrap();
    let mut scored: Vec<(usize, f32)> = units
        .iter()
        .enumerate()
        .map(|(i, u)| (i, cosine_normalized(u, &uq).unwrap()))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
    let normalized_order: Vec<VectorId> = scored
        .into_iter()
        .map(|(i, _)| VectorId::U64(i as u64))
        .collect();

    assert_eq!(cosine_order, normalized_order);
}
