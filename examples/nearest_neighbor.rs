//! A minimal top-`k` nearest-neighbour search — the core loop every index
//! crate builds on. Scores a query against a set of candidates with
//! [`compute_batch`], then ranks smaller-is-nearer.
//!
//! Note the `DotProduct` handling: it is a similarity (larger is closer), so an
//! index negates it at the boundary to keep one ordering for every metric. This
//! is exactly what `iqdb-flat` / `iqdb-hnsw` / `iqdb-ivf` do.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example nearest_neighbor
//! ```

use iqdb_distance::compute_batch;
use iqdb_types::DistanceMetric;

fn top_k(
    metric: DistanceMetric,
    query: &[f32],
    candidates: &[&[f32]],
    k: usize,
) -> Result<Vec<(usize, f32)>, iqdb_types::IqdbError> {
    let mut distances = vec![0.0_f32; candidates.len()];
    compute_batch(metric, query, candidates, &mut distances)?;

    // DotProduct: larger is more similar → negate so smaller is nearer.
    if matches!(metric, DistanceMetric::DotProduct) {
        for d in distances.iter_mut() {
            *d = -*d;
        }
    }

    let mut ranked: Vec<(usize, f32)> = distances.into_iter().enumerate().collect();
    ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(k);
    Ok(ranked)
}

fn main() -> Result<(), iqdb_types::IqdbError> {
    let query = [0.2_f32, 0.1, 0.0];
    let candidates: [&[f32]; 4] = [
        &[1.0, 0.0, 0.0],
        &[0.2, 0.1, 0.0],
        &[0.0, 0.0, 1.0],
        &[0.25, 0.05, 0.0],
    ];

    println!("query = {query:?}");
    for (rank, (idx, dist)) in top_k(DistanceMetric::Euclidean, &query, &candidates, 3)?
        .into_iter()
        .enumerate()
    {
        println!("  #{}: candidate {idx} at distance {dist:.4}", rank + 1);
    }

    Ok(())
}
