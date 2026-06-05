//! Choosing the metric at runtime with the [`compute`] / [`compute_batch`]
//! dispatchers — the entry point an engine uses when it does not know the
//! metric at compile time.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example runtime_dispatch
//! ```

use iqdb_distance::{compute, compute_batch};
use iqdb_types::DistanceMetric;

fn main() -> Result<(), iqdb_types::IqdbError> {
    // The metric arrives as a value — e.g. from a query, a config, or a stored
    // index header — not as a compile-time type.
    let metric = DistanceMetric::Euclidean;

    let query = [0.0_f32, 0.0, 0.0];
    let candidate = [3.0_f32, 4.0, 0.0];
    println!(
        "single: {:?} distance = {:.4}",
        metric,
        compute(metric, &query, &candidate)?,
    );

    // Score one query against many candidates into a caller-owned buffer.
    let candidates: [&[f32]; 3] = [&[1.0, 0.0, 0.0], &[0.0, 2.0, 0.0], &[3.0, 4.0, 0.0]];
    let mut out = [0.0_f32; 3];
    compute_batch(metric, &query, &candidates, &mut out)?;
    println!("batch:  {out:?}");

    // Iterate every metric the same way — the dispatcher routes each one.
    for metric in [
        DistanceMetric::Cosine,
        DistanceMetric::DotProduct,
        DistanceMetric::Euclidean,
        DistanceMetric::Manhattan,
        DistanceMetric::Hamming,
    ] {
        let d = compute(metric, &query, &candidate)?;
        println!("  {metric:?}: {d:.4}");
    }

    Ok(())
}
