//! The normalized-cosine fast path: [`normalize`] embeddings once at ingest,
//! then compare with [`cosine_normalized`] — which skips the norm, square root,
//! and division of the general cosine kernel.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example normalized_search
//! ```

use iqdb_distance::{Cosine, Distance, cosine_normalized, normalize};

fn main() -> Result<(), iqdb_types::IqdbError> {
    let raw_query = [1.0_f32, 2.0, 2.0];
    let raw_corpus: [&[f32]; 3] = [&[2.0, 4.0, 4.0], &[-1.0, 0.0, 3.0], &[5.0, 1.0, 1.0]];

    // Normalize once. In a real store you'd keep the unit vectors.
    let query = normalize(&raw_query)?;
    let corpus: Vec<Vec<f32>> = raw_corpus
        .iter()
        .map(|v| normalize(v))
        .collect::<Result<_, _>>()?;

    println!("normalized-cosine distances (smaller is nearer):");
    for (i, doc) in corpus.iter().enumerate() {
        let fast = cosine_normalized(doc, &query)?;
        // It matches the general kernel on unit-length inputs.
        let full = Cosine::compute(doc, &query)?;
        debug_assert!((fast - full).abs() < 1e-6);
        println!("  doc {i}: {fast:.4}");
    }

    // The first doc is a positive multiple of the query → distance ~ 0.
    let first = cosine_normalized(&corpus[0], &query)?;
    assert!(first.abs() < 1e-6);

    // A zero-magnitude vector cannot be normalized.
    assert!(normalize(&[0.0_f32, 0.0, 0.0]).is_err());

    Ok(())
}
