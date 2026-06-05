//! The five metrics through the type-level [`Distance`] trait.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example metrics
//! ```

use iqdb_distance::{Cosine, Distance, DotProduct, Euclidean, Hamming, Manhattan};

fn main() -> Result<(), iqdb_types::IqdbError> {
    let a = [1.0_f32, 2.0, 3.0];
    let b = [4.0_f32, -5.0, 6.0];

    // True distances: smaller is nearer.
    println!("Euclidean (L2): {:.4}", Euclidean::compute(&a, &b)?);
    println!("Manhattan (L1): {:.4}", Manhattan::compute(&a, &b)?);
    println!("Cosine:         {:.4}", Cosine::compute(&a, &b)?);

    // Dot product is a *similarity*: larger is more similar, and it is signed.
    println!("DotProduct:     {:.4}", DotProduct::compute(&a, &b)?);

    // Hamming on f32 counts bit-distinct positions (binary codes as 0.0/1.0).
    let code_a = [0.0_f32, 1.0, 0.0, 1.0, 1.0];
    let code_b = [0.0_f32, 0.0, 0.0, 1.0, 0.0];
    println!("Hamming:        {:.0}", Hamming::compute(&code_a, &code_b)?);

    // Identical vectors: 0 for every true distance.
    assert_eq!(Euclidean::compute(&a, &a)?, 0.0);
    assert_eq!(Manhattan::compute(&a, &a)?, 0.0);

    Ok(())
}
