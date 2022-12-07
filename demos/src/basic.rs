mod utils;

use quantization::{encoded_vectors::EncodedVectors, scorer::Scorer, simple_scorer::SimpleScorer};
use rand::Rng;

use crate::utils::euclid_similarity;

fn main() {
    // generate vector data and query
    let vectors_count = 100;
    let vector_dim = 64;

    let mut rng = rand::thread_rng();
    let mut vector_data: Vec<Vec<f32>> = Vec::new();
    for _ in 0..vectors_count {
        let vector: Vec<f32> = (0..vector_dim).map(|_| rng.gen()).collect();
        vector_data.push(vector);
    }
    let query: Vec<f32> = (0..vector_dim).map(|_| rng.gen()).collect();

    // First step, divide the vector dimension into chunks
    let chunks = EncodedVectors::create_dim_partition(vector_dim, 2);

    // Second step, encode the vector data
    let encoded = EncodedVectors::new(
        vector_data.iter().map(|v| v.as_slice()),
        vectors_count,
        vector_dim,
        &chunks,
    )
    .unwrap();

    // Third step, create lookup table - LUT. That's an encoding of the query
    let scorer: SimpleScorer = encoded.scorer(&query, euclid_similarity);

    for i in 0..vectors_count {
        // encoded score
        let score = scorer.score_point(i);
        let orginal_score = euclid_similarity(&query, &vector_data[i]);
        // decoded vector
        let decoded = encoded.decode_vector(i);
        // check if the decoded vector is the same as the original vector
        let diff: f32 = decoded
            .iter()
            .zip(vector_data[i].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, |a, b| if a > b { a } else { b });

        println!(
            "Encoded score {}, orig score {},  diff {}",
            score, orginal_score, diff
        );
        assert!(diff < 0.3);
        assert!((score - orginal_score).abs() < 2.0);
    }
}
