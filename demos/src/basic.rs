mod utils;
use quantization::i8_encoder::I8EncodedVectors;
use rand::Rng;

use crate::utils::euclid_similarity;

fn main() {
    // generate vector data and query
    let vectors_count = 100;
    let vector_dim = 64;
    let error = vector_dim as f32 * 0.1;

    let mut rng = rand::thread_rng();
    let mut vector_data: Vec<Vec<f32>> = Vec::new();
    for _ in 0..vectors_count {
        let vector: Vec<f32> = (0..vector_dim).map(|_| rng.gen()).collect();
        vector_data.push(vector);
    }
    let query: Vec<f32> = (0..vector_dim).map(|_| rng.gen()).collect();

    // Second step, encode the vector data
    let encoded = I8EncodedVectors::new(
        vector_data.iter().map(|v| v.as_slice()),
        vectors_count,
        vector_dim,
    )
    .unwrap();

    // Third step, create lookup table - LUT. That's an encoding of the query
    let query_u8 = I8EncodedVectors::encode_query(&query);

    // score query
    for i in 0..vectors_count {
        // encoded score
        let score = encoded.score_point_dot_sse(&query_u8, i);
        let orginal_score = euclid_similarity(&query, &vector_data[i]);
        assert!((score - orginal_score).abs() < error);
    }
}
