mod utils;

use quantization::{encoded_vectors::EncodedVectors, lut::Lut};
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
    let chunks = EncodedVectors::divide_dim(vector_dim, 2);

    // Second step, encode the vector data
    let encoded = EncodedVectors::new(vector_data.iter().map(|v| v.as_slice()), vectors_count, vector_dim, &chunks).unwrap();

    // Third step, create lookup table - LUT. That's an encoding of the query
    let lut = Lut::new(&encoded, &query, euclid_similarity);

    // Fourth step, score all vectors
    for i in 0..vectors_count {
        let encoded_vector = encoded.get(i);
        let score = lut.dist(encoded_vector);

        let orginal_score = euclid_similarity(&query, &vector_data[i]);
        println!("{} {}", score, orginal_score);
        assert!((score - orginal_score).abs() < 2.0);
    }
}
