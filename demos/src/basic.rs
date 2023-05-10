use quantization::{
    encoded_vectors::{DistanceType, EncodedVectors, VectorParameters},
    encoded_vectors_u8::EncodedVectorsU8,
};
use rand::{Rng, SeedableRng};

fn dot_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2).map(|(a, b)| a * b).sum()
}

fn main() {
    let vectors_count = 128;
    let vector_dim = 64;
    let error = vector_dim as f32 * 0.1;

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut vector_data: Vec<Vec<f32>> = Vec::new();
    for _ in 0..vectors_count {
        let vector: Vec<f32> = (0..vector_dim).map(|_| rng.gen()).collect();
        vector_data.push(vector);
    }
    let query: Vec<f32> = (0..vector_dim).map(|_| rng.gen()).collect();

    let encoded = EncodedVectorsU8::encode(
        vector_data.iter().map(|v| v.as_slice()),
        Vec::<u8>::new(),
        &VectorParameters {
            dim: vector_dim,
            count: vectors_count,
            distance_type: DistanceType::Dot,
            invert: false,
        },
        None,
        || false,
    )
    .unwrap();
    let query_u8 = encoded.encode_query(&query);

    for (index, vector) in vector_data.iter().enumerate() {
        let score = encoded.score_point(&query_u8, index as u32);
        let orginal_score = dot_similarity(&query, vector);
        assert!((score - orginal_score).abs() < error);
    }

    for (index, vector) in vector_data.iter().enumerate() {
        let score = encoded.score_internal(0, index as u32);
        let orginal_score = dot_similarity(&vector_data[0], vector);
        assert!((score - orginal_score).abs() < error);
    }
}
