use quantization::encoder::EncodedVectors;
use rand::{Rng, SeedableRng};

use quantization::utils::dot_similarity;

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

    let encoded = EncodedVectors::new(
        vector_data.iter().map(|v| v.as_slice()),
        vectors_count,
        vector_dim,
    )
    .unwrap();
    let query_u8 = encoded.encode_query(&query);

    let indexes = (0..vectors_count).collect::<Vec<_>>();
    let mut scores = vec![0.0; vectors_count];
    encoded.score_points_dot_simple(&query_u8, &indexes, &mut scores);

    for i in 0..vectors_count {
        let score = encoded.score_point_dot_simple(&query_u8, i);
        let score2 = scores[i];
        let orginal_score = dot_similarity(&query, &vector_data[i]);
        assert!((score - orginal_score).abs() < error);
        assert!((score2 - orginal_score).abs() < error);
    }
}
