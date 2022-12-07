mod utils;

use quantization::encoded_vectors::EncodedVectors;
use rand::Rng;

use crate::utils::euclid_similarity;

fn main() {
    let vectors_count = 10_000;
    let vector_dim = 64;
    let queries_count = 100;

    let mut rng = rand::thread_rng();
    let mut vector_data: Vec<Vec<f32>> = Vec::new();
    for _ in 0..vectors_count {
        let vector: Vec<f32> = (0..vector_dim).map(|_| rng.gen()).collect();
        vector_data.push(vector);
    }

    let timer = std::time::Instant::now();
    let chunks = EncodedVectors::create_dim_partition(vector_dim, 1);
    let encoder = EncodedVectors::new(
        vector_data.iter().map(|v| v.as_slice()),
        vectors_count,
        vector_dim,
        &chunks,
    )
    .unwrap();
    println!("encoding time: {}ms", timer.elapsed().as_millis());

    let mut queries_data: Vec<Vec<f32>> = Vec::new();
    for _ in 0..queries_count {
        let query: Vec<f32> = (0..vector_dim).map(|_| rng.gen()).collect();
        queries_data.push(query);
    }

    utils::run_knn_queries(
        vectors_count,
        queries_data.iter().map(|q| q.as_slice()),
        |i| vector_data[i].as_slice(),
        &encoder,
        euclid_similarity,
    );
}