use criterion::{criterion_group, criterion_main, Criterion};
use rand::Rng;
use quantization::pq::pq_vectors::{CLUSTER_COUNT, PQVectors, ScoreType};

const NUM_VECTORS: usize = 100_000;
const QDIM: usize = 128; // 1024 compressed 8x


fn get_fixture_pq_vectors(count: usize, qdim: usize) -> PQVectors {

    // Generate random lookup table of size CLUSTER_COUNT * CLUSTER_COUNT
    let lookup_table = (0..CLUSTER_COUNT * CLUSTER_COUNT)
        .map(|_| rand::random::<ScoreType>())
        .collect::<Vec<ScoreType>>();

    // Generate random quantized vectors of size dim * count in range [0, CLUSTER_COUNT]
    let quantized_vectors = (0..qdim * count)
        .map(|_| rand::thread_rng().gen_range(0..CLUSTER_COUNT) as u8)
        .collect::<Vec<u8>>();

    PQVectors::new(
        lookup_table,
        quantized_vectors,
        qdim,
        count,
    )
}

fn pq_scoring(c: &mut Criterion) {
    let mut group = c.benchmark_group("pq-scoring");

    let pq_vectors = get_fixture_pq_vectors(NUM_VECTORS, QDIM);

    let mut total_score = 0. as ScoreType;
    let query = (0..QDIM).map(|_| rand::thread_rng().gen_range(0..CLUSTER_COUNT) as u8).collect::<Vec<u8>>();

    group.bench_function("pq-scoring", |b| {
        b.iter(|| {
            let random_idx = rand::random::<usize>() % NUM_VECTORS;
            total_score += pq_vectors.score(&query, random_idx);
        })
    });

    eprintln!("total_score = {:?}", total_score);
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = pq_scoring
}

criterion_main!(benches);
