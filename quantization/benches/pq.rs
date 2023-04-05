use criterion::{criterion_group, criterion_main, Criterion};
use rand::Rng;
use quantization::pq::pq_vectors::{CLUSTER_COUNT, PQVectors, ScoreType};

const NUM_VECTORS: usize = 100_000;
const QDIM: usize = 128; // 1024 compressed 8x
const BUCKET_SIZE: usize = 2; // 8x compression is 2 floats per u8

fn get_fixture_orig_vectors(count: usize, qdim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let mut list: Vec<Vec<f32>> = Vec::new();
    for _ in 0..count {
        let vector: Vec<f32> = (0..qdim * BUCKET_SIZE).map(|_| rng.gen()).collect();
        list.push(vector);
    }
    list
}

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
    let orig_vectors = get_fixture_orig_vectors(NUM_VECTORS, QDIM);

    let mut total_score = 0. as ScoreType;
    let mut orig_score = 0.;
    let pq_query = (0..QDIM).map(|_| rand::thread_rng().gen_range(0..CLUSTER_COUNT) as u8).collect::<Vec<u8>>();
    let orig_query = (0..QDIM * BUCKET_SIZE).map(|_| rand::thread_rng().gen_range(0.0..1.0) as f32).collect::<Vec<f32>>();

    group.bench_function("pq-scoring", |b| {
        b.iter(|| {
            let random_idx = rand::random::<usize>() % NUM_VECTORS;
            total_score += pq_vectors.score(&pq_query, random_idx);
        })
    });

    group.bench_function("avx-scoring", |b| {
        b.iter(|| {
            let random_idx = rand::random::<usize>() % NUM_VECTORS;
            orig_score += unsafe { dot_avx(&orig_query, &orig_vectors[random_idx]) };
        })
    });

    eprintln!("orig_score = {:?}", orig_score);
    eprintln!("total_score = {:?}", total_score);
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = pq_scoring
}

criterion_main!(benches);

use std::arch::x86_64::*;

#[target_feature(enable = "avx")]
#[target_feature(enable = "fma")]
#[allow(clippy::missing_safety_doc)]
pub(crate) unsafe fn hsum256_ps_avx(x: __m256) -> f32 {
    let x128: __m128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    let x64: __m128 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    let x32: __m128 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    _mm_cvtss_f32(x32)
}

#[target_feature(enable = "avx")]
#[target_feature(enable = "fma")]
#[allow(clippy::missing_safety_doc)]
pub unsafe fn dot_avx(v1: &[f32], v2: &[f32]) -> f32 {
    let n = v1.len();
    let m = n - (n % 32);
    let mut ptr1: *const f32 = v1.as_ptr();
    let mut ptr2: *const f32 = v2.as_ptr();
    let mut sum256_1: __m256 = _mm256_setzero_ps();
    let mut sum256_2: __m256 = _mm256_setzero_ps();
    let mut sum256_3: __m256 = _mm256_setzero_ps();
    let mut sum256_4: __m256 = _mm256_setzero_ps();
    let mut i: usize = 0;
    while i < m {
        sum256_1 = _mm256_fmadd_ps(_mm256_loadu_ps(ptr1), _mm256_loadu_ps(ptr2), sum256_1);
        sum256_2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(ptr1.add(8)),
            _mm256_loadu_ps(ptr2.add(8)),
            sum256_2,
        );
        sum256_3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(ptr1.add(16)),
            _mm256_loadu_ps(ptr2.add(16)),
            sum256_3,
        );
        sum256_4 = _mm256_fmadd_ps(
            _mm256_loadu_ps(ptr1.add(24)),
            _mm256_loadu_ps(ptr2.add(24)),
            sum256_4,
        );

        ptr1 = ptr1.add(32);
        ptr2 = ptr2.add(32);
        i += 32;
    }

    let mut result = hsum256_ps_avx(sum256_1)
        + hsum256_ps_avx(sum256_2)
        + hsum256_ps_avx(sum256_3)
        + hsum256_ps_avx(sum256_4);

    for i in 0..n - m {
        result += (*ptr1.add(i)) * (*ptr2.add(i));
    }
    result
}
