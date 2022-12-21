use criterion::{criterion_group, criterion_main, Criterion};
use permutation_iterator::Permutor;
use quantization::i8_encoder::I8EncodedVectors;
use rand::Rng;

use std::arch::x86_64::*;

#[target_feature(enable = "avx")]
#[target_feature(enable = "fma")]
unsafe fn hsum256_ps_avx(x: __m256) -> f32 {
    let x128: __m128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    let x64: __m128 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    let x32: __m128 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    _mm_cvtss_f32(x32)
}

unsafe fn hsum128_ps_sse(x: __m128) -> f32 {
    let x64: __m128 = _mm_add_ps(x, _mm_movehl_ps(x, x));
    let x32: __m128 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    _mm_cvtss_f32(x32)
}

#[target_feature(enable = "avx")]
#[target_feature(enable = "fma")]
pub(crate) unsafe fn dot_avx(v1: &[f32], v2: &[f32]) -> f32 {
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

pub(crate) unsafe fn dot_similarity_sse(v1: &[f32], v2: &[f32]) -> f32 {
    let n = v1.len();
    let m = n - (n % 16);
    let mut ptr1: *const f32 = v1.as_ptr();
    let mut ptr2: *const f32 = v2.as_ptr();
    let mut sum128_1: __m128 = _mm_setzero_ps();
    let mut sum128_2: __m128 = _mm_setzero_ps();
    let mut sum128_3: __m128 = _mm_setzero_ps();
    let mut sum128_4: __m128 = _mm_setzero_ps();

    let mut i: usize = 0;
    while i < m {
        sum128_1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ptr1), _mm_loadu_ps(ptr2)), sum128_1);

        sum128_2 = _mm_add_ps(
            _mm_mul_ps(_mm_loadu_ps(ptr1.add(4)), _mm_loadu_ps(ptr2.add(4))),
            sum128_2,
        );

        sum128_3 = _mm_add_ps(
            _mm_mul_ps(_mm_loadu_ps(ptr1.add(8)), _mm_loadu_ps(ptr2.add(8))),
            sum128_3,
        );

        sum128_4 = _mm_add_ps(
            _mm_mul_ps(_mm_loadu_ps(ptr1.add(12)), _mm_loadu_ps(ptr2.add(12))),
            sum128_4,
        );

        ptr1 = ptr1.add(16);
        ptr2 = ptr2.add(16);
        i += 16;
    }

    let mut result = hsum128_ps_sse(sum128_1)
        + hsum128_ps_sse(sum128_2)
        + hsum128_ps_sse(sum128_3)
        + hsum128_ps_sse(sum128_4);
    for i in 0..n - m {
        result += (*ptr1.add(i)) * (*ptr2.add(i));
    }
    result
}

fn encode_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode");

    let vectors_count = 1_000_000;
    let vector_dim = 512;
    let mut rng = rand::thread_rng();
    let mut list: Vec<f32> = Vec::new();
    for _ in 0..vectors_count {
        let vector: Vec<f32> = (0..vector_dim).map(|_| rng.gen()).collect();
        list.extend_from_slice(&vector);
    }

    let i8_encoded = I8EncodedVectors::new(
        (0..vectors_count)
            .into_iter()
            .map(|i| &list[i * vector_dim..(i + 1) * vector_dim]),
        vectors_count,
        vector_dim,
    )
    .unwrap();

    let query: Vec<f32> = (0..vector_dim).map(|_| rng.gen()).collect();
    let encoded_query = I8EncodedVectors::encode_query(&query);

    group.bench_function("score all u8 avx", |b| {
        b.iter(|| {
            let mut s = 0.0;
            for i in 0..vectors_count {
                s = i8_encoded.score_point_dot_avx(&encoded_query, i);
            }
        });
    });

    group.bench_function("score all u8 avx 2", |b| {
        b.iter(|| {
            let mut s = 0.0;
            for i in 0..vectors_count {
                s = i8_encoded.score_point_dot_avx_2(&encoded_query, i);
            }
        });
    });

    group.bench_function("score all u8 sse", |b| {
        b.iter(|| {
            let mut s = 0.0;
            for i in 0..vectors_count {
                s = i8_encoded.score_point_dot_sse(&encoded_query, i);
            }
        });
    });

    group.bench_function("score all avx", |b| {
        b.iter(|| unsafe {
            let mut s = 0.0;
            for i in 0..vectors_count {
                s = dot_avx(&query, &list[i * vector_dim..(i + 1) * vector_dim]);
            }
        });
    });

    group.bench_function("score all sse", |b| {
        b.iter(|| unsafe {
            let mut s = 0.0;
            for i in 0..vectors_count {
                s = dot_similarity_sse(&query, &list[i * vector_dim..(i + 1) * vector_dim]);
            }
        });
    });

    let permutor = Permutor::new(vectors_count as u64);
    let permutation: Vec<usize> = permutor.map(|i| i as usize).collect();

    group.bench_function("score random access u8 avx", |b| {
        b.iter(|| {
            let mut s = 0.0;
            for &i in &permutation {
                s = i8_encoded.score_point_dot_avx(&encoded_query, i);
            }
        });
    });

    group.bench_function("score random access u8 avx 2", |b| {
        b.iter(|| {
            let mut s = 0.0;
            for &i in &permutation {
                s = i8_encoded.score_point_dot_avx_2(&encoded_query, i);
            }
        });
    });

    group.bench_function("score random access u8 sse", |b| {
        let mut s = 0.0;
        b.iter(|| {
            for &i in &permutation {
                s = i8_encoded.score_point_dot_sse(&encoded_query, i);
            }
        });
    });

    group.bench_function("score random access avx", |b| {
        b.iter(|| unsafe {
            let mut s = 0.0;
            for &i in &permutation {
                s = dot_avx(&query, &list[i * vector_dim..(i + 1) * vector_dim]);
            }
        });
    });

    group.bench_function("score random access sse", |b| {
        let mut s = 0.0;
        b.iter(|| unsafe {
            for &i in &permutation {
                s = dot_similarity_sse(&query, &list[i * vector_dim..(i + 1) * vector_dim]);
            }
        });
    });

    group.bench_function("score random access blocks avx", |b| {
        let mut scores = vec![0.0; 16];
        b.iter(|| unsafe {
            for window in permutation.as_slice().chunks_exact(16) {
                for (i, idx) in window.iter().enumerate() {
                    scores[i] =
                        dot_similarity_sse(&query, &list[idx * vector_dim..(idx + 1) * vector_dim]);
                }
            }
        });
    });

    group.bench_function("score random access blocks i8 avx", |b| {
        let mut scores = vec![0.0; 16];
        b.iter(|| {
            for window in permutation.as_slice().chunks_exact(16) {
                for (i, idx) in window.iter().enumerate() {
                    scores[i] = i8_encoded.score_point_dot_avx(&encoded_query, *idx);
                }
            }
        });
    });

    group.bench_function("score random access blocks i8 avx 2", |b| {
        let mut scores = vec![0.0; 16];
        b.iter(|| {
            for window in permutation.as_slice().chunks_exact(16) {
                for (i, idx) in window.iter().enumerate() {
                    scores[i] = i8_encoded.score_point_dot_avx_2(&encoded_query, *idx);
                }
            }
        });
    });

    group.bench_function("score random access blocks i8 sse", |b| {
        let mut scores = vec![0.0; 16];
        b.iter(|| {
            for window in permutation.as_slice().chunks_exact(16) {
                for (i, idx) in window.iter().enumerate() {
                    scores[i] = i8_encoded.score_point_dot_sse(&encoded_query, *idx);
                }
            }
        });
    });

    group.bench_function("score random access blocks i8 sse blocked", |b| {
        let mut scores = vec![0.0; 16];
        b.iter(|| {
            for window in permutation.as_slice().chunks_exact(16) {
                i8_encoded.score_points_dot_sse(&encoded_query, window, &mut scores);
            }
        });
    });

    group.bench_function("score random access blocks i8 avx blocked", |b| {
        let mut scores = vec![0.0; 16];
        b.iter(|| {
            for window in permutation.as_slice().chunks_exact(16) {
                i8_encoded.score_points_dot_avx(&encoded_query, window, &mut scores);
            }
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = encode_bench
}

criterion_main!(benches);
