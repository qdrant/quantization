use criterion::{criterion_group, criterion_main, Criterion};
use permutation_iterator::Permutor;
use quantization::{encoder::EncodedVectorStorage, lut::Lut};
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

#[target_feature(enable = "avx")]
#[target_feature(enable = "fma")]
pub(crate) unsafe fn dot_avx(
    v1: &[f32],
    v2: &[f32],
) -> f32 {
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

fn encode_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode");

    let vectors_count = 10_000;
    let vector_dim = 512;
    let mut rng = rand::thread_rng();
    let mut list: Vec<Vec<f32>> = Vec::new();
    for _ in 0..vectors_count {
        let vector: Vec<f32> = (0..vector_dim).map(|_| rng.gen()).collect();
        list.push(vector);
    }

    let chunks = EncodedVectorStorage::divide_dim(vector_dim, 1);

    group.bench_function("encode", |b| {
        b.iter(|| EncodedVectorStorage::new(Box::new(list.iter().map(|v| v.as_slice())), &chunks));
    });

    let encoder =
        EncodedVectorStorage::new(Box::new(list.iter().map(|v| v.as_slice())), &chunks).unwrap();
    let metric = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f32>();
    let query: Vec<f32> = (0..vector_dim).map(|_| rng.gen()).collect();
    group.bench_function("score all quantized", |b| {
        b.iter(|| {
            let lut = Lut::new(&encoder, &query, metric);
            for i in 0..vectors_count {
                lut.dist(encoder.get(i));
            }
        });
    });

    group.bench_function("score all original", |b| {
        b.iter(|| {
            for v in &list {
                let mut _sum = 0.0;
                for i in 0..vector_dim {
                    _sum += query[i] * v[i];
                }
            }
        });
    });

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma")
        {
            group.bench_function("score all avx", |b| {
                b.iter(|| unsafe {
                    for v in &list {
                        dot_avx(&query, v);
                    }
                });
            });
        }
    }

    let permutor = Permutor::new(vectors_count as u64);
    let permutation: Vec<usize> = permutor.map(|i| i as usize).collect();

    group.bench_function("score random access quantized", |b| {
        b.iter(|| {
            let lut = Lut::new(&encoder, &query, metric);
            for &i in &permutation {
                lut.dist(encoder.get(i));
            }
        });
    });

    group.bench_function("score random access original", |b| {
        b.iter(|| {
            for &i in &permutation {
                let mut _sum = 0.0;
                let v = &list[i];
                for i in 0..vector_dim {
                    _sum += query[i] * v[i];
                }
            }
        });
    });

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma")
        {
            group.bench_function("score random access avx", |b| {
                b.iter(|| unsafe {
                    for &i in &permutation {
                        let v = &list[i];
                        dot_avx(&query, v);
                    }
                });
            });
        }
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = encode_bench
}

criterion_main!(benches);
