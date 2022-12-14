use criterion::{criterion_group, criterion_main, Criterion};
use permutation_iterator::Permutor;
use quantization::{
    encoded_vectors::EncodedVectors, lut16_2_scorer, lut16_scorer, scorer::Scorer,
    simple_scorer::SimpleScorer, sse_scorer::SseScorer,
};
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

fn encode_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode");

    let vectors_count = 320_000;
    const DIM: usize = 16;
    let mut rng = rand::thread_rng();
    let mut list: Vec<f32> = Vec::new();
    for _ in 0..vectors_count {
        let vector: Vec<f32> = (0..DIM).map(|_| rng.gen()).collect();
        list.extend_from_slice(&vector);
    }

    let chunks = EncodedVectors::create_dim_partition(DIM, 1);

    let encoder = EncodedVectors::new(
        (0..vectors_count)
            .into_iter()
            .map(|i| &list[DIM * i..DIM * (i + 1)]),
        vectors_count,
        DIM,
        &chunks,
    )
    .unwrap();
    let metric = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f32>();
    let query: Vec<f32> = (0..DIM).map(|_| rng.gen()).collect();

    let permutor = Permutor::new(vectors_count as u64);
    let permutation: Vec<usize> = permutor.map(|i| i as usize).collect();

    const WINDOW_SIZE: usize = 32;
    group.bench_function("Quantized simple bucket score", |b| {
        let mut scores = vec![0.0f32; WINDOW_SIZE];
        b.iter(|| {
            let mut scorer: SimpleScorer = encoder.scorer(&query, metric);
            for window in permutation.as_slice().chunks_exact(WINDOW_SIZE) {
                scorer.score_points(window, &mut scores);
            }
        });
    });

    /*
    group.bench_function("Quantized SSE one-by-one bucket score", |b| {
        let mut scores = vec![0.0f32; WINDOW_SIZE];
        b.iter(|| {
            let scorer: SseScorer = encoder.scorer(&query, metric);
            for window in permutation.as_slice().chunks_exact(WINDOW_SIZE) {
                for i in 0..WINDOW_SIZE {
                    scores[i] = scorer.score_point(window[i]);
                }
            }
        });
    });
    */

    group.bench_function("Quantized SSE bucket score", |b| {
        let mut scores = vec![0.0f32; WINDOW_SIZE];
        b.iter(|| {
            let mut scorer: SseScorer = encoder.scorer(&query, metric);
            for window in permutation.as_slice().chunks_exact(WINDOW_SIZE) {
                scorer.score_points(window, &mut scores);
            }
        });
    });

    group.bench_function("AVX bucket score", |b| {
        let mut scores = vec![0.0f32; WINDOW_SIZE];
        b.iter(|| unsafe {
            for window in permutation.as_slice().chunks_exact(WINDOW_SIZE) {
                for (i, &v) in window.iter().enumerate() {
                    scores[i] = dot_avx(&query, &list[DIM * v..DIM * (v + 1)]);
                }
            }
        });
    });

    group.bench_function("Original bucket score", |b| {
        let mut scores = vec![0.0f32; WINDOW_SIZE];
        b.iter(|| {
            for window in permutation.as_slice().chunks_exact(WINDOW_SIZE) {
                for (i, &v) in window.iter().enumerate() {
                    scores[i] = 0.0;
                    for j in 0..DIM {
                        scores[i] += query[j] * list[DIM * v + j];
                    }
                }
            }
        });
    });

    #[cfg(target_arch = "x86_64")]
    {
        group.bench_function("score random access quantized LUT16", |b| {
            b.iter(|| {
                let scorer: lut16_2_scorer::SseScorer = encoder.scorer(&query, metric);
                for &i in &permutation {
                    scorer.score_point(i);
                }
            });
        });
    }

    group.bench_function("score all quantized", |b| {
        b.iter(|| {
            let scorer: SimpleScorer = encoder.scorer(&query, metric);
            for i in 0..vectors_count {
                scorer.score_point(i);
            }
        });
    });

    #[cfg(target_arch = "x86_64")]
    {
        group.bench_function("score all quantized SSE", |b| {
            b.iter(|| {
                let scorer: SseScorer = encoder.scorer(&query, metric);
                for i in 0..vectors_count {
                    scorer.score_point(i);
                }
            });
        });
    }

    group.bench_function("score all original", |b| {
        b.iter(|| {
            for i in 0..vectors_count {
                let mut _sum = 0.0;
                for j in 0..DIM {
                    _sum += query[j] * list[DIM * i + j];
                }
            }
        });
    });

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
            group.bench_function("score all avx", |b| {
                b.iter(|| unsafe {
                    for i in 0..vectors_count {
                        dot_avx(&query, &list[DIM * i..DIM * (i + 1)]);
                    }
                });
            });
        }
    }

    group.bench_function("score random access quantized", |b| {
        b.iter(|| {
            let scorer: SimpleScorer = encoder.scorer(&query, metric);
            for &i in &permutation {
                scorer.score_point(i);
            }
        });
    });

    #[cfg(target_arch = "x86_64")]
    {
        group.bench_function("score random access quantized SSE", |b| {
            b.iter(|| {
                let scorer: SseScorer = encoder.scorer(&query, metric);
                for &i in &permutation {
                    scorer.score_point(i);
                }
            });
        });
    }

    group.bench_function("score random access original", |b| {
        b.iter(|| {
            for &i in &permutation {
                let mut _sum = 0.0;
                for j in 0..DIM {
                    _sum += query[j] * list[DIM * i + j];
                }
            }
        });
    });

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
            group.bench_function("score random access avx", |b| {
                b.iter(|| unsafe {
                    for &i in &permutation {
                        dot_avx(&query, &list[DIM * i..DIM * (i + 1)]);
                    }
                });
            });
        }
    }

    group.bench_function("copy mem", |b| {
        let mut mem = vec![0u8; 16 * DIM];
        let mut mem_t = vec![0u8; 16 * DIM];
        b.iter(|| unsafe {
            for window in permutation.as_slice().chunks_exact(16) {
                for (j, &i) in window.iter().enumerate() {
                    let v = encoder.get(i);
                    std::ptr::copy(
                        encoder.get(i).as_ptr(),
                        mem.as_mut_ptr().add(j * DIM / 2),
                        v.len(),
                    );
                }
                for i in 0..DIM {
                    for j in 0..16 {
                        mem_t[16 * i + j] = mem[DIM * j + i];
                    }
                }
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
