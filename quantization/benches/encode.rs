use criterion::{criterion_group, criterion_main, Criterion};
use permutation_iterator::Permutor;
use quantization::encoder::EncodedVectors;
#[cfg(target_arch = "x86_64")]
use quantization::utils::{dot_avx, dot_sse};
use rand::Rng;

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

    let i8_encoded = EncodedVectors::new(
        (0..vectors_count)
            .into_iter()
            .map(|i| &list[i * vector_dim..(i + 1) * vector_dim]),
        vectors_count,
        vector_dim,
    )
    .unwrap();

    let query: Vec<f32> = (0..vector_dim).map(|_| rng.gen()).collect();
    let encoded_query = i8_encoded.encode_query(&query);

    #[cfg(target_arch = "x86_64")]
    group.bench_function("score all u8 avx", |b| {
        b.iter(|| {
            let mut _s = 0.0;
            for i in 0..vectors_count {
                _s = i8_encoded.score_point_dot_avx(&encoded_query, i);
            }
        });
    });

    #[cfg(target_arch = "x86_64")]
    group.bench_function("score all u8 sse", |b| {
        b.iter(|| {
            let mut _s = 0.0;
            for i in 0..vectors_count {
                _s = i8_encoded.score_point_dot_sse(&encoded_query, i);
            }
        });
    });

    #[cfg(target_arch = "aarch64")]
    group.bench_function("score all u8 neon", |b| {
        b.iter(|| {
            let mut _s = 0.0;
            for i in 0..vectors_count {
                _s = i8_encoded.score_point_dot_neon(&encoded_query, i);
            }
        });
    });

    #[cfg(target_arch = "x86_64")]
    group.bench_function("score all avx", |b| {
        b.iter(|| unsafe {
            let mut _s = 0.0;
            for i in 0..vectors_count {
                _s = dot_avx(&query, &list[i * vector_dim..(i + 1) * vector_dim]);
            }
        });
    });

    #[cfg(target_arch = "x86_64")]
    group.bench_function("score all sse", |b| {
        b.iter(|| unsafe {
            let mut _s = 0.0;
            for i in 0..vectors_count {
                _s = dot_sse(&query, &list[i * vector_dim..(i + 1) * vector_dim]);
            }
        });
    });

    let permutor = Permutor::new(vectors_count as u64);
    let permutation: Vec<usize> = permutor.map(|i| i as usize).collect();

    #[cfg(target_arch = "x86_64")]
    group.bench_function("score random access u8 avx", |b| {
        b.iter(|| {
            let mut _s = 0.0;
            for &i in &permutation {
                _s = i8_encoded.score_point_dot_avx(&encoded_query, i);
            }
        });
    });

    #[cfg(target_arch = "x86_64")]
    group.bench_function("score random access u8 sse", |b| {
        let mut _s = 0.0;
        b.iter(|| {
            for &i in &permutation {
                _s = i8_encoded.score_point_dot_sse(&encoded_query, i);
            }
        });
    });

    #[cfg(target_arch = "aarch64")]
    group.bench_function("score random access u8 sse", |b| {
        let mut _s = 0.0;
        b.iter(|| {
            for &i in &permutation {
                _s = i8_encoded.score_point_dot_neon(&encoded_query, i);
            }
        });
    });

    #[cfg(target_arch = "x86_64")]
    group.bench_function("score random access avx", |b| {
        b.iter(|| unsafe {
            let mut _s = 0.0;
            for &i in &permutation {
                _s = dot_avx(&query, &list[i * vector_dim..(i + 1) * vector_dim]);
            }
        });
    });

    #[cfg(target_arch = "x86_64")]
    group.bench_function("score random access sse", |b| {
        let mut _s = 0.0;
        b.iter(|| unsafe {
            for &i in &permutation {
                _s = dot_sse(&query, &list[i * vector_dim..(i + 1) * vector_dim]);
            }
        });
    });

    #[cfg(target_arch = "x86_64")]
    group.bench_function("score random access blocks avx", |b| {
        let mut scores = vec![0.0; 16];
        b.iter(|| unsafe {
            for window in permutation.as_slice().chunks_exact(16) {
                for (i, idx) in window.iter().enumerate() {
                    scores[i] = dot_sse(&query, &list[idx * vector_dim..(idx + 1) * vector_dim]);
                }
            }
        });
    });

    #[cfg(target_arch = "x86_64")]
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

    #[cfg(target_arch = "x86_64")]
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

    #[cfg(target_arch = "x86_64")]
    group.bench_function("score random access blocks i8 sse blocked", |b| {
        let mut scores = vec![0.0; 16];
        b.iter(|| {
            for window in permutation.as_slice().chunks_exact(16) {
                i8_encoded.score_points_dot_sse(&encoded_query, window, &mut scores);
            }
        });
    });

    #[cfg(target_arch = "x86_64")]
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
