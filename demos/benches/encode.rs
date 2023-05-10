use criterion::{criterion_group, criterion_main, Criterion};
use permutation_iterator::Permutor;
use quantization::encoded_vectors::{DistanceType, EncodedVectors, VectorParameters};
use quantization::encoded_vectors_u8::EncodedVectorsU8;
use rand::Rng;

#[cfg(target_arch = "x86_64")]
use demos::metrics::utils_avx2::dot_avx;

#[cfg(target_arch = "x86_64")]
use demos::metrics::utils_sse::dot_sse;

fn encode_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode");

    let vectors_count = 100_000;
    let vector_dim = 1024;
    let mut rng = rand::thread_rng();
    let mut list: Vec<f32> = Vec::new();
    for _ in 0..vectors_count {
        let vector: Vec<f32> = (0..vector_dim).map(|_| rng.gen()).collect();
        list.extend_from_slice(&vector);
    }

    let i8_encoded = EncodedVectorsU8::encode(
        (0..vectors_count).map(|i| &list[i * vector_dim..(i + 1) * vector_dim]),
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

    let query: Vec<f32> = (0..vector_dim).map(|_| rng.gen()).collect();
    let encoded_query = i8_encoded.encode_query(&query);

    #[cfg(target_arch = "x86_64")]
    group.bench_function("score all u8 avx", |b| {
        b.iter(|| {
            let mut _s = 0.0;
            for i in 0..vectors_count as u32 {
                _s = i8_encoded.score_point_avx(&encoded_query, i);
            }
        });
    });

    #[cfg(target_arch = "x86_64")]
    group.bench_function("score all u8 sse", |b| {
        b.iter(|| {
            let mut _s = 0.0;
            for i in 0..vectors_count as u32 {
                _s = i8_encoded.score_point_sse(&encoded_query, i);
            }
        });
    });

    #[cfg(target_arch = "aarch64")]
    group.bench_function("score all u8 neon", |b| {
        b.iter(|| {
            let mut _s = 0.0;
            for i in 0..vectors_count as u32 {
                _s = i8_encoded.score_point_neon(&encoded_query, i);
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
    let permutation: Vec<u32> = permutor.map(|i| i as u32).collect();

    #[cfg(target_arch = "x86_64")]
    group.bench_function("score random access u8 avx", |b| {
        b.iter(|| {
            let mut _s = 0.0;
            for &i in &permutation {
                _s = i8_encoded.score_point_avx(&encoded_query, i);
            }
        });
    });

    #[cfg(target_arch = "x86_64")]
    group.bench_function("score random access u8 sse", |b| {
        let mut _s = 0.0;
        b.iter(|| {
            for &i in &permutation {
                _s = i8_encoded.score_point_sse(&encoded_query, i);
            }
        });
    });

    #[cfg(target_arch = "aarch64")]
    group.bench_function("score random access u8 neon", |b| {
        let mut _s = 0.0;
        b.iter(|| {
            for &i in &permutation {
                _s = i8_encoded.score_point_neon(&encoded_query, i);
            }
        });
    });

    #[cfg(target_arch = "x86_64")]
    group.bench_function("score random access avx", |b| {
        b.iter(|| unsafe {
            let mut _s = 0.0;
            for &i in &permutation {
                let i = i as usize;
                _s = dot_avx(&query, &list[i * vector_dim..(i + 1) * vector_dim]);
            }
        });
    });

    #[cfg(target_arch = "x86_64")]
    group.bench_function("score random access sse", |b| {
        let mut _s = 0.0;
        b.iter(|| unsafe {
            for &i in &permutation {
                let i = i as usize;
                _s = dot_sse(&query, &list[i * vector_dim..(i + 1) * vector_dim]);
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
