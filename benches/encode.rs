use criterion::{criterion_group, criterion_main, Criterion};
use permutation_iterator::Permutor;
use quantization::{encoder::EncodedVectorStorage, lut::Lut};
use rand::Rng;

fn serde_formats_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode");

    let vectors_count = 10_000;
    let vector_dim = 64;
    let mut rng = rand::thread_rng();
    let mut list: Vec<Vec<f32>> = Vec::new();
    for _ in 0..vectors_count {
        let vector: Vec<f32> = (0..vector_dim).map(|_| rng.gen()).collect();
        list.push(vector);
    }

    let chunks = EncodedVectorStorage::divide_dim(vector_dim, 2);

    let metric = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f32>();
    group.bench_function("encode", |b| {
        b.iter(|| {
            EncodedVectorStorage::new(Box::new(list.iter().map(|v| v.as_slice())), &chunks, metric)
        });
    });

    let encoder =
        EncodedVectorStorage::new(Box::new(list.iter().map(|v| v.as_slice())), &chunks, metric)
            .unwrap();
    let metric = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f32>();
    let query: Vec<f32> = (0..vector_dim).map(|_| rng.gen()).collect();
    let lut = Lut::new(&encoder, &query, metric);
    group.bench_function("score all quantized", |b| {
        b.iter(|| {
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

    let permutor = Permutor::new(vectors_count as u64);
    let permutation: Vec<usize> = permutor.map(|i| i as usize).collect();

    group.bench_function("score random access quantized", |b| {
        b.iter(|| {
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
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = serde_formats_bench
}

criterion_main!(benches);
