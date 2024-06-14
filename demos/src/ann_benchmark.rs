mod ann_benchmark_data;
mod metrics;

use quantization::encoded_vectors::{DistanceType, EncodedVectors};
use quantization::encoded_vectors_pq::{CentroidsParameters, EncodedVectorsPQ};
use quantization::{EncodedVectorsU8, VectorParameters};

#[cfg(target_arch = "x86_64")]
use crate::metrics::utils_avx2::dot_avx;
#[cfg(target_arch = "x86_64")]
use demos::metrics::utils_sse::dot_sse;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::metrics::utils_neon::dot_neon;

use crate::ann_benchmark_data::AnnBenchmarkData;

use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[clap(version, about)]
pub struct Args {
    #[clap(long, default_value = "")]
    pub dataset: String,

    #[clap(long, default_value_t = false)]
    pub bench: bool,

    #[clap(long, default_value_t = false)]
    pub bench_simd: bool,

    #[clap(long, default_value_t = false)]
    pub test_acc: bool,

    // pq or u8
    #[clap(long, default_value = "u8")]
    pub method: String,

    #[clap(long)]
    pub quantile: Option<f32>,

    #[clap(long, default_value_t = 1)]
    pub chunk_size: usize,
}

const DATASETS: [(&str, &str, DistanceType); 11] = [
    (
        "test_data/glove-200-angular.hdf5",
        "http://ann-benchmarks.com/glove-200-angular.hdf5",
        DistanceType::Dot,
    ),
    (
        "test_data/glove-100-angular.hdf5",
        "http://ann-benchmarks.com/glove-100-angular.hdf5",
        DistanceType::Dot,
    ),
    (
        "test_data/glove-50-angular.hdf5",
        "http://ann-benchmarks.com/glove-50-angular.hdf5",
        DistanceType::Dot,
    ),
    (
        "test_data/glove-25-angular.hdf5",
        "http://ann-benchmarks.com/glove-25-angular.hdf5",
        DistanceType::Dot,
    ),
    (
        "test_data/DEEP1B.hdf5",
        "http://ann-benchmarks.com/deep-image-96-angular.hdf5",
        DistanceType::Dot,
    ),
    (
        "test_data/NYTimes.hdf5",
        "http://ann-benchmarks.com/nytimes-256-angular.hdf5",
        DistanceType::Dot,
    ),
    (
        "test_data/LastFM.hdf5",
        "http://ann-benchmarks.com/lastfm-64-dot.hdf5",
        DistanceType::Dot,
    ),
    (
        "test_data/Fashion-MNIST.hdf5",
        "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
        DistanceType::L2,
    ),
    (
        "test_data/GIST.hdf5",
        "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
        DistanceType::L2,
    ),
    (
        "test_data/MNIST.hdf5",
        "http://ann-benchmarks.com/mnist-784-euclidean.hdf5",
        DistanceType::L2,
    ),
    (
        "test_data/SIFT.hdf5",
        "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
        DistanceType::L2,
    ),
];

fn main() {
    let args = Args::parse();

    for dataset in &DATASETS {
        if !dataset.0.contains(&args.dataset) {
            continue;
        }

        let mut data = AnnBenchmarkData::new(dataset.0, dataset.1);
        println!("Original data size: {}", data.vectors_count * data.dim * 4);
        if dataset.2 == DistanceType::Dot {
            data.cosine_preprocess();
        }

        let timer = std::time::Instant::now();
        let data_iter = data
            .vectors
            .rows()
            .into_iter()
            .map(|row| row.to_slice().unwrap());
        let vector_parameters = VectorParameters {
            dim: data.dim,
            count: data.vectors_count,
            distance_type: dataset.2,
            invert: false,
        };

        match args.method.as_str() {
            "pq" => {
                println!("Start encoding:");
                let encoded = EncodedVectorsPQ::encode(
                    data_iter,
                    Vec::<u8>::new(),
                    &vector_parameters,
                    CentroidsParameters::KMeans {
                        chunk_size: args.chunk_size,
                    },
                    num_cpus::get(),
                    || false,
                )
                .unwrap();
                println!("encoding time: {:?}", timer.elapsed());
                run_test(&data, &encoded, dataset.2, &args);
            }
            "u8" => {
                println!("Start encoding:");
                let encoded = EncodedVectorsU8::encode(
                    data_iter,
                    Vec::<u8>::new(),
                    &vector_parameters,
                    args.quantile,
                    || false,
                )
                .unwrap();
                println!("encoding time: {:?}", timer.elapsed());
                run_test(&data, &encoded, dataset.2, &args);
            }
            _ => panic!("Unknown method"),
        };
    }
}

fn run_test<TEncodedQuery, TEncodedVectors: EncodedVectors<TEncodedQuery>>(
    data: &AnnBenchmarkData,
    encoded: &TEncodedVectors,
    distance_type: DistanceType,
    args: &Args,
) {
    let permutor = permutation_iterator::Permutor::new(data.vectors_count as u64);

    if args.test_acc {
        println!("Estimate knn accuracy");
        if distance_type == DistanceType::Dot {
            data.test_knn(encoded, |x| 1.0 - x);
        } else {
            data.test_knn(encoded, |x| x);
        }
    }

    if !args.bench && !args.bench_simd {
        return;
    }

    let queries_count = data.queries_count / 10;
    let random_indices: Vec<u32> = permutor.map(|i| i as u32).collect();
    let linear_indices: Vec<u32> = (0..data.vectors_count as u32).collect();

    #[cfg(target_arch = "x86_64")]
    if args.bench_simd {
        println!("Measure AVX2 linear access");
        data.measure_scoring(queries_count, |query, scores| {
            for &index in &linear_indices {
                scores[index as usize] =
                    unsafe { dot_avx(query, data.vectors.row(index as usize).as_slice().unwrap()) };
            }
        });

        println!("Measure SSE linear access");
        data.measure_scoring(queries_count, |query, scores| {
            for &index in &linear_indices {
                scores[index as usize] =
                    unsafe { dot_sse(query, data.vectors.row(index as usize).as_slice().unwrap()) };
            }
        });

        println!("Measure AVX2 random access");
        data.measure_scoring(queries_count, |query, scores| {
            for &index in &random_indices {
                scores[index as usize] =
                    unsafe { dot_avx(query, data.vectors.row(index as usize).as_slice().unwrap()) };
            }
        });

        println!("Measure SSE random access");
        data.measure_scoring(queries_count, |query, scores| {
            for &index in &random_indices {
                scores[index as usize] =
                    unsafe { dot_sse(query, data.vectors.row(index as usize).as_slice().unwrap()) };
            }
        });
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    if args.bench_simd {
        println!("Measure NEON linear access");
        data.measure_scoring(queries_count, |query, scores| {
            for &index in &linear_indices {
                scores[index as usize] = unsafe {
                    dot_neon(query, data.vectors.row(index as usize).as_slice().unwrap())
                };
            }
        });

        println!("Measure NEON random access");
        data.measure_scoring(queries_count, |query, scores| {
            for &index in &random_indices {
                scores[index as usize] = unsafe {
                    dot_neon(query, data.vectors.row(index as usize).as_slice().unwrap())
                };
            }
        });
    }

    if args.bench {
        println!("Measure Quantized linear access");
        data.measure_scoring(queries_count, |query, scores| {
            let query_u8 = encoded.encode_query(query);
            for &index in &linear_indices {
                scores[index as usize] = encoded.score_point(&query_u8, index);
            }
        });

        println!("Measure Quantized random access");
        data.measure_scoring(queries_count, |query, scores| {
            let query_u8 = encoded.encode_query(query);
            for &index in &random_indices {
                scores[index as usize] = encoded.score_point(&query_u8, index);
            }
        });
    }
}
