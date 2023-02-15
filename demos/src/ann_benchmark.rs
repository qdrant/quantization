mod ann_benchmark_data;
mod metrics;

use quantization::encoded_vectors::{EncodedVectors, SimilarityType};

#[cfg(target_arch = "x86_64")]
use crate::metrics::utils_avx2::dot_avx;

#[cfg(target_arch = "x86_64")]
use crate::metrics::utils_sse::dot_sse;

#[cfg(target_arch = "aarch64")]
#[cfg(target_feature = "neon")]
use crate::metrics::utils_neon::dot_neon;

use crate::ann_benchmark_data::AnnBenchmarkData;

const DATASETS: [(&str, &str, SimilarityType); 11] = [
    (
        "test_data/glove-200-angular.hdf5",
        "http://ann-benchmarks.com/glove-200-angular.hdf5",
        SimilarityType::Dot,
    ),
    (
        "test_data/glove-100-angular.hdf5",
        "http://ann-benchmarks.com/glove-100-angular.hdf5",
        SimilarityType::Dot,
    ),
    (
        "test_data/glove-50-angular.hdf5",
        "http://ann-benchmarks.com/glove-50-angular.hdf5",
        SimilarityType::Dot,
    ),
    (
        "test_data/glove-25-angular.hdf5",
        "http://ann-benchmarks.com/glove-25-angular.hdf5",
        SimilarityType::Dot,
    ),
    (
        "test_data/DEEP1B.hdf5",
        "http://ann-benchmarks.com/deep-image-96-angular.hdf5",
        SimilarityType::Dot,
    ),
    (
        "test_data/NYTimes.hdf5",
        "http://ann-benchmarks.com/nytimes-256-angular.hdf5",
        SimilarityType::Dot,
    ),
    (
        "test_data/LastFM.hdf5",
        "http://ann-benchmarks.com/lastfm-64-dot.hdf5",
        SimilarityType::Dot,
    ),
    (
        "test_data/Fashion-MNIST.hdf5",
        "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
        SimilarityType::L2,
    ),
    (
        "test_data/GIST.hdf5",
        "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
        SimilarityType::L2,
    ),
    (
        "test_data/MNIST.hdf5",
        "http://ann-benchmarks.com/mnist-784-euclidean.hdf5",
        SimilarityType::L2,
    ),
    (
        "test_data/SIFT.hdf5",
        "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
        SimilarityType::L2,
    ),
];

fn main() {
    for dataset in &DATASETS {
        let mut data = AnnBenchmarkData::new(dataset.0, dataset.1);
        if dataset.2 == SimilarityType::Dot {
            data.cosine_preprocess();
        }

        let encoded = data.encode_data(dataset.2, None);

        let permutor = permutation_iterator::Permutor::new(data.vectors_count as u64);
        let random_indices: Vec<u32> = permutor.map(|i| i as u32).collect();
        let linear_indices: Vec<u32> = (0..data.vectors_count as u32).collect();

        #[cfg(target_arch = "x86_64")]
        {
            println!("Measure AVX2 linear access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                for &index in &linear_indices {
                    scores[index as usize] = unsafe {
                        dot_avx(query, data.vectors.row(index as usize).as_slice().unwrap())
                    };
                }
            });
        }

        #[cfg(target_arch = "x86_64")]
        {
            println!("Measure Quantized AVX2 linear access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                let query_u8 = encoded.encode_query(query);
                for &index in &linear_indices {
                    scores[index as usize] = encoded.score_point_avx(&query_u8, index);
                }
            });
        }

        #[cfg(target_arch = "x86_64")]
        {
            println!("Measure SSE linear access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                for &index in &linear_indices {
                    scores[index as usize] = unsafe {
                        dot_sse(query, data.vectors.row(index as usize).as_slice().unwrap())
                    };
                }
            });
        }

        #[cfg(target_arch = "x86_64")]
        {
            println!("Measure Quantized SSE linear access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                let query_u8 = encoded.encode_query(query);
                for &index in &linear_indices {
                    scores[index as usize] = encoded.score_point_sse(&query_u8, index);
                }
            });
        }

        #[cfg(target_arch = "aarch64")]
        #[cfg(target_feature = "neon")]
        {
            println!("Measure NEON linear access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                for &index in &linear_indices {
                    scores[index as usize] = unsafe {
                        dot_neon(query, data.vectors.row(index as usize).as_slice().unwrap())
                    };
                }
            });
        }

        #[cfg(target_arch = "aarch64")]
        #[cfg(target_feature = "neon")]
        {
            println!("Measure Quantized NEON linear access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                let query_u8 = encoded.encode_query(query);
                for &index in &linear_indices {
                    scores[index as usize] = encoded.score_point_neon(&query_u8, index);
                }
            });
        }

        #[cfg(target_arch = "x86_64")]
        {
            println!("Measure AVX2 random access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                for &index in &random_indices {
                    scores[index as usize] = unsafe {
                        dot_avx(query, data.vectors.row(index as usize).as_slice().unwrap())
                    };
                }
            });
        }

        #[cfg(target_arch = "x86_64")]
        {
            println!("Measure Quantized AVX2 random access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                let query_u8 = encoded.encode_query(query);
                for &index in &random_indices {
                    scores[index as usize] = encoded.score_point_avx(&query_u8, index);
                }
            });
        }

        #[cfg(target_arch = "x86_64")]
        {
            println!("Measure SSE random access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                for &index in &random_indices {
                    scores[index as usize] = unsafe {
                        dot_sse(query, data.vectors.row(index as usize).as_slice().unwrap())
                    };
                }
            });
        }

        #[cfg(target_arch = "x86_64")]
        {
            println!("Measure Quantized SSE random access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                let query_u8 = encoded.encode_query(query);
                for &index in &random_indices {
                    scores[index as usize] = encoded.score_point_sse(&query_u8, index);
                }
            });
        }

        #[cfg(target_arch = "aarch64")]
        #[cfg(target_feature = "neon")]
        {
            println!("Measure NEON random access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                for &index in &random_indices {
                    scores[index as usize] = unsafe {
                        dot_neon(query, data.vectors.row(index as usize).as_slice().unwrap())
                    };
                }
            });
        }

        #[cfg(target_arch = "aarch64")]
        #[cfg(target_feature = "neon")]
        {
            println!("Measure Quantized NEON random access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                let query_u8 = encoded.encode_query(query);
                for &index in &random_indices {
                    scores[index as usize] = encoded.score_point_neon(&query_u8, index);
                }
            });
        }

        println!("Estimate knn accuracy");
        if dataset.2 == SimilarityType::Dot {
            data.test_knn(&encoded, |x| 1.0 - x);
        } else {
            data.test_knn(&encoded, |x| x);
        }
    }
}
