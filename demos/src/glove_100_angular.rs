mod ann_benchmark_data;

use quantization::encoder::DistanceType;
#[cfg(target_arch = "x86_64")]
use quantization::utils_avx2::dot_avx;
#[cfg(target_arch = "x86_64")]
use quantization::utils_sse::dot_sse;

#[cfg(target_arch = "aarch64")]
#[cfg(target_feature = "neon")]
use quantization::utils_neon::dot_neon;

use crate::ann_benchmark_data::AnnBenchmarkData;

const DATASETS: [(&str, &str, DistanceType); 11] = [
    (
        "test_data/glove-200-angular.hdf5",
        "http://ann-benchmarks.com/glove-200-angular.hdf5",
        DistanceType::Cosine,
    ),
    (
        "test_data/glove-100-angular.hdf5",
        "http://ann-benchmarks.com/glove-100-angular.hdf5",
        DistanceType::Cosine,
    ),
    (
        "test_data/glove-50-angular.hdf5",
        "http://ann-benchmarks.com/glove-50-angular.hdf5",
        DistanceType::Cosine,
    ),
    (
        "test_data/glove-25-angular.hdf5",
        "http://ann-benchmarks.com/glove-25-angular.hdf5",
        DistanceType::Cosine,
    ),
    (
        "test_data/DEEP1B.hdf5",
        "http://ann-benchmarks.com/deep-image-96-angular.hdf5",
        DistanceType::Cosine,
    ),
    (
        "test_data/NYTimes.hdf5",
        "http://ann-benchmarks.com/nytimes-256-angular.hdf5",
        DistanceType::Cosine,
    ),
    (
        "test_data/LastFM.hdf5",
        "http://ann-benchmarks.com/lastfm-64-dot.hdf5",
        DistanceType::Cosine,
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
    for i in 0..11 {
        let dataset = &DATASETS[i];
        let mut data = AnnBenchmarkData::new(dataset.0, dataset.1);
        if dataset.2 == DistanceType::Cosine {
            data.cosine_preprocess();
        }

        let encoded = data.encode_data(dataset.2);

        let permutor = permutation_iterator::Permutor::new(data.vectors_count as u64);
        let random_indices: Vec<usize> = permutor.map(|i| i as usize).collect();
        let linear_indices: Vec<usize> = (0..data.vectors_count).collect();

        #[cfg(target_arch = "x86_64")]
        {
            println!("Measure AVX2 linear access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                for &index in &linear_indices {
                    scores[index] =
                        unsafe { dot_avx(&query, data.vectors.row(index).as_slice().unwrap()) };
                }
            });
        }

        #[cfg(target_arch = "x86_64")]
        {
            println!("Measure Quantized AVX2 linear access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                let query_u8 = encoded.encode_query(&query);
                for &index in &linear_indices {
                    scores[index] = encoded.score_point_dot_avx(&query_u8, index);
                }
            });
        }

        #[cfg(target_arch = "x86_64")]
        {
            println!("Measure Quantized chunked AVX2 linear access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                let encoded_query = encoded.encode_query(&query);
                encoded.score_points_dot_avx(&encoded_query, &linear_indices, scores);
            });
        }

        #[cfg(target_arch = "x86_64")]
        {
            println!("Measure SSE linear access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                for &index in &linear_indices {
                    scores[index] =
                        unsafe { dot_sse(&query, data.vectors.row(index).as_slice().unwrap()) };
                }
            });
        }

        #[cfg(target_arch = "x86_64")]
        {
            println!("Measure Quantized SSE linear access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                let query_u8 = encoded.encode_query(&query);
                for &index in &linear_indices {
                    scores[index] = encoded.score_point_dot_sse(&query_u8, index);
                }
            });
        }

        #[cfg(target_arch = "x86_64")]
        {
            println!("Measure Quantized chunked SSE linear access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                let encoded_query = encoded.encode_query(&query);
                encoded.score_points_dot_sse(&encoded_query, &linear_indices, scores);
            });
        }

        #[cfg(target_arch = "aarch64")]
        #[cfg(target_feature = "neon")]
        {
            println!("Measure NEON linear access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                for &index in &linear_indices {
                    scores[index] =
                        unsafe { dot_neon(&query, data.vectors.row(index).as_slice().unwrap()) };
                }
            });
        }

        #[cfg(target_arch = "aarch64")]
        #[cfg(target_feature = "neon")]
        {
            println!("Measure Quantized NEON linear access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                let query_u8 = encoded.encode_query(&query);
                for &index in &linear_indices {
                    scores[index] = encoded.score_point_dot_neon(&query_u8, index);
                }
            });
        }

        #[cfg(target_arch = "aarch64")]
        #[cfg(target_feature = "neon")]
        {
            println!("Measure Quantized chunked NOEN linear access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                let encoded_query = encoded.encode_query(&query);
                encoded.score_points_dot_neon(&encoded_query, &linear_indices, scores);
            });
        }

        #[cfg(target_arch = "x86_64")]
        {
            println!("Measure AVX2 random access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                for &index in &random_indices {
                    scores[index] =
                        unsafe { dot_avx(&query, data.vectors.row(index).as_slice().unwrap()) };
                }
            });
        }

        #[cfg(target_arch = "x86_64")]
        {
            println!("Measure Quantized AVX2 random access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                let query_u8 = encoded.encode_query(&query);
                for &index in &random_indices {
                    scores[index] = encoded.score_point_dot_avx(&query_u8, index);
                }
            });
        }

        #[cfg(target_arch = "x86_64")]
        {
            println!("Measure Quantized chunked AVX2 random access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                let encoded_query = encoded.encode_query(&query);
                encoded.score_points_dot_avx(&encoded_query, &random_indices, scores);
            });
        }

        #[cfg(target_arch = "x86_64")]
        {
            println!("Measure SSE random access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                for &index in &random_indices {
                    scores[index] =
                        unsafe { dot_sse(&query, data.vectors.row(index).as_slice().unwrap()) };
                }
            });
        }

        #[cfg(target_arch = "x86_64")]
        {
            println!("Measure Quantized SSE random access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                let query_u8 = encoded.encode_query(&query);
                for &index in &random_indices {
                    scores[index] = encoded.score_point_dot_sse(&query_u8, index);
                }
            });
        }

        #[cfg(target_arch = "x86_64")]
        {
            println!("Measure Quantized chunked SSE random access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                let encoded_query = encoded.encode_query(&query);
                encoded.score_points_dot_sse(&encoded_query, &random_indices, scores);
            });
        }

        #[cfg(target_arch = "aarch64")]
        #[cfg(target_feature = "neon")]
        {
            println!("Measure NEON random access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                for &index in &random_indices {
                    scores[index] =
                        unsafe { dot_neon(&query, data.vectors.row(index).as_slice().unwrap()) };
                }
            });
        }

        #[cfg(target_arch = "aarch64")]
        #[cfg(target_feature = "neon")]
        {
            println!("Measure Quantized NEON random access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                let query_u8 = encoded.encode_query(&query);
                for &index in &random_indices {
                    scores[index] = encoded.score_point_dot_neon(&query_u8, index);
                }
            });
        }

        #[cfg(target_arch = "aarch64")]
        #[cfg(target_feature = "neon")]
        {
            println!("Measure Quantized chunked NEON random access");
            data.measure_scoring(data.queries_count / 10, |query, scores| {
                let encoded_query = encoded.encode_query(&query);
                encoded.score_points_dot_neon(&encoded_query, &random_indices, scores);
            });
        }

        println!("Estimate knn accuracy");
        if dataset.2 == DistanceType::Cosine {
            data.test_knn(&encoded, |x| 1.0 - x);
        } else {
            data.test_knn(&encoded, |x| x);
        }
    }
}
