mod ann_benchmark_data;

use quantization::encoder::EncodedVectors;
use quantization::utils::{dot_avx, dot_sse};

use crate::ann_benchmark_data::AnnBenchmarkData;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceType {
    Euclid,
    Cosine,
}

const DATASETS: [(&str, &str, DistanceType); 7] = [
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
];

fn main() {
    let dataset = &DATASETS[0];
    let mut data = AnnBenchmarkData::new(dataset.0, dataset.1);
    if dataset.2 == DistanceType::Cosine {
        data.cosine_preprocess();
    }

    let encoded = data.encode_data();

    let permutor = permutation_iterator::Permutor::new(data.vectors_count as u64);
    let random_indices: Vec<usize> = permutor.map(|i| i as usize).collect();
    let linear_indices: Vec<usize> = (0..data.vectors_count).collect();

    println!("Measure AVX2 linear access");
    data.measure_scoring(
        data.queries_count / 10,
        |query, scores| {
            for &index in &linear_indices {
                scores[index] = unsafe { dot_avx(&query, data.vectors.row(index).as_slice().unwrap()) };
            }
        }
    );

    println!("Measure Quantized AVX2 linear access");
    data.measure_scoring(
        data.queries_count / 10,
        |query, scores| {
            let query_u8 = EncodedVectors::encode_query(&query);
            for &index in &linear_indices {
                scores[index] = encoded.score_point_dot_avx(&query_u8, index);
            }
        }
    );

    println!("Measure Quantized chunked AVX2 linear access");
    data.measure_scoring(
        data.queries_count / 10,
        |query, scores| {
            let encoded_query = EncodedVectors::encode_query(&query);
            encoded.score_points_dot_avx(&encoded_query, &linear_indices, scores);
        }
    );

    println!("Measure SSE linear access");
    data.measure_scoring(
        data.queries_count / 10,
        |query, scores| {
            for &index in &linear_indices {
                scores[index] = unsafe { dot_sse(&query, data.vectors.row(index).as_slice().unwrap()) };
            }
        }
    );

    println!("Measure Quantized SSE linear access");
    data.measure_scoring(
        data.queries_count / 10,
        |query, scores| {
            let query_u8 = EncodedVectors::encode_query(&query);
            for &index in &linear_indices {
                scores[index] = encoded.score_point_dot_sse(&query_u8, index);
            }
        }
    );

    println!("Measure Quantized chunked SSE linear access");
    data.measure_scoring(
        data.queries_count / 10,
        |query, scores| {
            let encoded_query = EncodedVectors::encode_query(&query);
            encoded.score_points_dot_sse(&encoded_query, &linear_indices, scores);
        }
    );

    println!("Measure AVX2 random access");
    data.measure_scoring(
        data.queries_count / 10,
        |query, scores| {
            for &index in &random_indices {
                scores[index] = unsafe { dot_avx(&query, data.vectors.row(index).as_slice().unwrap()) };
            }
        }
    );

    println!("Measure Quantized AVX2 random access");
    data.measure_scoring(
        data.queries_count / 10,
        |query, scores| {
            let query_u8 = EncodedVectors::encode_query(&query);
            for &index in &random_indices {
                scores[index] = encoded.score_point_dot_avx(&query_u8, index);
            }
        }
    );

    println!("Measure Quantized chunked AVX2 random access");
    data.measure_scoring(
        data.queries_count / 10,
        |query, scores| {
            let encoded_query = EncodedVectors::encode_query(&query);
            encoded.score_points_dot_avx(&encoded_query, &random_indices, scores);
        }
    );

    println!("Measure SSE random access");
    data.measure_scoring(
        data.queries_count / 10,
        |query, scores| {
            for &index in &random_indices {
                scores[index] = unsafe { dot_sse(&query, data.vectors.row(index).as_slice().unwrap()) };
            }
        }
    );

    println!("Measure Quantized SSE random access");
    data.measure_scoring(
        data.queries_count / 10,
        |query, scores| {
            let query_u8 = EncodedVectors::encode_query(&query);
            for &index in &random_indices {
                scores[index] = encoded.score_point_dot_sse(&query_u8, index);
            }
        }
    );

    println!("Measure Quantized chunked SSE random access");
    data.measure_scoring(
        data.queries_count / 10,
        |query, scores| {
            let encoded_query = EncodedVectors::encode_query(&query);
            encoded.score_points_dot_sse(&encoded_query, &random_indices, scores);
        }
    );

    println!("Estimate knn accuracy");
    data.test_knn(&encoded, |x| 1.0 - x);
}
