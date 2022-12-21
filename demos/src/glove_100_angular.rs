mod ann_benchmark_data;
mod utils;

use crate::ann_benchmark_data::AnnBenchmarkData;

fn main() {
    let mut data = AnnBenchmarkData::new(
        "test_data/glove-100-angular.hdf5",
        "http://ann-benchmarks.com/glove-100-angular.hdf5",
    );
    data.cosine_preprocess();
    let encoded = data.encode_data();
    println!("Measure scoring direct access");
    data.measure_scoring_time(&encoded, false, data.queries_count / 10);
    println!("Measure scoring random access");
    data.measure_scoring_time(&encoded, true, data.queries_count / 10);
    println!("Estimate knn accuracy");
    data.test_knn_encoded(&encoded, |x| 1.0 - x);
}
