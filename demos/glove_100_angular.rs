mod utils;
mod ann_benchmark_data;

use crate::ann_benchmark_data::AnnBenchmarkData;

fn main() {
    let mut data = AnnBenchmarkData::new(
        "test_data/glove-100-angular.hdf5",
        "http://ann-benchmarks.com/glove-100-angular.hdf5",
    );
    data.cosine_preprocess();
    let encoded = data.encode_data();
    data.test_encoded(&encoded, utils::dot_similarity);
}
