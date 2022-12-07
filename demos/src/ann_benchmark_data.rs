use std::collections::BinaryHeap;

use crate::utils::{cosine_preprocess, same_count, Score};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use quantization::{encoded_vectors::EncodedVectors, lut::Lut};

pub struct AnnBenchmarkData {
    pub dim: usize,
    pub vectors: ndarray::Array2<f32>,
    pub vectors_count: usize,
    pub queries: ndarray::Array2<f32>,
    pub queries_count: usize,
    pub neighbors: Vec<Vec<Score>>,
}

impl AnnBenchmarkData {
    pub fn new(filename: &str, url: &str) -> Self {
        println!("Test {}", url);
        if !std::path::Path::new(filename).exists() {
            Self::download_file(filename, url);
        }
        let dataset = hdf5::File::open(filename).unwrap();
        let vectors_dataset = dataset.dataset("train").unwrap();
        let vectors = vectors_dataset.read_2d::<f32>().unwrap();
        let vectors_count = vectors.shape()[0];
        let dim = vectors.shape()[1];

        let query_dataset = dataset.dataset("test").unwrap();
        let queries = query_dataset.read_2d::<f32>().unwrap();
        let queries_count = queries.shape()[0];

        let neighbors_dataset = dataset.dataset("neighbors").unwrap();
        let neighbors = neighbors_dataset.read_2d::<usize>().unwrap();

        let neighbor_distances_dataset = dataset.dataset("distances").unwrap();
        let neighbor_distances = neighbor_distances_dataset.read_2d::<f32>().unwrap();

        let neighbors = neighbors
            .rows()
            .into_iter()
            .zip(neighbor_distances.rows().into_iter())
            .map(|(neighbors, neighbor_distances)| {
                neighbors
                    .iter()
                    .take(10)
                    .zip(neighbor_distances.iter().take(10))
                    .map(|(index, distance)| Score {
                        index: *index,
                        score: *distance,
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        Self {
            dim,
            vectors,
            vectors_count,
            queries,
            queries_count,
            neighbors,
        }
    }

    pub fn cosine_preprocess(&mut self) {
        self.vectors.rows_mut().into_iter().for_each(|mut row| {
            cosine_preprocess(row.as_slice_mut().unwrap());
        });
        self.queries.rows_mut().into_iter().for_each(|mut row| {
            cosine_preprocess(row.as_slice_mut().unwrap());
        });
    }

    pub fn encode_data(&self, chunk_size: usize) -> EncodedVectors {
        println!("Start encoding:");
        let timer = std::time::Instant::now();
        let chunks = EncodedVectors::divide_dim(self.dim, chunk_size);
        let encoded_data = EncodedVectors::new(
            self.vectors
                .rows()
                .into_iter()
                .map(|row| row.to_slice().unwrap()),
            self.vectors_count,
            self.dim,
            &chunks,
        )
        .unwrap();
        println!("encoding time: {:?}", timer.elapsed());
        println!("Original data size: {}", self.vectors_count * self.dim * 4);
        println!("Encoded data size: {}", encoded_data.data_size());
        encoded_data
    }

    pub fn test_encoded(&self, encoded: &EncodedVectors, similarity: fn(&[f32], &[f32]) -> f32) {
        let multiprogress = MultiProgress::new();
        let sent_bar = multiprogress.add(ProgressBar::new(self.queries_count as u64));
        let progress_style = ProgressStyle::default_bar()
            .template("{msg} [{elapsed_precise}] {wide_bar} [{per_sec:>3}] {pos}/{len} (eta:{eta})")
            .expect("Failed to create progress style");
        sent_bar.set_style(progress_style);

        let mut same_10 = 0.0;
        let mut same_30 = 0.0;
        let mut same_100 = 0.0;
        let mut timings = Vec::new();
        for (j, query) in self
            .queries
            .rows()
            .into_iter()
            .map(|q| q.to_slice().unwrap())
            .enumerate()
        {
            let timer = std::time::Instant::now();
            let lut = Lut::new(&encoded, query, similarity);
            let mut heap: BinaryHeap<Score> = BinaryHeap::new();
            for index in 0..self.vectors_count {
                let encoded_vector = encoded.get(index);
                let score = 1.0 - lut.dist(encoded_vector);
                let score = Score { index, score };
                if heap.len() == 100 {
                    let mut top = heap.peek_mut().unwrap();
                    if top.score > score.score {
                        *top = score;
                    }
                } else {
                    heap.push(score);
                }
            }
            let knn = heap.into_sorted_vec();
            same_10 += same_count(&knn[0..10], &self.neighbors[j]) as f32;
            same_30 += same_count(&knn[0..30], &self.neighbors[j]) as f32;
            same_100 += same_count(&knn, &self.neighbors[j]) as f32;

            timings.push(timer.elapsed().as_millis() as f64);
            sent_bar.inc(1);
        }
        sent_bar.finish();

        same_10 /= self.queries_count as f32;
        same_30 /= self.queries_count as f32;
        same_100 /= self.queries_count as f32;
        println!("queries count: {}", self.queries_count);
        println!("same_10: {}", same_10);
        println!("same_30: {}", same_30);
        println!("same_100: {}", same_100);
        Self::print_timings(&mut timings);
    }

    fn download_file(name: &str, url: &str) {
        let path = std::path::Path::new(name);
        if !path.exists() {
            let dir = path.parent().unwrap();
            if !dir.is_dir() {
                std::fs::create_dir_all(dir).unwrap();
            }
            println!("Start downloading from {}", url);
            let mut resp = reqwest::blocking::get(url).unwrap();
            let mut file = std::fs::File::create(name).unwrap();
            std::io::copy(&mut resp, &mut file).unwrap();
            println!("File {} created", name);
        }
    }

    fn print_timings(timings: &mut Vec<f64>) {
        if timings.is_empty() {
            return;
        }
        // sort timings in ascending order
        timings.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let avg_time: f64 = timings.iter().sum::<f64>() / timings.len() as f64;
        let min_time: f64 = timings.first().copied().unwrap_or(0.0);
        let max_time: f64 = timings.last().copied().unwrap_or(0.0);
        let p95_time: f64 = timings[(timings.len() as f32 * 0.95) as usize];
        let p99_time: f64 = timings[(timings.len() as f32 * 0.99) as usize];

        println!("Min search time: {}ms", min_time);
        println!("Avg search time: {}ms", avg_time);
        println!("p95 search time: {}ms", p95_time);
        println!("p99 search time: {}ms", p99_time);
        println!("Max search time: {}ms", max_time);
    }
}
