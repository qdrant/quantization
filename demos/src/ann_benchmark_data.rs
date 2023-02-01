use std::collections::{BinaryHeap, HashSet};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use quantization::encoder::{EncodedVectors, EncodingParameters, SimilarityType};

pub struct AnnBenchmarkData {
    pub dim: usize,
    pub vectors: ndarray::Array2<f32>,
    pub vectors_count: usize,
    pub queries: ndarray::Array2<f32>,
    pub queries_count: usize,
    pub neighbors: Vec<Vec<Score>>,
}

#[derive(PartialEq, Clone, Debug, Default)]
pub struct Score {
    pub index: usize,
    pub score: f32,
}

impl Eq for Score {}

impl std::cmp::PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::cmp::Ord for Score {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.partial_cmp(&other.score).unwrap()
    }
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

    pub fn encode_data(
        &self,
        distance_type: SimilarityType,
        confidence_level: Option<f32>,
    ) -> EncodedVectors<Vec<u8>> {
        println!("Start encoding:");
        let timer = std::time::Instant::now();
        let encoded_data = EncodedVectors::encode(
            self.vectors
                .rows()
                .into_iter()
                .map(|row| row.to_slice().unwrap()),
            Vec::<u8>::new(),
            EncodingParameters {
                distance_type,
                confidence_level,
                ..Default::default()
            },
        )
        .unwrap();
        println!("encoding time: {:?}", timer.elapsed());
        println!("Original data size: {}", self.vectors_count * self.dim * 4);
        //println!("Encoded data size: {}", encoded_data.data_size());
        encoded_data
    }

    pub fn measure_scoring<F>(&self, queries_count: usize, query_function: F)
    where
        F: Fn(&[f32], &mut [f32]),
    {
        let multiprogress = MultiProgress::new();
        let sent_bar = multiprogress.add(ProgressBar::new(queries_count as u64));
        let progress_style = ProgressStyle::default_bar()
            .template("{msg} [{elapsed_precise}] {wide_bar} [{per_sec:>3}] {pos}/{len} (eta:{eta})")
            .expect("Failed to create progress style");
        sent_bar.set_style(progress_style);

        let mut scores = vec![0.0; self.vectors_count];
        let mut timings = Vec::new();

        for query in self
            .queries
            .rows()
            .into_iter()
            .take(queries_count)
            .map(|q| q.to_slice().unwrap())
        {
            let timer = std::time::Instant::now();
            query_function(query, &mut scores);
            timings.push(timer.elapsed().as_millis() as f64);
            sent_bar.inc(1);
        }
        sent_bar.finish();
        Self::print_timings(&mut timings);
    }

    pub fn test_knn<F>(&self, encoded: &EncodedVectors<Vec<u8>>, postprocess: F)
    where
        F: Fn(f32) -> f32,
    {
        let multiprogress = MultiProgress::new();
        let sent_bar = multiprogress.add(ProgressBar::new(self.queries_count as u64));
        let progress_style = ProgressStyle::default_bar()
            .template("{msg} [{elapsed_precise}] {wide_bar} [{per_sec:>3}] {pos}/{len} (eta:{eta})")
            .expect("Failed to create progress style");
        sent_bar.set_style(progress_style);

        let mut same_10 = 0.0;
        let mut same_20 = 0.0;
        let mut same_30 = 0.0;
        let mut timings = Vec::new();
        for (j, query) in self
            .queries
            .rows()
            .into_iter()
            .map(|q| q.to_slice().unwrap())
            .enumerate()
        {
            let timer = std::time::Instant::now();
            let query_u8 = encoded.encode_query(query);
            let mut heap: BinaryHeap<Score> = BinaryHeap::new();
            for index in 0..self.vectors_count as u32 {
                let score = postprocess(encoded.score_point(&query_u8, index));
                let score = Score {
                    index: index as usize,
                    score,
                };
                if heap.len() == 30 {
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
            same_20 += same_count(&knn[0..20], &self.neighbors[j]) as f32;
            same_30 += same_count(&knn, &self.neighbors[j]) as f32;

            timings.push(timer.elapsed().as_millis() as f64);
            sent_bar.inc(1);
        }
        sent_bar.finish();

        same_10 /= self.queries_count as f32;
        same_20 /= self.queries_count as f32;
        same_30 /= self.queries_count as f32;
        println!("queries count: {}", self.queries_count);
        println!("same_10: {}", same_10);
        println!("same_20: {}", same_20);
        println!("same_30: {}", same_30);
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

pub fn cosine_preprocess(vector: &mut [f32]) {
    let mut length: f32 = vector.iter().map(|x| x * x).sum();
    if length < f32::EPSILON {
        return;
    }
    length = length.sqrt();
    vector.iter_mut().for_each(|x| *x /= length);
}

pub fn same_count(a: &[Score], b: &[Score]) -> usize {
    let a = a.iter().map(|s| s.index).collect::<HashSet<_>>();
    let b = b.iter().map(|s| s.index).collect::<HashSet<_>>();
    a.intersection(&b).count()
}
