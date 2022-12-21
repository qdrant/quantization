use std::collections::BinaryHeap;

use crate::utils::{cosine_preprocess, same_count, Score};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use quantization::i8_encoder::I8EncodedVectors;

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

    pub fn encode_data(&self) -> I8EncodedVectors {
        println!("Start encoding:");
        let timer = std::time::Instant::now();
        let encoded_data = I8EncodedVectors::new(
            self.vectors
                .rows()
                .into_iter()
                .map(|row| row.to_slice().unwrap()),
            self.vectors_count,
            self.dim,
        )
        .unwrap();
        println!("encoding time: {:?}", timer.elapsed());
        println!("Original data size: {}", self.vectors_count * self.dim * 4);
        //println!("Encoded data size: {}", encoded_data.data_size());
        encoded_data
    }

    pub fn measure_scoring_time(
        &self,
        encoded: &I8EncodedVectors,
        random_access: bool,
        queries_count: usize,
    ) {
        let multiprogress = MultiProgress::new();
        let sent_bar = multiprogress.add(ProgressBar::new(3 * queries_count as u64));
        let progress_style = ProgressStyle::default_bar()
            .template("{msg} [{elapsed_precise}] {wide_bar} [{per_sec:>3}] {pos}/{len} (eta:{eta})")
            .expect("Failed to create progress style");
        sent_bar.set_style(progress_style);

        let permutation: Vec<usize> = if random_access {
            let permutor = permutation_iterator::Permutor::new(self.vectors_count as u64);
            permutor.map(|i| i as usize).collect()
        } else {
            (0..self.vectors_count).collect()
        };

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
            let query_u8 = I8EncodedVectors::encode_query(&query);
            for &index in &permutation {
                scores[index] = encoded.score_point_dot_avx_2(&query_u8, index);
            }
            timings.push(timer.elapsed().as_millis() as f64);
            sent_bar.inc(1);
        }
        let avg_encoded = timings.iter().sum::<f64>() / timings.len() as f64;
        timings.clear();

        /*
        for query in self
            .queries
            .rows()
            .into_iter()
            .take(queries_count)
            .map(|q| q.to_slice().unwrap())
        {
            let timer = std::time::Instant::now();
            let query_u8 = I8EncodedVectors::encode_query(&query);
            encoded.score_points_dot(&query_u8, &permutation, &mut scores);
            timings.push(timer.elapsed().as_millis() as f64);
            sent_bar.inc(1);
        }
        */
        let avg_encoded_block = timings.iter().sum::<f64>() / timings.len() as f64;
        timings.clear();

        for query in self
            .queries
            .rows()
            .into_iter()
            .take(queries_count)
            .map(|q| q.to_slice().unwrap())
        {
            let timer = std::time::Instant::now();
            for &index in &permutation {
                scores[index] = unsafe { dot_avx(&query, self.vectors.row(index).as_slice().unwrap()) };
            }
            timings.push(timer.elapsed().as_millis() as f64);
            sent_bar.inc(1);
        }
        let avg_avx = timings.iter().sum::<f64>() / timings.len() as f64;
        timings.clear();

        sent_bar.finish();
        println!("Scoring time: AVX: {:.2}ms, encoded: {:.2}ms, encoded block: {:.2}ms", avg_avx, avg_encoded, avg_encoded_block);
    }

    pub fn test_knn_encoded<F>(&self, encoded: &I8EncodedVectors, postprocess: F)
    where F: Fn(f32) -> f32
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
            let query_u8 = I8EncodedVectors::encode_query(&query);
            let mut heap: BinaryHeap<Score> = BinaryHeap::new();
            for index in 0..self.vectors_count {
                let score = postprocess(encoded.score_point_dot_avx_2(&query_u8, index));
                //let score = 1.0 - unsafe { dot_avx(&query, self.vectors.row(index).as_slice().unwrap()) };
                //let score = 1.0 - unsafe { dot_similarity_sse(&query, self.vectors.row(index).as_slice().unwrap()) };
                let score = Score { index, score };
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

use std::arch::x86_64::*;

#[target_feature(enable = "avx")]
#[target_feature(enable = "fma")]
unsafe fn hsum256_ps_avx(x: __m256) -> f32 {
    let x128: __m128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    let x64: __m128 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    let x32: __m128 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    _mm_cvtss_f32(x32)
}

unsafe fn hsum128_ps_sse(x: __m128) -> f32 {
    let x64: __m128 = _mm_add_ps(x, _mm_movehl_ps(x, x));
    let x32: __m128 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    _mm_cvtss_f32(x32)
}

#[target_feature(enable = "avx")]
#[target_feature(enable = "fma")]
pub(crate) unsafe fn dot_avx(v1: &[f32], v2: &[f32]) -> f32 {
    let n = v1.len();
    let m = n - (n % 32);
    let mut ptr1: *const f32 = v1.as_ptr();
    let mut ptr2: *const f32 = v2.as_ptr();
    let mut sum256_1: __m256 = _mm256_setzero_ps();
    let mut sum256_2: __m256 = _mm256_setzero_ps();
    let mut sum256_3: __m256 = _mm256_setzero_ps();
    let mut sum256_4: __m256 = _mm256_setzero_ps();
    let mut i: usize = 0;
    while i < m {
        sum256_1 = _mm256_fmadd_ps(_mm256_loadu_ps(ptr1), _mm256_loadu_ps(ptr2), sum256_1);
        sum256_2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(ptr1.add(8)),
            _mm256_loadu_ps(ptr2.add(8)),
            sum256_2,
        );
        sum256_3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(ptr1.add(16)),
            _mm256_loadu_ps(ptr2.add(16)),
            sum256_3,
        );
        sum256_4 = _mm256_fmadd_ps(
            _mm256_loadu_ps(ptr1.add(24)),
            _mm256_loadu_ps(ptr2.add(24)),
            sum256_4,
        );

        ptr1 = ptr1.add(32);
        ptr2 = ptr2.add(32);
        i += 32;
    }

    let mut result = hsum256_ps_avx(sum256_1)
        + hsum256_ps_avx(sum256_2)
        + hsum256_ps_avx(sum256_3)
        + hsum256_ps_avx(sum256_4);

    for i in 0..n - m {
        result += (*ptr1.add(i)) * (*ptr2.add(i));
    }
    result
}

pub(crate) unsafe fn dot_similarity_sse(v1: &[f32], v2: &[f32]) -> f32 {
    let n = v1.len();
    let m = n - (n % 16);
    let mut ptr1: *const f32 = v1.as_ptr();
    let mut ptr2: *const f32 = v2.as_ptr();
    let mut sum128_1: __m128 = _mm_setzero_ps();
    let mut sum128_2: __m128 = _mm_setzero_ps();
    let mut sum128_3: __m128 = _mm_setzero_ps();
    let mut sum128_4: __m128 = _mm_setzero_ps();

    let mut i: usize = 0;
    while i < m {
        sum128_1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ptr1), _mm_loadu_ps(ptr2)), sum128_1);

        sum128_2 = _mm_add_ps(
            _mm_mul_ps(_mm_loadu_ps(ptr1.add(4)), _mm_loadu_ps(ptr2.add(4))),
            sum128_2,
        );

        sum128_3 = _mm_add_ps(
            _mm_mul_ps(_mm_loadu_ps(ptr1.add(8)), _mm_loadu_ps(ptr2.add(8))),
            sum128_3,
        );

        sum128_4 = _mm_add_ps(
            _mm_mul_ps(_mm_loadu_ps(ptr1.add(12)), _mm_loadu_ps(ptr2.add(12))),
            sum128_4,
        );

        ptr1 = ptr1.add(16);
        ptr2 = ptr2.add(16);
        i += 16;
    }

    let mut result = hsum128_ps_sse(sum128_1)
        + hsum128_ps_sse(sum128_2)
        + hsum128_ps_sse(sum128_3)
        + hsum128_ps_sse(sum128_4);
    for i in 0..n - m {
        result += (*ptr1.add(i)) * (*ptr2.add(i));
    }
    result
}
