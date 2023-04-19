use rayon::prelude::{IndexedParallelIterator, ParallelBridge, ParallelExtend, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::prelude::*;
use std::ops::Range;
use std::path::Path;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
#[cfg(target_feature = "neon")]
use std::arch::aarch64::*;
use std::sync::{Arc, Mutex};

use crate::kmeans::kmeans;
use crate::{
    encoded_storage::{EncodedStorage, EncodedStorageBuilder},
    encoded_vectors::{EncodedVectors, VectorParameters},
    EncodingError,
};

pub const KMEANS_SAMPLE_SIZE: usize = 10_000;
pub const KMEANS_MAX_ITERATIONS: usize = 100;
pub const KMEANS_ACCURACY: f32 = 1e-5;

pub struct EncodedVectorsPQ<TStorage: EncodedStorage> {
    encoded_vectors: TStorage,
    metadata: Metadata,
}

pub struct EncodedQueryPQ {
    lut: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct Metadata {
    centroids: Vec<Vec<f32>>,
    vector_division: Vec<Range<usize>>,
    vector_parameters: VectorParameters,
}

impl<TStorage: EncodedStorage> EncodedVectorsPQ<TStorage> {
    pub fn encode<'a>(
        orig_data: impl Iterator<Item = &'a [f32]> + Clone + Send,
        mut storage_builder: impl EncodedStorageBuilder<TStorage> + Send,
        vector_parameters: &VectorParameters,
        bucket_size: usize,
        max_kmeans_threads: usize,
    ) -> Result<Self, EncodingError> {
        let vector_division = Self::get_vector_division(vector_parameters.dim, bucket_size);

        let centroids_count = 256;
        let centroids = Self::find_centroids(
            orig_data.clone(),
            &vector_division,
            vector_parameters,
            centroids_count,
            max_kmeans_threads,
        )?;

        // let centroids_count = 256;
        // let centroids = Self::find_centroids_rayon(
        //     // TODO: we need an indexed parallel iterator here
        //     orig_data.par_bridge(),
        //     &vector_division,
        //     vector_parameters,
        //     centroids_count,
        //     max_kmeans_threads,
        // )?;

        // #[allow(clippy::redundant_clone)]
        // Self::encode_storage(
        //     orig_data,
        //     &mut storage_builder,
        //     vector_parameters,
        //     &vector_division,
        //     &centroids,
        //     max_kmeans_threads,
        // );

        #[allow(clippy::redundant_clone)]
        Self::encode_storage_rayon(
            // TODO: according to docs this does not preserve order, we can resolve this by using parallel iterator everywhere
            orig_data.par_bridge(),
            Arc::new(Mutex::new(&mut storage_builder)),
            vector_parameters,
            &vector_division,
            &centroids,
            max_kmeans_threads,
        );

        let storage = storage_builder.build();

        #[cfg(feature = "dump_image")]
        Self::dump_to_image(orig_data, &storage, &centroids, &vector_division);

        Ok(Self {
            encoded_vectors: storage,
            metadata: Metadata {
                centroids,
                vector_division,
                vector_parameters: vector_parameters.clone(),
            },
        })
    }

    pub fn get_quantized_vector_size(
        vector_parameters: &VectorParameters,
        bucket_size: usize,
    ) -> usize {
        let vector_division = Self::get_vector_division(vector_parameters.dim, bucket_size);
        vector_division.len()
    }

    fn get_vector_division(dim: usize, bucket_size: usize) -> Vec<Range<usize>> {
        (0..dim)
            .step_by(bucket_size)
            .map(|i| i..std::cmp::min(i + bucket_size, dim))
            .collect::<Vec<_>>()
    }

    #[allow(unused)]
    fn encode_storage<'a>(
        data: impl Iterator<Item = &'a [f32]>,
        storage_builder: &mut impl EncodedStorageBuilder<TStorage>,
        vector_parameters: &VectorParameters,
        vector_division: &[Range<usize>],
        centroids: &[Vec<f32>],
        max_kmeans_threads: usize,
    ) {
        let threads = (0..max_kmeans_threads)
            .map(|_| {
                let (vector_sender, vector_receiver) =
                    std::sync::mpsc::channel::<(Vec<f32>, Vec<u8>)>();
                let (encoded_sender, encoded_receiver) =
                    std::sync::mpsc::channel::<(Vec<f32>, Vec<u8>)>();
                let vector_division = vector_division.to_vec();
                let centroids = centroids.to_vec();
                let handle = std::thread::spawn(move || {
                    while let Ok((vector_data, mut encoded_vector)) = vector_receiver.recv() {
                        if vector_data.is_empty() {
                            break;
                        }
                        encoded_vector.clear();
                        for range in vector_division.iter() {
                            let subvector_data = &vector_data[range.clone()];
                            let mut min_distance = f32::MAX;
                            let mut min_centroid_index = 0;
                            for (centroid_index, centroid) in centroids.iter().enumerate() {
                                let centroid_data = &centroid[range.clone()];
                                let distance = subvector_data
                                    .iter()
                                    .zip(centroid_data.iter())
                                    .map(|(a, b)| (a - b).powi(2))
                                    .sum::<f32>();
                                if distance < min_distance {
                                    min_distance = distance;
                                    min_centroid_index = centroid_index;
                                }
                            }
                            encoded_vector.push(min_centroid_index as u8);
                        }
                        encoded_sender.send((vector_data, encoded_vector)).unwrap();
                    }
                });
                EncodingThread {
                    handle: Some(handle),
                    vector_sender,
                    encoded_receiver,
                }
            })
            .collect::<Vec<_>>();

        let mut encoded_pool: Vec<Vec<u8>> = vec![];
        let mut vectors_pool: Vec<Vec<f32>> = vec![];
        let mut start_index = 0;
        let mut end_index = 0;
        let next = |i: usize| -> usize { (i + 1) % max_kmeans_threads };
        let mut busy_threads_count = 0;
        for vector_data in data.into_iter() {
            if busy_threads_count > 0 && start_index == end_index {
                let (vector, encoded) = threads[end_index].encoded_receiver.recv().unwrap();
                storage_builder.push_vector_data(&encoded);
                encoded_pool.push(encoded);
                vectors_pool.push(vector);
                end_index = next(end_index);
                busy_threads_count -= 1;
            }

            let encoded = encoded_pool.pop().unwrap_or_default();
            let mut v = vectors_pool
                .pop()
                .unwrap_or_else(|| vec![0.0; vector_parameters.dim]);
            v.copy_from_slice(vector_data);
            threads[start_index]
                .vector_sender
                .send((v, encoded))
                .unwrap();
            start_index = next(start_index);
            busy_threads_count += 1;
        }
        for _ in 0..busy_threads_count {
            let (_, encoded) = threads[end_index].encoded_receiver.recv().unwrap();
            storage_builder.push_vector_data(&encoded);
            end_index = next(end_index);
        }
    }

    #[allow(unused)]
    fn encode_storage_rayon<'a, SB>(
        data: impl ParallelIterator<Item = &'a [f32]>,
        storage_builder: Arc<Mutex<&mut SB>>,
        _vector_parameters: &VectorParameters,
        vector_division: &[Range<usize>],
        centroids: &[Vec<f32>],
        max_kmeans_threads: usize,
    ) where
        SB: EncodedStorageBuilder<TStorage> + Send,
    {
        let pool = rayon::ThreadPoolBuilder::new()
            .thread_name(|idx| format!("pq-{idx}"))
            .num_threads(max_kmeans_threads)
            .build()
            .map_err(|e| EncodingError {
                description: format!("Failed PQ encoding while thread pool init: {e}"),
            })
            .unwrap();

        // TODO: minimize allocation, use buffer types

        pool.install(|| {
            data.map(|vector_data| {
                let mut encoded_vector = vec![];
                for range in vector_division.iter() {
                    let subvector_data = &vector_data[range.clone()];
                    let mut min_distance = f32::MAX;
                    let mut min_centroid_index = 0;
                    for (centroid_index, centroid) in centroids.iter().enumerate() {
                        let centroid_data = &centroid[range.clone()];
                        let distance = subvector_data
                            .iter()
                            .zip(centroid_data.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f32>();
                        if distance < min_distance {
                            min_distance = distance;
                            min_centroid_index = centroid_index;
                        }
                    }
                    encoded_vector.push(min_centroid_index as u8);
                }
                encoded_vector
            })
            // TODO: must be ordered, check this
            .for_each_with(storage_builder, |storage_builder, encoded_vector| {
                storage_builder
                    .lock()
                    .unwrap()
                    .push_vector_data(&encoded_vector);
            });
        });
    }

    #[allow(unused)]
    pub fn find_centroids<'a>(
        data: impl IntoIterator<Item = &'a [f32]> + Clone,
        vector_division: &[Range<usize>],
        vector_parameters: &VectorParameters,
        centroids_count: usize,
        max_kmeans_threads: usize,
    ) -> Result<Vec<Vec<f32>>, EncodingError> {
        let sample_size = KMEANS_SAMPLE_SIZE.min(vector_parameters.count);
        let mut result = vec![vec![]; centroids_count];

        // if there are not enough vectors, set centroids as point positions
        if vector_parameters.count <= centroids_count {
            for (i, vector_data) in data.into_iter().enumerate() {
                result[i] = vector_data.to_vec();
            }
            for r in result
                .iter_mut()
                .take(centroids_count)
                .skip(vector_parameters.count)
            {
                *r = vec![0.0; vector_parameters.dim];
            }
            return Ok(result);
        }

        // generate random subset of data
        let permutor = permutation_iterator::Permutor::new(vector_parameters.count as u64);
        let mut selected_vectors: Vec<usize> =
            permutor.map(|i| i as usize).take(sample_size).collect();
        selected_vectors.sort_unstable();

        for range in vector_division.iter() {
            let mut data_subset = Vec::with_capacity(sample_size * range.len());
            let mut selected_index: usize = 0;
            for (vector_index, vector_data) in data.clone().into_iter().enumerate() {
                if vector_index == selected_vectors[selected_index] {
                    data_subset.extend_from_slice(&vector_data[range.clone()]);
                    selected_index += 1;
                    if selected_index == sample_size {
                        break;
                    }
                }
            }

            let centroids = kmeans(
                &data_subset,
                centroids_count,
                range.len(),
                KMEANS_MAX_ITERATIONS,
                max_kmeans_threads,
                KMEANS_ACCURACY,
            )?;
            for (centroid_index, centroid_data) in centroids.chunks_exact(range.len()).enumerate() {
                result[centroid_index].extend_from_slice(centroid_data);
            }
        }

        Ok(result)
    }

    #[allow(unused)]
    pub fn find_centroids_rayon<'a>(
        data: impl IndexedParallelIterator<Item = &'a [f32]> + Clone,
        vector_division: &[Range<usize>],
        vector_parameters: &VectorParameters,
        centroids_count: usize,
        max_kmeans_threads: usize,
    ) -> Result<Vec<Vec<f32>>, EncodingError> {
        let sample_size = KMEANS_SAMPLE_SIZE.min(vector_parameters.count);
        let mut result = vec![vec![]; centroids_count];

        // if there are not enough vectors, set centroids as point positions
        if vector_parameters.count <= centroids_count {
            data.map(|vector_data| vector_data.to_vec())
                .collect_into_vec(&mut result);

            for r in result
                .iter_mut()
                .take(centroids_count)
                .skip(vector_parameters.count)
            {
                *r = vec![0.0; vector_parameters.dim];
            }
            return Ok(result);
        }

        // generate random subset of data
        let permutor = permutation_iterator::Permutor::new(vector_parameters.count as u64);
        let mut selected_vectors: Vec<usize> =
            permutor.map(|i| i as usize).take(sample_size).collect();
        selected_vectors.sort_unstable();

        for range in vector_division.iter() {
            let mut data_subset = Vec::with_capacity(sample_size * range.len());

            // TODO: not sure if this impl is the same as before
            let mut par_iter = data
                .clone()
                .enumerate()
                .filter(|(vector_index, _vector_data)| {
                    // TODO: what we did before with selected_index is probably more efficient?
                    selected_vectors.binary_search(vector_index).is_ok()
                })
                .take_any(sample_size)
                .flat_map(|(_vector_index, vector_data)| &vector_data[range.clone()]);
            data_subset.par_extend(par_iter);

            let centroids = kmeans(
                &data_subset,
                centroids_count,
                range.len(),
                KMEANS_MAX_ITERATIONS,
                max_kmeans_threads,
                KMEANS_ACCURACY,
            )?;
            for (centroid_index, centroid_data) in centroids.chunks_exact(range.len()).enumerate() {
                result[centroid_index].extend_from_slice(centroid_data);
            }
        }

        Ok(result)
    }

    #[cfg(feature = "dump_image")]
    pub fn dump_to_image<'a>(
        data: impl IntoIterator<Item = &'a [f32]> + Clone,
        storage: &TStorage,
        centroids: &[Vec<f32>],
        vector_division: &[Range<usize>],
    ) {
        let (min, max, _count, _dim) =
            crate::quantile::find_min_max_size_dim_from_iter(data.clone());

        let colors_r: Vec<_> = (0..256).map(|_| rand::random::<u8>()).collect();
        let colors_g: Vec<_> = (0..256).map(|_| rand::random::<u8>()).collect();
        let colors_b: Vec<_> = (0..256).map(|_| rand::random::<u8>()).collect();
        for (range_i, range) in vector_division.iter().enumerate() {
            if range.len() < 2 {
                continue;
            }

            let mut centroids_counter = (0..centroids.len()).map(|_| 0usize).collect::<Vec<_>>();

            let imgx = 1000;
            let imgy = 1000;
            let mut imgbuf = image::ImageBuffer::new(imgx, imgy);
            for (_x, _y, pixel) in imgbuf.enumerate_pixels_mut() {
                *pixel = image::Rgb([255u8, 255u8, 255u8]);
            }

            for (i, vector_data) in data.clone().into_iter().enumerate() {
                let subvector_data = &vector_data[range.clone()];
                let centroid_index =
                    storage.get_vector_data(i, vector_division.len())[range_i] as usize;
                centroids_counter[centroid_index] += 1;
                let x = (((subvector_data[0] - min) / (max - min)) * imgx as f32)
                    .clamp(0., imgx as f32 - 1.0) as u32;
                let y = (((subvector_data[1] - min) / (max - min)) * imgy as f32)
                    .clamp(0., imgy as f32 - 1.0) as u32;
                *imgbuf.get_pixel_mut(x, y) = image::Rgb([
                    colors_r[centroid_index],
                    colors_g[centroid_index],
                    colors_b[centroid_index],
                ]);
            }

            for centroid in centroids {
                let subvector_data = &centroid[range.clone()];
                let x = (((subvector_data[0] - min) / (max - min)) * imgx as f32)
                    .clamp(0., imgx as f32 - 2.0) as u32;
                let y = (((subvector_data[1] - min) / (max - min)) * imgy as f32)
                    .clamp(0., imgy as f32 - 2.0) as u32;
                *imgbuf.get_pixel_mut(x, y) = image::Rgb([255u8, 0u8, 0u8]);
                *imgbuf.get_pixel_mut(x + 1, y) = image::Rgb([255u8, 0u8, 0u8]);
                *imgbuf.get_pixel_mut(x + 1, y + 1) = image::Rgb([255u8, 0u8, 0u8]);
                *imgbuf.get_pixel_mut(x, y + 1) = image::Rgb([255u8, 0u8, 0u8]);
            }

            imgbuf.save(&format!("kmeans-{range_i}.png")).unwrap();
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse4.1")]
    unsafe fn score_point_sse(&self, query: &EncodedQueryPQ, i: u32) -> f32 {
        let centroids = self
            .encoded_vectors
            .get_vector_data(i as usize, self.metadata.vector_division.len());
        let len = centroids.len();
        let centroids_count = self.metadata.centroids.len();

        let mut centroids = centroids.as_ptr();
        let mut lut = query.lut.as_ptr();
        let mut sum128: __m128 = _mm_setzero_ps();
        for _ in 0..len / 4 {
            let buffer = [
                *lut.add(*centroids as usize),
                *lut.add(centroids_count + *centroids.add(1) as usize),
                *lut.add(2 * centroids_count + *centroids.add(2) as usize),
                *lut.add(3 * centroids_count + *centroids.add(3) as usize),
            ];
            let c = _mm_loadu_ps(buffer.as_ptr());
            sum128 = _mm_add_ps(sum128, c);

            centroids = centroids.add(4);
            lut = lut.add(4 * centroids_count);
        }
        let sum64: __m128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32: __m128 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55));
        let mut sum = _mm_cvtss_f32(sum32);

        for _ in 0..len % 4 {
            sum += *lut.add(*centroids as usize);
            centroids = centroids.add(1);
            lut = lut.add(centroids_count);
        }
        sum
    }

    #[cfg(target_arch = "aarch64")]
    #[cfg(target_feature = "neon")]
    unsafe fn score_point_neon(&self, query: &EncodedQueryPQ, i: u32) -> f32 {
        let centroids = self
            .encoded_vectors
            .get_vector_data(i as usize, self.metadata.vector_division.len());
        let len = centroids.len();
        let centroids_count = self.metadata.centroids.len();

        let mut centroids = centroids.as_ptr();
        let mut lut = query.lut.as_ptr();
        let mut sum128 = vdupq_n_f32(0.);
        for _ in 0..len / 4 {
            let buffer = [
                *lut.add(*centroids as usize),
                *lut.add(centroids_count + *centroids.add(1) as usize),
                *lut.add(2 * centroids_count + *centroids.add(2) as usize),
                *lut.add(3 * centroids_count + *centroids.add(3) as usize),
            ];
            let c = vld1q_f32(buffer.as_ptr());
            sum128 = vaddq_f32(sum128, c);

            centroids = centroids.add(4);
            lut = lut.add(4 * centroids_count);
        }
        let mut sum = vaddvq_f32(sum128);

        for _ in 0..len % 4 {
            sum += *lut.add(*centroids as usize);
            centroids = centroids.add(1);
            lut = lut.add(centroids_count);
        }
        sum
    }

    fn score_point_simple(&self, query: &EncodedQueryPQ, i: u32) -> f32 {
        unsafe {
            let centroids = self
                .encoded_vectors
                .get_vector_data(i as usize, self.metadata.vector_division.len());
            let len = centroids.len();
            let centroids_count = self.metadata.centroids.len();

            let mut centroids = centroids.as_ptr();
            let mut lut = query.lut.as_ptr();

            let mut sum = 0.0;
            for _ in 0..len {
                sum += *lut.add(*centroids as usize);
                centroids = centroids.add(1);
                lut = lut.add(centroids_count);
            }
            sum
        }
    }
}

impl<TStorage: EncodedStorage> EncodedVectors<EncodedQueryPQ> for EncodedVectorsPQ<TStorage> {
    fn save(&self, data_path: &Path, meta_path: &Path) -> std::io::Result<()> {
        let metadata_bytes = serde_json::to_vec(&self.metadata)?;
        meta_path.parent().map(std::fs::create_dir_all);
        let mut buffer = File::create(meta_path)?;
        buffer.write_all(&metadata_bytes)?;

        data_path.parent().map(std::fs::create_dir_all);
        self.encoded_vectors.save_to_file(data_path)?;
        Ok(())
    }

    fn load(
        data_path: &Path,
        meta_path: &Path,
        vector_parameters: &VectorParameters,
    ) -> std::io::Result<Self> {
        let mut contents = String::new();
        let mut file = File::open(meta_path)?;
        file.read_to_string(&mut contents)?;
        let metadata: Metadata = serde_json::from_str(&contents)?;
        let quantized_vector_size = metadata.vector_division.len();
        let encoded_vectors =
            TStorage::from_file(data_path, quantized_vector_size, vector_parameters.count)?;
        let result = Self {
            metadata,
            encoded_vectors,
        };
        Ok(result)
    }

    fn encode_query(&self, query: &[f32]) -> EncodedQueryPQ {
        let lut_capacity = self.metadata.vector_division.len() * self.metadata.centroids.len();
        let mut lut = Vec::with_capacity(lut_capacity);
        for range in &self.metadata.vector_division {
            let subquery = &query[range.clone()];
            for i in 0..self.metadata.centroids.len() {
                let centroid = &self.metadata.centroids[i];
                let subcentroid = &centroid[range.clone()];
                let distance = self
                    .metadata
                    .vector_parameters
                    .distance_type
                    .distance(subquery, subcentroid);
                let distance = if self.metadata.vector_parameters.invert {
                    -distance
                } else {
                    distance
                };
                lut.push(distance);
            }
        }
        EncodedQueryPQ { lut }
    }

    fn score_point(&self, query: &EncodedQueryPQ, i: u32) -> f32 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if is_x86_feature_detected!("sse4.1") {
                return self.score_point_sse(query, i);
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return self.score_point_neon(query, i);
            }
        }

        self.score_point_simple(query, i)
    }

    fn score_internal(&self, i: u32, j: u32) -> f32 {
        let centroids_i = self
            .encoded_vectors
            .get_vector_data(i as usize, self.metadata.vector_division.len());
        let centroids_j = self
            .encoded_vectors
            .get_vector_data(j as usize, self.metadata.vector_division.len());
        let distance: f32 = centroids_i
            .iter()
            .zip(centroids_j)
            .enumerate()
            .map(|(range_index, (&c_i, &c_j))| {
                let range = &self.metadata.vector_division[range_index];
                let data_i = &self.metadata.centroids[c_i as usize][range.clone()];
                let data_j = &self.metadata.centroids[c_j as usize][range.clone()];
                self.metadata
                    .vector_parameters
                    .distance_type
                    .distance(data_i, data_j)
            })
            .sum();
        if self.metadata.vector_parameters.invert {
            -distance
        } else {
            distance
        }
    }
}

struct EncodingThread {
    handle: Option<std::thread::JoinHandle<()>>,
    vector_sender: std::sync::mpsc::Sender<(Vec<f32>, Vec<u8>)>,
    encoded_receiver: std::sync::mpsc::Receiver<(Vec<f32>, Vec<u8>)>,
}

impl Drop for EncodingThread {
    fn drop(&mut self) {
        if self.vector_sender.send((vec![], vec![])).is_ok() {
            if let Some(handle) = self.handle.take() {
                handle.join().unwrap();
            }
        }
    }
}
