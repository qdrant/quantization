use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::prelude::*;
use std::ops::Range;
use std::path::Path;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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
        orig_data: impl IntoIterator<Item = &'a [f32]> + Clone,
        mut storage_builder: impl EncodedStorageBuilder<TStorage>,
        vector_parameters: &VectorParameters,
        bucket_size: usize,
    ) -> Result<Self, EncodingError> {
        let vector_division = Self::get_vector_division(vector_parameters.dim, bucket_size);

        let centroids_count = 256;
        let centroids = Self::find_centroids(
            orig_data.clone(),
            &vector_division,
            vector_parameters.count,
            centroids_count,
        )?;

        #[allow(clippy::redundant_clone)]
        Self::encode_storage(
            orig_data.clone(),
            &mut storage_builder,
            &vector_division,
            &centroids,
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

    fn encode_storage<'a>(
        data: impl IntoIterator<Item = &'a [f32]> + Clone,
        storage_builder: &mut impl EncodedStorageBuilder<TStorage>,
        vector_division: &[Range<usize>],
        centroids: &[Vec<f32>],
    ) {
        let mut encoded_vector = vec![0u8; vector_division.len()];
        for vector_data in data.into_iter() {
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
            storage_builder.push_vector_data(&encoded_vector);
        }
    }

    pub fn find_centroids<'a>(
        data: impl IntoIterator<Item = &'a [f32]> + Clone,
        vector_division: &[Range<usize>],
        count: usize,
        centroids_count: usize,
    ) -> Result<Vec<Vec<f32>>, EncodingError> {
        // generate random subset of data
        let sample_size = KMEANS_SAMPLE_SIZE.min(count);
        let permutor = permutation_iterator::Permutor::new(count as u64);
        let mut selected_vectors: Vec<usize> =
            permutor.map(|i| i as usize).take(sample_size).collect();
        selected_vectors.sort_unstable();

        let mut result = vec![vec![]; centroids_count];

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
                KMEANS_ACCURACY,
            );
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
        let mut lut = Vec::new();
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
