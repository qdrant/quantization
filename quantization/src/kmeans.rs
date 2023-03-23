use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::{
    quantile::find_min_max_size_dim_from_iter, EncodedStorage, EncodedStorageBuilder, EncodingError,
};

pub type Centroid = Vec<f32>;

#[derive(Serialize, Deserialize)]
pub struct KMeans {
    pub bucket_size: usize,
    pub centroids: Vec<Vec<f32>>,
    pub vector_division: Vec<Range<usize>>,
}

impl KMeans {
    pub fn run<'a, TStorage: EncodedStorage>(
        data: impl IntoIterator<Item = &'a [f32]> + Clone,
        storage_builder: &mut impl EncodedStorageBuilder<TStorage>,
        bucket_size: usize,
        centroids_count: u8,
    ) -> Result<Self, EncodingError> {
        let (min, max, _count, dim) = find_min_max_size_dim_from_iter(data.clone());

        let vector_division = (0..dim)
            .step_by(bucket_size)
            .map(|i| i..std::cmp::min(i + bucket_size, dim))
            .collect::<Vec<_>>();
        let centroids = (0..centroids_count)
            .map(|_| {
                (0..dim)
                    .map(|_| rand::random::<f32>() * (max - min) + min)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut kmeans = Self {
            bucket_size,
            centroids,
            vector_division,
        };

        for _ in 0..30 {
            kmeans.update_indexes(data.clone(), storage_builder);
            if kmeans.update_centroids(data.clone(), storage_builder) {
                break;
            }
        }

        kmeans.update_indexes(data, storage_builder);
        Ok(kmeans)
    }

    fn update_centroids<'a, TStorage: EncodedStorage>(
        &mut self,
        data: impl IntoIterator<Item = &'a [f32]> + Clone,
        storage_builder: &impl EncodedStorageBuilder<TStorage>,
    ) -> bool {
        let mut byte_index = 0;
        let mut centroids_counter =
            vec![vec![0usize; self.vector_division[0].len()]; self.centroids.len()];
        let mut centroids_acc = vec![vec![0.0_f64; self.centroids[0].len()]; self.centroids.len()];
        for vector_data in data.into_iter() {
            for (i, range) in self.vector_division.iter().enumerate() {
                let subvector_data = &vector_data[range.clone()];
                let centroid_index = storage_builder.get(byte_index) as usize;
                let centroid_data = &mut self.centroids[centroid_index][range.clone()];
                for (c, v) in centroid_data.iter_mut().zip(subvector_data.iter()) {
                    *c += v;
                }
                centroids_counter[centroid_index][i] += 1;
                byte_index += 1;
            }
        }
        for (centroid_index, centroid) in centroids_acc.iter_mut().enumerate() {
            for (i, range) in self.vector_division.iter().enumerate() {
                let centroid_data = &mut centroid[range.clone()];
                for c in centroid_data.iter_mut() {
                    *c /= centroids_counter[centroid_index][i] as f64;
                }
            }
        }
        self.centroids = centroids_acc
            .iter()
            .map(|v| v.iter().map(|f| *f as f32).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        false
    }

    fn update_indexes<'a, TStorage: EncodedStorage>(
        &mut self,
        data: impl IntoIterator<Item = &'a [f32]> + Clone,
        storage_builder: &mut impl EncodedStorageBuilder<TStorage>,
    ) {
        let mut byte_index = 0;
        for vector_data in data.into_iter() {
            for range in &self.vector_division {
                let subvector_data = &vector_data[range.clone()];
                let mut min_distance = f32::MAX;
                let mut min_centroid_index = 0;
                for (centroid_index, centroid) in self.centroids.iter().enumerate() {
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
                storage_builder.set(byte_index, min_centroid_index as u8);
                byte_index += 1;
            }
        }
    }
}
