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
        let (min, max, count, dim) = find_min_max_size_dim_from_iter(data.clone());

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

        let empty_vector = vec![0u8; vector_division.len()];
        for _ in 0..count {
            storage_builder.push_vector_data(&empty_vector);
        }

        let mut kmeans = Self {
            bucket_size,
            centroids,
            vector_division,
        };

        for _ in 0..10 {
            println!("KMeans Iteration");
            kmeans.update_indexes(data.clone(), storage_builder);
            if kmeans.update_centroids(data.clone(), storage_builder) {
                break;
            }
        }

        kmeans.update_indexes(data.clone(), storage_builder);
        //kmeans.build_image(data, storage_builder, min, max);
        Ok(kmeans)
    }

    fn update_centroids<'a, TStorage: EncodedStorage>(
        &mut self,
        data: impl IntoIterator<Item = &'a [f32]> + Clone,
        storage_builder: &impl EncodedStorageBuilder<TStorage>,
    ) -> bool {
        println!("Update Centroids");
        let mut byte_index = 0;
        let mut centroids_counter =
            vec![vec![0usize; self.vector_division.len()]; self.centroids.len()];
        let mut centroids_acc = vec![vec![0.0_f64; self.centroids[0].len()]; self.centroids.len()];
        let mut rand_vectors = Vec::with_capacity(self.centroids.len());
        for vector_data in data.into_iter() {
            if rand_vectors.len() < self.centroids.len() {
                rand_vectors.push(vector_data);
            }
            for (i, range) in self.vector_division.iter().enumerate() {
                let subvector_data = &vector_data[range.clone()];
                let centroid_index = storage_builder.get(byte_index) as usize;
                let centroid_data = &mut centroids_acc[centroid_index][range.clone()];
                for (c, v) in centroid_data.iter_mut().zip(subvector_data.iter()) {
                    *c += *v as f64;
                }
                centroids_counter[centroid_index][i] += 1;
                byte_index += 1;
            }
        }
//        println!("centroids_acc {:?}", centroids_acc);
//        println!("centroids_counter {:?}", centroids_counter);
        for (centroid_index, centroid) in centroids_acc.iter_mut().enumerate() {
            for (i, range) in self.vector_division.iter().enumerate() {
                let centroid_data = &mut centroid[range.clone()];
                let rand_vector = &rand_vectors[centroid_index][range.clone()];
                for (c, r) in centroid_data.iter_mut().zip(rand_vector.iter()) {
                    if centroids_counter[centroid_index][i] == 0 {
                        *c = *r as f64;
                    } else {
                        *c /= centroids_counter[centroid_index][i] as f64;
                    }
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
        println!("Update Indexes");
        //println!("Vector division {:?}", self.vector_division);
        let mut byte_index = 0;
        for vector_data in data.into_iter() {
            for range in self.vector_division.iter() {
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

/*
    fn build_image<'a, TStorage: EncodedStorage>(
        &self,
        data: impl IntoIterator<Item = &'a [f32]> + Clone,
        storage_builder: &mut impl EncodedStorageBuilder<TStorage>,
        min: f32,
        max: f32,
    ) {
        //println!("All centroids {:?}", self.centroids);
        println!("Build Image");
        let colors_r: Vec<_> = (0..256).map(|_| rand::random::<u8>()).collect();
        let colors_g: Vec<_> = (0..256).map(|_| rand::random::<u8>()).collect();
        let colors_b: Vec<_> = (0..256).map(|_| rand::random::<u8>()).collect();
        for (range_i, range) in self.vector_division.iter().enumerate() {
            if range.len() < 2 {
                continue;
            }

            //println!("Range {:?}", range);
            let mut centroids_counter = (0..self.centroids.len())
                .map(|_| 0usize)
                .collect::<Vec<_>>();

            let imgx = 1000;
            let imgy = 1000;
            let mut imgbuf = image::ImageBuffer::new(imgx, imgy);
            for (_x, _y, pixel) in imgbuf.enumerate_pixels_mut() {
                *pixel = image::Rgb([255u8, 255u8, 255u8]);
            }

            for (i, vector_data) in data.clone().into_iter().enumerate() {
                let subvector_data = &vector_data[range.clone()];
                let storage_builder_index = self.vector_division.len() * i + range_i;
                let centroid_index = storage_builder.get(storage_builder_index) as usize;
                centroids_counter[centroid_index] += 1;
                let x = (((subvector_data[0] - min) / (max - min)) * imgx as f32).clamp(0., imgx as f32 - 1.0) as u32;
                let y = (((subvector_data[1] - min) / (max - min)) * imgy as f32).clamp(0., imgy as f32 - 1.0) as u32;
                *imgbuf.get_pixel_mut(x, y) = image::Rgb([
                    colors_r[centroid_index],
                    colors_g[centroid_index],
                    colors_b[centroid_index],
                ]);
            }

            for centroid in &self.centroids {
                let subvector_data = &centroid[range.clone()];
                let x = (((subvector_data[0] - min) / (max - min)) * imgx as f32).clamp(0., imgx as f32 - 2.0) as u32;
                let y = (((subvector_data[1] - min) / (max - min)) * imgy as f32).clamp(0., imgy as f32 - 2.0) as u32;
                *imgbuf.get_pixel_mut(x, y) = image::Rgb([255u8, 0u8, 0u8]);
                *imgbuf.get_pixel_mut(x + 1, y) = image::Rgb([255u8, 0u8, 0u8]);
                *imgbuf.get_pixel_mut(x + 1, y + 1) = image::Rgb([255u8, 0u8, 0u8]);
                *imgbuf.get_pixel_mut(x, y + 1) = image::Rgb([255u8, 0u8, 0u8]);
            }

            //println!("Centroids distribution {:?}", centroids_counter);
            imgbuf.save(&format!("target/kmeans-{range_i}.png")).unwrap();
        }
    }
*/
}
