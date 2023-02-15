use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

use crate::quantile::find_quantile_interval;
use crate::{
    encoded_storage::{EncodedStorage, EncodedStorageBuilder},
    encoded_vectors::{EncodedVectors, SimilarityType, VectorParameters},
    EncodingError,
};

pub const ALIGHMENT: usize = 16;

pub struct EncodedVectorsU8<TStorage: EncodedStorage> {
    encoded_vectors: TStorage,
    metadata: Metadata,
}

pub struct EncodedQueryU8 {
    offset: f32,
    encoded_query: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
struct Metadata {
    actual_dim: usize,
    alpha: f32,
    offset: f32,
    multiplier: f32,
    vector_parameters: VectorParameters,
}

impl<TStorage: EncodedStorage> EncodedVectorsU8<TStorage> {
    pub fn encode<'a>(
        orig_data: impl IntoIterator<Item = &'a [f32]> + Clone,
        mut storage_builder: impl EncodedStorageBuilder<TStorage>,
        vector_parameters: &VectorParameters,
        quantile: Option<f32>,
    ) -> Result<Self, EncodingError> {
        let (alpha, offset, count, dim) = Self::find_alpha_offset_size_dim(orig_data.clone());
        let (alpha, offset) = if let Some(quantile) = quantile {
            if let Some((min, max)) =
                find_quantile_interval(orig_data.clone(), dim, count, quantile)
            {
                Self::alpha_offset_from_min_max(min, max)
            } else {
                (alpha, offset)
            }
        } else {
            (alpha, offset)
        };

        let actual_dim = Self::get_actual_dim(vector_parameters);
        for vector in orig_data {
            let mut encoded_vector = Vec::with_capacity(actual_dim + std::mem::size_of::<f32>());
            encoded_vector.extend_from_slice(&f32::default().to_ne_bytes());
            for &value in vector {
                let endoded = Self::f32_to_u8(value, alpha, offset);
                encoded_vector.push(endoded);
            }
            if dim % ALIGHMENT != 0 {
                for _ in 0..(ALIGHMENT - dim % ALIGHMENT) {
                    let placeholder = match vector_parameters.distance_type {
                        SimilarityType::Dot => 0.0,
                        SimilarityType::L2 => offset,
                    };
                    let endoded = Self::f32_to_u8(placeholder, alpha, offset);
                    encoded_vector.push(endoded);
                }
            }
            let vector_offset = match vector_parameters.distance_type {
                SimilarityType::Dot => {
                    actual_dim as f32 * offset * offset
                        + encoded_vector.iter().map(|&x| x as f32).sum::<f32>() * alpha * offset
                }
                SimilarityType::L2 => {
                    actual_dim as f32 * offset * offset
                        + encoded_vector
                            .iter()
                            .map(|&x| x as f32 * x as f32)
                            .sum::<f32>()
                            * alpha
                            * alpha
                }
            };
            let vector_offset = if vector_parameters.invert {
                -vector_offset
            } else {
                vector_offset
            };
            encoded_vector[0..std::mem::size_of::<f32>()]
                .copy_from_slice(&vector_offset.to_ne_bytes());
            storage_builder.push_vector_data(&encoded_vector);
        }
        let multiplier = match vector_parameters.distance_type {
            SimilarityType::Dot => alpha * alpha,
            SimilarityType::L2 => -2.0 * alpha * alpha,
        };
        let multiplier = if vector_parameters.invert {
            -multiplier
        } else {
            multiplier
        };

        Ok(EncodedVectorsU8 {
            encoded_vectors: storage_builder.build(),
            metadata: Metadata {
                actual_dim,
                alpha,
                offset,
                multiplier,
                vector_parameters: vector_parameters.clone(),
            },
        })
    }

    pub fn score_point_simple(&self, query: &EncodedQueryU8, i: u32) -> f32 {
        unsafe {
            let (vector_offset, v_ptr) = self.get_vec_ptr(i);
            let mut mul = 0i32;
            for i in 0..self.metadata.actual_dim {
                mul += query.encoded_query[i] as i32 * (*v_ptr.add(i)) as i32;
            }
            self.metadata.multiplier * mul as f32 + query.offset + vector_offset
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    pub fn score_point_neon(&self, query: &EncodedQueryU8, i: u32) -> f32 {
        unsafe {
            let (vector_offset, v_ptr) = self.get_vec_ptr(i);
            let score = impl_score_dot_neon(
                query.encoded_query.as_ptr() as *const u8,
                v_ptr,
                self.metadata.vector_parameters.dim as u32,
            );
            self.metadata.multiplier * score + query.offset + vector_offset
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn score_point_sse(&self, query: &EncodedQueryU8, i: u32) -> f32 {
        unsafe {
            let (vector_offset, v_ptr) = self.get_vec_ptr(i);
            let score = impl_score_dot_sse(
                query.encoded_query.as_ptr() as *const u8,
                v_ptr,
                self.metadata.actual_dim as u32,
            );
            self.metadata.multiplier * score + query.offset + vector_offset
        }
    }

    #[cfg(target_arch = "x86_64")]
    pub fn score_point_avx(&self, query: &EncodedQueryU8, i: u32) -> f32 {
        unsafe {
            let (vector_offset, v_ptr) = self.get_vec_ptr(i);
            let score = impl_score_dot_avx(
                query.encoded_query.as_ptr() as *const u8,
                v_ptr,
                self.metadata.actual_dim as u32,
            );
            self.metadata.multiplier * score + query.offset + vector_offset
        }
    }

    fn find_alpha_offset_size_dim<'a>(
        orig_data: impl IntoIterator<Item = &'a [f32]>,
    ) -> (f32, f32, usize, usize) {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        let mut count: usize = 0;
        let mut dim: usize = 0;
        for vector in orig_data {
            count += 1;
            dim = dim.max(vector.len());
            for &value in vector {
                if value < min {
                    min = value;
                }
                if value > max {
                    max = value;
                }
            }
        }
        let (alpha, offset) = Self::alpha_offset_from_min_max(min, max);
        (alpha, offset, count, dim)
    }

    fn alpha_offset_from_min_max(min: f32, max: f32) -> (f32, f32) {
        let alpha = (max - min) / 127.0;
        let offset = min;
        (alpha, offset)
    }

    fn f32_to_u8(i: f32, alpha: f32, offset: f32) -> u8 {
        let i = (i - offset) / alpha;
        i.clamp(0.0, 127.0) as u8
    }

    #[inline]
    fn get_vec_ptr(&self, i: u32) -> (f32, *const u8) {
        unsafe {
            let vector_data_size = self.metadata.actual_dim + std::mem::size_of::<f32>();
            let v_ptr = self
                .encoded_vectors
                .get_vector_data(i as usize, vector_data_size)
                .as_ptr();
            let vector_offset = *(v_ptr as *const f32);
            (vector_offset, v_ptr.add(std::mem::size_of::<f32>()))
        }
    }

    pub fn get_quantized_vector_size(vector_parameters: &VectorParameters) -> usize {
        let actual_dim = Self::get_actual_dim(vector_parameters);
        actual_dim + std::mem::size_of::<f32>()
    }

    pub fn get_actual_dim(vector_parameters: &VectorParameters) -> usize {
        vector_parameters.dim + (ALIGHMENT - vector_parameters.dim % ALIGHMENT) % ALIGHMENT
    }
}

impl<TStorage: EncodedStorage> EncodedVectors<EncodedQueryU8> for EncodedVectorsU8<TStorage> {
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
        let quantized_vector_size = Self::get_quantized_vector_size(vector_parameters);
        let encoded_vectors = TStorage::from_file(data_path, quantized_vector_size)?;
        let result = Self {
            metadata,
            encoded_vectors,
        };
        Ok(result)
    }

    fn encode_query(&self, query: &[f32]) -> EncodedQueryU8 {
        let dim = query.len();
        let mut query: Vec<_> = query
            .iter()
            .map(|&v| Self::f32_to_u8(v, self.metadata.alpha, self.metadata.offset))
            .collect();
        if dim % ALIGHMENT != 0 {
            for _ in 0..(ALIGHMENT - dim % ALIGHMENT) {
                let placeholder = match self.metadata.vector_parameters.distance_type {
                    SimilarityType::Dot => 0.0,
                    SimilarityType::L2 => self.metadata.offset,
                };
                let endoded =
                    Self::f32_to_u8(placeholder, self.metadata.alpha, self.metadata.offset);
                query.push(endoded);
            }
        }
        let offset = match self.metadata.vector_parameters.distance_type {
            SimilarityType::Dot => {
                query.iter().map(|&x| x as f32).sum::<f32>()
                    * self.metadata.alpha
                    * self.metadata.offset
            }
            SimilarityType::L2 => {
                query.iter().map(|&x| x as f32 * x as f32).sum::<f32>()
                    * self.metadata.alpha
                    * self.metadata.alpha
            }
        };
        let offset = if self.metadata.vector_parameters.invert {
            -offset
        } else {
            offset
        };
        EncodedQueryU8 {
            offset,
            encoded_query: query,
        }
    }

    fn score_point(&self, query: &EncodedQueryU8, i: u32) -> f32 {
        let q_ptr = query.encoded_query.as_ptr() as *const u8;
        let (vector_offset, v_ptr) = self.get_vec_ptr(i);

        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
                let score = impl_score_dot_avx(q_ptr, v_ptr, self.metadata.actual_dim as u32);
                return self.metadata.multiplier * score + query.offset + vector_offset;
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if is_x86_feature_detected!("sse") {
                let score = impl_score_dot_sse(q_ptr, v_ptr, self.metadata.actual_dim as u32);
                return self.metadata.multiplier * score + query.offset + vector_offset;
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            if std::arch::is_aarch64_feature_detected!("neon") {
                let score =
                    impl_score_dot_neon(q_ptr, v_ptr, self.metadata.vector_parameters.dim as u32);
                return self.metadata.multiplier * score + query.offset + vector_offset;
            }
        }

        self.score_point_simple(query, i)
    }

    fn score_internal(&self, i: u32, j: u32) -> f32 {
        let (query_offset, q_ptr) = self.get_vec_ptr(i);
        let (vector_offset, v_ptr) = self.get_vec_ptr(j);
        let diff = self.metadata.actual_dim as f32 * self.metadata.offset * self.metadata.offset;
        let diff = if self.metadata.vector_parameters.invert {
            -diff
        } else {
            diff
        };
        let offset = query_offset + vector_offset - diff;

        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
                let score = impl_score_dot_avx(q_ptr, v_ptr, self.metadata.actual_dim as u32);
                return self.metadata.multiplier * score + offset;
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if is_x86_feature_detected!("sse") {
                let score = impl_score_dot_sse(q_ptr, v_ptr, self.metadata.actual_dim as u32);
                return self.metadata.multiplier * score + offset;
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            if std::arch::is_aarch64_feature_detected!("neon") {
                let score =
                    impl_score_dot_neon(q_ptr, v_ptr, self.metadata.vector_parameters.dim as u32);
                return self.metadata.multiplier * score + offset;
            }
        }

        unsafe {
            let mut mul = 0i32;
            for i in 0..self.metadata.actual_dim {
                mul += (*q_ptr.add(i)) as i32 * (*v_ptr.add(i)) as i32;
            }
            self.metadata.multiplier * mul as f32 + offset
        }
    }
}

#[cfg(target_arch = "x86_64")]
extern "C" {
    fn impl_score_dot_avx(query_ptr: *const u8, vector_ptr: *const u8, dim: u32) -> f32;

    fn impl_score_dot_sse(query_ptr: *const u8, vector_ptr: *const u8, dim: u32) -> f32;
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
extern "C" {
    fn impl_score_dot_neon(query_ptr: *const u8, vector_ptr: *const u8, dim: u32) -> f32;
}
