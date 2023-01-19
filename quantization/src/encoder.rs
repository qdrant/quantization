use std::path::Path;
use std::io::prelude::*;
use std::fs::File;

use atomicwrites::AtomicFile;
use atomicwrites::OverwriteBehavior::AllowOverwrite;
use serde::{Serialize, Deserialize};

pub const ALIGHMENT: usize = 16;
pub const FILE_HEADER_MAGIC_NUMBER: u64 = 0x00_DD_91_12_FA_BB_09_01;

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimilarityType {
    L2,
    Dot,
}

pub trait Storage {
    fn ptr(&self) -> *const u8;

    fn as_slice(&self) -> &[u8];

    fn from_file(path: &Path) -> std::io::Result<Self> where Self: Sized;
}

pub trait StorageBuilder<TStorage: Storage> {
    fn build(self) -> TStorage;

    fn extend_from_slice(&mut self, other: &[u8]);
}

pub struct EncodedVectors<TStorage: Storage> {
    encoded_vectors: TStorage,
    metadata: Metadata,
}

pub struct EncodedQuery {
    offset: f32,
    encoded_query: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
struct Metadata {
    dim: usize,
    alpha: f32,
    offset: f32,
    multiplier: f32,
    distance_type: SimilarityType,
}

impl<TStorage: Storage> EncodedVectors<TStorage> {
    pub fn save(&self, data_path: &Path, meta_path: &Path) -> std::io::Result<()> {
        let mut buffer = File::create(data_path)?;
        buffer.write_all(self.encoded_vectors.as_slice())?;

        let af = AtomicFile::new(meta_path, AllowOverwrite);
        let state_bytes = serde_json::to_vec(&self.metadata)?;
        af.write(|f| f.write_all(&state_bytes))?;
        Ok(())
    }

    pub fn load(data_path: &Path, meta_path: &Path) -> std::io::Result<Self>
    {
        let mut contents = String::new();
        let mut file = File::open(meta_path)?;
        file.read_to_string(&mut contents)?;
        let metadata: Metadata = serde_json::from_str(&contents)?;
        let encoded_vectors = TStorage::from_file(data_path)?;
        let result = Self {
            metadata,
            encoded_vectors,
        };
        Ok(result)
    }

    pub fn encode<'a>(
        orig_data: impl IntoIterator<Item = &'a [f32]> + Clone,
        mut storage_builder: impl StorageBuilder<TStorage>,
        distance_type: SimilarityType,
    ) -> Result<Self, String> {
        let (alpha, offset, _, dim) = Self::find_alpha_offset_size_dim(orig_data.clone());
        let extended_dim = dim + (ALIGHMENT - dim % ALIGHMENT) % ALIGHMENT;
        for vector in orig_data {
            let mut encoded_vector = Vec::new();
            for &value in vector {
                let endoded = Self::f32_to_u8(value, alpha, offset);
                encoded_vector.push(endoded);
            }
            if dim % ALIGHMENT != 0 {
                for _ in 0..(ALIGHMENT - dim % ALIGHMENT) {
                    let placeholder = match distance_type {
                        SimilarityType::Dot => 0.0,
                        SimilarityType::L2 => offset,
                    };
                    let endoded = Self::f32_to_u8(placeholder, alpha, offset);
                    encoded_vector.push(endoded);
                }
            }
            let vector_offset = match distance_type {
                SimilarityType::Dot => {
                    extended_dim as f32 * offset * offset
                        + encoded_vector.iter().map(|&x| x as f32).sum::<f32>() * alpha * offset
                }
                SimilarityType::L2 => {
                    extended_dim as f32 * offset * offset
                        + encoded_vector
                            .iter()
                            .map(|&x| x as f32 * x as f32)
                            .sum::<f32>()
                            * alpha
                            * alpha
                }
            };
            storage_builder.extend_from_slice(&vector_offset.to_ne_bytes());
            storage_builder.extend_from_slice(&encoded_vector);
        }
        let multiplier = match distance_type {
            SimilarityType::Dot => alpha * alpha,
            SimilarityType::L2 => -2.0 * alpha * alpha,
        };

        Ok(EncodedVectors {
            encoded_vectors: storage_builder.build(),
            metadata: Metadata {
                dim: extended_dim,
                alpha,
                offset,
                distance_type,
                multiplier,
            },
        })
    }

    pub fn encode_query(&self, query: &[f32]) -> EncodedQuery {
        let dim = query.len();
        let mut query: Vec<_> = query
            .iter()
            .map(|&v| Self::f32_to_u8(v, self.metadata.alpha, self.metadata.offset))
            .collect();
        if dim % ALIGHMENT != 0 {
            for _ in 0..(ALIGHMENT - dim % ALIGHMENT) {
                let placeholder = match self.metadata.distance_type {
                    SimilarityType::Dot => 0.0,
                    SimilarityType::L2 => self.metadata.offset,
                };
                let endoded = Self::f32_to_u8(placeholder, self.metadata.alpha, self.metadata.offset);
                query.push(endoded);
            }
        }
        let offset = match self.metadata.distance_type {
            SimilarityType::Dot => {
                query.iter().map(|&x| x as f32).sum::<f32>() * self.metadata.alpha * self.metadata.offset
            }
            SimilarityType::L2 => {
                query.iter().map(|&x| x as f32 * x as f32).sum::<f32>() * self.metadata.alpha * self.metadata.alpha
            }
        };
        EncodedQuery {
            offset,
            encoded_query: query,
        }
    }

    pub fn score_point(&self, query: &EncodedQuery, i: usize) -> f32 {
        let q_ptr = query.encoded_query.as_ptr() as *const u8;
        let (vector_offset, v_ptr) = self.get_vec_ptr(i);

        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
                let score = impl_score_dot_avx(q_ptr, v_ptr, self.metadata.dim as u32);
                return self.metadata.multiplier * score + query.offset + vector_offset;
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if is_x86_feature_detected!("sse") {
                let score = impl_score_dot_sse(q_ptr, v_ptr, self.metadata.dim as u32);
                return self.metadata.multiplier * score + query.offset + vector_offset;
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            if std::arch::is_aarch64_feature_detected!("neon") {
                let score = impl_score_dot_neon(q_ptr, v_ptr, self.dim as u32);
                self.multiplier * score + query.offset + vector_offset
            }
        }

        self.score_point_simple(query, i)
    }

    pub fn score_points(&self, query: &EncodedQuery, i: &[usize], scores: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
                return self.score_points_avx(query, i, scores);
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("sse") {
                return self.score_points_sse(query, i, scores);
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return self.score_points_neon(query, i, scores);
            }
        }

        self.score_points_simple(query, i, scores)
    }

    pub fn score_internal(&self, i: usize, j: usize) -> f32 {
        let (query_offset, q_ptr) = self.get_vec_ptr(i);
        let (vector_offset, v_ptr) = self.get_vec_ptr(j);

        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
                let score = impl_score_dot_avx(q_ptr, v_ptr, self.metadata.dim as u32);
                return self.metadata.multiplier * score + query_offset + vector_offset;
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if is_x86_feature_detected!("sse") {
                let score = impl_score_dot_sse(q_ptr, v_ptr, self.metadata.dim as u32);
                return self.metadata.multiplier * score + query_offset + vector_offset;
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            if std::arch::is_aarch64_feature_detected!("neon") {
                let score = impl_score_dot_neon(q_ptr, v_ptr, self.dim as u32);
                return self.multiplier * score + query.offset + vector_offset;
            }
        }

        unsafe {
            let mut mul = 0i32;
            for i in 0..self.metadata.dim {
                mul += (*q_ptr.add(i)) as i32 * (*v_ptr.add(i)) as i32;
            }
            self.metadata.multiplier * mul as f32 + query_offset + vector_offset
        }
    }

    pub fn score_point_simple(&self, query: &EncodedQuery, i: usize) -> f32 {
        unsafe {
            let (vector_offset, v_ptr) = self.get_vec_ptr(i);
            let mut mul = 0i32;
            for i in 0..self.metadata.dim {
                mul += query.encoded_query[i] as i32 * (*v_ptr.add(i)) as i32;
            }
            self.metadata.multiplier * mul as f32 + query.offset + vector_offset
        }
    }

    pub fn score_points_simple(&self, query: &EncodedQuery, i: &[usize], scores: &mut [f32]) {
        for (i, score) in i.iter().zip(scores.iter_mut()) {
            *score = self.score_point_simple(query, *i);
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    pub fn score_point_neon(&self, query: &EncodedQuery, i: usize) -> f32 {
        unsafe {
            let (vector_offset, v_ptr) = self.get_vec_ptr(i);
            let score = impl_score_dot_neon(
                query.encoded_query.as_ptr() as *const u8,
                v_ptr,
                self.dim as u32,
            );
            self.multiplier * score + query.offset + vector_offset
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    pub fn score_points_neon(&self, query: &EncodedQuery, indexes: &[usize], scores: &mut [f32]) {
        unsafe {
            for (indexes, scores) in indexes.chunks_exact(2).zip(scores.chunks_exact_mut(2)) {
                let (vector1_offset, v1_ptr) = self.get_vec_ptr(indexes[0]);
                let (vector2_offset, v2_ptr) = self.get_vec_ptr(indexes[1]);
                impl_score_pair_dot_neon(
                    query.encoded_query.as_ptr() as *const u8,
                    v1_ptr,
                    v2_ptr,
                    self.dim as u32,
                    scores.as_mut_ptr(),
                );
                scores[0] = self.multiplier * scores[0] + query.offset + vector1_offset;
                scores[1] = self.multiplier * scores[1] + query.offset + vector2_offset;
            }
            if indexes.len() % 2 == 1 {
                let idx = indexes.len() - 1;
                scores[idx] = self.score_point_neon(query, indexes[idx]);
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn score_point_sse(&self, query: &EncodedQuery, i: usize) -> f32 {
        unsafe {
            let (vector_offset, v_ptr) = self.get_vec_ptr(i);
            let score = impl_score_dot_sse(
                query.encoded_query.as_ptr() as *const u8,
                v_ptr,
                self.metadata.dim as u32,
            );
            self.metadata.multiplier * score + query.offset + vector_offset
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn score_points_sse(&self, query: &EncodedQuery, indexes: &[usize], scores: &mut [f32]) {
        unsafe {
            for (indexes, scores) in indexes.chunks_exact(2).zip(scores.chunks_exact_mut(2)) {
                let (vector1_offset, v1_ptr) = self.get_vec_ptr(indexes[0]);
                let (vector2_offset, v2_ptr) = self.get_vec_ptr(indexes[1]);
                impl_score_pair_dot_sse(
                    query.encoded_query.as_ptr() as *const u8,
                    v1_ptr,
                    v2_ptr,
                    self.metadata.dim as u32,
                    scores.as_mut_ptr(),
                );
                scores[0] = self.metadata.multiplier * scores[0] + query.offset + vector1_offset;
                scores[1] = self.metadata.multiplier * scores[1] + query.offset + vector2_offset;
            }
            if indexes.len() % 2 == 1 {
                let idx = indexes.len() - 1;
                scores[idx] = self.score_point_avx(query, indexes[idx]);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    pub fn score_point_avx(&self, query: &EncodedQuery, i: usize) -> f32 {
        unsafe {
            let (vector_offset, v_ptr) = self.get_vec_ptr(i);
            let score = impl_score_dot_avx(
                query.encoded_query.as_ptr() as *const u8,
                v_ptr,
                self.metadata.dim as u32,
            );
            self.metadata.multiplier * score + query.offset + vector_offset
        }
    }

    #[cfg(target_arch = "x86_64")]
    pub fn score_points_avx(&self, query: &EncodedQuery, indexes: &[usize], scores: &mut [f32]) {
        unsafe {
            for (indexes, scores) in indexes.chunks_exact(2).zip(scores.chunks_exact_mut(2)) {
                let (vector1_offset, v1_ptr) = self.get_vec_ptr(indexes[0]);
                let (vector2_offset, v2_ptr) = self.get_vec_ptr(indexes[1]);
                impl_score_pair_dot_avx(
                    query.encoded_query.as_ptr() as *const u8,
                    v1_ptr,
                    v2_ptr,
                    self.metadata.dim as u32,
                    scores.as_mut_ptr(),
                );
                scores[0] = self.metadata.multiplier * scores[0] + query.offset + vector1_offset;
                scores[1] = self.metadata.multiplier * scores[1] + query.offset + vector2_offset;
            }
            if indexes.len() % 2 == 1 {
                let idx = indexes.len() - 1;
                scores[idx] = self.score_point_avx(query, indexes[idx]);
            }
        }
    }

    fn find_alpha_offset_size_dim<'a>(orig_data: impl IntoIterator<Item = &'a [f32]>) -> (f32, f32, usize, usize) {
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
        let alpha = (max - min) / 127.0;
        let offset = min;
        (alpha, offset, count, dim)
    }

    fn f32_to_u8(i: f32, alpha: f32, offset: f32) -> u8 {
        let i = (i - offset) / alpha;
        i.clamp(0.0, 127.0) as u8
    }

    #[inline]
    fn get_vec_ptr(&self, i: usize) -> (f32, *const u8) {
        unsafe {
            let vector_data_size = self.metadata.dim + std::mem::size_of::<f32>();
            let v_ptr = self.encoded_vectors.ptr().add(i * vector_data_size);
            let vector_offset = *(v_ptr as *const f32);
            (vector_offset, v_ptr.add(std::mem::size_of::<f32>()))
        }
    }
}

impl Storage for Vec<u8> {
    fn ptr(&self) -> *const u8 {
        self.as_ptr()
    }

    fn as_slice(&self) -> &[u8] {
        self.as_slice()
    }

    fn from_file(path: &Path) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        Ok(buffer)
    }
}

impl StorageBuilder<Vec<u8>> for Vec<u8> {
    fn build(self) -> Vec<u8> {
        self
    }

    fn extend_from_slice(&mut self, other: &[u8]) {
        self.extend_from_slice(other);
    }
}

#[cfg(target_arch = "x86_64")]
extern "C" {
    fn impl_score_dot_avx(query_ptr: *const u8, vector_ptr: *const u8, dim: u32) -> f32;

    fn impl_score_pair_dot_avx(
        query_ptr: *const u8,
        vector1_ptr: *const u8,
        vector2_ptr: *const u8,
        dim: u32,
        result: *mut f32,
    );

    fn impl_score_dot_sse(query_ptr: *const u8, vector_ptr: *const u8, dim: u32) -> f32;

    fn impl_score_pair_dot_sse(
        query_ptr: *const u8,
        vector1_ptr: *const u8,
        vector2_ptr: *const u8,
        dim: u32,
        result: *mut f32,
    );
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
extern "C" {
    fn impl_score_dot_neon(query_ptr: *const u8, vector_ptr: *const u8, dim: u32) -> f32;

    fn impl_score_pair_dot_neon(
        query_ptr: *const u8,
        vector1_ptr: *const u8,
        vector2_ptr: *const u8,
        dim: u32,
        result: *mut f32,
    );
}
