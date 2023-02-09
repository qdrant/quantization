use permutation_iterator::Permutor;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

pub const ALIGHMENT: usize = 16;
pub const FILE_HEADER_MAGIC_NUMBER: u64 = 0x00_DD_91_12_FA_BB_09_01;
pub const QUANTILE_SAMPLE_SIZE: usize = 100_000;

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimilarityType {
    Dot,
    L2,
}

#[derive(Serialize, Deserialize)]
pub struct EncodingParameters {
    pub dim: usize,
    pub distance_type: SimilarityType,
    pub invert: bool,
    pub quantile: Option<f32>,
}

pub trait Storage {
    fn get_vector_data(&self, index: usize, vector_size: usize) -> &[u8];

    fn from_file(path: &Path, encoding_parameters: &EncodingParameters) -> std::io::Result<Self>
    where
        Self: Sized;

    fn save_to_file(&self, path: &Path) -> std::io::Result<()>;
}

pub trait StorageBuilder<TStorage: Storage> {
    fn build(self) -> TStorage;

    fn push_vector_data(&mut self, other: &[u8]);
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
    actual_dim: usize,
    alpha: f32,
    offset: f32,
    multiplier: f32,
    encoding_parameters: EncodingParameters,
}

impl<TStorage: Storage> EncodedVectors<TStorage> {
    pub fn save(&self, data_path: &Path, meta_path: &Path) -> std::io::Result<()> {
        let metadata_bytes = serde_json::to_vec(&self.metadata)?;
        meta_path.parent().map(|p| std::fs::create_dir_all(p));
        let mut buffer = File::create(meta_path)?;
        buffer.write(metadata_bytes.as_slice())?;

        data_path.parent().map(|p| std::fs::create_dir_all(p));
        self.encoded_vectors.save_to_file(data_path)?;
        Ok(())
    }

    pub fn load(
        data_path: &Path,
        meta_path: &Path,
        encoding_parameters: &EncodingParameters,
    ) -> std::io::Result<Self> {
        let mut contents = String::new();
        let mut file = File::open(meta_path)?;
        file.read_to_string(&mut contents)?;
        let metadata: Metadata = serde_json::from_str(&contents)?;
        let encoded_vectors = TStorage::from_file(data_path, encoding_parameters)?;
        let result = Self {
            metadata,
            encoded_vectors,
        };
        Ok(result)
    }

    pub fn encode<'a>(
        orig_data: impl IntoIterator<Item = &'a [f32]> + Clone,
        mut storage_builder: impl StorageBuilder<TStorage>,
        encoding_parameters: EncodingParameters,
    ) -> Result<Self, String> {
        let (alpha, offset, count, dim) = Self::find_alpha_offset_size_dim(orig_data.clone());
        let (alpha, offset) = if let Some(quantile) = encoding_parameters.quantile {
            Self::find_quantile_interval(orig_data.clone(), dim, count, quantile, alpha, offset)
        } else {
            (alpha, offset)
        };

        let actual_dim = encoding_parameters.get_actual_dim();
        for vector in orig_data {
            let mut encoded_vector = Vec::with_capacity(actual_dim + std::mem::size_of::<f32>());
            encoded_vector.extend_from_slice(&f32::default().to_ne_bytes());
            for &value in vector {
                let endoded = Self::f32_to_u8(value, alpha, offset);
                encoded_vector.push(endoded);
            }
            if dim % ALIGHMENT != 0 {
                for _ in 0..(ALIGHMENT - dim % ALIGHMENT) {
                    let placeholder = match encoding_parameters.distance_type {
                        SimilarityType::Dot => 0.0,
                        SimilarityType::L2 => offset,
                    };
                    let endoded = Self::f32_to_u8(placeholder, alpha, offset);
                    encoded_vector.push(endoded);
                }
            }
            let vector_offset = match encoding_parameters.distance_type {
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
            let vector_offset = if encoding_parameters.invert {
                -vector_offset
            } else {
                vector_offset
            };
            encoded_vector[0..std::mem::size_of::<f32>()]
                .copy_from_slice(&vector_offset.to_ne_bytes());
            storage_builder.push_vector_data(&encoded_vector);
        }
        let multiplier = match encoding_parameters.distance_type {
            SimilarityType::Dot => alpha * alpha,
            SimilarityType::L2 => -2.0 * alpha * alpha,
        };
        let multiplier = if encoding_parameters.invert {
            -multiplier
        } else {
            multiplier
        };

        Ok(EncodedVectors {
            encoded_vectors: storage_builder.build(),
            metadata: Metadata {
                actual_dim,
                alpha,
                offset,
                multiplier,
                encoding_parameters,
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
                let placeholder = match self.metadata.encoding_parameters.distance_type {
                    SimilarityType::Dot => 0.0,
                    SimilarityType::L2 => self.metadata.offset,
                };
                let endoded =
                    Self::f32_to_u8(placeholder, self.metadata.alpha, self.metadata.offset);
                query.push(endoded);
            }
        }
        let offset = match self.metadata.encoding_parameters.distance_type {
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
        let offset = if self.metadata.encoding_parameters.invert {
            -offset
        } else {
            offset
        };
        EncodedQuery {
            offset,
            encoded_query: query,
        }
    }

    pub fn score_point(&self, query: &EncodedQuery, i: u32) -> f32 {
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
                let score = impl_score_dot_neon(q_ptr, v_ptr, self.metadata.dim as u32);
                return self.metadata.multiplier * score + query.offset + vector_offset;
            }
        }

        self.score_point_simple(query, i)
    }

    pub fn score_points(&self, query: &EncodedQuery, i: &[u32], scores: &mut [f32]) {
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

    pub fn score_internal(&self, i: u32, j: u32) -> f32 {
        let (query_offset, q_ptr) = self.get_vec_ptr(i);
        let (vector_offset, v_ptr) = self.get_vec_ptr(j);
        let diff = self.metadata.actual_dim as f32 * self.metadata.offset * self.metadata.offset;
        let diff = if self.metadata.encoding_parameters.invert {
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
                let score = impl_score_dot_neon(q_ptr, v_ptr, self.metadata.dim as u32);
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

    pub fn score_point_simple(&self, query: &EncodedQuery, i: u32) -> f32 {
        unsafe {
            let (vector_offset, v_ptr) = self.get_vec_ptr(i);
            let mut mul = 0i32;
            for i in 0..self.metadata.actual_dim {
                mul += query.encoded_query[i] as i32 * (*v_ptr.add(i)) as i32;
            }
            self.metadata.multiplier * mul as f32 + query.offset + vector_offset
        }
    }

    pub fn score_points_simple(&self, query: &EncodedQuery, i: &[u32], scores: &mut [f32]) {
        for (i, score) in i.iter().zip(scores.iter_mut()) {
            *score = self.score_point_simple(query, *i);
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    pub fn score_point_neon(&self, query: &EncodedQuery, i: u32) -> f32 {
        unsafe {
            let (vector_offset, v_ptr) = self.get_vec_ptr(i);
            let score = impl_score_dot_neon(
                query.encoded_query.as_ptr() as *const u8,
                v_ptr,
                self.metadata.dim as u32,
            );
            self.metadata.multiplier * score + query.offset + vector_offset
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    pub fn score_points_neon(&self, query: &EncodedQuery, indexes: &[u32], scores: &mut [f32]) {
        unsafe {
            for (indexes, scores) in indexes.chunks_exact(2).zip(scores.chunks_exact_mut(2)) {
                let (vector1_offset, v1_ptr) = self.get_vec_ptr(indexes[0]);
                let (vector2_offset, v2_ptr) = self.get_vec_ptr(indexes[1]);
                impl_score_pair_dot_neon(
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
                scores[idx] = self.score_point_neon(query, indexes[idx]);
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn score_point_sse(&self, query: &EncodedQuery, i: u32) -> f32 {
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

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn score_points_sse(&self, query: &EncodedQuery, indexes: &[u32], scores: &mut [f32]) {
        unsafe {
            for (indexes, scores) in indexes.chunks_exact(2).zip(scores.chunks_exact_mut(2)) {
                let (vector1_offset, v1_ptr) = self.get_vec_ptr(indexes[0]);
                let (vector2_offset, v2_ptr) = self.get_vec_ptr(indexes[1]);
                impl_score_pair_dot_sse(
                    query.encoded_query.as_ptr() as *const u8,
                    v1_ptr,
                    v2_ptr,
                    self.metadata.actual_dim as u32,
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
    pub fn score_point_avx(&self, query: &EncodedQuery, i: u32) -> f32 {
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

    #[cfg(target_arch = "x86_64")]
    pub fn score_points_avx(&self, query: &EncodedQuery, indexes: &[u32], scores: &mut [f32]) {
        unsafe {
            for (indexes, scores) in indexes.chunks_exact(2).zip(scores.chunks_exact_mut(2)) {
                let (vector1_offset, v1_ptr) = self.get_vec_ptr(indexes[0]);
                let (vector2_offset, v2_ptr) = self.get_vec_ptr(indexes[1]);
                impl_score_pair_dot_avx(
                    query.encoded_query.as_ptr() as *const u8,
                    v1_ptr,
                    v2_ptr,
                    self.metadata.actual_dim as u32,
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

    pub fn score_pair(&self, query: &EncodedQuery, i1: u32, i2: u32) -> [f32; 2] {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
                let (vector1_offset, v1_ptr) = self.get_vec_ptr(i1);
                let (vector2_offset, v2_ptr) = self.get_vec_ptr(i2);
                let mut scores = [0.0, 0.0];
                impl_score_pair_dot_avx(
                    query.encoded_query.as_ptr() as *const u8,
                    v1_ptr,
                    v2_ptr,
                    self.metadata.actual_dim as u32,
                    scores.as_mut_ptr(),
                );
                return [
                    self.metadata.multiplier * scores[0] + query.offset + vector1_offset,
                    self.metadata.multiplier * scores[1] + query.offset + vector2_offset,
                ];
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
                let (vector1_offset, v1_ptr) = self.get_vec_ptr(i1);
                let (vector2_offset, v2_ptr) = self.get_vec_ptr(i2);
                let mut scores = [0.0, 0.0];
                impl_score_pair_dot_sse(
                    query.encoded_query.as_ptr() as *const u8,
                    v1_ptr,
                    v2_ptr,
                    self.metadata.actual_dim as u32,
                    scores.as_mut_ptr(),
                );
                return [
                    self.metadata.multiplier * scores[0] + query.offset + vector1_offset,
                    self.metadata.multiplier * scores[1] + query.offset + vector2_offset,
                ];
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
                let (vector1_offset, v1_ptr) = self.get_vec_ptr(i1);
                let (vector2_offset, v2_ptr) = self.get_vec_ptr(i2);
                let mut scores = [0.0, 0.0];
                impl_score_pair_dot_neon(
                    query.encoded_query.as_ptr() as *const u8,
                    v1_ptr,
                    v2_ptr,
                    self.metadata.dim as u32,
                    scores.as_mut_ptr(),
                );
                return [
                    self.metadata.multiplier * scores[0] + query.offset + vector1_offset,
                    self.metadata.multiplier * scores[1] + query.offset + vector2_offset,
                ];
            }
        }

        [
            self.score_point_simple(query, i1),
            self.score_point_simple(query, i2),
        ]
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

    fn find_quantile_interval<'a>(
        orig_data: impl IntoIterator<Item = &'a [f32]>,
        dim: usize,
        count: usize,
        quantile: f32,
        alpha: f32,
        offset: f32,
    ) -> (f32, f32) {
        let slice_size = std::cmp::min(count, QUANTILE_SAMPLE_SIZE);
        let permutor = Permutor::new(count as u64);
        let mut selected_vectors: Vec<usize> =
            permutor.map(|i| i as usize).take(slice_size).collect();
        selected_vectors.sort_unstable();

        let mut data_slice = Vec::with_capacity(slice_size * dim);
        let mut vector_index: usize = 0;
        let mut selected_index: usize = 0;
        for vector in orig_data {
            if vector_index == selected_vectors[selected_index] {
                data_slice.extend_from_slice(vector);
                selected_index += 1;
                if selected_index == slice_size {
                    break;
                }
            }
            vector_index += 1;
        }

        let cut_index = (slice_size as f32 * (1.0 - quantile) / 2.0) as usize;
        let comparator = |a: &f32, b: &f32| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
        let data_slice_len = data_slice.len();
        let (selected_values, _, _) =
            data_slice.select_nth_unstable_by(data_slice_len - cut_index, comparator);
        let (_, _, selected_values) = selected_values.select_nth_unstable_by(cut_index, comparator);

        if selected_values.len() < 2 {
            return (alpha, offset);
        }

        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for value in selected_values {
            if *value < min {
                min = *value;
            }
            if *value > max {
                max = *value;
            }
        }

        Self::alpha_offset_from_min_max(min, max)
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
}

impl EncodingParameters {
    pub fn get_vector_data_size(&self) -> usize {
        let actual_dim = self.get_actual_dim();
        actual_dim + std::mem::size_of::<f32>()
    }

    pub fn get_actual_dim(&self) -> usize {
        self.dim + (ALIGHMENT - self.dim % ALIGHMENT) % ALIGHMENT
    }
}

impl Storage for Vec<u8> {
    fn get_vector_data(&self, index: usize, vector_size: usize) -> &[u8] {
        &self[vector_size * index..vector_size * (index + 1)]
    }

    fn from_file(path: &Path, _encoding_parameters: &EncodingParameters) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        Ok(buffer)
    }

    fn save_to_file(&self, path: &Path) -> std::io::Result<()> {
        let mut buffer = File::create(path)?;
        buffer.write_all(self.as_slice())?;
        buffer.flush()?;
        Ok(())
    }
}

impl StorageBuilder<Vec<u8>> for Vec<u8> {
    fn build(self) -> Vec<u8> {
        self
    }

    fn push_vector_data(&mut self, other: &[u8]) {
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
