pub const ALIGHMENT: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceType {
    L2,
    Cosine,
}

pub struct EncodedVectors {
    pub encoded_vectors: Vec<u8>,
    pub dim: usize,
    pub actual_dim: usize,
    pub alpha: f32,
    pub offset: f32,
    pub distance_type: DistanceType,
    pub multiplier: f32,
}

pub struct EncodedQuery {
    pub offset: f32,
    pub encoded_query: Vec<u8>,
}

impl EncodedVectors {
    pub fn new<'a>(
        orig_data: impl IntoIterator<Item = &'a [f32]> + Clone,
        vectors_count: usize,
        dim: usize,
        distance_type: DistanceType,
    ) -> Result<EncodedVectors, String> {
        let (alpha, offset) = Self::find_alpha_offset(orig_data.clone());
        let extended_dim = dim + (ALIGHMENT - dim % ALIGHMENT) % ALIGHMENT;
        let mut encoded_vectors = Vec::with_capacity(vectors_count * dim);
        for vector in orig_data {
            let mut encoded_vector = Vec::new();
            for &value in vector {
                let endoded = Self::f32_to_u8(value, alpha, offset);
                encoded_vector.push(endoded);
            }
            if dim % ALIGHMENT != 0 {
                for _ in 0..(ALIGHMENT - dim % ALIGHMENT) {
                    let placeholder = match distance_type {
                        DistanceType::Cosine => 0.0,
                        DistanceType::L2 => offset,
                    };
                    let endoded = Self::f32_to_u8(placeholder, alpha, offset);
                    encoded_vector.push(endoded);
                }
            }
            let vector_offset = match distance_type {
                DistanceType::Cosine => {
                    extended_dim as f32 * offset * offset
                        + encoded_vector.iter().map(|&x| x as f32).sum::<f32>() * alpha * offset
                }
                DistanceType::L2 => {
                    extended_dim as f32 * offset * offset
                        + encoded_vector
                            .iter()
                            .map(|&x| x as f32 * x as f32)
                            .sum::<f32>()
                            * alpha
                            * alpha
                }
            };
            encoded_vectors.extend_from_slice(&vector_offset.to_ne_bytes());
            encoded_vectors.extend_from_slice(&encoded_vector);
        }
        let multiplier = match distance_type {
            DistanceType::Cosine => alpha * alpha,
            DistanceType::L2 => -2.0 * alpha * alpha,
        };

        Ok(EncodedVectors {
            encoded_vectors,
            dim: extended_dim,
            actual_dim: dim,
            alpha,
            offset,
            distance_type,
            multiplier,
        })
    }

    fn find_alpha_offset<'a>(orig_data: impl IntoIterator<Item = &'a [f32]>) -> (f32, f32) {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for vector in orig_data {
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
        (alpha, offset)
    }

    fn f32_to_u8(i: f32, alpha: f32, offset: f32) -> u8 {
        let i = (i - offset) / alpha;
        let i = if i > 127.0 {
            127.0
        } else if i < 0.0 {
            0.0
        } else {
            i
        };
        i as u8
    }

    pub fn encode_query(&self, query: &[f32]) -> EncodedQuery {
        let dim = query.len();
        let mut query: Vec<_> = query
            .iter()
            .map(|&v| Self::f32_to_u8(v, self.alpha, self.offset))
            .collect();
        if dim % ALIGHMENT != 0 {
            for _ in 0..(ALIGHMENT - dim % ALIGHMENT) {
                let placeholder = match self.distance_type {
                    DistanceType::Cosine => 0.0,
                    DistanceType::L2 => self.offset,
                };
                let endoded = Self::f32_to_u8(placeholder, self.alpha, self.offset);
                query.push(endoded);
            }
        }
        let offset = match self.distance_type {
            DistanceType::Cosine => {
                query.iter().map(|&x| x as f32).sum::<f32>() * self.alpha * self.offset
            }
            DistanceType::L2 => {
                query.iter().map(|&x| x as f32 * x as f32).sum::<f32>() * self.alpha * self.alpha
            }
        };
        EncodedQuery {
            offset,
            encoded_query: query,
        }
    }

    pub fn score_point_dot(&self, query: &EncodedQuery, i: usize) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
                return self.score_point_dot_avx(query, i);
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("sse") {
                return self.score_point_dot_sse(query, i);
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return self.score_point_dot_neon(query, i);
            }
        }

        self.score_point_dot_simple(query, i)
    }

    pub fn score_points_dot(&self, query: &EncodedQuery, i: &[usize], scores: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
                return self.score_points_dot_avx(query, i, scores);
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("sse") {
                return self.score_points_dot_sse(query, i, scores);
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return self.score_points_dot_neon(query, i, scores);
            }
        }

        self.score_points_dot_simple(query, i, scores)
    }

    pub fn score_point_dot_simple(&self, query: &EncodedQuery, i: usize) -> f32 {
        unsafe {
            let (vector_offset, v_ptr) = self.get_vec_ptr(i);
            let mut mul = 0i32;
            for i in 0..self.dim {
                mul += query.encoded_query[i] as i32 * (*v_ptr.add(i)) as i32;
            }
            self.multiplier * mul as f32 + query.offset + vector_offset
        }
    }

    pub fn score_points_dot_simple(&self, query: &EncodedQuery, i: &[usize], scores: &mut [f32]) {
        for (i, score) in i.iter().zip(scores.iter_mut()) {
            *score = self.score_point_dot_simple(query, *i);
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    pub fn score_point_dot_neon(&self, query: &EncodedQuery, i: usize) -> f32 {
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
    pub fn score_points_dot_neon(
        &self,
        query: &EncodedQuery,
        indexes: &[usize],
        scores: &mut [f32],
    ) {
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
                scores[idx] = self.score_point_dot_neon(query, indexes[idx]);
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn score_point_dot_sse(&self, query: &EncodedQuery, i: usize) -> f32 {
        unsafe {
            let (vector_offset, v_ptr) = self.get_vec_ptr(i);
            let score = impl_score_dot_sse(
                query.encoded_query.as_ptr() as *const u8,
                v_ptr,
                self.dim as u32,
            );
            self.multiplier * score + query.offset + vector_offset
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn score_points_dot_sse(
        &self,
        query: &EncodedQuery,
        indexes: &[usize],
        scores: &mut [f32],
    ) {
        unsafe {
            for (indexes, scores) in indexes.chunks_exact(2).zip(scores.chunks_exact_mut(2)) {
                let (vector1_offset, v1_ptr) = self.get_vec_ptr(indexes[0]);
                let (vector2_offset, v2_ptr) = self.get_vec_ptr(indexes[1]);
                impl_score_pair_dot_sse(
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
                scores[idx] = self.score_point_dot_avx(query, indexes[idx]);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    pub fn score_point_dot_avx(&self, query: &EncodedQuery, i: usize) -> f32 {
        unsafe {
            let (vector_offset, v_ptr) = self.get_vec_ptr(i);
            let score = impl_score_dot_avx(
                query.encoded_query.as_ptr() as *const u8,
                v_ptr,
                self.dim as u32,
            );
            self.multiplier * score + query.offset + vector_offset
        }
    }

    #[cfg(target_arch = "x86_64")]
    pub fn score_points_dot_avx(
        &self,
        query: &EncodedQuery,
        indexes: &[usize],
        scores: &mut [f32],
    ) {
        unsafe {
            for (indexes, scores) in indexes.chunks_exact(2).zip(scores.chunks_exact_mut(2)) {
                let (vector1_offset, v1_ptr) = self.get_vec_ptr(indexes[0]);
                let (vector2_offset, v2_ptr) = self.get_vec_ptr(indexes[1]);
                impl_score_pair_dot_avx(
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
                scores[idx] = self.score_point_dot_avx(query, indexes[idx]);
            }
        }
    }

    #[inline]
    fn get_vec_ptr(&self, i: usize) -> (f32, *const u8) {
        unsafe {
            let vector_data_size = self.dim + std::mem::size_of::<f32>();
            let v_ptr = self.encoded_vectors.as_ptr().add(i * vector_data_size);
            let vector_offset = *(v_ptr as *const f32);
            (vector_offset, v_ptr.add(std::mem::size_of::<f32>()))
        }
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
