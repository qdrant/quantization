use std::arch::x86_64::*;

pub const ALIGHMENT: usize = 16;
pub const ALPHA: f32 = 1.0 / 63.0;
pub const OFFSET: f32 = -1.0;
pub const ALPHA_QUERY: f32 = 1.0 / 63.0;
pub const OFFSET_QUERY: f32 = -1.0;

pub struct I8EncodedVectors {
    pub encoded_vectors: Vec<u8>,
    pub dim: usize,
    pub actual_dim: usize,
}

pub struct EncodedQuery {
    pub offset: f32,
    pub encoded_query: Vec<i8>,
}

impl I8EncodedVectors {
    pub fn new<'a>(
        orig_data: impl IntoIterator<Item = &'a [f32]> + Clone,
        vectors_count: usize,
        dim: usize,
    ) -> Result<I8EncodedVectors, String> {
        let extended_dim = dim + (ALIGHMENT - dim % ALIGHMENT) % ALIGHMENT;
        let mut encoded_vectors = Vec::with_capacity(vectors_count * dim);
        for vector in orig_data {
            let mut encoded_vector = Vec::new();
            for &value in vector {
                let endoded = Self::f32_to_u8(value);
                encoded_vector.push(endoded);
            }
            if dim % ALIGHMENT != 0 {
                for _ in 0..(ALIGHMENT - dim % ALIGHMENT) {
                    let endoded = Self::f32_to_u8(0.0);
                    encoded_vector.push(endoded);
                }
            }
            let offset = extended_dim as f32 * OFFSET_QUERY * OFFSET + 
                encoded_vector.iter().map(|&x| x as f32).sum::<f32>() * ALPHA * OFFSET_QUERY;
            encoded_vectors.extend_from_slice(&offset.to_ne_bytes());
            encoded_vectors.extend_from_slice(&encoded_vector);
        }

        Ok(I8EncodedVectors {
            encoded_vectors,
            dim: extended_dim,
            actual_dim: dim,
        })
    }

    pub fn f32_to_u8(i: f32) -> u8 {
        let i = (i - OFFSET) / ALPHA;
        let i = if i > 127.0 {
            127.0
        } else if i < 0.0 {
            0.0
        } else {
            i
        };
        i as u8
    }

    pub fn f32_to_i8(i: f32) -> i8 {
        let i = (i - OFFSET_QUERY) / ALPHA_QUERY;
        let i = if i > 127.0 {
            127.0
        } else if i < 0.0 {
            0.0
        } else {
            i
        };
        i as i8
    }

    pub fn encode_query(query: &[f32]) -> EncodedQuery {
        let dim = query.len();
        let mut query: Vec<_> = query.iter().map(|&v| Self::f32_to_i8(v)).collect();
        if dim % ALIGHMENT != 0 {
            for _ in 0..(ALIGHMENT - dim % ALIGHMENT) {
                let endoded = Self::f32_to_i8(0.0);
                query.push(endoded);
            }
        }
        let offset = query.iter().map(|&x| x as f32).sum::<f32>() * ALPHA_QUERY * OFFSET;
        EncodedQuery {
            offset,
            encoded_query: query,
        }
    }

    pub fn score_point_dot(&self, query: &EncodedQuery, i: usize) -> f32 {
        self.score_point_dot_avx(query, i)
    }

    pub fn score_points_dot(&self, query: &EncodedQuery, i: &[usize], scores: &mut [f32]) {
        self.score_points_dot_avx(query, i, scores)
    }

    pub fn score_points_dot_sse(&self, query: &EncodedQuery, indexes: &[usize], scores: &mut [f32]) {
        unsafe {
            for (indexes, scores) in indexes
                .chunks_exact(2)
                .zip(scores.chunks_exact_mut(2))
            {
                let mut q_ptr = query.encoded_query.as_ptr() as *const __m128i;
                let (vector1_offset, v1_ptr) = self.get_vec_ptr(indexes[0]);
                let (vector2_offset, v2_ptr) = self.get_vec_ptr(indexes[1]);
                let mut v1_ptr = v1_ptr as *const __m128i;
                let mut v2_ptr = v2_ptr as *const __m128i;

                let mut mul1 = _mm_setzero_si128();
                let mut mul2 = _mm_setzero_si128();
                for _ in 0..self.dim / 16 {
                    let v1 = _mm_loadu_si128(v1_ptr);
                    let v2 = _mm_loadu_si128(v2_ptr);
                    let q = _mm_loadu_si128(q_ptr);
                    v1_ptr = v1_ptr.add(1);
                    v2_ptr = v2_ptr.add(1);
                    q_ptr = q_ptr.add(1);
    
                    let s1 = _mm_maddubs_epi16(v1, q);
                    let s2 = _mm_maddubs_epi16(v2, q);
                    let s1_low = _mm_cvtepi16_epi32(s1);
                    let s1_high = _mm_cvtepi16_epi32(_mm_srli_si128(s1, 8));
                    let s2_low = _mm_cvtepi16_epi32(s2);
                    let s2_high = _mm_cvtepi16_epi32(_mm_srli_si128(s2, 8));
                    mul1 = _mm_add_epi32(mul1, s1_low);
                    mul1 = _mm_add_epi32(mul1, s1_high);
                    mul2 = _mm_add_epi32(mul2, s2_low);
                    mul2 = _mm_add_epi32(mul2, s2_high);
                }
                let mul1 = Self::hsum128_ps_sse(_mm_cvtepi32_ps(mul1));
                let mul2 = Self::hsum128_ps_sse(_mm_cvtepi32_ps(mul2));
                scores[0] = ALPHA * ALPHA_QUERY * mul1 + query.offset + vector1_offset;
                scores[1] = ALPHA * ALPHA_QUERY * mul2 + query.offset + vector2_offset;
            }
        }
    }

    pub fn score_point_dot_sse(&self, query: &EncodedQuery, i: usize) -> f32 {
        unsafe {
            let (vector_offset, v_ptr) = self.get_vec_ptr(i);
            let mut v_ptr = v_ptr as *const __m128i;
            let mut q_ptr = query.encoded_query.as_ptr() as *const __m128i;

            let mut mul = _mm_setzero_si128();
            for _ in 0..self.dim / 16 {
                let v = _mm_loadu_si128(v_ptr);
                let q = _mm_loadu_si128(q_ptr);
                v_ptr = v_ptr.add(1);
                q_ptr = q_ptr.add(1);

                let s = _mm_maddubs_epi16(v, q);
                let s_low = _mm_cvtepi16_epi32(s);
                let s_high = _mm_cvtepi16_epi32(_mm_srli_si128(s, 8));
                mul = _mm_add_epi32(mul, s_low);
                mul = _mm_add_epi32(mul, s_high);
            }
            let mul = Self::hsum128_ps_sse(_mm_cvtepi32_ps(mul));
            ALPHA * ALPHA_QUERY * mul + query.offset + vector_offset
        }
    }

    pub fn score_points_dot_avx(&self, query: &EncodedQuery, indexes: &[usize], scores: &mut [f32]) {
        unsafe {
            for (indexes, scores) in indexes
                .chunks_exact(2)
                .zip(scores.chunks_exact_mut(2))
            {
                let (vector1_offset, v1_ptr) = self.get_vec_ptr(indexes[0]);
                let (vector2_offset, v2_ptr) = self.get_vec_ptr(indexes[1]);
                impl_score_pair_dot_avx(
                    query.encoded_query.as_ptr() as *const u8,
                    v1_ptr,
                    v2_ptr,
                    self.dim as u32,
                    ALPHA * ALPHA,
                    query.offset + vector1_offset,
                    query.offset + vector2_offset,
                    scores.as_mut_ptr(),
                );
            }
        }
    }

    pub fn score_point_dot_avx(&self, query: &EncodedQuery, i: usize) -> f32 {
        unsafe {
            let (vector_offset, v_ptr) = self.get_vec_ptr(i);
            impl_score_dot_avx(
                query.encoded_query.as_ptr() as *const u8,
                v_ptr,
                self.dim as u32,
                ALPHA * ALPHA_QUERY,
                query.offset + vector_offset,
            )
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

    #[inline]
    unsafe fn hsum128_ps_sse(x: __m128) -> f32 {
        let x64: __m128 = _mm_add_ps(x, _mm_movehl_ps(x, x));
        let x32: __m128 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
        _mm_cvtss_f32(x32)
    }
}

extern "C" {
    fn impl_score_dot_avx(
        query_ptr: *const u8,
        vector_ptr: *const u8,
        dim: u32,
        alpha: f32,
        offset: f32,
    ) -> f32;

    fn impl_score_pair_dot_avx(
        query_ptr: *const u8,
        vector1_ptr: *const u8,
        vector2_ptr: *const u8,
        dim: u32,
        alpha: f32,
        offset1: f32,
        offset2: f32,
        result: *mut f32,
    );
}
