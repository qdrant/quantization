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

    pub fn score_points_dot_avx(&self, query: &EncodedQuery, indexes: &[usize], scores: &mut [f32]) {
        const CHUNK_SIZE: usize = 2;
        let encoded_vectors_ptr = self.encoded_vectors.as_ptr();
        unsafe {
            for (indexes, scores) in indexes
                .chunks_exact(CHUNK_SIZE)
                .zip(scores.chunks_exact_mut(CHUNK_SIZE))
            {
                let v1_ptr = encoded_vectors_ptr.add(indexes[0] * self.dim);
                let v2_ptr = encoded_vectors_ptr.add(indexes[1] * self.dim);
                impl_score_pair_dot_avx(
                    query.encoded_query.as_ptr() as *const u8,
                    v1_ptr,
                    v2_ptr,
                    self.dim as u32,
                    ALPHA,
                    OFFSET,
                    scores.as_mut_ptr(),
                );
            }
        }
    }

    pub fn score_points_dot_sse(&self, query: &EncodedQuery, indexes: &[usize], scores: &mut [f32]) {
        const CHUNK_SIZE: usize = 2;
        let encoded_vectors_ptr = self.encoded_vectors.as_ptr();
        unsafe {
            for (indexes, scores) in indexes
                .chunks_exact(CHUNK_SIZE)
                .zip(scores.chunks_exact_mut(CHUNK_SIZE))
            {
                let mut q_ptr = query.encoded_query.as_ptr() as *const __m128i;
                let mut v1_ptr = encoded_vectors_ptr.add(indexes[0] * self.dim) as *const __m128i;
                let mut v2_ptr = encoded_vectors_ptr.add(indexes[1] * self.dim) as *const __m128i;

                let mut sum1 = _mm_setzero_si128();
                let mut sum2 = _mm_setzero_si128();
                let mut mul1 = _mm_setzero_si128();
                let mut mul2 = _mm_setzero_si128();
                let mask_epu16 = _mm_set1_epi16(0xFF);
                let mask_epu32 = _mm_set1_epi32(0xFFFF);
                for _ in 0..self.dim / 16 {
                    let q = _mm_loadu_si128(q_ptr);
                    q_ptr = q_ptr.add(1);
                    let q1 = _mm_and_si128(q, mask_epu16);
                    let q2 = _mm_srli_epi16(q, 8);

                    {
                        let v = _mm_loadu_si128(v1_ptr);
                        v1_ptr = v1_ptr.add(1);

                        let v1 = _mm_and_si128(v, mask_epu16);

                        let m1 = _mm_mullo_epi16(v1, q1);
                        let s1 = _mm_adds_epu16(v1, q1);

                        sum1 = _mm_add_epi32(sum1, _mm_and_si128(s1, mask_epu32));
                        sum1 = _mm_add_epi32(sum1, _mm_srli_epi32(s1, 16));

                        mul1 = _mm_add_epi32(mul1, _mm_and_si128(m1, mask_epu32));
                        mul1 = _mm_add_epi32(mul1, _mm_srli_epi32(m1, 16));

                        let v2 = _mm_srli_epi16(v, 8);

                        let m2 = _mm_mullo_epi16(v2, q2);
                        let s2 = _mm_adds_epu16(v2, q2);

                        sum1 = _mm_add_epi32(sum1, _mm_and_si128(s2, mask_epu32));
                        sum1 = _mm_add_epi32(sum1, _mm_srli_epi32(s2, 16));

                        mul1 = _mm_add_epi32(mul1, _mm_and_si128(m2, mask_epu32));
                        mul1 = _mm_add_epi32(mul1, _mm_srli_epi32(m2, 16));
                    }

                    {
                        let v = _mm_loadu_si128(v2_ptr);
                        v2_ptr = v2_ptr.add(1);

                        let v1 = _mm_and_si128(v, mask_epu16);

                        let m1 = _mm_mullo_epi16(v1, q1);
                        let s1 = _mm_adds_epu16(v1, q1);

                        sum2 = _mm_add_epi32(sum2, _mm_and_si128(s1, mask_epu32));
                        sum2 = _mm_add_epi32(sum2, _mm_srli_epi32(s1, 16));

                        mul2 = _mm_add_epi32(mul2, _mm_and_si128(m1, mask_epu32));
                        mul2 = _mm_add_epi32(mul2, _mm_srli_epi32(m1, 16));

                        let v2 = _mm_srli_epi16(v, 8);

                        let m2 = _mm_mullo_epi16(v2, q2);
                        let s2 = _mm_adds_epu16(v2, q2);

                        sum2 = _mm_add_epi32(sum2, _mm_and_si128(s2, mask_epu32));
                        sum2 = _mm_add_epi32(sum2, _mm_srli_epi32(s2, 16));

                        mul2 = _mm_add_epi32(mul2, _mm_and_si128(m2, mask_epu32));
                        mul2 = _mm_add_epi32(mul2, _mm_srli_epi32(m2, 16));
                    }
                }
                let mul1 = Self::hsum128_ps_sse(_mm_cvtepi32_ps(mul1));
                let sum1 = Self::hsum128_ps_sse(_mm_cvtepi32_ps(sum1));
                let mul2 = Self::hsum128_ps_sse(_mm_cvtepi32_ps(mul2));
                let sum2 = Self::hsum128_ps_sse(_mm_cvtepi32_ps(sum2));
                scores[0] = ALPHA * ALPHA * mul1
                    + ALPHA * OFFSET * sum1
                    + OFFSET * OFFSET * self.dim as f32;
                scores[1] = ALPHA * ALPHA * mul2
                    + ALPHA * OFFSET * sum2
                    + OFFSET * OFFSET * self.dim as f32;
            }
        }
    }

    pub fn score_point_dot_sse(&self, query: &EncodedQuery, i: usize) -> f32 {
        unsafe {
            let (vector_offset, v_ptr) = self.get_vec_ptr(i);
            let mut v_ptr = v_ptr as *const __m128i;
            let mut q_ptr = query.encoded_query.as_ptr() as *const __m128i;

            let mut mul1 = _mm_setzero_si128();
            let mut mul2 = _mm_setzero_si128();
            for _ in 0..self.dim / 16 {
                let v = _mm_loadu_si128(v_ptr);
                let q = _mm_loadu_si128(q_ptr);
                v_ptr = v_ptr.add(1);
                q_ptr = q_ptr.add(1);

                let s = _mm_maddubs_epi16(v, q);
                let s_low = _mm_cvtepi16_epi32(s);
                let s_high = _mm_cvtepi16_epi32(_mm_srli_si128(s, 8));
                mul1 = _mm_add_epi32(mul1, s_low);
                mul2 = _mm_add_epi32(mul2, s_high);
            }
            let mul = _mm_add_epi32(mul1, mul2);
            let mul = Self::hsum128_ps_sse(_mm_cvtepi32_ps(mul));
            ALPHA * ALPHA * mul + query.offset + vector_offset
        }
    }

    pub fn score_point_dot_avx(&self, query: &EncodedQuery, i: usize) -> f32 {
        unsafe {
            impl_score_dot_avx(
                query.encoded_query.as_ptr() as *const u8,
                self.encoded_vectors.as_ptr().add(i * self.dim),
                self.dim as u32,
                ALPHA,
                OFFSET,
            )
        }
    }

    pub fn score_point_dot_avx_2(&self, query: &EncodedQuery, i: usize) -> f32 {
        unsafe {
            let (vector_offset, v_ptr) = self.get_vec_ptr(i);
            impl_score_dot_avx_2(
                query.encoded_query.as_ptr() as *const u8,
                v_ptr,
                self.dim as u32,
                ALPHA * ALPHA,
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

    fn impl_score_dot_avx_2(
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
        offset: f32,
        result: *mut f32,
    );
}
