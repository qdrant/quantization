use std::arch::x86_64::*;

pub const ALIGHMENT: usize = 16;
pub const ALPHA: f32 = 1.0 / 127.0;
pub const OFFSET: f32 = -1.0;

pub struct I8EncodedVectors {
    pub encoded_vectors: Vec<u8>,
    pub dim: usize,
    pub actual_dim: usize,
}

impl I8EncodedVectors {
    pub fn new<'a>(
        orig_data: impl IntoIterator<Item = &'a [f32]> + Clone,
        vectors_count: usize,
        dim: usize,
    ) -> Result<I8EncodedVectors, String> {
        let mut encoded_vectors = Vec::with_capacity(vectors_count * dim);
        for vector in orig_data {
            for &value in vector {
                let endoded = Self::f32_to_u8(value);
                encoded_vectors.push(endoded);
            }
            if dim % ALIGHMENT != 0 {
                for _ in 0..(ALIGHMENT - dim % ALIGHMENT) {
                    let endoded = Self::f32_to_u8(0.0);
                    encoded_vectors.push(endoded);
                }
            }
        }

        Ok(I8EncodedVectors {
            encoded_vectors,
            dim: dim + (ALIGHMENT - dim % ALIGHMENT) % ALIGHMENT,
            actual_dim: dim,
        })
    }

    pub fn f32_to_u8(i: f32) -> u8 {
        let i = (i - OFFSET) / ALPHA;
        let i = if i > 255.0 {
            255.0
        } else if i < 0.0 {
            0.0
        } else {
            i
        };
        i as u8
    }

    pub fn encode_query(query: &[f32]) -> Vec<u8> {
        query.iter().map(|&v| Self::f32_to_u8(v)).collect()
    }

    pub fn score_point_dot(&self, query: &[u8], i: usize) -> f32 {
        self.score_point_dot_sse(query, i)
    }

    pub fn score_point_dot_sse(&self, query: &[u8], i: usize) -> f32 {
        unsafe {
            let mut v_ptr = self.encoded_vectors.as_ptr().add(i * self.dim) as *const __m128i;
            let mut q_ptr = query.as_ptr() as *const __m128i;

            let mut sum1 = _mm_setzero_si128();
            let mut sum2 = _mm_setzero_si128();
            let mut sum3 = _mm_setzero_si128();
            let mut sum4 = _mm_setzero_si128();

            let mut mul1 = _mm_setzero_si128();
            let mut mul2 = _mm_setzero_si128();
            let mut mul3 = _mm_setzero_si128();
            let mut mul4 = _mm_setzero_si128();
            let mask_epu16 = _mm_set1_epi16(0xFF);
            let mask_epu32 = _mm_set1_epi32(0xFFFF);
            for _ in 0..self.dim / 16 {
                let v = _mm_loadu_si128(v_ptr);
                let q = _mm_loadu_si128(q_ptr);
                v_ptr = v_ptr.add(1);
                q_ptr = q_ptr.add(1);

                let v1 = _mm_and_si128(v, mask_epu16);
                let q1 = _mm_and_si128(q, mask_epu16);

                let m1 = _mm_mullo_epi16(v1, q1);
                let s1 = _mm_adds_epu16(v1, q1);

                sum1 = _mm_add_epi32(sum1, _mm_and_si128(s1, mask_epu32));
                sum2 = _mm_add_epi32(sum2, _mm_srli_epi32(s1, 16));

                mul1 = _mm_add_epi32(mul1, _mm_and_si128(m1, mask_epu32));
                mul2 = _mm_add_epi32(mul2, _mm_srli_epi32(m1, 16));
                
                let v2 = _mm_srli_epi16(v, 8);
                let q2 = _mm_srli_epi16(q, 8);

                let m2 = _mm_mullo_epi16(v2, q2);
                let s2 = _mm_adds_epu16(v2, q2);

                sum3 = _mm_add_epi32(sum3, _mm_and_si128(s2, mask_epu32));
                sum4 = _mm_add_epi32(sum4, _mm_srli_epi32(s2, 16));

                mul3 = _mm_add_epi32(mul3, _mm_and_si128(m2, mask_epu32));
                mul4 = _mm_add_epi32(mul4, _mm_srli_epi32(m2, 16));
            }
            let mul = _mm_add_epi32(_mm_add_epi32(mul1, mul2), _mm_add_epi32(mul3, mul4));
            let sum = _mm_add_epi32(_mm_add_epi32(sum1, sum2), _mm_add_epi32(sum3, sum4));
            let mul = Self::hsum128_ps_sse(_mm_cvtepi32_ps(mul));
            let sum = Self::hsum128_ps_sse(_mm_cvtepi32_ps(sum));
            ALPHA * ALPHA * mul + ALPHA * OFFSET * sum + OFFSET * OFFSET * self.dim as f32
        }
    }

    pub fn score_point_dot_avx(&self, query: &[u8], i: usize) -> f32 {
        unsafe {
            let mut v_ptr = self.encoded_vectors.as_ptr().add(i * self.dim) as *const __m128i;
            let mut q_ptr = query.as_ptr() as *const __m128i;
            let mut sum1 = _mm256_setzero_si256();
            let mut sum2 = _mm256_setzero_si256();
            let mut sum3 = _mm256_setzero_si256();
            let mut sum4 = _mm256_setzero_si256();

            let mut mul1 = _mm256_setzero_si256();
            let mut mul2 = _mm256_setzero_si256();
            let mut mul3 = _mm256_setzero_si256();
            let mut mul4 = _mm256_setzero_si256();
            let mask_epu32 = _mm256_set1_epi32(0xFFFF);
            for _ in 0..self.dim / 32 {
                let v1 = _mm_loadu_si128(v_ptr);
                let q1 = _mm_loadu_si128(q_ptr);
                v_ptr = v_ptr.add(1);
                q_ptr = q_ptr.add(1);
                let v2 = _mm_loadu_si128(v_ptr);
                let q2 = _mm_loadu_si128(q_ptr);
                v_ptr = v_ptr.add(1);
                q_ptr = q_ptr.add(1);

                let v1 = _mm256_cvtepu8_epi16(v1);
                let q1 = _mm256_cvtepu8_epi16(q1);
                let m1 = _mm256_mullo_epi16(v1, q1);
                let s1 = _mm256_adds_epu16(v1, q1);

                sum1 = _mm256_add_epi32(sum1, _mm256_and_si256(s1, mask_epu32));
                sum2 = _mm256_add_epi32(sum2, _mm256_srli_epi32(s1, 16));

                mul1 = _mm256_add_epi32(mul1, _mm256_and_si256(m1, mask_epu32));
                mul2 = _mm256_add_epi32(mul2, _mm256_srli_epi32(m1, 16));

                let v2 = _mm256_cvtepu8_epi16(v2);
                let q2 = _mm256_cvtepu8_epi16(q2);
                let m2 = _mm256_mullo_epi16(v2, q2);
                let s2 = _mm256_adds_epu16(v2, q2);

                sum3 = _mm256_add_epi32(sum3, _mm256_and_si256(s2, mask_epu32));
                sum4 = _mm256_add_epi32(sum4, _mm256_srli_epi32(s2, 16));

                mul3 = _mm256_add_epi32(mul3, _mm256_and_si256(m2, mask_epu32));
                mul4 = _mm256_add_epi32(mul4, _mm256_srli_epi32(m2, 16));
            }
            let mul = _mm256_add_epi32(_mm256_add_epi32(mul1, mul2), _mm256_add_epi32(mul3, mul4));
            let sum = _mm256_add_epi32(_mm256_add_epi32(sum1, sum2), _mm256_add_epi32(sum3, sum4));
            let mul = Self::hsum256_ps_avx(_mm256_cvtepi32_ps(mul));
            let sum = Self::hsum256_ps_avx(_mm256_cvtepi32_ps(sum));
            ALPHA * ALPHA * mul + ALPHA * OFFSET * sum + OFFSET * OFFSET * self.dim as f32
        }
    }

    #[inline]
    unsafe fn hsum128_ps_sse(x: __m128) -> f32 {
        let x64: __m128 = _mm_add_ps(x, _mm_movehl_ps(x, x));
        let x32: __m128 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
        _mm_cvtss_f32(x32)
    }

    #[inline]
    unsafe fn hsum256_ps_avx(x: __m256) -> f32 {
        let x128: __m128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
        let x64: __m128 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
        let x32: __m128 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
        _mm_cvtss_f32(x32)
    }
}
