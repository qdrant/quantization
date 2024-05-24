use std::arch::x86_64::*;

use super::hsum256_epi32_avx;

#[target_feature(enable = "avx2")]
#[allow(clippy::missing_safety_doc)]
pub unsafe fn impl_score_dot_avx(query_ptr: *const u8, vector_ptr: *const u8, dim: u32) -> f32 {
    let mut v_ptr = vector_ptr as *const __m256i;
    let mut q_ptr = query_ptr as *const __m256i;

    let mut mul1 = _mm256_setzero_si256();
    let mask_epu32 = _mm256_set1_epi32(0xFFFF);
    for _ in 0..dim / 32 {
        let v = _mm256_loadu_si256(v_ptr);
        let q = _mm256_loadu_si256(q_ptr);
        v_ptr = v_ptr.add(1);
        q_ptr = q_ptr.add(1);

        let s = _mm256_maddubs_epi16(v, q);
        let s_low = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(s));
        let s_high = _mm256_cvtepi16_epi32(_mm256_extractf128_si256(s, 1));
        mul1 = _mm256_add_epi32(mul1, s_low);
        mul1 = _mm256_add_epi32(mul1, s_high);
    }

    // the vector sizes are assumed to be multiples of 16, check if one last 16-element part remaining
    if dim % 32 != 0 {
        let v_short = _mm_loadu_si128(v_ptr as *const __m128i);
        let q_short = _mm_loadu_si128(q_ptr as *const __m128i);

        let v1 = _mm256_cvtepu8_epi16(v_short);
        let q1 = _mm256_cvtepu8_epi16(q_short);

        let s = _mm256_mullo_epi16(v1, q1);
        mul1 = _mm256_add_epi32(mul1, _mm256_and_si256(s, mask_epu32));
        mul1 = _mm256_add_epi32(mul1, _mm256_srli_epi32(s, 16));
    }

    hsum256_epi32_avx(mul1) as f32
}
