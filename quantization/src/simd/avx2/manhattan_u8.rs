use std::arch::x86_64::*;

use super::hsum256_epi32_avx;

#[target_feature(enable = "avx2")]
#[allow(clippy::missing_safety_doc)]
pub unsafe fn impl_score_l1_avx(query_ptr: *const u8, vector_ptr: *const u8, dim: u32) -> f32 {
    let mut v_ptr = vector_ptr as *const __m256i;
    let mut q_ptr = query_ptr as *const __m256i;

    let mut sum256 = _mm256_setzero_si256();
    for _ in 0..dim / 32 {
        let v = _mm256_loadu_si256(v_ptr);
        let q = _mm256_loadu_si256(q_ptr);
        v_ptr = v_ptr.add(1);
        q_ptr = q_ptr.add(1);

        // Compute the difference in both directions and take the maximum for abs
        let diff1 = _mm256_subs_epu8(v, q);
        let diff2 = _mm256_subs_epu8(q, v);

        let abs_diff = _mm256_max_epu8(diff1, diff2);

        let abs_diff16_lo = _mm256_unpacklo_epi8(abs_diff, _mm256_setzero_si256());
        let abs_diff16_hi = _mm256_unpackhi_epi8(abs_diff, _mm256_setzero_si256());

        sum256 = _mm256_add_epi16(sum256, abs_diff16_lo);
        sum256 = _mm256_add_epi16(sum256, abs_diff16_hi);
    }

    // the vector sizes are assumed to be multiples of 16, check if one last 16-element part remaining
    if dim % 32 != 0 {
        let v_short = _mm_loadu_si128(v_ptr as *const __m128i);
        let q_short = _mm_loadu_si128(q_ptr as *const __m128i);

        let diff1 = _mm_subs_epu8(v_short, q_short);
        let diff2 = _mm_subs_epu8(q_short, v_short);

        let abs_diff = _mm_max_epu8(diff1, diff2);

        let abs_diff16_lo_128 = _mm_unpacklo_epi8(abs_diff, _mm_setzero_si128());
        let abs_diff16_hi_128 = _mm_unpackhi_epi8(abs_diff, _mm_setzero_si128());

        let abs_diff16_lo = _mm256_cvtepu16_epi32(abs_diff16_lo_128);
        let abs_diff16_hi = _mm256_cvtepu16_epi32(abs_diff16_hi_128);

        sum256 = _mm256_add_epi16(sum256, abs_diff16_lo);
        sum256 = _mm256_add_epi16(sum256, abs_diff16_hi);
    }

    let sum_epi32 = _mm256_add_epi32(
        _mm256_unpacklo_epi16(sum256, _mm256_setzero_si256()),
        _mm256_unpackhi_epi16(sum256, _mm256_setzero_si256()));

    hsum256_epi32_avx(sum_epi32) as f32
}
