use std::arch::x86_64::*;

use super::hsum128_epi16_sse;

#[target_feature(enable = "sse2")]
#[allow(clippy::missing_safety_doc)]
pub unsafe fn impl_score_l1_sse(query_ptr: *const u8, vector_ptr: *const u8, dim: u32) -> f32 {
    let mut v_ptr = vector_ptr as *const __m128i;
    let mut q_ptr = query_ptr as *const __m128i;

    let mut sum128 = _mm_setzero_si128();
    // the vector sizes are assumed to be multiples of 16, no remaining part here
    for _ in 0..dim / 16 {
        let vec2 = _mm_loadu_si128(v_ptr);
        let vec1 = _mm_loadu_si128(q_ptr);
        v_ptr = v_ptr.add(1);
        q_ptr = q_ptr.add(1);

        // Compute the difference in both directions
        let diff1 = _mm_subs_epu8(vec1, vec2);
        let diff2 = _mm_subs_epu8(vec2, vec1);

        // Take the maximum
        let abs_diff = _mm_max_epu8(diff1, diff2);

        let abs_diff16_low = _mm_unpacklo_epi8(abs_diff, _mm_setzero_si128());
        let abs_diff16_high = _mm_unpackhi_epi8(abs_diff, _mm_setzero_si128());

        sum128 = _mm_add_epi16(sum128, abs_diff16_low);
        sum128 = _mm_add_epi16(sum128, abs_diff16_high);
    }

    // Convert 16-bit sums to 32-bit and sum them up
    let sum_epi32 = _mm_add_epi32(
        _mm_unpacklo_epi16(sum128, _mm_setzero_si128()),
        _mm_unpackhi_epi16(sum128, _mm_setzero_si128()));

    hsum128_epi16_sse(sum_epi32) as f32
}
