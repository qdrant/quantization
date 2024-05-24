use std::arch::x86_64::*;

use super::hsum128_epi16_sse;

#[target_feature(enable = "sse2")]
#[allow(clippy::missing_safety_doc)]
pub unsafe fn impl_score_dot_sse(query_ptr: *const u8, vector_ptr: *const u8, dim: u32) -> f32 {
    let mut v_ptr = vector_ptr as *const __m128i;
    let mut q_ptr = query_ptr as *const __m128i;

    let mut mul = _mm_setzero_si128();
    for _ in 0..dim / 16 {
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
    hsum128_epi16_sse(mul) as f32
}
