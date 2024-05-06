pub mod dot_u8;
pub mod manhattan_u8;

use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
#[allow(clippy::missing_safety_doc)]
pub unsafe fn hsum256_epi32_avx(x: __m256i) -> i32 {
    let x128: __m128i = _mm_add_epi32(_mm256_extractf128_si256(x, 1), _mm256_castsi256_si128(x));
    let x64: __m128i = _mm_add_epi32(x128, _mm_srli_si128(x128, 8));
    let x32: __m128i = _mm_add_epi32(x64, _mm_srli_si128(x64, 4));
    _mm_cvtsi128_si32(x32) as i32
}
