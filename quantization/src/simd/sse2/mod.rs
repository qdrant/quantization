pub mod dot_u8;
pub mod manhattan_u8;
pub mod xor_popcnt;

use std::arch::x86_64::*;

#[target_feature(enable = "sse2")]
#[allow(clippy::missing_safety_doc)]
pub unsafe fn hsum128_epi16_sse(x: __m128i) -> i32 {
    let x64 = _mm_add_epi16(x, _mm_srli_si128(x, 8));
    let x32 = _mm_add_epi16(x64, _mm_srli_si128(x64, 4));
    _mm_extract_epi16(x32, 0) + _mm_extract_epi16(x32, 1)
}
