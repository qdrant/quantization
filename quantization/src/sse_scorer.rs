use crate::{encoded_vectors::CompressedLookupTable, scorer::Scorer};
use std::arch::x86_64::*;

pub struct SseScorer<'a> {
    lut: CompressedLookupTable<'a>,
}

impl Scorer for SseScorer<'_> {
    //#[inline]
    fn score_point(&self, point: usize) -> f32 {
        unsafe {
            let v = self.lut.encoded_vectors.get(point as usize);
            let codes_count = v.len();

            let mut codes_ptr = v.as_ptr() as *const __m128i;
            let mut lut_ptr = self.lut.centroid_distances.as_ptr();

            let mut sum: __m128i = _mm_setzero_si128();
            for _ in 0..codes_count / 16 {
                let codes = _mm_loadu_si128(codes_ptr);
                let low_4bits_mask = _mm_set1_epi8(0x0F);
                let mut codes_low = _mm_and_si128(codes, low_4bits_mask);
                let codes_shft = _mm_srli_epi16(codes, 4);
                let mut codes_high = _mm_and_si128(codes_shft, low_4bits_mask);

                let high_mask = _mm_set_epi32(0xFF, 0, 0xFF, 0);
                let low_mask = _mm_set_epi32(0, 0xFF, 0, 0xFF);
                for _ in 0..16 {
                    let alpha1 = *(lut_ptr as *const i32);
                    lut_ptr = lut_ptr.add(4);
                    let lut1 = _mm_loadu_si128(lut_ptr as *const __m128i);
                    lut_ptr = lut_ptr.add(16);

                    let alpha2 = *(lut_ptr as *const i32);
                    lut_ptr = lut_ptr.add(4);
                    let lut2 = _mm_loadu_si128(lut_ptr as *const __m128i);
                    lut_ptr = lut_ptr.add(16);

                    let dists1 = _mm_shuffle_epi8(lut1, codes_high);
                    let dists1 = _mm_and_si128(dists1, high_mask);

                    let dists2 = _mm_shuffle_epi8(lut2, codes_low);
                    let dists2 = _mm_and_si128(dists2, low_mask);

                    let dists = _mm_and_si128(dists1, dists2);
                    let dists = _mm_mul_epu32(dists, _mm_set_epi32(alpha1, alpha2, alpha1, alpha2));
                    sum = _mm_add_epi32(sum, dists);

                    codes_low = _mm_shuffle_epi32(codes_low, 0b01_10_11_00);
                    codes_high = _mm_shuffle_epi32(codes_high, 0b01_10_11_00);
                }

                codes_ptr = codes_ptr.add(1);
            }
            let sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0b01_00_11_10));
            _mm_cvtss_f32(_mm_cvtepi32_ps(sum)) * self.lut.alpha + self.lut.offset
        }
    }
}

impl<'a> From<CompressedLookupTable<'a>> for SseScorer<'a> {
    fn from(lut: CompressedLookupTable<'a>) -> Self {
        Self { lut }
    }
}
