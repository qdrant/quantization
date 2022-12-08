use crate::{encoded_vectors::CompressedLookupTable, scorer::Scorer};
use std::arch::x86_64::*;

pub struct SseScorer<'a> {
    lut: CompressedLookupTable<'a>,
}

impl Scorer for SseScorer<'_> {
    #[inline]
    fn score_point(&self, point: usize) -> f32 {
        // requires sse + sse2 + ssse3
        unsafe {
            let low_4bits_mask = _mm_set1_epi8(0x0F);
            let low_8bits_mask = _mm_set_epi32(255, 255, 255, 255);

            let v = self.lut.encoded_vectors.get(point as usize);
            let codes_count = v.len();

            let mut codes_ptr = v.as_ptr();
            let mut lut_ptr = self.lut.centroid_distances.as_ptr();
            let mut alphas_ptr = self.lut.alphas.as_ptr();

            let mut sum_low: __m128 = _mm_setzero_ps();
            let mut sum_high: __m128 = _mm_setzero_ps();
            for _ in 0..codes_count {
                let x_col = _mm_set1_epi8(*(codes_ptr as *const i8));
                codes_ptr = codes_ptr.add(1);

                let x_low = _mm_and_si128(x_col, low_4bits_mask);
                let x_shft = _mm_srli_epi16(x_col, 4);
                let x_high = _mm_and_si128(x_shft, low_4bits_mask);

                let lut = _mm_loadu_si128(lut_ptr as *const __m128i);
                lut_ptr = lut_ptr.add(16);
                let dists = _mm_shuffle_epi8(lut, x_high);
                let dists = _mm_and_si128(dists, low_8bits_mask);
                let dists = _mm_cvtepi32_ps(dists);
                let alpha = _mm_set1_ps(*alphas_ptr);
                alphas_ptr = alphas_ptr.add(1);
                sum_high = _mm_add_ps(sum_high, _mm_mul_ps(dists, alpha));

                let lut = _mm_loadu_si128(lut_ptr as *const __m128i);
                lut_ptr = lut_ptr.add(16);
                let dists = _mm_shuffle_epi8(lut, x_low);
                let dists = _mm_and_si128(dists, low_8bits_mask);
                let dists = _mm_cvtepi32_ps(dists);
                let alpha = _mm_set1_ps(*alphas_ptr);
                alphas_ptr = alphas_ptr.add(1);
                sum_low = _mm_add_ps(sum_low, _mm_mul_ps(dists, alpha));
            }
            let sum = _mm_add_ps(sum_low, sum_high);
            _mm_cvtss_f32(sum) + self.lut.total_offset
        }
    }
}

impl<'a> From<CompressedLookupTable<'a>> for SseScorer<'a> {
    fn from(lut: CompressedLookupTable<'a>) -> Self {
        Self { lut }
    }
}
