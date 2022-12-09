use crate::{encoded_vectors::CompressedLookupTable, scorer::Scorer};
use std::arch::x86_64::*;

pub struct SseScorer<'a> {
    lut: CompressedLookupTable<'a>,
}

impl Scorer for SseScorer<'_> {
    //#[inline]
    fn score_point(&self, point: usize) -> f32 {
        // requires sse + sse2 + ssse3
        unsafe {
            //let low_4bits_mask = _mm_set1_epi8(0x0F);
            //let low_8bits_mask = _mm_set_epi32(255, 255, 255, 255);

            let v = self.lut.encoded_vectors.get(point as usize);
            let codes_count = v.len();

            let mut codes_ptr = v.as_ptr();
            let mut lut_ptr = self.lut.centroid_distances.as_ptr();

            let mut sum: __m128i = _mm_setzero_si128();
            for _ in 0..codes_count {
                let v = *codes_ptr;
                codes_ptr = codes_ptr.add(1);
                let c1 = v >> 4;
                let c2 = v & 0x0F;

                let alpha = *(lut_ptr as *const i32);
                lut_ptr = lut_ptr.add(4);
                let d1 = alpha * (*lut_ptr.add(c1 as usize) as i32);
                lut_ptr = lut_ptr.add(16);

                let alpha = *(lut_ptr as *const i32);
                lut_ptr = lut_ptr.add(4);
                let d2 = alpha * (*lut_ptr.add(c2 as usize) as i32);
                lut_ptr = lut_ptr.add(16);

                sum = _mm_add_epi32(sum, _mm_set_epi32(d1, d2, 0, 0));
            }
            //let sum = _mm_add_ps(sum_low, sum_high);
            _mm_cvtsi128_si32(sum) as f32 * self.lut.alpha + self.lut.offset
        }
    }
}

impl<'a> From<CompressedLookupTable<'a>> for SseScorer<'a> {
    fn from(lut: CompressedLookupTable<'a>) -> Self {
        Self { lut }
    }
}
