use crate::{encoded_vectors::CompressedLookupTable, scorer::Scorer};
use std::arch::x86_64::*;

pub struct AvxScorer<'a> {
    lut: CompressedLookupTable<'a>,
}

impl Scorer for AvxScorer<'_> {
    #[inline]
    fn score_point(&self, point: usize) -> f32 {
        unsafe {
            let first_4bits_mask = _mm256_set1_epi8(0b00001111);
            let last_4bits_mask = _mm256_set1_epi8(0b00001111 << 4);

            let v = self.lut.encoded_vectors.get(point as usize);
            let codes_count = v.len();
            let codes_ptr = v.as_ptr();

            for _ in 0..codes_count / 32 {
                // _mm256_stream_load_si256
                let codes = _mm256_loadu_si256(codes_ptr as *const __m256i);
                let codes_first = _mm256_and_si256(codes, first_4bits_mask);

                let codes_last = _mm256_and_si256(codes, last_4bits_mask);
            }
        }
        0.0
    }
}

impl<'a> From<CompressedLookupTable<'a>> for AvxScorer<'a> {
    fn from(lut: CompressedLookupTable<'a>) -> Self {
        Self { lut }
    }
}
