use crate::{encoded_vectors::CompressedLookupTable, scorer::Scorer};
use std::arch::x86_64::*;

pub struct SseScorer<'a> {
    lut: CompressedLookupTable<'a>,
    vector_size: usize,
    points_data: Vec<u8>,
}

impl Scorer for SseScorer<'_> {
    //#[inline]

    fn score_point(&self, point: usize) -> f32 {
        unsafe {
            let v = self.lut.encoded_vectors.get(point as usize);
            let mut sum = 0u32;
            let mut ptr = v.as_ptr() as *const u64;
            let mut lut_ptr = self.lut.centroid_distances.as_ptr();
            let cnt = v.len() / 8;
            for _ in 0..cnt {
                let mut codes = *ptr;
                ptr = ptr.add(1);
                for _ in 0..8 {
                    let c1 = (codes >> 4) & 0x0F;
                    let c2 = codes & 0x0F;
                    codes >>= 8;

                    let alpha = *(lut_ptr as *const u32);
                    lut_ptr = lut_ptr.add(4);
                    sum += alpha * (*lut_ptr.add(c1 as usize) as u32);
                    lut_ptr = lut_ptr.add(16);

                    let alpha = *(lut_ptr as *const u32);
                    lut_ptr = lut_ptr.add(4);
                    sum += alpha * (*lut_ptr.add(c2 as usize) as u32);
                    lut_ptr = lut_ptr.add(16);
                }
            }
            sum as f32 * self.lut.alpha + self.lut.offset
        }
    }

    fn score_points(&mut self, points: &[usize], scores: &mut [f32]) {
        unsafe {
            let vector_size = self.vector_size;
            let mut lut_ptr = self.lut.centroid_distances.as_ptr();

            const CHUNK_SIZE: usize = 16;
            let mut points_ptr: [*const i8; CHUNK_SIZE] = [std::ptr::null(); CHUNK_SIZE];

            let distances_mask = _mm_set_epi64x(0, 0xFF);
            let low_4bits_mask = _mm_set1_epi8(0x0F);
            for (point_ids, scores) in points
                .chunks_exact(CHUNK_SIZE)
                .zip(scores.chunks_exact_mut(CHUNK_SIZE))
            {
                for i in 0..CHUNK_SIZE {
//                    std::ptr::copy_nonoverlapping(
//                        ,
//                        ,
//                        );
                    points_ptr[i] = self.lut.encoded_vectors.get_ptr(point_ids[i]) as *const i8;
                }

                let mut sum: __m128i = _mm_setzero_si128();
                for _ in 0..vector_size {
                    let alpha1 = *(lut_ptr as *const i32);
                    lut_ptr = lut_ptr.add(4);
                    let lut1 = _mm_loadu_si128(lut_ptr as *const __m128i);
                    lut_ptr = lut_ptr.add(16);

                    let alpha2 = *(lut_ptr as *const i32);
                    lut_ptr = lut_ptr.add(4);
                    let lut2 = _mm_loadu_si128(lut_ptr as *const __m128i);
                    lut_ptr = lut_ptr.add(16);

                    let codes = _mm_set_epi8(
                        **points_ptr.get_unchecked(0),
                        **points_ptr.get_unchecked(1),
                        **points_ptr.get_unchecked(2),
                        **points_ptr.get_unchecked(3),
                        **points_ptr.get_unchecked(4),
                        **points_ptr.get_unchecked(5),
                        **points_ptr.get_unchecked(6),
                        **points_ptr.get_unchecked(7),
                        **points_ptr.get_unchecked(8),
                        **points_ptr.get_unchecked(9),
                        **points_ptr.get_unchecked(10),
                        **points_ptr.get_unchecked(11),
                        **points_ptr.get_unchecked(12),
                        **points_ptr.get_unchecked(13),
                        **points_ptr.get_unchecked(14),
                        **points_ptr.get_unchecked(15),
                    );

                    let codes_low = _mm_and_si128(codes, low_4bits_mask);
                    let dists2 = _mm_shuffle_epi8(lut2, codes_low);
                    let dists2 = _mm_and_si128(dists2, distances_mask);
                    sum = _mm_add_epi32(sum, dists2);

                    let codes_shft = _mm_srli_epi16(codes, 4);
                    let codes_high = _mm_and_si128(codes_shft, low_4bits_mask);
                    let dists1 = _mm_shuffle_epi8(lut1, codes_high);
                    let dists1 = _mm_and_si128(dists1, distances_mask);
                    sum = _mm_add_epi32(sum, dists1);

                    for j in 0..CHUNK_SIZE {
                        points_ptr[j] = points_ptr[j].add(1);
                    }
                }

                let sum = hsum128_ps_sse(_mm_cvtepi32_ps(sum));
                for j in 0..CHUNK_SIZE {
                    scores[j] = sum * self.lut.alpha + self.lut.offset;
                }
            }
        }
    }
}

impl<'a> From<CompressedLookupTable<'a>> for SseScorer<'a> {
    fn from(lut: CompressedLookupTable<'a>) -> Self {
        Self {
            vector_size: lut.encoded_vectors.vector_size,
            points_data: vec![0u8; lut.encoded_vectors.vector_size * 8],
            lut,
        }
    }
}

unsafe fn hsum128_ps_sse(x: __m128) -> f32 {
    let x64: __m128 = _mm_add_ps(x, _mm_movehl_ps(x, x));
    let x32: __m128 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    _mm_cvtss_f32(x32)
}
