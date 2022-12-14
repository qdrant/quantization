use crate::{encoded_vectors::CompressedLookupTable, scorer::Scorer, CENTROIDS_COUNT};
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
            scores.fill(0.0);
            let vector_size = self.vector_size;
            let mut lut_ptr = self.lut.centroid_distances.as_ptr();

            let distances_mask = _mm_set_epi32(0xFF, 0xFF, 0xFF, 0xFF);
            let low_4bits_mask = _mm_set1_epi8(0x0F);
            for (point_ids, scores) in points
                .chunks_exact(CENTROIDS_COUNT)
                .zip(scores.chunks_exact_mut(CENTROIDS_COUNT))
            {
                let mut sum1: __m128i = _mm_setzero_si128();
                let mut sum2: __m128i = _mm_setzero_si128();
                let mut sum3: __m128i = _mm_setzero_si128();
                let mut sum4: __m128i = _mm_setzero_si128();

                const SUBVECTOR_SIZE: usize = 32;
                for sv in 0..vector_size / SUBVECTOR_SIZE {
                    for i in 0..CENTROIDS_COUNT {
                        std::ptr::copy_nonoverlapping(
                            self.lut.encoded_vectors.get_ptr(point_ids[i]),
                            self.points_data
                                .as_mut_ptr()
                                .add(i * vector_size + sv * SUBVECTOR_SIZE),
                            SUBVECTOR_SIZE,
                        );
                    }

                    let mut points_ptr = self.points_data.as_ptr() as *const i8;
                    for _ in 0..SUBVECTOR_SIZE {
                        let alpha1 = *(lut_ptr as *const i32);
                        lut_ptr = lut_ptr.add(4);
                        let lut1 = _mm_loadu_si128(lut_ptr as *const __m128i);
                        lut_ptr = lut_ptr.add(16);

                        let alpha2 = *(lut_ptr as *const i32);
                        lut_ptr = lut_ptr.add(4);
                        let lut2 = _mm_loadu_si128(lut_ptr as *const __m128i);
                        lut_ptr = lut_ptr.add(16);

                        let codes = _mm_set_epi8(
                            *points_ptr,
                            *points_ptr.add(SUBVECTOR_SIZE),
                            *points_ptr.add(2 * SUBVECTOR_SIZE),
                            *points_ptr.add(3 * SUBVECTOR_SIZE),
                            *points_ptr.add(4 * SUBVECTOR_SIZE),
                            *points_ptr.add(5 * SUBVECTOR_SIZE),
                            *points_ptr.add(6 * SUBVECTOR_SIZE),
                            *points_ptr.add(7 * SUBVECTOR_SIZE),
                            *points_ptr.add(8 * SUBVECTOR_SIZE),
                            *points_ptr.add(9 * SUBVECTOR_SIZE),
                            *points_ptr.add(10 * SUBVECTOR_SIZE),
                            *points_ptr.add(11 * SUBVECTOR_SIZE),
                            *points_ptr.add(12 * SUBVECTOR_SIZE),
                            *points_ptr.add(13 * SUBVECTOR_SIZE),
                            *points_ptr.add(14 * SUBVECTOR_SIZE),
                            *points_ptr.add(15 * SUBVECTOR_SIZE),
                        );

                        let codes_low = _mm_and_si128(codes, low_4bits_mask);
                        let alpha2 = _mm_set1_epi32(alpha2);
                        let dists = _mm_shuffle_epi8(lut2, codes_low);

                        let dists_part = _mm_and_si128(dists, distances_mask);
                        let dists_part = _mm_mullo_epi16(dists_part, alpha2);
                        sum1 = _mm_adds_epu16(sum1, dists_part);

                        let dists_part = _mm_srli_epi32(dists, 8);
                        let dists_part = _mm_and_si128(dists_part, distances_mask);
                        let dists_part = _mm_mullo_epi16(dists_part, alpha2);
                        sum2 = _mm_adds_epu16(sum2, dists_part);

                        let dists_part = _mm_srli_epi32(dists, 16);
                        let dists_part = _mm_and_si128(dists_part, distances_mask);
                        let dists_part = _mm_mullo_epi16(dists_part, alpha2);
                        sum3 = _mm_adds_epu16(sum3, dists_part);

                        let dists_part = _mm_srli_epi32(dists, 24);
                        let dists_part = _mm_and_si128(dists_part, distances_mask);
                        let dists_part = _mm_mullo_epi16(dists_part, alpha2);
                        sum4 = _mm_adds_epu16(sum4, dists_part);

                        let codes_shft = _mm_srli_epi16(codes, 4);
                        let codes_high = _mm_and_si128(codes_shft, low_4bits_mask);
                        let alpha1 = _mm_set1_epi32(alpha1);
                        let dists = _mm_shuffle_epi8(lut1, codes_high);

                        let dists_part = _mm_and_si128(dists, distances_mask);
                        let dists_part = _mm_mullo_epi16(dists_part, alpha1);
                        sum1 = _mm_adds_epu16(sum1, dists_part);

                        let dists_part = _mm_srli_epi32(dists, 8);
                        let dists_part = _mm_and_si128(dists_part, distances_mask);
                        let dists_part = _mm_mullo_epi16(dists_part, alpha1);
                        sum2 = _mm_adds_epu16(sum2, dists_part);

                        let dists_part = _mm_srli_epi32(dists, 16);
                        let dists_part = _mm_and_si128(dists_part, distances_mask);
                        let dists_part = _mm_mullo_epi16(dists_part, alpha1);
                        sum3 = _mm_adds_epu16(sum3, dists_part);

                        let dists_part = _mm_srli_epi32(dists, 24);
                        let dists_part = _mm_and_si128(dists_part, distances_mask);
                        let dists_part = _mm_mullo_epi16(dists_part, alpha1);
                        sum4 = _mm_adds_epu16(sum4, dists_part);

                        points_ptr = points_ptr.add(1);
                    }
                }

                let sum1 = _mm_cvtepi32_ps(sum1);
                scores[0] = f32::from_bits(_mm_extract_ps(sum1, 0) as u32) * self.lut.alpha + self.lut.offset;
                scores[1] = f32::from_bits(_mm_extract_ps(sum1, 1) as u32) * self.lut.alpha + self.lut.offset;
                scores[2] = f32::from_bits(_mm_extract_ps(sum1, 2) as u32) * self.lut.alpha + self.lut.offset;
                scores[3] = f32::from_bits(_mm_extract_ps(sum1, 3) as u32) * self.lut.alpha + self.lut.offset;

                let sum2 = _mm_cvtepi32_ps(sum2);
                scores[4] = f32::from_bits(_mm_extract_ps(sum2, 0) as u32) * self.lut.alpha + self.lut.offset;
                scores[5] = f32::from_bits(_mm_extract_ps(sum2, 1) as u32) * self.lut.alpha + self.lut.offset;
                scores[6] = f32::from_bits(_mm_extract_ps(sum2, 2) as u32) * self.lut.alpha + self.lut.offset;
                scores[7] = f32::from_bits(_mm_extract_ps(sum2, 3) as u32) * self.lut.alpha + self.lut.offset;

                let sum3 = _mm_cvtepi32_ps(sum3);
                scores[8] = f32::from_bits(_mm_extract_ps(sum3, 0) as u32) * self.lut.alpha + self.lut.offset;
                scores[9] = f32::from_bits(_mm_extract_ps(sum3, 1) as u32) * self.lut.alpha + self.lut.offset;
                scores[10] = f32::from_bits(_mm_extract_ps(sum3, 2) as u32) * self.lut.alpha + self.lut.offset;
                scores[11] = f32::from_bits(_mm_extract_ps(sum3, 3) as u32) * self.lut.alpha + self.lut.offset;
            
                let sum4 = _mm_cvtepi32_ps(sum4);
                scores[12] = f32::from_bits(_mm_extract_ps(sum4, 0) as u32) * self.lut.alpha + self.lut.offset;
                scores[13] = f32::from_bits(_mm_extract_ps(sum4, 1) as u32) * self.lut.alpha + self.lut.offset;
                scores[14] = f32::from_bits(_mm_extract_ps(sum4, 2) as u32) * self.lut.alpha + self.lut.offset;
                scores[15] = f32::from_bits(_mm_extract_ps(sum4, 3) as u32) * self.lut.alpha + self.lut.offset;
            }
        }
    }
}

impl<'a> From<CompressedLookupTable<'a>> for SseScorer<'a> {
    fn from(lut: CompressedLookupTable<'a>) -> Self {
        Self {
            vector_size: lut.encoded_vectors.vector_size,
            points_data: vec![0u8; lut.encoded_vectors.vector_size * 16],
            lut,
        }
    }
}
