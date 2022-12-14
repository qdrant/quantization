use crate::{encoded_vectors::CompressedLookupTable, scorer::Scorer};

pub struct SimpleScorer<'a> {
    lut: CompressedLookupTable<'a>,
    chunks_count: usize,
}

impl Scorer for SimpleScorer<'_> {
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

    /*
    fn score_points(&self, points: &[usize], scores: &mut [f32]) {
        const CHUNK_SIZE: usize = 16;
        let mut points_ptr: [*const u8; CHUNK_SIZE] = [std::ptr::null(); CHUNK_SIZE];
        let mut scores_int: [u32; CHUNK_SIZE] = [0; CHUNK_SIZE];
        unsafe {
            for (point_ids, scores) in points.chunks_exact(CHUNK_SIZE).zip(scores.chunks_exact_mut(CHUNK_SIZE)) {
                for i in 0..CHUNK_SIZE {
                    points_ptr[i] = self.lut.encoded_vectors.get_ptr(point_ids[i]);
                    scores_int[i] = 0;
                }
                let mut lut_ptr = self.lut.centroid_distances.as_ptr() as *const u128;
                for _ in 0..self.chunks_count {
                    let alpha1 = *(lut_ptr as *const u32);
                    let lut1 = *lut_ptr;
                    lut_ptr = lut_ptr.add(1);
                    let alpha2 = *(lut_ptr as *const u32);
                    let lut2 = *lut_ptr;
                    lut_ptr = lut_ptr.add(1);

                    for (i, point_ptr) in points_ptr.iter_mut().enumerate() {
                        let codes = **point_ptr;
                        *point_ptr = point_ptr.add(1);
                        let c1 = (codes >> 4) & 0x0F;
                        let c2 = codes & 0x0F;
                        scores_int[i] +=
                            alpha1 * ((lut1 >> c1) & 0xFF) as u32 +
                            alpha2 * ((lut2 >> c2) & 0xFF) as u32;
                    }
                }
                for i in 0..CHUNK_SIZE {
                    scores[i] = scores_int[i] as f32 * self.lut.alpha + self.lut.offset;
                }
            }
        }
    }
    */
}

impl<'a> From<CompressedLookupTable<'a>> for SimpleScorer<'a> {
    fn from(lut: CompressedLookupTable<'a>) -> Self {
        Self {
            chunks_count: lut.encoded_vectors.chunks.len(),
            lut,
        }
    }
}
