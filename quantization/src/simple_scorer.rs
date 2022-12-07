use crate::{encoded_vectors::CompressedLookupTable, scorer::Scorer};

pub struct SimpleScorer<'a> {
    lut: CompressedLookupTable<'a>,
}

impl Scorer for SimpleScorer<'_> {
    #[inline]
    fn score_point(&self, point: usize) -> f32 {
        unsafe {
            let v = self.lut.encoded_vectors.get(point as usize);
            let mut sum = self.lut.total_offset;
            let mut ptr = v.as_ptr();
            let mut alphas_ptr = self.lut.alphas.as_ptr();
            for chunk_pair in 0..v.len() {
                let v = *ptr;
                ptr = ptr.add(1);
                let c1 = v >> 4;
                let c2 = v % 16;
                sum +=
                    *alphas_ptr * self.lut.centroid_distances[32 * chunk_pair + c1 as usize] as f32;
                alphas_ptr = alphas_ptr.add(1);
                sum += *alphas_ptr
                    * self.lut.centroid_distances[32 * chunk_pair + 16 + c2 as usize] as f32;
                alphas_ptr = alphas_ptr.add(1);
            }
            sum
        }
    }

    fn score_internal<M>(&self, point_a: usize, point_b: usize, metric: M) -> f32
    where
        M: Fn(&[f32], &[f32]) -> f32,
    {
        let a = self.lut.encoded_vectors.decode_vector(point_a as usize);
        let b = self.lut.encoded_vectors.decode_vector(point_b as usize);
        metric(&a, &b)
    }
}

impl<'a> From<CompressedLookupTable<'a>> for SimpleScorer<'a> {
    fn from(lut: CompressedLookupTable<'a>) -> Self {
        Self { lut }
    }
}
