use crate::{encoded_vectors::CompressedLookupTable, scorer::Scorer};

pub struct SimpleScorer<'a> {
    lut: CompressedLookupTable<'a>,
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

                    let alpha1 = *(lut_ptr as *const u32);
                    lut_ptr = lut_ptr.add(4);
                    let alpha2 = *(lut_ptr as *const u32);
                    lut_ptr = lut_ptr.add(4);

                    sum += alpha1 * (*lut_ptr.add(c1 as usize) as u32);
                    lut_ptr = lut_ptr.add(16);

                    sum += alpha2 * (*lut_ptr.add(c2 as usize) as u32);
                    lut_ptr = lut_ptr.add(16);
                }
            }
            sum as f32 * self.lut.alpha + self.lut.offset
        }
    }
}

impl<'a> From<CompressedLookupTable<'a>> for SimpleScorer<'a> {
    fn from(lut: CompressedLookupTable<'a>) -> Self {
        Self {
            lut,
        }
    }
}
