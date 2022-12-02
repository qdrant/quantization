use crate::encoder::EncodedVectorStorage;

pub struct Lut {
    pub(crate) dist: Vec<u8>,
    pub(crate) alphas: Vec<f32>,
    pub(crate) offset: f32,
}

impl Lut {
    pub fn new<F>(encoder: &EncodedVectorStorage, query: &[f32], metric: F) -> Lut
    where
        F: Fn(&[f32], &[f32]) -> f32,
    {
        let mut dist = vec![0; encoder.centroids.len()];
        for (i, centroid) in encoder.centroids.iter().enumerate() {
            let d = metric(query, centroid) / encoder.alphas[i % 16];
            dist[i] = d as u8;
        }
        Lut {
            dist,
            alphas: encoder.alphas.clone(),
            offset: encoder.offsets.iter().fold(0.0, |a, &b| a + b),
        }
    }

    #[inline]
    pub fn dist(&self, v: &[u8]) -> f32 {
        unsafe {
            let mut sum = self.offset;
            let mut ptr = v.as_ptr();
            let mut alphas_ptr = self.alphas.as_ptr();
            for _ in 0..v.len() / 2 {
                let v = *ptr;
                ptr = ptr.add(1);
                let c1 = v >> 4;
                let c2 = v % 16;
                sum += *alphas_ptr * self.dist[c1 as usize] as f32;
                alphas_ptr = alphas_ptr.add(1);
                sum += *alphas_ptr * self.dist[c2 as usize] as f32;
                alphas_ptr = alphas_ptr.add(1);
            }
            sum
        }
    }
}
