use crate::{encoded_vectors::EncodedVectors, CENTROIDS_COUNT};

pub struct Lut {
    pub(crate) centroid_distances: Vec<u8>,
    pub(crate) alphas: Vec<f32>,
    pub(crate) total_offset: f32,
}

impl Lut {
    pub fn new<F>(encoder: &EncodedVectors, query: &[f32], metric: F) -> Lut
    where
        F: Fn(&[f32], &[f32]) -> f32,
    {
        let mut centroid_distances = Vec::with_capacity(CENTROIDS_COUNT * encoder.centroids.len());
        let mut alphas = Vec::with_capacity(encoder.centroids.len());
        let mut total_offset = 0.0;
        let mut start = 0;
        for (i, chunk_centroids) in encoder.centroids.iter().enumerate() {
            let query_chunk = &query[start..start + encoder.chunks[i]];
            start += encoder.chunks[i];
            let distances: Vec<f32> = chunk_centroids
                .iter()
                .map(|c| metric(c, query_chunk))
                .collect();

            let mut min = f32::MAX;
            let mut max = f32::MIN;
            for &d in &distances {
                if d < min {
                    min = d;
                }
                if d > max {
                    max = d;
                }
            }

            let alpha = (max - min) / 255.0;
            let offset = min;
            let byte_distances = distances
                .iter()
                .map(|&d| ((d - offset) / alpha) as u8)
                .collect::<Vec<_>>();

            centroid_distances.extend_from_slice(&byte_distances);
            alphas.push(alpha);
            total_offset += offset;
        }
        Lut {
            centroid_distances,
            alphas,
            total_offset,
        }
    }

    #[inline]
    pub fn dist(&self, v: &[u8]) -> f32 {
        unsafe {
            let mut sum = self.total_offset;
            let mut ptr = v.as_ptr();
            let mut alphas_ptr = self.alphas.as_ptr();
            for chunk_pair in 0..v.len() {
                let v = *ptr;
                ptr = ptr.add(1);
                let c1 = v >> 4;
                let c2 = v % 16;
                sum += *alphas_ptr * self.centroid_distances[32 * chunk_pair + c1 as usize] as f32;
                alphas_ptr = alphas_ptr.add(1);
                sum += *alphas_ptr
                    * self.centroid_distances[32 * chunk_pair + 16 + c2 as usize] as f32;
                alphas_ptr = alphas_ptr.add(1);
            }
            sum
        }
    }
}
