pub struct EncodedVectorStorage {
    pub(crate) data: Vec<u8>,
    pub(crate) vector_size: usize,
    pub(crate) centroids: Vec<Vec<f32>>,
    pub(crate) alphas: Vec<f32>,
    pub(crate) offsets: Vec<f32>,
}

impl EncodedVectorStorage {
    pub fn new<F>(
        orig_data: Box<dyn Iterator<Item = &[f32]> + '_>,
        chunks: &[usize],
        metric: F,
    ) -> Result<EncodedVectorStorage, String>
    where
        F: Fn(&[f32], &[f32]) -> f32,
    {
        let separated_data = Self::separate_data(orig_data, chunks)?;

        let vectors_count = separated_data
            .iter()
            .zip(chunks.iter().cloned())
            .map(|(chunk_data, chunk_size)| chunk_data.len() / chunk_size)
            .fold(0, |acc, x| std::cmp::max(acc, x));

        if chunks.len() % 2 == 1 {
            return Err("chunks.len() must be even".to_string());
        }
        let mut data = vec![0; vectors_count * chunks.len() / 2];
        let mut centroids = Vec::new();
        let mut alphas = Vec::new();
        let mut offsets = Vec::new();
        for (chunk_index, (chunk_data, chunk_size)) in separated_data
            .into_iter()
            .zip(chunks.iter().cloned())
            .enumerate()
        {
            let chunk_data =
                ndarray::Array2::from_shape_vec((vectors_count, chunk_size), chunk_data)
                    .map_err(|_| format!("Failed to create ndarray from chunk data"))?;
            let (chunk_centroids, indexes) = rkm::kmeans_lloyd(&chunk_data.view(), 16);
            let centroid = chunk_centroids.as_slice().unwrap();
            for i in 0..16 {
                centroids.push(centroid[chunk_size * i..chunk_size * (i + 1)].to_vec());
            }
            let (alpha, offset) = Self::fit_alpha_offset(
                &centroids[16 * chunk_index..16 * (chunk_index + 1)],
                &metric,
            );
            alphas.push(alpha);
            offsets.push(offset);
            for (vector_index, centroid_index) in indexes.into_iter().enumerate() {
                Self::add_encoded_value(
                    &mut data,
                    chunks.len(),
                    vector_index,
                    chunk_index,
                    centroid_index as u8,
                );
            }
        }
        Ok(EncodedVectorStorage {
            data,
            vector_size: chunks.len() / 2,
            centroids,
            alphas,
            offsets,
        })
    }

    pub fn divide_dim(dim: usize, chunk_size: usize) -> Vec<usize> {
        let mut chunks = Vec::new();
        let mut dim = dim;
        while dim > chunk_size {
            chunks.push(chunk_size);
            dim -= chunk_size;
        }
        if dim > 0 {
            chunks.push(dim);
        }
        chunks
    }

    #[inline]
    pub fn get(&self, index: usize) -> &[u8] {
        &self.data[index * self.vector_size..(index + 1) * self.vector_size]
    }

    pub fn data_size(&self) -> usize {
        self.data.len()
    }

    fn separate_data(
        orig_data: Box<dyn Iterator<Item = &[f32]> + '_>,
        chunks: &[usize],
    ) -> Result<Vec<Vec<f32>>, String> {
        let mut separated = vec![Vec::new(); chunks.len()];
        for v in orig_data {
            let mut start = 0;
            for (i, &chunk_size) in chunks.iter().enumerate() {
                let end = start + chunk_size;
                separated[i].extend_from_slice(&v[start..end]);
                start = end;
            }
        }
        Ok(separated)
    }

    fn fit_alpha_offset<F>(centroids: &[Vec<f32>], _metric: F) -> (f32, f32)
    where
        F: Fn(&[f32], &[f32]) -> f32,
    {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for centroid in centroids {
            for &v in centroid {
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }
        }
        let alpha = 255.0 / (max - min);
        let offset = -min * alpha;
        (alpha, offset)
    }

    fn add_encoded_value(
        data: &mut [u8],
        chunks_count: usize,
        vector_index: usize,
        chunk_index: usize,
        centroid_index: u8,
    ) {
        let byte_index = vector_index * chunks_count / 2 + chunk_index / 2;
        let bit_index = (chunk_index % 2) * 4;
        data[byte_index] |= (centroid_index as u8) << bit_index;
    }
}
