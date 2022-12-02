use crate::CENTROIDS_COUNT;

pub struct EncodedVectorStorage {
    pub(crate) data: Vec<u8>,
    pub(crate) vector_size: usize,
    pub(crate) centroids: Vec<Vec<Vec<f32>>>,
    pub(crate) chunks: Vec<usize>,
}

impl EncodedVectorStorage {
    pub fn new(
        orig_data: Box<dyn Iterator<Item = &[f32]> + '_>,
        chunks: &[usize],
    ) -> Result<EncodedVectorStorage, String> {
        let separated_data = Self::separate_data(orig_data, chunks)?;

        let vectors_count = separated_data[0].len();

        if chunks.len() % 2 == 1 {
            return Err("chunks.len() must be even".to_string());
        }
        let mut data = vec![0; vectors_count * chunks.len() / 2];
        let mut centroids = Vec::new();
        for (chunk_index, (chunk_data, _chunk_size)) in separated_data
            .into_iter()
            .zip(chunks.iter().cloned())
            .enumerate()
        {
            let (chunk_centroids, indexes) = Self::get_centroids(chunk_data)?;
            centroids.push(chunk_centroids);
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
            chunks: chunks.to_vec(),
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
    ) -> Result<Vec<Vec<Vec<f32>>>, String> {
        let mut separated = vec![Vec::new(); chunks.len()];
        for v in orig_data {
            let mut start = 0;
            for (i, &chunk_size) in chunks.iter().enumerate() {
                let end = start + chunk_size;
                separated[i].push(v[start..end].to_vec());
                start = end;
            }
        }
        Ok(separated)
    }

    pub fn get_centroids(points: Vec<Vec<f32>>) -> Result<(Vec<Vec<f32>>, Vec<usize>), String> {
        let vectors_count = points.len();
        let dim = points[0].len();
        let mut chunk_data = ndarray::Array2::<f32>::default((vectors_count, dim));
        for (i, mut row) in chunk_data.axis_iter_mut(ndarray::Axis(0)).enumerate() {
            for (j, col) in row.iter_mut().enumerate() {
                *col = points[i][j];
            }
        }

        let (chunk_centroids, indexes) = rkm::kmeans_lloyd(&chunk_data.view(), CENTROIDS_COUNT);

        let mut centroids = vec![Vec::new(); CENTROIDS_COUNT];
        for (i, row) in chunk_centroids.axis_iter(ndarray::Axis(0)).enumerate() {
            for col in &row {
                centroids[i].push(*col);
            }
        }
        Ok((centroids, indexes))
    }

    fn add_encoded_value(
        data: &mut [u8],
        chunks_count: usize,
        vector_index: usize,
        chunk_index: usize,
        centroid_index: u8,
    ) {
        let byte_index = vector_index * chunks_count / 2 + chunk_index / 2;
        let bit_index = (1 - (chunk_index % 2)) * 4;
        data[byte_index] |= (centroid_index as u8) << bit_index;
    }
}
