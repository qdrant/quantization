pub struct EncodedVectorStorage {
    pub(crate) data: Vec<u8>,
    pub(crate) vector_size: usize,
    pub(crate) centroids: Vec<Vec<Vec<f32>>>,
    pub(crate) chunks: Vec<usize>,
}

impl EncodedVectorStorage {
    pub fn new<'a>(
        orig_data: impl Iterator<Item = &'a [f32]>,
        chunks: &[usize],
    ) -> Result<EncodedVectorStorage, String> {
        let separated_data = Self::separate_data(orig_data, chunks)?;

        let vectors_count = separated_data[0].len();

        if chunks.len() % 2 == 1 {
            return Err("chunks.len() must be even".to_string());
        }
        let mut data = vec![0; vectors_count * chunks.len() / 2];
        let mut centroids = Vec::new();
        for (chunk_index, (chunk_data, chunk_size)) in separated_data
            .into_iter()
            .zip(chunks.iter().cloned())
            .enumerate()
        {
            let (chunk_centroids, indexes) = Self::get_centroids(chunk_data, chunk_size)?;
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

    fn separate_data<'a>(
        orig_data: impl Iterator<Item = &'a [f32]>,
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

    pub fn get_centroids(
        points: Vec<Vec<f32>>,
        chunk_size: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<usize>), String> {
        match chunk_size {
            1 => {
                let points: Vec<f32> = points.into_iter().flatten().collect();
                let (centroids, indexes) = crate::kmeans_1d::kmeans_1d(&points);
                return Ok((
                    centroids.into_iter().map(|v| vec![v]).collect(),
                    indexes,
                ))
            },
            2 => {
                let points: Vec<f32> = points.into_iter().flatten().collect();
                let (centroids, indexes) = crate::kmeans_2d::kmeans_2d(&points);
                return Ok((
                    centroids.chunks_exact(2).map(|v| vec![v[0], v[1]]).collect(),
                    indexes,
                ))
            },
            _ => Err("Only 1 and 2 dimensions are supported".to_string()),
        }
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
