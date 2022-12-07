use std::ops::Range;

pub struct EncodedVectors {
    pub(crate) data: Vec<u8>,
    pub(crate) vector_size: usize,
    pub(crate) centroids: Vec<Vec<Vec<f32>>>,
    pub(crate) chunks: Vec<usize>,
}

impl EncodedVectors {
    pub fn new<'a>(
        orig_data: impl IntoIterator<Item = &'a [f32]> + Clone,
        vectors_count: usize,
        dim: usize,
        chunks: &[usize],
    ) -> Result<EncodedVectors, String> {
        Self::validate_partition(dim, chunks)?;

        let mut data = vec![0; vectors_count * chunks.len() / 2];
        let mut centroids = Vec::new();
        let mut chunk_offset = 0;
        let mut byte_column = vec![0; vectors_count];
        for (chunk_index, &chunk) in chunks.iter().enumerate() {
            let chunk_centroids = Self::encode_chunk(
                &mut byte_column,
                orig_data.clone(),
                chunk_offset..chunk_offset + chunk,
                chunk_index,
            )?;
            centroids.push(chunk_centroids);
            chunk_offset += chunk;

            if chunk_index % 2 == 1 {
                let column_index = chunk_index / 2;
                let columns_count = chunks.len() / 2;
                for (vector_index, &byte) in byte_column.iter().enumerate() {
                    data[vector_index * columns_count + column_index] = byte;
                }
                byte_column.as_mut_slice().fill(0);
            }
        }

        Ok(EncodedVectors {
            data,
            vector_size: chunks.len() / 2,
            centroids,
            chunks: chunks.to_vec(),
        })
    }

    /// Check that the chunk sizes are valid.
    /// If the chunk is invalid, return an error with the reason of invalidness.
    pub fn validate_partition(dim: usize, chunks: &[usize]) -> Result<(), String> {
        if chunks.len() % 2 != 0 {
            return Err("chunks.len() must be even".to_string());
        }
        if !chunks.iter().all(|&v| v == 1 || v == 2) {
            return Err("Chunk must be only 1 and 2".to_string());
        }
        if chunks.iter().sum::<usize>() != dim {
            return Err("Chunks sum must be equal to dim".to_string());
        }
        Ok(())
    }

    pub fn divide_dim(dim: usize, chunk_size: usize) -> Vec<usize> {
        if chunk_size != 1 && chunk_size % 2 != 0 {
            panic!("chunk_size must be 1 or 2");
        }

        let chunks = match chunk_size {
            1 => {
                if dim % 2 == 0 {
                    vec![1; dim]
                } else {
                    let mut chunks = vec![1; dim - 2];
                    chunks.push(2);
                    chunks
                }
            }
            2 => {
                let mut chunks = Vec::new();
                let mut dim = dim;
                while dim > 2 {
                    chunks.push(2);
                    dim -= 2;
                }
                if dim > 0 {
                    if chunks.len() % 2 == 0 {
                        chunks.pop();
                        chunks.push(1);
                        chunks.push(1);
                    }
                    chunks.push(dim);
                } else {
                    if chunks.len() % 2 == 1 {
                        chunks.pop();
                        chunks.push(1);
                        chunks.push(1);
                    }
                }
                chunks
            }
            _ => unreachable!(),
        };
        assert!(Self::validate_partition(dim, &chunks).is_ok());
        chunks
    }

    fn encode_chunk<'a>(
        byte_column: &mut [u8],
        orig_data: impl IntoIterator<Item = &'a [f32]>,
        chunk: Range<usize>,
        chunk_index: usize,
    ) -> Result<Vec<Vec<f32>>, String> {
        let mut chunk_data = Vec::new();
        for v in orig_data {
            chunk_data.extend_from_slice(
                v.get(chunk.clone())
                    .ok_or(format!("Invalid chunk range",))?,
            );
        }

        let (centroids, indexes) = Self::get_centroids(&chunk_data, chunk.end - chunk.start)?;

        let bits_offset = (1 - (chunk_index % 2)) * 4;
        for (vector_index, centroid_index) in indexes.into_iter().enumerate() {
            byte_column[vector_index] |= (centroid_index as u8) << bits_offset;
        }

        Ok(centroids)
    }

    #[inline]
    pub fn get(&self, index: usize) -> &[u8] {
        &self.data[index * self.vector_size..(index + 1) * self.vector_size]
    }

    pub fn data_size(&self) -> usize {
        self.data.len()
    }

    pub fn get_centroids(
        points: &[f32],
        chunk_size: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<usize>), String> {
        match chunk_size {
            1 => {
                let (centroids, indexes) = crate::kmeans_1d::kmeans_1d(&points);
                return Ok((centroids.into_iter().map(|v| vec![v]).collect(), indexes));
            }
            2 => {
                let (centroids, indexes) = crate::kmeans_2d::kmeans_2d(&points);
                return Ok((
                    centroids
                        .chunks_exact(2)
                        .map(|v| vec![v[0], v[1]])
                        .collect(),
                    indexes,
                ));
            }
            _ => Err("Only 1 and 2 dimensions are supported".to_string()),
        }
    }
}
