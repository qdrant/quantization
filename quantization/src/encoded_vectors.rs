use std::ops::Range;

use crate::scorer::Scorer;

pub struct EncodedVectors {
    pub(crate) data: Vec<u8>,
    pub(crate) vector_size: usize,
    pub(crate) centroids: Vec<Vec<f32>>,
    pub(crate) chunks: Vec<usize>,
    pub(crate) dim: usize,
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
            dim,
        })
    }

    #[inline]
    pub fn get(&self, index: usize) -> &[u8] {
        &self.data[index * self.vector_size..(index + 1) * self.vector_size]
    }

    pub fn data_size(&self) -> usize {
        self.data.len()
    }

    pub fn decode_vector(&self, index: usize) -> Vec<f32> {
        let mut vector = Vec::with_capacity(self.dim);
        let encoded = self.get(index);
        for (byte_index, &byte) in encoded.iter().enumerate() {
            let centroid = (byte >> 4) as usize;
            let chunk_index = 2 * byte_index;
            let chunk_dim = self.chunks[chunk_index];
            let range = chunk_dim * centroid..chunk_dim * (centroid + 1);
            vector.extend_from_slice(&self.centroids[chunk_index][range]);

            let centroid = (byte & 0b0000_1111) as usize;
            let chunk_index = 2 * byte_index + 1;
            let chunk_dim = self.chunks[chunk_index];
            let range = chunk_dim * centroid..chunk_dim * (centroid + 1);
            vector.extend_from_slice(&self.centroids[chunk_index][range]);
        }
        vector
    }

    pub fn score_between_points<M>(&self, point_a: usize, point_b: usize, metric: M) -> f32
    where
        M: Fn(&[f32], &[f32]) -> f32,
    {
        let a = self.decode_vector(point_a as usize);
        let b = self.decode_vector(point_b as usize);
        metric(&a, &b)
    }

    pub fn scorer<'a, TScorer, M>(&'a self, query: &[f32], metric: M) -> TScorer
    where
        TScorer: Scorer + From<CompressedLookupTable<'a>>,
        M: Fn(&[f32], &[f32]) -> f32,
    {
        CompressedLookupTable::new(self, query, metric).into()
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

    pub fn create_dim_partition(dim: usize, chunk_size: usize) -> Vec<usize> {
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
    ) -> Result<Vec<f32>, String> {
        let mut chunk_data = Vec::new();
        for v in orig_data {
            chunk_data.extend_from_slice(
                v.get(chunk.clone())
                    .ok_or(format!("Invalid chunk range",))?,
            );
        }

        let (centroids, indexes) = crate::kmeans::kmeans(&chunk_data, chunk.end - chunk.start);

        let bits_offset = (1 - (chunk_index % 2)) * 4;
        for (vector_index, centroid_index) in indexes.into_iter().enumerate() {
            byte_column[vector_index] |= (centroid_index as u8) << bits_offset;
        }

        Ok(centroids)
    }
}

pub struct CompressedLookupTable<'a> {
    pub(crate) encoded_vectors: &'a EncodedVectors,
    pub(crate) centroid_distances: Vec<u8>,
    pub(crate) alphas: Vec<f32>,
    pub(crate) total_offset: f32,
}

impl<'a> CompressedLookupTable<'a> {
    fn new<M>(encoded_vectors: &'a EncodedVectors, query: &[f32], metric: M) -> Self
    where
        M: Fn(&[f32], &[f32]) -> f32,
    {
        let mut centroid_distances =
            Vec::with_capacity(crate::CENTROIDS_COUNT * encoded_vectors.centroids.len());
        let mut alphas = Vec::with_capacity(encoded_vectors.centroids.len());
        let mut total_offset = 0.0;
        let mut start = 0;
        for (i, chunk_centroids) in encoded_vectors.centroids.iter().enumerate() {
            let chunk_size = encoded_vectors.chunks[i];
            let query_chunk = &query[start..start + chunk_size];
            start += chunk_size;
            let distances: Vec<f32> = chunk_centroids
                .as_slice()
                .chunks_exact(chunk_size)
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
        Self {
            encoded_vectors,
            centroid_distances,
            alphas,
            total_offset,
        }
    }
}
