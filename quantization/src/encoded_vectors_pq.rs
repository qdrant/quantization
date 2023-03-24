use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

use crate::kmeans::KMeans;
use crate::{
    encoded_storage::{EncodedStorage, EncodedStorageBuilder},
    encoded_vectors::{EncodedVectors, VectorParameters},
    EncodingError,
};

pub struct EncodedVectorsPQ<TStorage: EncodedStorage> {
    encoded_vectors: TStorage,
    metadata: Metadata,
}
pub struct EncodedQueryPQ {
    lut: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct Metadata {
    quantized_vector_size: usize,
    kmeans: KMeans,
    vector_parameters: VectorParameters,
}

impl<TStorage: EncodedStorage> EncodedVectorsPQ<TStorage> {
    pub fn encode<'a>(
        orig_data: impl IntoIterator<Item = &'a [f32]> + Clone,
        mut storage_builder: impl EncodedStorageBuilder<TStorage>,
        vector_parameters: &VectorParameters,
        bucket_size: usize,
    ) -> Result<Self, EncodingError> {
        let kmeans = KMeans::run(orig_data, &mut storage_builder, bucket_size, 255)?;
        let storage = storage_builder.build();
        Ok(Self {
            encoded_vectors: storage,
            metadata: Metadata {
                quantized_vector_size: kmeans.vector_division.len(),
                kmeans,
                vector_parameters: vector_parameters.clone(),
            },
        })
    }
}

impl<TStorage: EncodedStorage> EncodedVectors<EncodedQueryPQ> for EncodedVectorsPQ<TStorage> {
    fn save(&self, data_path: &Path, meta_path: &Path) -> std::io::Result<()> {
        let metadata_bytes = serde_json::to_vec(&self.metadata)?;
        meta_path.parent().map(std::fs::create_dir_all);
        let mut buffer = File::create(meta_path)?;
        buffer.write_all(&metadata_bytes)?;

        data_path.parent().map(std::fs::create_dir_all);
        self.encoded_vectors.save_to_file(data_path)?;
        Ok(())
    }

    fn load(
        data_path: &Path,
        meta_path: &Path,
        vector_parameters: &VectorParameters,
    ) -> std::io::Result<Self> {
        let mut contents = String::new();
        let mut file = File::open(meta_path)?;
        file.read_to_string(&mut contents)?;
        let metadata: Metadata = serde_json::from_str(&contents)?;
        let quantized_vector_size = metadata.kmeans.vector_division.len();
        let encoded_vectors =
            TStorage::from_file(data_path, quantized_vector_size, vector_parameters.count)?;
        let result = Self {
            metadata,
            encoded_vectors,
        };
        Ok(result)
    }

    fn encode_query(&self, query: &[f32]) -> EncodedQueryPQ {
        let mut lut = Vec::new();
        for range in &self.metadata.kmeans.vector_division {
            let subquery = &query[range.clone()];
            for i in 0..self.metadata.kmeans.centroids.len() {
                let centroid = &self.metadata.kmeans.centroids[i];
                let subcentroid = &centroid[range.clone()];
                let distance = self
                    .metadata
                    .vector_parameters
                    .distance_type
                    .distance(subquery, subcentroid);
                let distance = if self.metadata.vector_parameters.invert {
                    -distance
                } else {
                    distance
                };
                lut.push(distance);
            }
        }
        EncodedQueryPQ { lut }
    }

    fn score_point(&self, query: &EncodedQueryPQ, i: u32) -> f32 {
        let centroids = self
            .encoded_vectors
            .get_vector_data(i as usize, self.metadata.quantized_vector_size);
        let centroids_count = self.metadata.kmeans.centroids.len();
        centroids
            .iter()
            .enumerate()
            .map(|(i, &c)| query.lut[centroids_count * i + c as usize])
            .sum()
    }

    fn score_internal(&self, i: u32, j: u32) -> f32 {
        let centroids_i = self
            .encoded_vectors
            .get_vector_data(i as usize, self.metadata.quantized_vector_size);
        let centroids_j = self
            .encoded_vectors
            .get_vector_data(j as usize, self.metadata.quantized_vector_size);
        let distance: f32 = centroids_i
            .iter()
            .zip(centroids_j)
            .enumerate()
            .map(|(range_index, (&c_i, &c_j))| {
                let range = &self.metadata.kmeans.vector_division[range_index];
                let data_i = &self.metadata.kmeans.centroids[c_i as usize][range.clone()];
                let data_j = &self.metadata.kmeans.centroids[c_j as usize][range.clone()];
                self.metadata
                    .vector_parameters
                    .distance_type
                    .distance(data_i, data_j)
            })
            .sum();
        if self.metadata.vector_parameters.invert {
            -distance
        } else {
            distance
        }
    }
}
