use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceType {
    Dot,
    L2,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct VectorParameters {
    pub dim: usize,
    pub count: usize,
    pub distance_type: DistanceType,
    pub invert: bool,
}

pub trait EncodedVectors<TEncodedQuery: Sized>: Sized {
    fn save(&self, data_path: &Path, meta_path: &Path) -> std::io::Result<()>;

    fn load(
        data_path: &Path,
        meta_path: &Path,
        vector_parameters: &VectorParameters,
    ) -> std::io::Result<Self>;

    fn encode_query(&self, query: &[f32]) -> TEncodedQuery;

    fn score_point(&self, query: &TEncodedQuery, i: u32) -> f32;

    fn score_internal(&self, i: u32, j: u32) -> f32;
}

impl DistanceType {
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            DistanceType::Dot => a.iter().zip(b.iter()).map(|(a, b)| a * b).sum(),
            DistanceType::L2 => a.iter().zip(b.iter()).map(|(a, b)| (a - b) * (a - b)).sum(),
        }
    }
}
