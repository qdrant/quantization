pub mod encoded_storage;
pub mod encoded_vectors;
pub mod encoded_vectors_u8;
pub mod quantile;
pub mod pq;

use std::fmt::Display;

pub use encoded_vectors::DistanceType;
pub use encoded_vectors::EncodedVectors;
pub use encoded_vectors::VectorParameters;

pub use encoded_storage::EncodedStorage;
pub use encoded_storage::EncodedStorageBuilder;

pub use encoded_vectors_u8::EncodedQueryU8;
pub use encoded_vectors_u8::EncodedVectorsU8;

#[derive(Debug)]
pub struct EncodingError {
    pub description: String,
}

impl Display for EncodingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description)
    }
}
