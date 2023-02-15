pub mod encoded_storage;
pub mod encoded_vectors;
pub mod encoded_vectors_u8;
pub mod quantile;

pub use encoded_vectors::DistanceType;
pub use encoded_vectors::VectorParameters;
pub use encoded_vectors::EncodedVectors;

pub use encoded_storage::EncodedStorage;
pub use encoded_storage::EncodedStorageBuilder;

pub use encoded_vectors_u8::EncodedVectorsU8;
pub use encoded_vectors_u8::EncodedQueryU8;

#[derive(Debug)]
pub struct EncodingError {
    pub description: String,
}
