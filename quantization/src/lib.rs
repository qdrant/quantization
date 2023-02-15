pub mod encoded_storage;
pub mod encoded_vectors;
pub mod encoded_vectors_u8;
pub mod quantile;

#[derive(Debug)]
pub struct EncodingError {
    pub description: String,
}
