pub mod encoded_storage;
pub mod encoded_vectors;
pub mod encoded_vectors_pq;
pub mod encoded_vectors_u8;
pub mod kmeans;
pub mod quantile;

use std::fmt::Display;
use std::sync::Condvar;
use std::sync::Mutex;

pub use encoded_vectors::DistanceType;
pub use encoded_vectors::EncodedVectors;
pub use encoded_vectors::VectorParameters;

pub use encoded_storage::EncodedStorage;
pub use encoded_storage::EncodedStorageBuilder;

pub use encoded_vectors_u8::EncodedQueryU8;
pub use encoded_vectors_u8::EncodedVectorsU8;

pub use encoded_vectors_pq::EncodedQueryPQ;
pub use encoded_vectors_pq::EncodedVectorsPQ;

#[derive(Debug)]
pub struct EncodingError {
    pub description: String,
}

impl Display for EncodingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description)
    }
}

// ConditionalVariable is a wrapper around a mutex and a condvar
#[derive(Default)]
pub struct ConditionalVariable {
    mutex: Mutex<bool>,
    condvar: Condvar,
}

impl ConditionalVariable {
    pub fn wait(&self) {
        let mut guard = self.mutex.lock().unwrap();
        while !*guard {
            guard = self.condvar.wait(guard).unwrap();
        }
        *guard = false;
    }

    pub fn notify(&self) {
        *self.mutex.lock().unwrap() = true;
        self.condvar.notify_one();
    }
}
