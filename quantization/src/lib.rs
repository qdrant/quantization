pub mod encoded_storage;
pub mod encoded_vectors;
pub mod encoded_vectors_pq;
pub mod encoded_vectors_u8;
pub mod kmeans;
pub mod quantile;

use std::fmt::Display;
use std::sync::Arc;
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

#[derive(Default, PartialEq, Clone, Copy)]
enum ConditionalVariableState {
    #[default]
    Waiting,
    Notified,
}

// ConditionalVariable is a wrapper around a mutex and a condvar
#[derive(Default, Clone)]
pub struct ConditionalVariable {
    mutex: Arc<Mutex<ConditionalVariableState>>,
    condvar: Arc<Condvar>,
}

impl ConditionalVariable {
    pub fn wait(&self) -> bool {
        let mut guard = self.mutex.lock().unwrap();
        while *guard == ConditionalVariableState::Waiting && Arc::strong_count(&self.mutex) > 1 {
            guard = self.condvar.wait(guard).unwrap();
        }
        *guard = ConditionalVariableState::Waiting;
        Arc::strong_count(&self.mutex) == 1
    }

    pub fn notify(&self) {
        *self.mutex.lock().unwrap() = ConditionalVariableState::Notified;
        self.condvar.notify_all();
    }
}

impl Drop for ConditionalVariable {
    fn drop(&mut self) {
        self.condvar.notify_all();
    }
}
