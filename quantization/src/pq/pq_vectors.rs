pub const CLUSTER_COUNT: usize = 256;

pub type ScoreType = u16;

/// Struct that holds holds PQ-quantized vectors.
pub struct PQVectors {
    lookup_table: Vec<ScoreType>,
    quantized_vectors: Vec<u8>,
    // Quantized vector dimension
    qdim: usize,
    count: usize,
}

impl PQVectors {
    pub fn new(lookup_table: Vec<ScoreType>, quantized_vectors: Vec<u8>, dim: usize, count: usize) -> Self {
        Self {
            lookup_table,
            quantized_vectors,
            qdim: dim,
            count,
        }
    }

    pub fn score(&self, quantized_query: &[u8], idx: usize) -> ScoreType {
        debug_assert!(idx < self.count);
        let mut score = 0. as ScoreType;
        for (i, &q) in quantized_query.iter().enumerate() {
            // Stored vector element offset
            let offset = idx * self.qdim + i;
            let element = self.quantized_vectors[offset];
            let lut_offset = element as usize * CLUSTER_COUNT + q as usize;
            score += self.lookup_table[lut_offset];
        }
        score
    }
}