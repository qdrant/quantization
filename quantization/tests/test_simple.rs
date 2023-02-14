#[allow(unused)]
mod metrics;

#[cfg(test)]
mod tests {
    use quantization::{
        encoded_vectors::{EncodedVectors, SimilarityType, VectorParameters},
        encoded_vectors_u8::EncodedVectorsU8,
    };
    use rand::{Rng, SeedableRng};

    use crate::metrics::{dot_similarity, l2_similarity};

    #[test]
    fn test_dot_simple() {
        let vectors_count = 129;
        let vector_dim = 65;
        let error = vector_dim as f32 * 0.1;

        //let mut rng = rand::thread_rng();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<f32>> = Vec::new();
        for _ in 0..vectors_count {
            let vector: Vec<f32> = (0..vector_dim).map(|_| rng.gen()).collect();
            vector_data.push(vector);
        }
        let query: Vec<f32> = (0..vector_dim).map(|_| rng.gen()).collect();

        let encoded = EncodedVectorsU8::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            &VectorParameters {
                dim: vector_dim,
                distance_type: SimilarityType::Dot,
                invert: false,
            },
            None,
        )
        .unwrap();
        let query_u8 = encoded.encode_query(&query);

        for (index, vector) in vector_data.iter().enumerate() {
            let score = encoded.score_point_simple(&query_u8, index as u32);
            let orginal_score = dot_similarity(&query, vector);
            assert!((score - orginal_score).abs() < error);
        }
    }

    #[test]
    fn test_l2_simple() {
        let vectors_count = 129;
        let vector_dim = 65;
        let error = vector_dim as f32 * 0.1;

        //let mut rng = rand::thread_rng();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<f32>> = Vec::new();
        for _ in 0..vectors_count {
            let vector: Vec<f32> = (0..vector_dim).map(|_| rng.gen::<f32>()).collect();
            vector_data.push(vector);
        }
        let query: Vec<f32> = (0..vector_dim).map(|_| rng.gen::<f32>()).collect();

        let encoded = EncodedVectorsU8::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            &VectorParameters {
                dim: vector_dim,
                distance_type: SimilarityType::L2,
                invert: false,
            },
            None,
        )
        .unwrap();
        let query_u8 = encoded.encode_query(&query);

        for (index, vector) in vector_data.iter().enumerate() {
            let score = encoded.score_point_simple(&query_u8, index as u32);
            let orginal_score = l2_similarity(&query, vector);
            assert!((score - orginal_score).abs() < error);
        }
    }

    #[test]
    fn test_dot_inverted_simple() {
        let vectors_count = 129;
        let vector_dim = 65;
        let error = vector_dim as f32 * 0.1;

        //let mut rng = rand::thread_rng();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<f32>> = Vec::new();
        for _ in 0..vectors_count {
            let vector: Vec<f32> = (0..vector_dim).map(|_| rng.gen()).collect();
            vector_data.push(vector);
        }
        let query: Vec<f32> = (0..vector_dim).map(|_| rng.gen()).collect();

        let encoded = EncodedVectorsU8::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            &VectorParameters {
                dim: vector_dim,
                distance_type: SimilarityType::Dot,
                invert: true,
            },
            None,
        )
        .unwrap();
        let query_u8 = encoded.encode_query(&query);

        for (index, vector) in vector_data.iter().enumerate() {
            let score = encoded.score_point_simple(&query_u8, index as u32);
            let orginal_score = -dot_similarity(&query, vector);
            assert!((score - orginal_score).abs() < error);
        }
    }

    #[test]
    fn test_l2_inverted_simple() {
        let vectors_count = 129;
        let vector_dim = 65;
        let error = vector_dim as f32 * 0.1;

        //let mut rng = rand::thread_rng();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<f32>> = Vec::new();
        for _ in 0..vectors_count {
            let vector: Vec<f32> = (0..vector_dim).map(|_| rng.gen::<f32>()).collect();
            vector_data.push(vector);
        }
        let query: Vec<f32> = (0..vector_dim).map(|_| rng.gen::<f32>()).collect();

        let encoded = EncodedVectorsU8::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            &VectorParameters {
                dim: vector_dim,
                distance_type: SimilarityType::L2,
                invert: true,
            },
            None,
        )
        .unwrap();
        let query_u8 = encoded.encode_query(&query);

        for (index, vector) in vector_data.iter().enumerate() {
            let score = encoded.score_point_simple(&query_u8, index as u32);
            let orginal_score = -l2_similarity(&query, vector);
            assert!((score - orginal_score).abs() < error);
        }
    }

    #[test]
    fn test_dot_internal_simple() {
        let vectors_count: usize = 129;
        let vector_dim = 65;
        let error = vector_dim as f32 * 0.1;

        //let mut rng = rand::thread_rng();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<f32>> = Vec::new();
        for _ in 0..vectors_count {
            let vector: Vec<f32> = (0..vector_dim).map(|_| rng.gen()).collect();
            vector_data.push(vector);
        }

        let encoded = EncodedVectorsU8::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            &VectorParameters {
                dim: vector_dim,
                distance_type: SimilarityType::Dot,
                invert: false,
            },
            None,
        )
        .unwrap();

        for i in 1..vectors_count {
            let score = encoded.score_internal(0, i as u32);
            let orginal_score = dot_similarity(&vector_data[0], &vector_data[i]);
            assert!((score - orginal_score).abs() < error);
        }
    }

    #[test]
    fn test_dot_inverted_internal_simple() {
        let vectors_count: usize = 129;
        let vector_dim = 65;
        let error = vector_dim as f32 * 0.1;

        //let mut rng = rand::thread_rng();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<f32>> = Vec::new();
        for _ in 0..vectors_count {
            let vector: Vec<f32> = (0..vector_dim).map(|_| rng.gen()).collect();
            vector_data.push(vector);
        }

        let encoded = EncodedVectorsU8::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            &VectorParameters {
                dim: vector_dim,
                distance_type: SimilarityType::Dot,
                invert: true,
            },
            None,
        )
        .unwrap();

        for i in 1..vectors_count {
            let score = encoded.score_internal(0, i as u32);
            let orginal_score = -dot_similarity(&vector_data[0], &vector_data[i]);
            assert!((score - orginal_score).abs() < error);
        }
    }
}
