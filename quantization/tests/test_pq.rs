#[allow(unused)]
mod metrics;

#[cfg(test)]
mod tests {
    use quantization::{
        encoded_vectors::{DistanceType, EncodedVectors, VectorParameters},
        encoded_vectors_pq::EncodedVectorsPQ,
    };
    use rand::{Rng, SeedableRng};

    use crate::metrics::{dot_similarity, l2_similarity};

    const VECTORS_COUNT: usize = 513;
    const VECTOR_DIM: usize = 65;
    const ERROR: f32 = VECTOR_DIM as f32 * 0.05;

    #[test]
    fn test_pq_dot() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<f32>> = Vec::new();
        for _ in 0..VECTORS_COUNT {
            let vector: Vec<f32> = (0..VECTOR_DIM).map(|_| rng.gen()).collect();
            vector_data.push(vector);
        }
        let query: Vec<f32> = (0..VECTOR_DIM).map(|_| rng.gen()).collect();

        let encoded = EncodedVectorsPQ::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            &VectorParameters {
                dim: VECTOR_DIM,
                count: VECTORS_COUNT,
                distance_type: DistanceType::Dot,
                invert: false,
            },
            1,
            1,
        )
        .unwrap();
        let query_u8 = encoded.encode_query(&query);

        for (index, vector) in vector_data.iter().enumerate() {
            let score = encoded.score_point(&query_u8, index as u32);
            let orginal_score = dot_similarity(&query, vector);
            assert!((score - orginal_score).abs() < ERROR);
        }
    }

    #[test]
    fn test_pq_l2() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<f32>> = Vec::new();
        for _ in 0..VECTORS_COUNT {
            let vector: Vec<f32> = (0..VECTOR_DIM).map(|_| rng.gen::<f32>()).collect();
            vector_data.push(vector);
        }
        let query: Vec<f32> = (0..VECTOR_DIM).map(|_| rng.gen::<f32>()).collect();

        let encoded = EncodedVectorsPQ::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            &VectorParameters {
                dim: VECTOR_DIM,
                count: VECTORS_COUNT,
                distance_type: DistanceType::L2,
                invert: false,
            },
            1,
            1,
        )
        .unwrap();
        let query_u8 = encoded.encode_query(&query);

        for (index, vector) in vector_data.iter().enumerate() {
            let score = encoded.score_point(&query_u8, index as u32);
            let orginal_score = l2_similarity(&query, vector);
            assert!((score - orginal_score).abs() < ERROR);
        }
    }

    #[test]
    fn test_pq_dot_inverted() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<f32>> = Vec::new();
        for _ in 0..VECTORS_COUNT {
            let vector: Vec<f32> = (0..VECTOR_DIM).map(|_| rng.gen()).collect();
            vector_data.push(vector);
        }
        let query: Vec<f32> = (0..VECTOR_DIM).map(|_| rng.gen()).collect();

        let encoded = EncodedVectorsPQ::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            &VectorParameters {
                dim: VECTOR_DIM,
                count: VECTORS_COUNT,
                distance_type: DistanceType::Dot,
                invert: true,
            },
            1,
            1,
        )
        .unwrap();
        let query_u8 = encoded.encode_query(&query);

        for (index, vector) in vector_data.iter().enumerate() {
            let score = encoded.score_point(&query_u8, index as u32);
            let orginal_score = -dot_similarity(&query, vector);
            assert!((score - orginal_score).abs() < ERROR);
        }
    }

    #[test]
    fn test_pq_l2_inverted() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<f32>> = Vec::new();
        for _ in 0..VECTORS_COUNT {
            let vector: Vec<f32> = (0..VECTOR_DIM).map(|_| rng.gen::<f32>()).collect();
            vector_data.push(vector);
        }
        let query: Vec<f32> = (0..VECTOR_DIM).map(|_| rng.gen::<f32>()).collect();

        let encoded = EncodedVectorsPQ::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            &VectorParameters {
                dim: VECTOR_DIM,
                count: VECTORS_COUNT,
                distance_type: DistanceType::L2,
                invert: true,
            },
            1,
            1,
        )
        .unwrap();
        let query_u8 = encoded.encode_query(&query);

        for (index, vector) in vector_data.iter().enumerate() {
            let score = encoded.score_point(&query_u8, index as u32);
            let orginal_score = -l2_similarity(&query, vector);
            assert!((score - orginal_score).abs() < ERROR);
        }
    }

    #[test]
    fn test_pq_dot_internal() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<f32>> = Vec::new();
        for _ in 0..VECTORS_COUNT {
            let vector: Vec<f32> = (0..VECTOR_DIM).map(|_| rng.gen()).collect();
            vector_data.push(vector);
        }

        let encoded = EncodedVectorsPQ::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            &VectorParameters {
                dim: VECTOR_DIM,
                count: VECTORS_COUNT,
                distance_type: DistanceType::Dot,
                invert: false,
            },
            1,
            1,
        )
        .unwrap();

        for i in 1..VECTORS_COUNT {
            let score = encoded.score_internal(0, i as u32);
            let orginal_score = dot_similarity(&vector_data[0], &vector_data[i]);
            assert!((score - orginal_score).abs() < ERROR);
        }
    }

    #[test]
    fn test_pq_dot_inverted_internal() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<f32>> = Vec::new();
        for _ in 0..VECTORS_COUNT {
            let vector: Vec<f32> = (0..VECTOR_DIM).map(|_| rng.gen()).collect();
            vector_data.push(vector);
        }

        let encoded = EncodedVectorsPQ::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            &VectorParameters {
                dim: VECTOR_DIM,
                count: VECTORS_COUNT,
                distance_type: DistanceType::Dot,
                invert: true,
            },
            1,
            1,
        )
        .unwrap();

        for i in 1..VECTORS_COUNT {
            let score = encoded.score_internal(0, i as u32);
            let orginal_score = -dot_similarity(&vector_data[0], &vector_data[i]);
            assert!((score - orginal_score).abs() < ERROR);
        }
    }
}