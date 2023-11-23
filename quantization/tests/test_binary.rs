#[allow(unused)]
mod metrics;

#[cfg(test)]
mod tests {
    use quantization::{
        encoded_vectors::{DistanceType, EncodedVectors, VectorParameters},
        encoded_vectors_binary::EncodedVectorsBin,
    };
    use rand::{Rng, SeedableRng};

    use crate::metrics::{dot_similarity, l1_similarity, l2_similarity};

    fn generate_number(rng: &mut rand::rngs::StdRng) -> f32 {
        let n = f32::signum(rng.gen_range(-1.0..1.0));
        if n == 0.0 {
            1.0
        } else {
            n
        }
    }

    fn generate_vector(dim: usize, rng: &mut rand::rngs::StdRng) -> Vec<f32> {
        (0..dim).map(|_| generate_number(rng)).collect()
    }

    #[test]
    fn test_binary_dot() {
        let vectors_count = 128;
        let vector_dim = 3 * 128;
        let error = vector_dim as f32 * 0.01;

        //let mut rng = rand::thread_rng();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<f32>> = Vec::new();
        for _ in 0..vectors_count {
            vector_data.push(generate_vector(vector_dim, &mut rng));
        }

        let encoded = EncodedVectorsBin::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            &VectorParameters {
                dim: vector_dim,
                count: vectors_count,
                distance_type: DistanceType::Dot,
                invert: false,
            },
            || false,
        )
        .unwrap();

        let query: Vec<f32> = generate_vector(vector_dim, &mut rng);
        let query_u8 = encoded.encode_query(&query);

        for (index, vector) in vector_data.iter().enumerate() {
            let score = encoded.score_point(&query_u8, index as u32);
            let orginal_score = dot_similarity(&query, vector);
            assert!((score - orginal_score).abs() < error);
        }
    }

    #[test]
    fn test_binary_dot_inverted() {
        let vectors_count = 128;
        let vector_dim = 128;
        let error = vector_dim as f32 * 0.01;

        //let mut rng = rand::thread_rng();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<f32>> = Vec::new();
        for _ in 0..vectors_count {
            vector_data.push(generate_vector(vector_dim, &mut rng));
        }

        let encoded = EncodedVectorsBin::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            &VectorParameters {
                dim: vector_dim,
                count: vectors_count,
                distance_type: DistanceType::Dot,
                invert: true,
            },
            || false,
        )
        .unwrap();

        let query: Vec<f32> = generate_vector(vector_dim, &mut rng);
        let query_u8 = encoded.encode_query(&query);

        for (index, vector) in vector_data.iter().enumerate() {
            let score = encoded.score_point(&query_u8, index as u32);
            let orginal_score = -dot_similarity(&query, vector);
            assert!((score - orginal_score).abs() < error);
        }
    }

    #[test]
    fn test_binary_dot_internal() {
        let vectors_count = 128;
        let vector_dim = 128;
        let error = vector_dim as f32 * 0.01;

        //let mut rng = rand::thread_rng();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<f32>> = Vec::new();
        for _ in 0..vectors_count {
            vector_data.push(generate_vector(vector_dim, &mut rng));
        }

        let encoded = EncodedVectorsBin::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            &VectorParameters {
                dim: vector_dim,
                count: vectors_count,
                distance_type: DistanceType::Dot,
                invert: false,
            },
            || false,
        )
        .unwrap();

        for i in 1..vectors_count {
            let score = encoded.score_internal(0, i as u32);
            let orginal_score = dot_similarity(&vector_data[0], &vector_data[i]);
            assert!((score - orginal_score).abs() < error);
        }
    }

    #[test]
    fn test_binary_dot_inverted_internal() {
        let vectors_count = 128;
        let vector_dim = 128;
        let error = vector_dim as f32 * 0.01;

        //let mut rng = rand::thread_rng();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<f32>> = Vec::new();
        for _ in 0..vectors_count {
            vector_data.push(generate_vector(vector_dim, &mut rng));
        }

        let encoded = EncodedVectorsBin::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            &VectorParameters {
                dim: vector_dim,
                count: vectors_count,
                distance_type: DistanceType::Dot,
                invert: true,
            },
            || false,
        )
        .unwrap();

        for i in 1..vectors_count {
            let score = encoded.score_internal(0, i as u32);
            let orginal_score = -dot_similarity(&vector_data[0], &vector_data[i]);
            assert!((score - orginal_score).abs() < error);
        }
    }

    #[test]
    fn test_binary_l1() {
        let vectors_count = 128;
        let vector_dim = 3 * 128;
        let error = vector_dim as f32 * 0.01;

        //let mut rng = rand::thread_rng();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<f32>> = Vec::new();
        for _ in 0..vectors_count {
            vector_data.push(generate_vector(vector_dim, &mut rng));
        }

        let encoded = EncodedVectorsBin::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            &VectorParameters {
                dim: vector_dim,
                count: vectors_count,
                distance_type: DistanceType::L1,
                invert: false,
            },
            || false,
        )
        .unwrap();

        let query: Vec<f32> = generate_vector(vector_dim, &mut rng);
        let query_u8 = encoded.encode_query(&query);

        for (index, vector) in vector_data.iter().enumerate() {
            let score = encoded.score_point(&query_u8, index as u32);
            let orginal_score = l1_similarity(&query, vector);
            assert!((score - orginal_score).abs() < error);
        }
    }

    #[test]
    fn test_binary_l1_inverted() {
        let vectors_count = 128;
        let vector_dim = 128;
        let error = vector_dim as f32 * 0.01;

        //let mut rng = rand::thread_rng();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<f32>> = Vec::new();
        for _ in 0..vectors_count {
            vector_data.push(generate_vector(vector_dim, &mut rng));
        }

        let encoded = EncodedVectorsBin::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            &VectorParameters {
                dim: vector_dim,
                count: vectors_count,
                distance_type: DistanceType::L1,
                invert: true,
            },
            || false,
        )
        .unwrap();

        let query: Vec<f32> = generate_vector(vector_dim, &mut rng);
        let query_u8 = encoded.encode_query(&query);

        for (index, vector) in vector_data.iter().enumerate() {
            let score = encoded.score_point(&query_u8, index as u32);
            let orginal_score = (-l1_similarity(&query, vector)).exp();
            assert!((score - orginal_score).abs() < error);
        }
    }

    #[test]
    fn test_binary_l1_internal() {
        let vectors_count = 128;
        let vector_dim = 128;
        let error = vector_dim as f32 * 0.01;

        //let mut rng = rand::thread_rng();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<f32>> = Vec::new();
        for _ in 0..vectors_count {
            vector_data.push(generate_vector(vector_dim, &mut rng));
        }

        let encoded = EncodedVectorsBin::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            &VectorParameters {
                dim: vector_dim,
                count: vectors_count,
                distance_type: DistanceType::L1,
                invert: false,
            },
            || false,
        )
        .unwrap();

        for i in 1..vectors_count {
            let score = encoded.score_internal(0, i as u32);
            let orginal_score = l1_similarity(&vector_data[0], &vector_data[i]);
            assert!((score - orginal_score).abs() < error);
        }
    }

    #[test]
    fn test_binary_l1_inverted_internal() {
        let vectors_count = 128;
        let vector_dim = 128;
        let error = vector_dim as f32 * 0.01;

        //let mut rng = rand::thread_rng();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<f32>> = Vec::new();
        for _ in 0..vectors_count {
            vector_data.push(generate_vector(vector_dim, &mut rng));
        }

        let encoded = EncodedVectorsBin::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            &VectorParameters {
                dim: vector_dim,
                count: vectors_count,
                distance_type: DistanceType::L1,
                invert: true,
            },
            || false,
        )
        .unwrap();

        for i in 1..vectors_count {
            let score = encoded.score_internal(0, i as u32);
            let orginal_score = (-l1_similarity(&vector_data[0], &vector_data[i])).exp();
            assert!((score - orginal_score).abs() < error);
        }
    }

    #[test]
    fn test_binary_l2() {
        let vectors_count = 128;
        let vector_dim = 3 * 128;
        let error = vector_dim as f32 * 0.01;

        //let mut rng = rand::thread_rng();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<f32>> = Vec::new();
        for _ in 0..vectors_count {
            vector_data.push(generate_vector(vector_dim, &mut rng));
        }

        let encoded = EncodedVectorsBin::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            &VectorParameters {
                dim: vector_dim,
                count: vectors_count,
                distance_type: DistanceType::L2,
                invert: false,
            },
            || false,
        )
        .unwrap();

        let query: Vec<f32> = generate_vector(vector_dim, &mut rng);
        let query_u8 = encoded.encode_query(&query);

        for (index, vector) in vector_data.iter().enumerate() {
            let score = encoded.score_point(&query_u8, index as u32);
            let orginal_score = l2_similarity(&query, vector);
            assert!((score - orginal_score).abs() < error);
        }
    }

    #[test]
    fn test_binary_l2_inverted() {
        let vectors_count = 128;
        let vector_dim = 128;
        let error = vector_dim as f32 * 0.01;

        //let mut rng = rand::thread_rng();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<f32>> = Vec::new();
        for _ in 0..vectors_count {
            vector_data.push(generate_vector(vector_dim, &mut rng));
        }

        let encoded = EncodedVectorsBin::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            &VectorParameters {
                dim: vector_dim,
                count: vectors_count,
                distance_type: DistanceType::L2,
                invert: true,
            },
            || false,
        )
        .unwrap();

        let query: Vec<f32> = generate_vector(vector_dim, &mut rng);
        let query_u8 = encoded.encode_query(&query);

        for (index, vector) in vector_data.iter().enumerate() {
            let score = encoded.score_point(&query_u8, index as u32);
            let orginal_score = (-l2_similarity(&query, vector)).exp();
            assert!((score - orginal_score).abs() < error);
        }
    }

    #[test]
    fn test_binary_l2_internal() {
        let vectors_count = 128;
        let vector_dim = 128;
        let error = vector_dim as f32 * 0.01;

        //let mut rng = rand::thread_rng();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<f32>> = Vec::new();
        for _ in 0..vectors_count {
            vector_data.push(generate_vector(vector_dim, &mut rng));
        }

        let encoded = EncodedVectorsBin::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            &VectorParameters {
                dim: vector_dim,
                count: vectors_count,
                distance_type: DistanceType::L2,
                invert: false,
            },
            || false,
        )
        .unwrap();

        for i in 1..vectors_count {
            let score = encoded.score_internal(0, i as u32);
            let orginal_score = l2_similarity(&vector_data[0], &vector_data[i]);
            assert!((score - orginal_score).abs() < error);
        }
    }

    #[test]
    fn test_binary_l2_inverted_internal() {
        let vectors_count = 128;
        let vector_dim = 128;
        let error = vector_dim as f32 * 0.01;

        //let mut rng = rand::thread_rng();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<f32>> = Vec::new();
        for _ in 0..vectors_count {
            vector_data.push(generate_vector(vector_dim, &mut rng));
        }

        let encoded = EncodedVectorsBin::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            &VectorParameters {
                dim: vector_dim,
                count: vectors_count,
                distance_type: DistanceType::L2,
                invert: true,
            },
            || false,
        )
        .unwrap();

        for i in 1..vectors_count {
            let score = encoded.score_internal(0, i as u32);
            let orginal_score = (-l2_similarity(&vector_data[0], &vector_data[i])).exp();
            assert!((score - orginal_score).abs() < error);
        }
    }
}
