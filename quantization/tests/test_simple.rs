#[cfg(test)]
mod tests {
    use quantization::{
        encoder::{EncodedVectors, EncodingParameters, SimilarityType},
        utils::{dot_similarity, l2_similarity},
    };
    use rand::{Rng, SeedableRng};

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

        let encoded = EncodedVectors::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            EncodingParameters {
                distance_type: SimilarityType::Dot,
                ..Default::default()
            },
        )
        .unwrap();
        let query_u8 = encoded.encode_query(&query);

        let indexes = (0..vectors_count as u32).collect::<Vec<_>>();
        let mut scores = vec![0.0; vectors_count];
        encoded.score_points_simple(&query_u8, &indexes, &mut scores);

        for i in 0..vectors_count {
            let score = encoded.score_point_simple(&query_u8, i as u32);
            let score2 = scores[i];
            let orginal_score = dot_similarity(&query, &vector_data[i]);
            assert!((score - orginal_score).abs() < error);
            assert!((score2 - orginal_score).abs() < error);
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

        let encoded = EncodedVectors::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            EncodingParameters {
                distance_type: SimilarityType::L2,
                ..Default::default()
            },
        )
        .unwrap();
        let query_u8 = encoded.encode_query(&query);

        let indexes = (0..vectors_count as u32).collect::<Vec<_>>();
        let mut scores = vec![0.0; vectors_count];
        encoded.score_points_simple(&query_u8, &indexes, &mut scores);

        for i in 0..vectors_count {
            let score = encoded.score_point_simple(&query_u8, i as u32);
            let score2 = scores[i];
            let orginal_score = l2_similarity(&query, &vector_data[i]);
            assert!((score - orginal_score).abs() < error);
            assert!((score2 - orginal_score).abs() < error);
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

        let encoded = EncodedVectors::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            EncodingParameters {
                distance_type: SimilarityType::Dot,
                invert: true,
                ..Default::default()
            },
        )
        .unwrap();
        let query_u8 = encoded.encode_query(&query);

        let indexes = (0..vectors_count as u32).collect::<Vec<_>>();
        let mut scores = vec![0.0; vectors_count];
        encoded.score_points_simple(&query_u8, &indexes, &mut scores);

        for i in 0..vectors_count {
            let score = encoded.score_point_simple(&query_u8, i as u32);
            let score2 = scores[i];
            let orginal_score = -dot_similarity(&query, &vector_data[i]);
            assert!((score - orginal_score).abs() < error);
            assert!((score2 - orginal_score).abs() < error);
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

        let encoded = EncodedVectors::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            EncodingParameters {
                distance_type: SimilarityType::L2,
                invert: true,
                ..Default::default()
            },
        )
        .unwrap();
        let query_u8 = encoded.encode_query(&query);

        let indexes = (0..vectors_count as u32).collect::<Vec<_>>();
        let mut scores = vec![0.0; vectors_count];
        encoded.score_points_simple(&query_u8, &indexes, &mut scores);

        for i in 0..vectors_count {
            let score = encoded.score_point_simple(&query_u8, i as u32);
            let score2 = scores[i];
            let orginal_score = -l2_similarity(&query, &vector_data[i]);
            assert!((score - orginal_score).abs() < error);
            assert!((score2 - orginal_score).abs() < error);
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

        let encoded = EncodedVectors::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            EncodingParameters {
                distance_type: SimilarityType::Dot,
                ..Default::default()
            },
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

        let encoded = EncodedVectors::encode(
            vector_data.iter().map(|v| v.as_slice()),
            Vec::<u8>::new(),
            EncodingParameters {
                distance_type: SimilarityType::Dot,
                invert: true,
                ..Default::default()
            },
        )
        .unwrap();

        for i in 1..vectors_count {
            let score = encoded.score_internal(0, i as u32);
            let orginal_score = -dot_similarity(&vector_data[0], &vector_data[i]);
            assert!((score - orginal_score).abs() < error);
        }
    }
}
