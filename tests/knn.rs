use quantization::{encoder::EncodedVectorStorage, lut::Lut};
use rand::Rng;

fn metric(a: &[f32], b: &[f32]) -> f32 {
    assert!(a.len() == b.len());
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

fn knn(scores: &[(usize, f32)], k: usize) -> Vec<usize> {
    let mut scores = scores.to_vec();
    scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scores.iter().take(k).map(|(i, _)| *i).collect()
}

fn same_count(a: &[usize], b: &[usize]) -> usize {
    let mut count = 0;
    for i in a {
        if b.contains(i) {
            count += 1;
        }
    }
    count
}

#[test]
fn knn_test() {
    let vectors_count = 1_000;
    let vector_dim = 64;
    let mut rng = rand::thread_rng();
    let mut vector_data: Vec<Vec<f32>> = Vec::new();
    for _ in 0..vectors_count {
        let vector: Vec<f32> = (0..vector_dim).map(|_| rng.gen()).collect();
        vector_data.push(vector);
    }

    let timer = std::time::Instant::now();
    let chunks = EncodedVectorStorage::divide_dim(vector_dim, 2);
    let encoder =
        EncodedVectorStorage::new(Box::new(vector_data.iter().map(|v| v.as_slice())), &chunks)
            .unwrap();
    println!("encoding time: {}ms", timer.elapsed().as_millis());

    let queries_count = 10;

    let mut same_10 = 0.0;
    let mut same_30 = 0.0;
    let mut same_100 = 0.0;
    for _ in 0..queries_count {
        let query: Vec<f32> = (0..vector_dim).map(|_| rng.gen()).collect();
        let lut = Lut::new(&encoder, &query, metric);
        let mut scores_orig: Vec<(usize, f32)> = Vec::new();
        let mut scores_encoded: Vec<(usize, f32)> = Vec::new();
        for i in 0..vector_data.len() {
            let distance = metric(&query, &vector_data[i]);
            let encoded_distance = lut.dist(encoder.get(i));
            scores_orig.push((i, distance));
            scores_encoded.push((i, encoded_distance));
        }
        let knn_orig = knn(&scores_orig, 10);
        assert!(knn_orig.len() == 10);
        let knn_encoded_10 = knn(&scores_encoded, 10);
        assert!(knn_encoded_10.len() == 10);
        let knn_encoded_30 = knn(&scores_encoded, 30);
        assert!(knn_encoded_30.len() == 30);
        let knn_encoded_100 = knn(&scores_encoded, 100);
        assert!(knn_encoded_100.len() == 100);
        same_10 += same_count(&knn_orig, &knn_encoded_10) as f32;
        same_30 += same_count(&knn_orig, &knn_encoded_30) as f32;
        same_100 += same_count(&knn_orig, &knn_encoded_100) as f32;
    }
    same_10 /= queries_count as f32;
    same_30 /= queries_count as f32;
    same_100 /= queries_count as f32;

    println!(
        "compression: {}",
        (vector_dim * vectors_count * std::mem::size_of::<f32>()) as f32
            / encoder.data_size() as f32
    );
    println!("same_10: {}", same_10);
    println!("same_30: {}", same_30);
    println!("same_100: {}", same_100);
    assert!(same_10 > 6.0);
    assert!(same_30 > 8.0);
    assert!(same_100 > 9.5);
}
