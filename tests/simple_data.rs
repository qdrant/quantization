use quantization::{encoder::EncodedVectorStorage, lut::Lut};

fn metric(a: &[f32], b: &[f32]) -> f32 {
    assert!(a.len() == b.len());
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

#[test]
fn simple_data_test() {
    let vector_data: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0],
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![3.0, 3.0],
        vec![4.0, 4.0],
        vec![5.0, 5.0],
        vec![6.0, 6.0],
        vec![7.0, 7.0],
        vec![8.0, 8.0],
        vec![9.0, 9.0],
        vec![10.0, 10.0],
        vec![11.0, 11.0],
        vec![12.0, 12.0],
        vec![13.0, 13.0],
        vec![14.0, 14.0],
        vec![15.0, 15.0],
    ];
    let chunks = EncodedVectorStorage::divide_dim(2, 1);
    let encoder =
        EncodedVectorStorage::new(Box::new(vector_data.iter().map(|v| v.as_slice())), &chunks)
            .unwrap();
    let query: Vec<f32> = vec![2.0, 2.0];
    let lut = Lut::new(&encoder, &query, metric);
    for i in 0..vector_data.len() {
        let distance = metric(&query, &vector_data[i]);
        let encoded_distance = lut.dist(encoder.get(i));
        assert_eq!(distance, encoded_distance);
    }
}
