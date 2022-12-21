use std::collections::{BinaryHeap, HashSet};

use quantization::encoder::EncodedVectors;

#[derive(PartialEq, Clone, Debug, Default)]
pub struct Score {
    pub index: usize,
    pub score: f32,
}

impl Eq for Score {}

impl std::cmp::PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::cmp::Ord for Score {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.partial_cmp(&other.score).unwrap()
    }
}

#[allow(unused)]
pub fn euclid_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    let s: f32 = v1
        .iter()
        .copied()
        .zip(v2.iter().copied())
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    -s
}

#[allow(unused)]
pub fn dot_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2).map(|(a, b)| a * b).sum()
}

#[allow(unused)]
pub fn cosine_preprocess(vector: &mut [f32]) {
    let mut length: f32 = vector.iter().map(|x| x * x).sum();
    if length < f32::EPSILON {
        return;
    }
    length = length.sqrt();
    vector.iter_mut().for_each(|x| *x = *x / length);
}

#[allow(unused)]
pub fn knn<I>(count: usize, scores: I) -> Vec<Score>
where
    I: Iterator<Item = Score>,
{
    let mut heap: BinaryHeap<Score> = BinaryHeap::new();
    for score in scores {
        if heap.len() == count {
            let mut top = heap.peek_mut().unwrap();
            if top.score > score.score {
                *top = score;
            }
        } else {
            heap.push(score);
        }
    }
    heap.into_sorted_vec()
}

#[allow(unused)]
pub fn same_count(a: &[Score], b: &[Score]) -> usize {
    let a = a.iter().map(|s| s.index).collect::<HashSet<_>>();
    let b = b.iter().map(|s| s.index).collect::<HashSet<_>>();
    a.intersection(&b).count()
}

#[allow(unused)]
pub fn run_knn_queries<'a, I, F, M>(
    vectors_count: usize,
    queries: I,
    orig_data: F,
    encoded_data: &EncodedVectors,
    metric: M,
) -> (f32, f32, f32)
where
    I: Iterator<Item = &'a [f32]>,
    F: Fn(usize) -> &'a [f32],
    M: Fn(&[f32], &[f32]) -> f32,
{
    let mut same_10 = 0.0;
    let mut same_30 = 0.0;
    let mut same_100 = 0.0;
    let mut queries_count = 0;
    for query in queries {
        let query_u8 = EncodedVectors::encode_query(query);
        queries_count += 1;
        let mut scores_orig: Vec<Score> = Vec::new();
        let mut scores_encoded: Vec<Score> = Vec::new();
        for index in 0..vectors_count {
            let distance = metric(&query, orig_data(index));
            let encoded_distance = encoded_data.score_point_dot_sse(&query_u8, index);
            scores_orig.push(Score {
                index,
                score: distance,
            });
            scores_encoded.push(Score {
                index,
                score: encoded_distance,
            });
        }
        let knn_orig = knn(10, scores_orig.iter().cloned());
        assert!(knn_orig.len() == 10);
        let knn_encoded_10 = knn(10, scores_encoded.iter().cloned());
        assert!(knn_encoded_10.len() == 10);
        let knn_encoded_30 = knn(30, scores_encoded.iter().cloned());
        assert!(knn_encoded_30.len() == 30);
        let knn_encoded_100 = knn(100, scores_encoded.iter().cloned());
        assert!(knn_encoded_100.len() == 100);
        same_10 += same_count(&knn_orig, &knn_encoded_10) as f32;
        same_30 += same_count(&knn_orig, &knn_encoded_30) as f32;
        same_100 += same_count(&knn_orig, &knn_encoded_100) as f32;
    }
    same_10 /= queries_count as f32;
    same_30 /= queries_count as f32;
    same_100 /= queries_count as f32;

    println!("queries count: {}", queries_count);
    println!("same_10: {}", same_10);
    println!("same_30: {}", same_30);
    println!("same_100: {}", same_100);
    (same_10, same_30, same_100)
}
