pub fn dot_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2).map(|(a, b)| a * b).sum()
}

pub fn l2_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter()
        .copied()
        .zip(v2.iter().copied())
        .map(|(a, b)| (a - b).powi(2))
        .sum()
}
