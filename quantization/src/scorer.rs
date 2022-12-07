pub trait Scorer {
    fn score_point(&self, point: usize) -> f32;

    fn score_internal<M>(&self, point_a: usize, point_b: usize, metric: M) -> f32
    where
        M: Fn(&[f32], &[f32]) -> f32;
}
