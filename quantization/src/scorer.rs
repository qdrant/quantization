pub trait Scorer {
    fn score_point(&self, point: usize) -> f32;
}
