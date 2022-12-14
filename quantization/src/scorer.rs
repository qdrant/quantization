pub trait Scorer {
    fn score_point(&self, point: usize) -> f32;

    fn score_points(&mut self, points: &[usize], scores: &mut [f32]) {
        for (i, &point) in points.iter().enumerate() {
            scores[i] = self.score_point(point);
        }
    }
}
