use crate::CENTROIDS_COUNT;

#[derive(Clone, Copy)]
struct Vec2 {
    x: f32,
    y: f32,
}

impl Vec2 {
    fn new(x: f32, y: f32) -> Vec2 {
        Vec2 { x, y }
    }

    fn dist2(&self, other: &Vec2) -> f32 {
        (self.x - other.x).powi(2) + (self.y - other.y).powi(2)
    }

    fn add(&self, other: &Vec2) -> Vec2 {
        Vec2::new(self.x + other.x, self.y + other.y)
    }

    fn min(&self, other: &Vec2) -> Vec2 {
        Vec2::new(self.x.min(other.x), self.y.min(other.y))
    }

    fn max(&self, other: &Vec2) -> Vec2 {
        Vec2::new(self.x.max(other.x), self.y.max(other.y))
    }

    fn div(&self, other: f32) -> Vec2 {
        Vec2::new(self.x / other, self.y / other)
    }
}

pub fn kmeans_2d(array: &[f32]) -> (Vec<f32>, Vec<usize>) {
    let array = array.chunks_exact(2).map(|v| Vec2::new(v[0], v[1])).collect::<Vec<_>>();

    let mut min = Vec2::new(f32::MAX, f32::MAX);
    let mut max = Vec2::new(f32::MIN, f32::MIN);
    for p in &array {
        min = min.min(p);
        max = max.max(p);
    }

    let mut centroids: [Vec2; CENTROIDS_COUNT] = [Vec2::new(0.0, 0.0); CENTROIDS_COUNT];
    let centroids_sqrt = (CENTROIDS_COUNT as f32).sqrt() as usize;
    for i in 0..centroids_sqrt {
        for j in 0..centroids_sqrt {
            centroids[i * centroids_sqrt + j] = Vec2::new(
                min.x + i as f32 * (max.x - min.x) / centroids_sqrt as f32,
                min.y + j as f32 * (max.y - min.y) / centroids_sqrt as f32,
            );
        }
    }

    let mut indexes = vec![0u8; array.len()];
    for _ in 0..200 {
        update_indexes(&array, &centroids, &mut indexes);
        update_centroids(&array, &mut centroids, &indexes);
    }

    let mut c = vec![];
    for i in 0..centroids.len() {
        c.push(centroids[i].x);
        c.push(centroids[i].y);
    }
    (c, indexes.iter().map(|&i| i as usize).collect())
}

fn update_centroids(
    array: &[Vec2],
    centroids: &mut [Vec2; CENTROIDS_COUNT],
    indexes: &[u8],
) {
    let mut cluster_size = [0u32; CENTROIDS_COUNT];
    centroids.iter_mut().for_each(|c| *c = Vec2::new(0.0, 0.0));
    cluster_size.iter_mut().for_each(|c| *c = 0);
    for (i, v) in array.iter().enumerate() {
        cluster_size[indexes[i] as usize] += 1;
        centroids[indexes[i] as usize] = centroids[indexes[i] as usize].add(v);
    }
    for (c, &s) in centroids.iter_mut().zip(cluster_size.iter()) {
        if s > 0 {
            *c = c.div(s as f32);
        }
    }
}

fn update_indexes(
    array: &[Vec2],
    centroids: &[Vec2; CENTROIDS_COUNT],
    indexes: &mut [u8],
) {
    for (i, v) in array.iter().enumerate() {
        let mut min_dist = f32::MAX;
        let mut min_index = 0u8;
        for (j, c) in centroids.iter().enumerate() {
            let dist = v.dist2(c);
            if dist < min_dist {
                min_dist = dist;
                min_index = j as u8;
                assert!(min_index < CENTROIDS_COUNT as u8);
            }
        }
        indexes[i] = min_index;
    }
}
