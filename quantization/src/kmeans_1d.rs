use crate::CENTROIDS_COUNT;

pub fn kmeans_1d(array: &[f32]) -> (Vec<f32>, Vec<usize>) {
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    for &v in array {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    let mut centroids = [0.0; CENTROIDS_COUNT];
    for i in 0..CENTROIDS_COUNT {
        centroids[i] = min + i as f32 * (max - min) / CENTROIDS_COUNT as f32;
    }
    let mut indexes = vec![0u8; array.len()];

    let mut cluster_size = [0u32; CENTROIDS_COUNT];
    for _ in 0..50 {
        update_indexes(array, &centroids, &mut indexes);
        update_centroids(array, &mut centroids, &indexes, &mut cluster_size);
    }

    (
        centroids.to_vec(),
        indexes.iter().map(|&i| i as usize).collect(),
    )
}

fn update_centroids(
    array: &[f32],
    centroids: &mut [f32; CENTROIDS_COUNT],
    indexes: &[u8],
    cluster_size: &mut [u32; CENTROIDS_COUNT],
) {
    centroids.iter_mut().for_each(|c| *c = 0.0);
    cluster_size.iter_mut().for_each(|c| *c = 0);
    for (i, &v) in array.iter().enumerate() {
        cluster_size[indexes[i] as usize] += 1;
        centroids[indexes[i] as usize] += v;
    }
    for (c, &s) in centroids.iter_mut().zip(cluster_size.iter()) {
        if s > 0 {
            *c /= s as f32;
        }
    }
}

fn update_indexes(array: &[f32], centroids: &[f32; CENTROIDS_COUNT], indexes: &mut [u8]) {
    for (i, &v) in array.iter().enumerate() {
        let mut min_dist = f32::MAX;
        let mut min_index = 0u8;
        for (j, &c) in centroids.iter().enumerate() {
            let dist = (v - c).abs();
            if dist < min_dist {
                min_dist = dist;
                min_index = j as u8;
            }
        }
        indexes[i] = min_index;
    }
}
