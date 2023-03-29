
pub fn kmeans(
    data: &[f32],
    centroids_count: usize,
    dim: usize,
    max_iterations: usize,
) -> Vec<f32> {
    let mut centroids = data[0..centroids_count * dim].to_vec();
    let mut centroid_indexes = vec![0u32; data.len() / dim];

    for _ in 0..max_iterations {
        update_indexes(data, &mut centroid_indexes, &centroids);
        if update_centroids(data, &centroid_indexes, &mut centroids) {
            break;
        }
    }
    update_indexes(data, &mut centroid_indexes, &centroids);
    centroids
}

fn update_centroids(
    data: &[f32],
    centroid_indexes: &[u32],
    centroids: &mut [f32],
) -> bool {
    let dim = data.len() / centroid_indexes.len();
    let centroids_count = centroids.len() / dim;

    let mut centroids_counter = vec![0usize; centroids_count];
    let mut centroids_acc = vec![0.0_f64; centroids.len()];
    let mut rand_vectors = Vec::with_capacity(centroids_count);

    for (i, vector_data) in data.chunks_exact(dim).enumerate() {
        // take some vector for case when centroid is not updated
        if rand_vectors.len() < centroids_count {
            rand_vectors.push(vector_data);
        }

        let centroid_index = centroid_indexes[i] as usize;
        centroids_counter[centroid_index] += 1;
        let centroid_data = &mut centroids_acc[dim * centroid_index..dim * (centroid_index + 1)];
        for (c, v) in centroid_data.iter_mut().zip(vector_data.iter()) {
            *c += *v as f64;
        }
    }

    for (centroid_index, centroid_data) in centroids_acc.chunks_exact_mut(dim).enumerate() {
        let rand_vector = &rand_vectors[centroid_index];
        for (c, r) in centroid_data.iter_mut().zip(rand_vector.iter()) {
            if centroids_counter[centroid_index] == 0 {
                *c = *r as f64;
            } else {
                *c /= centroids_counter[centroid_index] as f64;
            }
        }
    }

    for (c, c_acc) in centroids.iter_mut().zip(centroids_acc.iter()) {
        *c = *c_acc as f32;
    }
    false
}

fn update_indexes(
    data: &[f32],
    centroid_indexes: &mut [u32],
    centroids: &[f32],
) {
    let dim = data.len() / centroid_indexes.len();
    for (i, vector_data) in data.chunks_exact(dim).enumerate() {
        let mut min_distance = f32::MAX;
        let mut min_centroid_index = 0;
        for (centroid_index, centroid_data) in centroids.chunks_exact(dim).enumerate() {
            let distance = vector_data
                .iter()
                .zip(centroid_data.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>();
            if distance < min_distance {
                min_distance = distance;
                min_centroid_index = centroid_index;
            }
        }
        centroid_indexes[i] = min_centroid_index as u32;
    }
}
