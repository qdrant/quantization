use crate::encoded_vectors::validate_vector_parameters;
use crate::simd::sse2::xor_popcnt::impl_xor_popcnt_sse;
use crate::utils::{transmute_from_u8_to_slice, transmute_to_u8_slice};
use crate::{
    DistanceType, EncodedStorage, EncodedStorageBuilder, EncodedVectors, EncodingError,
    VectorParameters,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

type BitsStoreType = u128;

const BITS_STORE_TYPE_SIZE: usize = std::mem::size_of::<BitsStoreType>() * 8;

pub struct EncodedVectorsBin<TStorage: EncodedStorage> {
    encoded_vectors: TStorage,
    metadata: Metadata,
}

pub struct EncodedBinVector {
    encoded_vector: Vec<BitsStoreType>,
}

#[derive(Serialize, Deserialize)]
struct Metadata {
    vector_parameters: VectorParameters,
}

impl<TStorage: EncodedStorage> EncodedVectorsBin<TStorage> {
    pub fn encode<'a>(
        orig_data: impl Iterator<Item = impl AsRef<[f32]> + 'a> + Clone,
        mut storage_builder: impl EncodedStorageBuilder<TStorage>,
        vector_parameters: &VectorParameters,
        stop_condition: impl Fn() -> bool,
    ) -> Result<Self, EncodingError> {
        debug_assert!(validate_vector_parameters(orig_data.clone(), vector_parameters).is_ok());

        for vector in orig_data {
            if stop_condition() {
                return Err(EncodingError::Stopped);
            }

            let encoded_vector = Self::_encode_vector(vector.as_ref());
            let encoded_vector_slice = encoded_vector.encoded_vector.as_slice();
            let bytes = transmute_to_u8_slice(encoded_vector_slice);
            storage_builder.push_vector_data(bytes);
        }

        Ok(Self {
            encoded_vectors: storage_builder.build(),
            metadata: Metadata {
                vector_parameters: vector_parameters.clone(),
            },
        })
    }

    fn _encode_vector(vector: &[f32]) -> EncodedBinVector {
        let mut encoded_vector = vec![0; Self::get_storage_size(vector.len())];

        for (i, &v) in vector.iter().enumerate() {
            // flag is true if the value is positive
            // It's expected that the vector value is in range [-1; 1]
            if v > 0.0 {
                encoded_vector[i / BITS_STORE_TYPE_SIZE] |= 1 << (i % BITS_STORE_TYPE_SIZE);
            }
        }

        EncodedBinVector { encoded_vector }
    }

    /// Xor vectors and return the number of bits set to 1
    ///
    /// Assume that `v1` and `v2` are aligned to `BITS_STORE_TYPE_SIZE` with both with zeros
    /// So it does not affect the resulting number of bits set to 1
    fn xor_product(v1: &[BitsStoreType], v2: &[BitsStoreType]) -> usize {
        debug_assert!(v1.len() == v2.len());

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if is_x86_feature_detected!("sse4.1") {
            unsafe {
                return impl_xor_popcnt_sse(
                    v1.as_ptr() as *const u64,
                    v2.as_ptr() as *const u64,
                    v1.len() as u32,
                ) as usize;
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                return impl_xor_popcnt_neon(
                    v1.as_ptr() as *const u64,
                    v2.as_ptr() as *const u64,
                    v1.len() as u32,
                ) as usize;
            }
        }

        let mut result = 0;
        for (b1, b2) in v1.iter().zip(v2.iter()) {
            result += (b1 ^ b2).count_ones() as usize;
        }
        result
    }

    /// Estimates how many `StorageType` elements are needed to store `size` bits
    fn get_storage_size(size: usize) -> usize {
        let mut result = size / BITS_STORE_TYPE_SIZE;
        if size % BITS_STORE_TYPE_SIZE != 0 {
            result += 1;
        }
        result
    }

    pub fn get_quantized_vector_size_from_params(vector_parameters: &VectorParameters) -> usize {
        Self::get_storage_size(vector_parameters.dim) * std::mem::size_of::<BitsStoreType>()
    }

    fn get_quantized_vector_size(&self) -> usize {
        Self::get_quantized_vector_size_from_params(&self.metadata.vector_parameters)
    }

    fn calculate_metric(&self, v1: &[BitsStoreType], v2: &[BitsStoreType]) -> f32 {
        // Dot product in a range [-1; 1] is approximated by NXOR in a range [0; 1]
        // L1 distance in range [-1; 1] (alpha=2) is approximated by alpha*XOR in a range [0; 1]
        // L2 distance in range [-1; 1] (alpha=2) is approximated by alpha*sqrt(XOR) in a range [0; 1]
        // For example:

        // |  A   |  B   | Dot product | L1 | L2 |
        // | -0.5 | -0.5 |  0.25       | 0  | 0  |
        // | -0.5 |  0.5 | -0.25       | 1  | 1  |
        // |  0.5 | -0.5 | -0.25       | 1  | 1  |
        // |  0.5 |  0.5 |  0.25       | 0  | 0  |

        // | A | B | NXOR | XOR
        // | 0 | 0 | 1    | 0
        // | 0 | 1 | 0    | 1
        // | 1 | 0 | 0    | 1
        // | 1 | 1 | 1    | 0

        let xor_product = Self::xor_product(v1, v2) as f32;

        let dim = self.metadata.vector_parameters.dim as f32;
        let zeros_count = dim - xor_product;

        match (
            self.metadata.vector_parameters.distance_type,
            self.metadata.vector_parameters.invert,
        ) {
            // So if `invert` is true we return XOR, otherwise we return (dim - XOR)
            (DistanceType::Dot, true) => xor_product - zeros_count,
            (DistanceType::Dot, false) => zeros_count - xor_product,
            // This also results in exact ordering as L1 and L2 but reversed.
            (DistanceType::L1 | DistanceType::L2, true) => zeros_count - xor_product,
            (DistanceType::L1 | DistanceType::L2, false) => xor_product - zeros_count,
        }
    }
}

impl<TStorage: EncodedStorage> EncodedVectors<EncodedBinVector> for EncodedVectorsBin<TStorage> {
    fn save(&self, data_path: &Path, meta_path: &Path) -> std::io::Result<()> {
        let metadata_bytes = serde_json::to_vec(&self.metadata)?;
        meta_path.parent().map(std::fs::create_dir_all);
        std::fs::write(meta_path, metadata_bytes)?;

        data_path.parent().map(std::fs::create_dir_all);
        self.encoded_vectors.save_to_file(data_path)?;
        Ok(())
    }

    fn load(
        data_path: &Path,
        meta_path: &Path,
        vector_parameters: &VectorParameters,
    ) -> std::io::Result<Self> {
        let contents = std::fs::read_to_string(meta_path)?;
        let metadata: Metadata = serde_json::from_str(&contents)?;
        let quantized_vector_size = Self::get_quantized_vector_size_from_params(vector_parameters);
        let encoded_vectors =
            TStorage::from_file(data_path, quantized_vector_size, vector_parameters.count)?;
        let result = Self {
            metadata,
            encoded_vectors,
        };
        Ok(result)
    }

    fn encode_query(&self, query: &[f32]) -> EncodedBinVector {
        debug_assert!(query.len() == self.metadata.vector_parameters.dim);
        Self::_encode_vector(query)
    }

    fn score_point(&self, query: &EncodedBinVector, i: u32) -> f32 {
        let vector_data_1 = self
            .encoded_vectors
            .get_vector_data(i as _, self.get_quantized_vector_size());
        let vector_data_usize_1 = transmute_from_u8_to_slice(vector_data_1);

        self.calculate_metric(vector_data_usize_1, &query.encoded_vector)
    }

    fn score_internal(&self, i: u32, j: u32) -> f32 {
        let vector_data_1 = self
            .encoded_vectors
            .get_vector_data(i as _, self.get_quantized_vector_size());
        let vector_data_2 = self
            .encoded_vectors
            .get_vector_data(j as _, self.get_quantized_vector_size());

        let vector_data_usize_1 = transmute_from_u8_to_slice(vector_data_1);
        let vector_data_usize_2 = transmute_from_u8_to_slice(vector_data_2);

        self.calculate_metric(vector_data_usize_1, vector_data_usize_2)
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
extern "C" {
    fn impl_xor_popcnt_neon(query_ptr: *const u64, vector_ptr: *const u64, count: u32) -> u32;
}
