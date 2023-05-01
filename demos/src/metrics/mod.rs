#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod utils_sse;

#[cfg(target_arch = "x86_64")]
pub mod utils_avx2;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub mod utils_neon;
