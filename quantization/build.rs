fn main() {
    let mut builder = cc::Build::new();

    #[cfg(target_arch = "x86_64")]
    {
        builder.file("cpp/avx2.c");
        builder.flag("-march=haswell");
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        builder.file("cpp/neon.c");
    }

    builder.compile("simd_utils");
}
