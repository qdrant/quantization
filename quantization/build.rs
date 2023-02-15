fn main() {
    let mut builder = cc::Build::new();

    #[cfg(target_arch = "x86_64")]
    {
        builder.file("cpp/sse.c");
        builder.file("cpp/avx2.c");
        if builder.get_compiler().is_like_msvc() {
            builder.flag("/arch:AVX");
            builder.flag("/arch:AVX2");
            builder.flag("/arch:SSE");
            builder.flag("/arch:SSE2");
        } else {
            builder.flag("-march=haswell");
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        builder.file("cpp/neon.c");
    }

    builder.compile("simd_utils");
}
