fn main() {
    cc::Build::new()
        .file("cpp/avx2.c")
        .flag("-march=haswell")
        .compile("simd_utils");
}
