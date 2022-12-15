#include <stdint.h>
#include <mmintrin.h>  // MMX
#include <xmmintrin.h> // SSE
#include <emmintrin.h> // SSE2
#include <pmmintrin.h> // SSE3
#include <tmmintrin.h> // SSSE3
#include <smmintrin.h> // SSE4.1
#include <nmmintrin.h> // SSE4.2
#include <ammintrin.h> // SSE4A
#include <wmmintrin.h> // AES
#include <immintrin.h> // AVX, AVX2, FMA

#define EXPORT __attribute__((visibility("default")))

#define HSUM256_PS(X, R) \
    float R = 0.0f; \
    { \
    __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(X, 1), _mm256_castps256_ps128(X)); \
    __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128)); \
    __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55)); \
    R = _mm_cvtss_f32(x32); \
    }

EXPORT float impl_score_dot_avx(
    const uint8_t* query_ptr,
    const uint8_t* vector_ptr,
    uint32_t dim,
    float alpha,
    float offset
) {
    __m128i* v_ptr = (__m128i*)vector_ptr;
    __m128i* q_ptr = (__m128i*)query_ptr;
    __m256i sum1 = _mm256_setzero_si256();
    __m256i sum2 = _mm256_setzero_si256();
    __m256i sum3 = _mm256_setzero_si256();
    __m256i sum4 = _mm256_setzero_si256();

    __m256i mul1 = _mm256_setzero_si256();
    __m256i mul2 = _mm256_setzero_si256();
    __m256i mul3 = _mm256_setzero_si256();
    __m256i mul4 = _mm256_setzero_si256();
    __m256i mask_epu32 = _mm256_set1_epi32(0xFFFF);
    for (uint32_t _i = 0; _i < dim / 32; _i++) {
        __m128i v1_src = _mm_loadu_si128(v_ptr);
        v_ptr++;
        __m128i v2_src = _mm_loadu_si128(v_ptr);
        v_ptr++;

        __m128i q1_src = _mm_loadu_si128(q_ptr);
        q_ptr++;
        __m128i q2_src = _mm_loadu_si128(q_ptr);
        q_ptr++;

        __m256i v1 = _mm256_cvtepu8_epi16(v1_src);
        __m256i q1 = _mm256_cvtepu8_epi16(q1_src);
        __m256i m1 = _mm256_mullo_epi16(v1, q1);
        __m256i s1 = _mm256_adds_epu16(v1, q1);

        sum1 = _mm256_add_epi32(sum1, _mm256_and_si256(s1, mask_epu32));
        sum2 = _mm256_add_epi32(sum2, _mm256_srli_epi32(s1, 16));

        mul1 = _mm256_add_epi32(mul1, _mm256_and_si256(m1, mask_epu32));
        mul2 = _mm256_add_epi32(mul2, _mm256_srli_epi32(m1, 16));

        __m256i v2 = _mm256_cvtepu8_epi16(v2_src);
        __m256i q2 = _mm256_cvtepu8_epi16(q2_src);
        __m256i m2 = _mm256_mullo_epi16(v2, q2);
        __m256i s2 = _mm256_adds_epu16(v2, q2);

        sum3 = _mm256_add_epi32(sum3, _mm256_and_si256(s2, mask_epu32));
        sum4 = _mm256_add_epi32(sum4, _mm256_srli_epi32(s2, 16));

        mul3 = _mm256_add_epi32(mul3, _mm256_and_si256(m2, mask_epu32));
        mul4 = _mm256_add_epi32(mul4, _mm256_srli_epi32(m2, 16));
    }
    __m256i mul = _mm256_add_epi32(_mm256_add_epi32(mul1, mul2), _mm256_add_epi32(mul3, mul4));
    __m256i sum = _mm256_add_epi32(_mm256_add_epi32(sum1, sum2), _mm256_add_epi32(sum3, sum4));
    __m256 mul_ps = _mm256_cvtepi32_ps(mul);
    __m256 sum_ps = _mm256_cvtepi32_ps(sum);
    HSUM256_PS(mul_ps, mul_scalar);
    HSUM256_PS(sum_ps, sum_scalar);
    return alpha * alpha * mul_scalar + alpha * offset * sum_scalar + offset * offset * dim;
}
