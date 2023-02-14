#include <stdint.h>
#include <immintrin.h>

#include "export_macro.h"

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
    uint32_t dim
) {
    const __m256i* v_ptr = (const __m256i*)vector_ptr;
    const __m256i* q_ptr = (const __m256i*)query_ptr;

    __m256i mul1 = _mm256_setzero_si256();
    __m256i mask_epu32 = _mm256_set1_epi32(0xFFFF);
    for (uint32_t _i = 0; _i < dim / 32; _i++) {
        __m256i v = _mm256_loadu_si256(v_ptr);
        __m256i q = _mm256_loadu_si256(q_ptr);
        v_ptr++;
        q_ptr++;

        __m256i s = _mm256_maddubs_epi16(v, q);
        __m256i s_low = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(s));
        __m256i s_high = _mm256_cvtepi16_epi32(_mm256_extractf128_si256(s, 1));
        mul1 = _mm256_add_epi32(mul1, s_low);
        mul1 = _mm256_add_epi32(mul1, s_high);
    }
    if (dim % 32 != 0) {
        __m128i v_short = _mm_loadu_si128((const __m128i*)v_ptr);
        __m128i q_short = _mm_loadu_si128((const __m128i*)q_ptr);

        __m256i v1 = _mm256_cvtepu8_epi16(v_short);
        __m256i q1 = _mm256_cvtepu8_epi16(q_short);

        __m256i s = _mm256_mullo_epi16(v1, q1);
        mul1 = _mm256_add_epi32(mul1, _mm256_and_si256(s, mask_epu32));
        mul1 = _mm256_add_epi32(mul1, _mm256_srli_epi32(s, 16));
    }
    __m256 mul_ps = _mm256_cvtepi32_ps(mul1);
    HSUM256_PS(mul_ps, mul_scalar);
    return mul_scalar;
}
