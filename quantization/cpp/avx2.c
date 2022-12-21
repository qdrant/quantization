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
        mul1 = _mm256_add_epi32(mul1, _mm256_and_si256(s, mask_epu32));
        mul1 = _mm256_add_epi32(mul1, _mm256_srli_epi32(s, 16));
        
        //__m256i s_low = _mm256_cvtepi16_epi32(s);
        //__m256i s_high = _mm256_cvtepi16_epi32(_mm256_srli_si256(s, 16));
        //mul1 = _mm256_add_epi32(mul1, s_low);
        //mul2 = _mm256_add_epi32(mul2, s_high);
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
    return alpha * mul_scalar + offset;
}

EXPORT void impl_score_pair_dot_avx(
    const uint8_t* query_ptr,
    const uint8_t* vector1_ptr,
    const uint8_t* vector2_ptr,
    uint32_t dim,
    float alpha,
    float offset1,
    float offset2,
    float* result
) {
    const __m256i* v1_ptr = (const __m256i*)vector1_ptr;
    const __m256i* v2_ptr = (const __m256i*)vector2_ptr;
    const __m256i* q_ptr = (const __m256i*)query_ptr;

    __m256i mul1 = _mm256_setzero_si256();
    __m256i mul2 = _mm256_setzero_si256();
    __m256i mask_epu32 = _mm256_set1_epi32(0xFFFF);
    for (uint32_t _i = 0; _i < dim / 32; _i++) {
        __m256i v1 = _mm256_loadu_si256(v1_ptr);
        __m256i v2 = _mm256_loadu_si256(v2_ptr);
        __m256i q = _mm256_loadu_si256(q_ptr);
        v1_ptr++;
        v2_ptr++;
        q_ptr++;

        __m256i s1 = _mm256_maddubs_epi16(v1, q);
        __m256i s2 = _mm256_maddubs_epi16(v2, q);
        mul1 = _mm256_add_epi32(mul1, _mm256_and_si256(s1, mask_epu32));
        mul1 = _mm256_add_epi32(mul1, _mm256_srli_epi32(s1, 16));
        mul2 = _mm256_add_epi32(mul2, _mm256_and_si256(s2, mask_epu32));
        mul2 = _mm256_add_epi32(mul2, _mm256_srli_epi32(s2, 16));

        //__m256i s_low = _mm256_cvtepi16_epi32(s);
        //__m256i s_high = _mm256_cvtepi16_epi32(_mm256_srli_si256(s, 16));
        //mul1 = _mm256_add_epi32(mul1, s_low);
        //mul2 = _mm256_add_epi32(mul2, s_high);
    }
    if (dim % 32 != 0) {
        __m128i v1_short = _mm_loadu_si128((const __m128i*)v1_ptr);
        __m128i v2_short = _mm_loadu_si128((const __m128i*)v2_ptr);
        __m128i q_short = _mm_loadu_si128((const __m128i*)q_ptr);

        __m256i v1 = _mm256_cvtepu8_epi16(v1_short);
        __m256i v2 = _mm256_cvtepu8_epi16(v2_short);
        __m256i q1 = _mm256_cvtepu8_epi16(q_short);

        __m256i s1 = _mm256_mullo_epi16(v1, q1);
        __m256i s2 = _mm256_mullo_epi16(v2, q1);
        mul1 = _mm256_add_epi32(mul1, _mm256_and_si256(s1, mask_epu32));
        mul1 = _mm256_add_epi32(mul1, _mm256_srli_epi32(s1, 16));
        mul2 = _mm256_add_epi32(mul2, _mm256_and_si256(s2, mask_epu32));
        mul2 = _mm256_add_epi32(mul2, _mm256_srli_epi32(s2, 16));
    }
    __m256 mul1_ps = _mm256_cvtepi32_ps(mul1);
    __m256 mul2_ps = _mm256_cvtepi32_ps(mul2);
    HSUM256_PS(mul1_ps, mul1_scalar);
    HSUM256_PS(mul2_ps, mul2_scalar);
    result[0] = alpha * mul1_scalar + offset1;
    result[1] = alpha * mul2_scalar + offset2;
}

#define HSUM128_PS(X, R) \
    float R = 0.0f; \
    { \
    __m128 x64 = _mm_add_ps(X, _mm_movehl_ps(X, X)); \
    __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55)); \
    R = _mm_cvtss_f32(x32); \
    }

EXPORT float impl_score_dot_sse(
    const uint8_t* query_ptr,
    const uint8_t* vector_ptr,
    uint32_t dim,
    float alpha,
    float offset
) {
    const __m128i* v_ptr = (const __m128i*)vector_ptr;
    const __m128i* q_ptr = (const __m128i*)query_ptr;

    __m128i mul = _mm_setzero_si128();
    for (uint32_t _i = 0; _i < dim / 16; _i++) {
        __m128i v = _mm_loadu_si128(v_ptr);
        __m128i q = _mm_loadu_si128(q_ptr);
        v_ptr++;
        q_ptr++;

        __m128i s = _mm_maddubs_epi16(v, q);
        __m128i s_low = _mm_cvtepi16_epi32(s);
        __m128i s_high = _mm_cvtepi16_epi32(_mm_srli_si128(s, 8));
        mul = _mm_add_epi32(mul, s_low);
        mul = _mm_add_epi32(mul, s_high);
    }
    __m128 mul_ps = _mm_cvtepi32_ps(mul);
    HSUM128_PS(mul_ps, mul_scalar);
    return alpha * mul_scalar + offset;
}

EXPORT void impl_score_pair_dot_sse(
    const uint8_t* query_ptr,
    const uint8_t* vector1_ptr,
    const uint8_t* vector2_ptr,
    uint32_t dim,
    float alpha,
    float offset1,
    float offset2,
    float* result
) {
    const __m128i* v1_ptr = (const __m128i*)vector1_ptr;
    const __m128i* v2_ptr = (const __m128i*)vector2_ptr;
    const __m128i* q_ptr = (const __m128i*)query_ptr;

    __m128i mul1 = _mm_setzero_si128();
    __m128i mul2 = _mm_setzero_si128();
    for (uint32_t _i = 0; _i < dim / 16; _i++) {
        __m128i v1 = _mm_loadu_si128(v1_ptr);
        __m128i v2 = _mm_loadu_si128(v2_ptr);
        __m128i q = _mm_loadu_si128(q_ptr);
        v1_ptr++;
        v2_ptr++;
        q_ptr++;

        __m128i s1 = _mm_maddubs_epi16(v1, q);
        __m128i s2 = _mm_maddubs_epi16(v2, q);

        __m128i s1_low = _mm_cvtepi16_epi32(s1);
        __m128i s1_high = _mm_cvtepi16_epi32(_mm_srli_si128(s1, 8));
        mul1 = _mm_add_epi32(mul1, s1_low);
        mul1 = _mm_add_epi32(mul1, s1_high);

        __m128i s2_low = _mm_cvtepi16_epi32(s2);
        __m128i s2_high = _mm_cvtepi16_epi32(_mm_srli_si128(s2, 8));
        mul2 = _mm_add_epi32(mul2, s2_low);
        mul2 = _mm_add_epi32(mul2, s2_high);
    }
    __m128 mul1_ps = _mm_cvtepi32_ps(mul1);
    __m128 mul2_ps = _mm_cvtepi32_ps(mul2);
    HSUM128_PS(mul1_ps, mul1_scalar);
    HSUM128_PS(mul2_ps, mul2_scalar);
    result[0] = alpha * mul1_scalar + offset1;
    result[1] = alpha * mul2_scalar + offset2;
}
