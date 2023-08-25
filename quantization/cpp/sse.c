#include <stdint.h>
#include <immintrin.h>

#include "export_macro.h"

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
    uint32_t dim
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
    return mul_scalar;
}

EXPORT uint64_t impl_xor_popcnt_sse(
    const uint64_t* query_ptr,
    const uint64_t* vector_ptr,
    uint32_t count
) {
    const int64_t* v_ptr = (const int64_t*)vector_ptr;
    const int64_t* q_ptr = (const int64_t*)query_ptr;
    int64_t result = 0;
    for (uint32_t _i = 0; _i < count; _i++) {
        uint64_t x = (*v_ptr) ^ (*q_ptr);
        result += _mm_popcnt_u64(x);
        v_ptr++;
        q_ptr++;
    }
    return (uint32_t)result;
}
