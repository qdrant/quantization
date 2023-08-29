#include <arm_neon.h>

#include "export_macro.h"

EXPORT float impl_score_dot_neon(
    const uint8_t* query_ptr,
    const uint8_t* vector_ptr,
    uint32_t dim
) {
    uint32x4_t mul1 = vdupq_n_u32(0);
    uint32x4_t mul2 = vdupq_n_u32(0);
    for (uint32_t _i = 0; _i < dim / 16; _i++) {
        uint8x16_t q = vld1q_u8(query_ptr);
        uint8x16_t v = vld1q_u8(vector_ptr);
        query_ptr += 16;
        vector_ptr += 16;
        uint16x8_t mul_low = vmull_u8(vget_low_u8(q), vget_low_u8(v));
        uint16x8_t mul_high = vmull_u8(vget_high_u8(q), vget_high_u8(v));
        mul1 = vpadalq_u16(mul1, mul_low);
        mul2 = vpadalq_u16(mul2, mul_high);
    }
    return (float)vaddvq_u32(vaddq_u32(mul1, mul2));
}

EXPORT uint64_t impl_xor_popcnt_neon(
    const uint64_t* query_ptr,
    const uint64_t* vector_ptr,
    uint32_t count
) {
    const uint8_t* v_ptr = (const uint8_t*)vector_ptr;
    const uint8_t* q_ptr = (const uint8_t*)query_ptr;
    uint32x4_t result = vdupq_n_u32(0);
    for (uint32_t _i = 0; _i < count / 2; _i++) {
        uint8x16_t v1 = vld1q_u8(v_ptr);
        uint8x16_t q1 = vld1q_u8(q_ptr);
        uint8x16_t v2 = vld1q_u8(v_ptr + 16);
        uint8x16_t q2 = vld1q_u8(q_ptr + 16);

        uint8x16_t x1 = veorq_u8(q1, v1);
        uint8x16_t x2 = veorq_u8(q2, v2);
        uint8x16_t popcnt1 = vcntq_u8(x1);
        uint8x16_t popcnt2 = vcntq_u8(x2);
        uint8x16_t popcnt = vaddq_u8(popcnt1, popcnt2);
        uint8x8_t popcnt_low = vget_low_u8(popcnt);
        uint8x8_t popcnt_high = vget_high_u8(popcnt);
        uint16x8_t sum = vaddl_u8(popcnt_low, popcnt_high);
        result = vpadalq_u16(result, sum);

        v_ptr += 32;
        q_ptr += 32;
    }

    if (count % 2 == 1) {
        uint8x16_t v = vld1q_u8(v_ptr);
        uint8x16_t q = vld1q_u8(q_ptr);
        uint8x16_t x = veorq_u8(q, v);
        uint8x16_t popcnt = vcntq_u8(x);
        uint8x8_t popcnt_low = vget_low_u8(popcnt);
        uint8x8_t popcnt_high = vget_high_u8(popcnt);
        uint16x8_t sum = vaddl_u8(popcnt_low, popcnt_high);
        result = vpadalq_u16(result, sum);
    }

    return (uint64_t)vaddvq_u32(result);
}
