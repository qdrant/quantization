#include <arm_neon.h>

#define EXPORT __attribute__((visibility("default")))

EXPORT float impl_score_dot_neon(
    const uint8_t* query_ptr,
    const uint8_t* vector_ptr,
    uint32_t dim,
    float alpha,
    float offset
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
    float mul = vaddvq_u32(vaddq_u32(mul1, mul2));
    return alpha * mul + offset;
}

EXPORT void impl_score_pair_dot_neon(
    const uint8_t* query_ptr,
    const uint8_t* vector1_ptr,
    const uint8_t* vector2_ptr,
    uint32_t dim,
    float alpha,
    float offset1,
    float offset2,
    float* result
) {
    uint32x4_t mul1 = vdupq_n_u32(0);
    uint32x4_t mul2 = vdupq_n_u32(0);
    for (uint32_t _i = 0; _i < dim / 16; _i++) {
        uint8x16_t q = vld1q_u8(query_ptr);
        uint8x16_t v1 = vld1q_u8(vector1_ptr);
        uint8x16_t v2 = vld1q_u8(vector2_ptr);
        query_ptr += 16;
        vector1_ptr += 16;
        vector2_ptr += 16;
        uint8x8_t q_low = vget_low_u8(q);
        uint8x8_t q_high = vget_high_u8(q);
        uint16x8_t mul1_low = vmull_u8(q_low, vget_low_u8(v1));
        uint16x8_t mul1_high = vmull_u8(q_high, vget_high_u8(v1));
        uint16x8_t mul2_low = vmull_u8(q_low, vget_low_u8(v2));
        uint16x8_t mul2_high = vmull_u8(q_high, vget_high_u8(v2));
        mul1 = vpadalq_u16(mul1, mul1_low);
        mul1 = vpadalq_u16(mul1, mul1_high);
        mul2 = vpadalq_u16(mul2, mul2_low);
        mul2 = vpadalq_u16(mul2, mul2_high);
    }
    float mul1_scalar = vaddvq_u32(mul1);
    float mul2_scalar = vaddvq_u32(mul2);
    result[0] = alpha * mul1_scalar + offset1;
    result[1] = alpha * mul2_scalar + offset2;
}
