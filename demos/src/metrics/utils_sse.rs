use std::arch::x86_64::*;

#[allow(clippy::missing_safety_doc)]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn hsum128_ps_sse(x: __m128) -> f32 {
    let x64: __m128 = _mm_add_ps(x, _mm_movehl_ps(x, x));
    let x32: __m128 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    _mm_cvtss_f32(x32)
}

#[target_feature(enable = "sse4.1")]
#[allow(clippy::missing_safety_doc, dead_code)]
pub unsafe fn dot_sse(v1: &[f32], v2: &[f32]) -> f32 {
    let n = v1.len();
    let m = n - (n % 16);
    let mut ptr1: *const f32 = v1.as_ptr();
    let mut ptr2: *const f32 = v2.as_ptr();
    let mut sum128_1: __m128 = _mm_setzero_ps();
    let mut sum128_2: __m128 = _mm_setzero_ps();
    let mut sum128_3: __m128 = _mm_setzero_ps();
    let mut sum128_4: __m128 = _mm_setzero_ps();

    let mut i: usize = 0;
    while i < m {
        sum128_1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ptr1), _mm_loadu_ps(ptr2)), sum128_1);

        sum128_2 = _mm_add_ps(
            _mm_mul_ps(_mm_loadu_ps(ptr1.add(4)), _mm_loadu_ps(ptr2.add(4))),
            sum128_2,
        );

        sum128_3 = _mm_add_ps(
            _mm_mul_ps(_mm_loadu_ps(ptr1.add(8)), _mm_loadu_ps(ptr2.add(8))),
            sum128_3,
        );

        sum128_4 = _mm_add_ps(
            _mm_mul_ps(_mm_loadu_ps(ptr1.add(12)), _mm_loadu_ps(ptr2.add(12))),
            sum128_4,
        );

        ptr1 = ptr1.add(16);
        ptr2 = ptr2.add(16);
        i += 16;
    }

    let mut result = hsum128_ps_sse(sum128_1)
        + hsum128_ps_sse(sum128_2)
        + hsum128_ps_sse(sum128_3)
        + hsum128_ps_sse(sum128_4);
    for i in 0..n - m {
        result += (*ptr1.add(i)) * (*ptr2.add(i));
    }
    result
}
