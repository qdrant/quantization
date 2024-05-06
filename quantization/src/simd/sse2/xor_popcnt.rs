use std::arch::x86_64::*;


#[target_feature(enable = "popcnt")]
#[allow(clippy::missing_safety_doc)]
#[inline]
pub unsafe fn impl_xor_popcnt_sse(query_ptr: *const u64, vector_ptr: *const u64, count: u32) -> u32 {
    let mut v_ptr = vector_ptr as *const i64;
    let mut q_ptr = query_ptr as *const i64;
    let mut result = 0;
    for _ in 0..2 * count {
        let x = (*v_ptr) ^ (*q_ptr);
        result += _popcnt64(x);
        v_ptr = v_ptr.add(1);
        q_ptr = q_ptr.add(1);
    }
    result as u32
}
