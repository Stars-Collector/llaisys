#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    ASSERT(q->isContiguous() && k->isContiguous() && v->isContiguous() && attn_val->isContiguous(), 
           "self_attention: all tensors must be contiguous.");

    // q: [qlen, num_heads, head_dim]
    // k: [kvlen, kv_num_heads, head_dim]
    // v: [kvlen, kv_num_heads, v_head_dim]
    // attn_val: [qlen, num_heads, v_head_dim]
    ASSERT(q->ndim() == 3, "self_attention: q must be 3D tensor");
    ASSERT(k->ndim() == 3, "self_attention: k must be 3D tensor");
    ASSERT(v->ndim() == 3, "self_attention: v must be 3D tensor");
    ASSERT(attn_val->ndim() == 3, "self_attention: attn_val must be 3D tensor");

    size_t qlen = q->shape()[0];
    size_t kvlen = k->shape()[0];
    size_t num_heads = q->shape()[1];
    size_t kv_num_heads = k->shape()[1];
    size_t head_dim = q->shape()[2];       // d
    size_t v_head_dim = v->shape()[2];     // dv

    ASSERT(k->shape()[2] == head_dim, 
           "self_attention: k head_dim (d) must match q head_dim");
    ASSERT(v->shape()[0] == kvlen && v->shape()[1] == kv_num_heads, 
           "self_attention: v length/heads must match k");
    
    // Output (attn_val) 应该匹配 V 的最后一维 (dv)
    ASSERT(attn_val->shape()[0] == qlen && attn_val->shape()[1] == num_heads && 
           attn_val->shape()[2] == v_head_dim, 
           "self_attention: attn_val shape must match q shape (len/heads) and v shape (dim)");

    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    // always support cpu calculation
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                 attn_val->dtype(), qlen, kvlen, num_heads, kv_num_heads, 
                                 head_dim, v_head_dim, scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                 attn_val->dtype(), qlen, kvlen, num_heads, kv_num_heads, 
                                 head_dim, v_head_dim, scale);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops