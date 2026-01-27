#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    ASSERT(in->isContiguous() && pos_ids->isContiguous() && out->isContiguous(), "rope: all tensors must be contiguous.");

    // in: [seq_len, n_heads, head_dim]
    // pos_ids: [seq_len]
    // out: [seq_len, n_heads, head_dim]
    ASSERT(in->ndim() == 3, "rope: in must be 3D tensor");
    ASSERT(pos_ids->ndim() == 1, "rope: pos_ids must be 1D tensor");
    ASSERT(out->ndim() == 3, "rope: out must be 3D tensor");

    size_t seq_len = in->shape()[0];
    size_t n_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];

    ASSERT(pos_ids->shape()[0] == seq_len, "rope: pos_ids length must match seq_len");
    ASSERT(out->shape()[0] == seq_len && out->shape()[1] == n_heads && out->shape()[2] == head_dim, 
           "rope: out shape must match in shape");

    ASSERT(head_dim % 2 == 0, "rope: head_dim must be even");
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "rope: pos_ids must be int64 type");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), 
                        out->dtype(), seq_len, n_heads, head_dim, theta);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), 
                        out->dtype(), seq_len, n_heads, head_dim, theta);
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
