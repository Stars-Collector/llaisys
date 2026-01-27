#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    ASSERT(in->isContiguous() && weight->isContiguous() && out->isContiguous(), "rms_norm: all tensors must be contiguous.");

    // in: [batch_size, hidden_size]
    // weight: [hidden_size]
    // out: [batch_size, hidden_size]
    ASSERT(in->ndim() == 2, "rms_norm: in must be 2D tensor");
    ASSERT(weight->ndim() == 1, "rms_norm: weight must be 1D tensor");
    ASSERT(out->ndim() == 2, "rms_norm: out must be 2D tensor");

    size_t batch_size = in->shape()[0];
    size_t hidden_size = in->shape()[1];

    ASSERT(weight->shape()[0] == hidden_size, "rms_norm: weight shape must match hidden_size");
    ASSERT(out->shape()[0] == batch_size && out->shape()[1] == hidden_size, 
           "rms_norm: out shape must match in shape");

    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), 
                           out->dtype(), batch_size, hidden_size, eps);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), 
                           out->dtype(), batch_size, hidden_size, eps);
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
