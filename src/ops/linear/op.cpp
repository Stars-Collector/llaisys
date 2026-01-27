#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias != nullptr) {
        CHECK_SAME_DEVICE(out, bias);
    }

    ASSERT(in->isContiguous() && weight->isContiguous() && out->isContiguous(), "linear: all tensors must be contiguous.");
    if (bias != nullptr) {
        ASSERT(bias->isContiguous(), "linear: bias must be contiguous.");
    }

    // in: [batch_size, in_features]
    // weight: [out_features, in_features]
    // bias: [out_features] (optional)
    // out: [batch_size, out_features]
    ASSERT(in->ndim() == 2, "linear: in must be 2D tensor");
    ASSERT(weight->ndim() == 2, "linear: weight must be 2D tensor");
    ASSERT(out->ndim() == 2, "linear: out must be 2D tensor");

    size_t batch_size = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = out->shape()[1];

    ASSERT(weight->shape()[0] == out_features && weight->shape()[1] == in_features, 
           "linear: weight shape must be [out_features, in_features]");

    if (bias != nullptr) {
        ASSERT(bias->ndim() == 1 && bias->shape()[0] == out_features, 
               "linear: bias must be 1D tensor with shape [out_features]");
    }

    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if (bias != nullptr) {
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), 
                          bias != nullptr ? bias->data() : nullptr,
                          out->dtype(), batch_size, in_features, out_features);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), 
                          bias != nullptr ? bias->data() : nullptr,
                          out->dtype(), batch_size, in_features, out_features);
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
