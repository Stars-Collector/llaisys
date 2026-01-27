#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(index->isContiguous() && weight->isContiguous() && out->isContiguous(), "embedding: all tensors must be contiguous.");

    // index should be 1D, out should be 2D [num_indices, embedding_dim]
    // weight should be 2D [vocab_size, embedding_dim]
    ASSERT(index->ndim() == 1, "embedding: index must be 1D tensor");
    ASSERT(weight->ndim() == 2, "embedding: weight must be 2D tensor");
    ASSERT(out->ndim() == 2, "embedding: out must be 2D tensor");

    size_t num_indices = index->numel();
    size_t embedding_dim = weight->shape()[1];

    ASSERT(out->shape()[0] == num_indices, "embedding: out shape[0] must match index numel");
    ASSERT(out->shape()[1] == embedding_dim, "embedding: out shape[1] must match weight shape[1]");

    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), 
                             index->dtype(), weight->dtype(), 
                             num_indices, embedding_dim);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), 
                             index->dtype(), weight->dtype(), 
                             num_indices, embedding_dim);
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
