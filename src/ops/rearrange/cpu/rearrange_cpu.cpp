#include "rearrange_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
void rearrange_(T *out, const T *in,
                const std::vector<size_t>& in_shape,
                const std::vector<ptrdiff_t>& in_strides,
                const std::vector<size_t>& out_shape,
                const std::vector<ptrdiff_t>& out_strides,
                size_t out_numel) {
    size_t ndim = static_cast<size_t>(in_shape.size());

    for (size_t out_idx = 0; out_idx < out_numel; out_idx++) {
        // Convert linear index to output coordinates
        std::vector<size_t> out_coords(ndim);
        size_t temp = out_idx;
        for (size_t d = ndim; d > 0; d--) {
            size_t idx = d - 1;
            out_coords[idx] = temp % out_shape[idx];
            temp /= out_shape[idx];
        }

        // Calculate input offset
        size_t in_offset = 0;
        for (size_t d = 0; d < ndim; d++) {
            in_offset += out_coords[d] * static_cast<size_t>(in_strides[d]);
        }

        // Copy element
        out[out_idx] = in[in_offset];
    }
}

namespace llaisys::ops::cpu {
void rearrange(std::byte *out, const std::byte *in,
               llaisysDataType_t dtype,
               const std::vector<size_t>& in_shape,
               const std::vector<ptrdiff_t>& in_strides,
               const std::vector<size_t>& out_shape,
               const std::vector<ptrdiff_t>& out_strides,
               size_t out_numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rearrange_(reinterpret_cast<float *>(out), 
                        reinterpret_cast<const float *>(in),
                        in_shape, in_strides, out_shape, out_strides, out_numel);
    case LLAISYS_DTYPE_BF16:
        return rearrange_(reinterpret_cast<llaisys::bf16_t *>(out), 
                        reinterpret_cast<const llaisys::bf16_t *>(in),
                        in_shape, in_strides, out_shape, out_strides, out_numel);
    case LLAISYS_DTYPE_F16:
        return rearrange_(reinterpret_cast<llaisys::fp16_t *>(out), 
                        reinterpret_cast<const llaisys::fp16_t *>(in),
                        in_shape, in_strides, out_shape, out_strides, out_numel);
    case LLAISYS_DTYPE_I32:
        return rearrange_(reinterpret_cast<int32_t *>(out), 
                        reinterpret_cast<const int32_t *>(in),
                        in_shape, in_strides, out_shape, out_strides, out_numel);
    case LLAISYS_DTYPE_I64:
        return rearrange_(reinterpret_cast<int64_t *>(out), 
                        reinterpret_cast<const int64_t *>(in),
                        in_shape, in_strides, out_shape, out_strides, out_numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
