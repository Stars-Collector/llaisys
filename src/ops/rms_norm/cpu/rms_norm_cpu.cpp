#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight,
               size_t batch_size, size_t hidden_size, float eps) {
    for (size_t b = 0; b < batch_size; b++) {
        const T *batch_in = in + b * hidden_size;
        T *batch_out = out + b * hidden_size;

        // Compute RMS
        float sum_sq = 0.0f;
        for (size_t i = 0; i < hidden_size; i++) {
            float val = llaisys::utils::cast<float>(batch_in[i]);
            sum_sq += val * val;
        }
        float rms = std::sqrt(sum_sq / hidden_size + eps);

        // Apply normalization and weight
        for (size_t i = 0; i < hidden_size; i++) {
            float val = llaisys::utils::cast<float>(batch_in[i]);
            float w = llaisys::utils::cast<float>(weight[i]);
            batch_out[i] = llaisys::utils::cast<T>((val / rms) * w);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t dtype, size_t batch_size, size_t hidden_size, float eps) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), 
                        reinterpret_cast<const float *>(in), 
                        reinterpret_cast<const float *>(weight),
                        batch_size, hidden_size, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), 
                        reinterpret_cast<const llaisys::bf16_t *>(in), 
                        reinterpret_cast<const llaisys::bf16_t *>(weight),
                        batch_size, hidden_size, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), 
                        reinterpret_cast<const llaisys::fp16_t *>(in), 
                        reinterpret_cast<const llaisys::fp16_t *>(weight),
                        batch_size, hidden_size, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
