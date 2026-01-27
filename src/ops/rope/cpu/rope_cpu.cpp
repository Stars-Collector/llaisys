#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids,
           size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    size_t half_dim = head_dim / 2;

    for (size_t s = 0; s < seq_len; s++) {
        int64_t pos = pos_ids[s];

        for (size_t h = 0; h < n_heads; h++) {
            size_t offset = (s * n_heads + h) * head_dim;

            for (size_t i = 0; i < half_dim; i++) {
                float freq = static_cast<float>(pos) / std::pow(theta, static_cast<float>(2 * i) / head_dim);
                float cos_val = std::cos(freq);
                float sin_val = std::sin(freq);

                float x = llaisys::utils::cast<float>(in[offset + i]);
                float y = llaisys::utils::cast<float>(in[offset + i + half_dim]);

                out[offset + i] = llaisys::utils::cast<T>(x * cos_val - y * sin_val);
                out[offset + i + half_dim] = llaisys::utils::cast<T>(x * sin_val + y * cos_val);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t dtype, size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), 
                   reinterpret_cast<const float *>(in), 
                   reinterpret_cast<const int64_t *>(pos_ids),
                   seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), 
                   reinterpret_cast<const llaisys::bf16_t *>(in), 
                   reinterpret_cast<const int64_t *>(pos_ids),
                   seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), 
                   reinterpret_cast<const llaisys::fp16_t *>(in), 
                   reinterpret_cast<const int64_t *>(pos_ids),
                   seq_len, n_heads, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
