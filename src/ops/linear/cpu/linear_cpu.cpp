#include "linear_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
            size_t batch_size, size_t in_features, size_t out_features) {
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t o = 0; o < out_features; o++) {
            // [Fix 1] 始终使用 float (fp32) 作为累加器，避免 fp16/bf16 的精度溢出
            float sum = 0.0f;
            
            for (size_t i = 0; i < in_features; i++) {
                // 计算正确的扁平化索引
                size_t in_idx = b * in_features + i;
                // [Fix 2] 权重索引修正：W 是 [out_features, in_features]
                // 应该是第 o 行，第 i 列 -> o * in_features + i
                size_t w_idx = o * in_features + i; 

                float in_val, weight_val;
                
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    in_val = llaisys::utils::cast<float>(in[in_idx]);
                    weight_val = llaisys::utils::cast<float>(weight[w_idx]);
                } else {
                    in_val = static_cast<float>(in[in_idx]);
                    weight_val = static_cast<float>(weight[w_idx]);
                }
                
                // 在 float 精度下累加
                sum += in_val * weight_val;
            }

            if (bias != nullptr) {
                float bias_val;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    bias_val = llaisys::utils::cast<float>(bias[o]);
                } else {
                    bias_val = static_cast<float>(bias[o]);
                }
                sum += bias_val;
            }

            // [Fix 1] 仅在写入内存时转回 T
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[b * out_features + o] = llaisys::utils::cast<T>(sum);
            } else {
                out[b * out_features + o] = static_cast<T>(sum);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t dtype, size_t batch_size, size_t in_features, size_t out_features) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), 
                      reinterpret_cast<const float *>(in), 
                      reinterpret_cast<const float *>(weight), 
                      reinterpret_cast<const float *>(bias),
                      batch_size, in_features, out_features);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), 
                      reinterpret_cast<const llaisys::bf16_t *>(in), 
                      reinterpret_cast<const llaisys::bf16_t *>(weight), 
                      reinterpret_cast<const llaisys::bf16_t *>(bias),
                      batch_size, in_features, out_features);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), 
                      reinterpret_cast<const llaisys::fp16_t *>(in), 
                      reinterpret_cast<const llaisys::fp16_t *>(weight), 
                      reinterpret_cast<const llaisys::fp16_t *>(bias),
                      batch_size, in_features, out_features);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu