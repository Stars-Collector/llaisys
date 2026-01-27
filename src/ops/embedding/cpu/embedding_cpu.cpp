#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

template <typename IndexT, typename WeightT>
void embedding_(WeightT *out, const IndexT *index, const WeightT *weight, 
                size_t num_indices, size_t embedding_dim) {
    for (size_t i = 0; i < num_indices; i++) {
        size_t idx = static_cast<size_t>(index[i]);
        const WeightT *src = weight + idx * embedding_dim;
        WeightT *dst = out + i * embedding_dim;

        if constexpr (std::is_same_v<WeightT, llaisys::bf16_t> || std::is_same_v<WeightT, llaisys::fp16_t>) {
            for (size_t j = 0; j < embedding_dim; j++) {
                dst[j] = src[j];
            }
        } else {
            std::memcpy(dst, src, embedding_dim * sizeof(WeightT));
        }
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, 
               llaisysDataType_t index_type, llaisysDataType_t weight_type, 
               size_t num_indices, size_t embedding_dim) {
    // Support int32 and int64 for index type
    if (index_type != LLAISYS_DTYPE_I32 && index_type != LLAISYS_DTYPE_I64) {
        EXCEPTION_UNSUPPORTED_DATATYPE(index_type);
    }

    // Support float32, bf16, fp16 for weight type
    if (weight_type != LLAISYS_DTYPE_F32 && weight_type != LLAISYS_DTYPE_BF16 && weight_type != LLAISYS_DTYPE_F16) {
        EXCEPTION_UNSUPPORTED_DATATYPE(weight_type);
    }

    if (index_type == LLAISYS_DTYPE_I32) {
        if (weight_type == LLAISYS_DTYPE_F32) {
            return embedding_(reinterpret_cast<float *>(out), 
                            reinterpret_cast<const int32_t *>(index), 
                            reinterpret_cast<const float *>(weight), 
                            num_indices, embedding_dim);
        } else if (weight_type == LLAISYS_DTYPE_BF16) {
            return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), 
                            reinterpret_cast<const int32_t *>(index), 
                            reinterpret_cast<const llaisys::bf16_t *>(weight), 
                            num_indices, embedding_dim);
        } else if (weight_type == LLAISYS_DTYPE_F16) {
            return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), 
                            reinterpret_cast<const int32_t *>(index), 
                            reinterpret_cast<const llaisys::fp16_t *>(weight), 
                            num_indices, embedding_dim);
        }
    } else if (index_type == LLAISYS_DTYPE_I64) {
        if (weight_type == LLAISYS_DTYPE_F32) {
            return embedding_(reinterpret_cast<float *>(out), 
                            reinterpret_cast<const int64_t *>(index), 
                            reinterpret_cast<const float *>(weight), 
                            num_indices, embedding_dim);
        } else if (weight_type == LLAISYS_DTYPE_BF16) {
            return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), 
                            reinterpret_cast<const int64_t *>(index), 
                            reinterpret_cast<const llaisys::bf16_t *>(weight), 
                            num_indices, embedding_dim);
        } else if (weight_type == LLAISYS_DTYPE_F16) {
            return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), 
                            reinterpret_cast<const int64_t *>(index), 
                            reinterpret_cast<const llaisys::fp16_t *>(weight), 
                            num_indices, embedding_dim);
        }
    }
}
} // namespace llaisys::ops::cpu
