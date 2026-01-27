#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                     size_t qlen, size_t kvlen, size_t num_heads, size_t kv_num_heads, 
                     size_t head_dim, size_t v_head_dim, float scale) {
    size_t q_stride = num_heads * head_dim;
    size_t k_stride = kv_num_heads * head_dim;
    
    size_t v_stride = kv_num_heads * v_head_dim;
    size_t out_stride = num_heads * v_head_dim;

    size_t offset = kvlen - qlen; 
    
    //计算每组包含多少个 Query Head
    size_t group_size = num_heads / kv_num_heads;

    for (size_t i = 0; i < qlen; i++) {
        size_t current_token_idx = offset + i; 

        for (size_t h = 0; h < num_heads; h++) {
            // 使用除法进行块状映射，而非取模
            size_t kv_h = h / group_size;

            std::vector<float> scores(kvlen);
            float max_score = -INFINITY;

            for (size_t j = 0; j < kvlen; j++) {
                if (j > current_token_idx) {
                    scores[j] = -INFINITY;
                    continue; 
                }

                float sum = 0.0f;
                for (size_t d = 0; d < head_dim; d++) {
                    float q_val = llaisys::utils::cast<float>(
                        q[i * q_stride + h * head_dim + d]);
                    float k_val = llaisys::utils::cast<float>(
                        k[j * k_stride + kv_h * head_dim + d]);
                    sum += q_val * k_val;
                }
                scores[j] = sum * scale;
                
                if (scores[j] > max_score) {
                    max_score = scores[j];
                }
            }

            float sum_scores = 0.0f;
            for (size_t j = 0; j < kvlen; j++) {
                if (scores[j] != -INFINITY) {
                    scores[j] = std::exp(scores[j] - max_score);
                    sum_scores += scores[j];
                } else {
                    scores[j] = 0.0f;
                }
            }

            for (size_t j = 0; j < kvlen; j++) {
                scores[j] /= sum_scores;
            }

            for (size_t dv = 0; dv < v_head_dim; dv++) {
                float val = 0.0f;
                for (size_t j = 0; j < kvlen; j++) {
                    if (scores[j] == 0.0f) continue;

                    float v_val = llaisys::utils::cast<float>(
                        v[j * v_stride + kv_h * v_head_dim + dv]);
                    val += scores[j] * v_val;
                }
                attn_val[i * out_stride + h * v_head_dim + dv] = llaisys::utils::cast<T>(val);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t dtype, size_t qlen, size_t kvlen, size_t num_heads, 
                    size_t kv_num_heads, size_t head_dim, size_t v_head_dim, float scale) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), 
                             reinterpret_cast<const float *>(q), 
                             reinterpret_cast<const float *>(k), 
                             reinterpret_cast<const float *>(v),
                             qlen, kvlen, num_heads, kv_num_heads, head_dim, v_head_dim, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), 
                             reinterpret_cast<const llaisys::bf16_t *>(q), 
                             reinterpret_cast<const llaisys::bf16_t *>(k), 
                             reinterpret_cast<const llaisys::bf16_t *>(v),
                             qlen, kvlen, num_heads, kv_num_heads, head_dim, v_head_dim, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), 
                             reinterpret_cast<const llaisys::fp16_t *>(q), 
                             reinterpret_cast<const llaisys::fp16_t *>(k), 
                             reinterpret_cast<const llaisys::fp16_t *>(v),
                             qlen, kvlen, num_heads, kv_num_heads, head_dim, v_head_dim, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu