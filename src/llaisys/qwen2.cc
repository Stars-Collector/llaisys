#include "llaisys/models/qwen2.h"
#include "llaisys_tensor.hpp"
#include "../core/llaisys_core.hpp"
#include "../utils.hpp"

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"

#include <cstring>
#include <memory>
#include <vector>
#include <cmath>

namespace llaisys::models {

struct Qwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    llaisysDeviceType_t device_type;
    int device_id;

    // KV Cache
    std::vector<tensor_t> k_cache;
    std::vector<tensor_t> v_cache;
    size_t cache_len;

    Qwen2Model(const LlaisysQwen2Meta& meta, llaisysDeviceType_t device_type, int device_id)
        : meta(meta), device_type(device_type), device_id(device_id), cache_len(0) {
        // Initialize KV cache
        k_cache.resize(meta.nlayer);
        v_cache.resize(meta.nlayer);

        for (size_t i = 0; i < meta.nlayer; ++i) {
            // k_cache: [maxseq, nkvh, dh]
            k_cache[i] = Tensor::create(
                {meta.maxseq, meta.nkvh, meta.dh},
                meta.dtype,
                device_type,
                device_id
            );

            // v_cache: [maxseq, nkvh, dh]
            v_cache[i] = Tensor::create(
                {meta.maxseq, meta.nkvh, meta.dh},
                meta.dtype,
                device_type,
                device_id
            );
        }
    }

    ~Qwen2Model() {
        // weights tensors are managed by LlaisysQwen2ModelDestroy manually
        // kv cache shared_ptrs handle themselves
    }

    void reset_kv_cache() {
        cache_len = 0;
    }
};

} // namespace llaisys::models

__C {
    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta,
        llaisysDeviceType_t device_type,
        int *device_ids,
        int ndevice) {

        auto model = new llaisys::models::Qwen2Model(*meta, device_type, ndevice > 0 ? device_ids[0] : 0);

        // Allocate weight tensors
        auto& weights = model->weights;
        int dev_id = ndevice > 0 ? device_ids[0] : 0;

        // Input embedding: [voc, di] (Row lookup)
        weights.in_embed = new LlaisysTensor{
            llaisys::Tensor::create({meta->voc, meta->di}, meta->dtype, device_type, dev_id)
        };

        // Output embedding (LM Head): [voc, di]
        // 注意：Linear层的权重形状通常是 [out_features, in_features]
        // 此处输入是 di (hidden_size)，输出是 voc (vocab_size)
        weights.out_embed = new LlaisysTensor{
            llaisys::Tensor::create({meta->voc, meta->di}, meta->dtype, device_type, dev_id)
        };

        // Output norm: [di]
        weights.out_norm_w = new LlaisysTensor{
            llaisys::Tensor::create({meta->di}, meta->dtype, device_type, dev_id)
        };

        // Allocate arrays for layers
        weights.attn_norm_w = new llaisysTensor_t[meta->nlayer];
        weights.attn_q_w = new llaisysTensor_t[meta->nlayer];
        weights.attn_q_b = new llaisysTensor_t[meta->nlayer];
        weights.attn_k_w = new llaisysTensor_t[meta->nlayer];
        weights.attn_k_b = new llaisysTensor_t[meta->nlayer];
        weights.attn_v_w = new llaisysTensor_t[meta->nlayer];
        weights.attn_v_b = new llaisysTensor_t[meta->nlayer];
        weights.attn_o_w = new llaisysTensor_t[meta->nlayer];
        weights.mlp_norm_w = new llaisysTensor_t[meta->nlayer];
        weights.mlp_gate_w = new llaisysTensor_t[meta->nlayer];
        weights.mlp_up_w = new llaisysTensor_t[meta->nlayer];
        weights.mlp_down_w = new llaisysTensor_t[meta->nlayer];

        for (size_t i = 0; i < meta->nlayer; ++i) {
            // Attention norm: [di]
            weights.attn_norm_w[i] = new LlaisysTensor{
                llaisys::Tensor::create({meta->di}, meta->dtype, device_type, dev_id)
            };

            // Q projection: [nh * dh, di] (out, in)
            weights.attn_q_w[i] = new LlaisysTensor{
                llaisys::Tensor::create({meta->nh * meta->dh, meta->di}, meta->dtype, device_type, dev_id)
            };
            weights.attn_q_b[i] = new LlaisysTensor{
                llaisys::Tensor::create({meta->nh * meta->dh}, meta->dtype, device_type, dev_id)
            };

            // K projection: [nkvh * dh, di] (out, in)
            weights.attn_k_w[i] = new LlaisysTensor{
                llaisys::Tensor::create({meta->nkvh * meta->dh, meta->di}, meta->dtype, device_type, dev_id)
            };
            weights.attn_k_b[i] = new LlaisysTensor{
                llaisys::Tensor::create({meta->nkvh * meta->dh}, meta->dtype, device_type, dev_id)
            };

            // V projection: [nkvh * dh, di] (out, in)
            weights.attn_v_w[i] = new LlaisysTensor{
                llaisys::Tensor::create({meta->nkvh * meta->dh, meta->di}, meta->dtype, device_type, dev_id)
            };
            weights.attn_v_b[i] = new LlaisysTensor{
                llaisys::Tensor::create({meta->nkvh * meta->dh}, meta->dtype, device_type, dev_id)
            };

            // O projection: [di, nh * dh] (out, in)
            weights.attn_o_w[i] = new LlaisysTensor{
                llaisys::Tensor::create({meta->di, meta->nh * meta->dh}, meta->dtype, device_type, dev_id)
            };

            // MLP norm: [di]
            weights.mlp_norm_w[i] = new LlaisysTensor{
                llaisys::Tensor::create({meta->di}, meta->dtype, device_type, dev_id)
            };

            // MLP gate: [intermediate_size, di] (out, in)
            weights.mlp_gate_w[i] = new LlaisysTensor{
                llaisys::Tensor::create({meta->intermediate_size, meta->di}, meta->dtype, device_type, dev_id)
            };

            // MLP up: [intermediate_size, di] (out, in)
            weights.mlp_up_w[i] = new LlaisysTensor{
                llaisys::Tensor::create({meta->intermediate_size, meta->di}, meta->dtype, device_type, dev_id)
            };

            // MLP down: [di, intermediate_size] (out, in)
            weights.mlp_down_w[i] = new LlaisysTensor{
                llaisys::Tensor::create({meta->di, meta->intermediate_size}, meta->dtype, device_type, dev_id)
            };
        }

        return reinterpret_cast<struct LlaisysQwen2Model*>(model);
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
        if (!model) return;

        auto cpp_model = reinterpret_cast<llaisys::models::Qwen2Model*>(model);
        auto& weights = cpp_model->weights;

        // Destroy weight tensors
        delete weights.in_embed;
        delete weights.out_embed;
        delete weights.out_norm_w;

        for (size_t i = 0; i < cpp_model->meta.nlayer; ++i) {
            delete weights.attn_norm_w[i];
            delete weights.attn_q_w[i];
            delete weights.attn_q_b[i];
            delete weights.attn_k_w[i];
            delete weights.attn_k_b[i];
            delete weights.attn_v_w[i];
            delete weights.attn_v_b[i];
            delete weights.attn_o_w[i];
            delete weights.mlp_norm_w[i];
            delete weights.mlp_gate_w[i];
            delete weights.mlp_up_w[i];
            delete weights.mlp_down_w[i];
        }

        delete[] weights.attn_norm_w;
        delete[] weights.attn_q_w;
        delete[] weights.attn_q_b;
        delete[] weights.attn_k_w;
        delete[] weights.attn_k_b;
        delete[] weights.attn_v_w;
        delete[] weights.attn_v_b;
        delete[] weights.attn_o_w;
        delete[] weights.mlp_norm_w;
        delete[] weights.mlp_gate_w;
        delete[] weights.mlp_up_w;
        delete[] weights.mlp_down_w;

        delete cpp_model;
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
        if (!model) return nullptr;
        auto cpp_model = reinterpret_cast<llaisys::models::Qwen2Model*>(model);
        return &cpp_model->weights;
    }

    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
        if (!model || !token_ids || ntoken == 0) return -1;

        auto cpp_model = reinterpret_cast<llaisys::models::Qwen2Model*>(model);
        auto& meta = cpp_model->meta;
        auto& weights = cpp_model->weights;

        // Check if this is a new sequence
        if (cpp_model->cache_len == 0 && ntoken > 0) {
            cpp_model->reset_kv_cache();
        }
        
        // 如果是生成阶段（ntoken=1），且cache不为空，说明是后续生成
        // 如果是prefill阶段（ntoken>1），通常cache_len为0

        // Create position IDs placeholder
        auto pos_ids = llaisys::Tensor::create({ntoken}, ::LLAISYS_DTYPE_I64, cpp_model->device_type, cpp_model->device_id);

        // Create input token tensor
        auto input_tokens = llaisys::Tensor::create({ntoken}, ::LLAISYS_DTYPE_I64, cpp_model->device_type, cpp_model->device_id);
        memcpy(input_tokens->data(), token_ids, ntoken * sizeof(int64_t));

        // Embedding lookup: [ntoken, di]
        auto hidden_states = llaisys::Tensor::create({ntoken, meta.di}, meta.dtype, cpp_model->device_type, cpp_model->device_id);
        llaisys::ops::embedding(hidden_states, input_tokens, weights.in_embed->tensor);

        // Process each token sequentially (Simulating causal masking)
        for (size_t step = 0; step < ntoken; ++step) {
            // Update current position based on cache length + current step
            int64_t current_pos_val = cpp_model->cache_len; // cache_len is incremented at end of loop

            auto current_hidden = hidden_states->slice(0, step, step + 1);  // [1, di]
            
            // Create single pos tensor for RoPE
            auto current_pos = llaisys::Tensor::create({1}, ::LLAISYS_DTYPE_I64, cpp_model->device_type, cpp_model->device_id);
            ((int64_t*)current_pos->data())[0] = current_pos_val;

            // Process through each layer
            for (size_t layer = 0; layer < meta.nlayer; ++layer) {
                // 1. Attention RMS norm
                auto attn_norm_out = llaisys::Tensor::create({1, meta.di}, meta.dtype, cpp_model->device_type, cpp_model->device_id);
                llaisys::ops::rms_norm(attn_norm_out, current_hidden, weights.attn_norm_w[layer]->tensor, meta.epsilon);

                // 2. QKV projections
                auto q_out = llaisys::Tensor::create({1, meta.nh * meta.dh}, meta.dtype, cpp_model->device_type, cpp_model->device_id);
                auto k_out = llaisys::Tensor::create({1, meta.nkvh * meta.dh}, meta.dtype, cpp_model->device_type, cpp_model->device_id);
                auto v_out = llaisys::Tensor::create({1, meta.nkvh * meta.dh}, meta.dtype, cpp_model->device_type, cpp_model->device_id);

                llaisys::ops::linear(q_out, attn_norm_out, weights.attn_q_w[layer]->tensor, weights.attn_q_b[layer]->tensor);
                llaisys::ops::linear(k_out, attn_norm_out, weights.attn_k_w[layer]->tensor, weights.attn_k_b[layer]->tensor);
                llaisys::ops::linear(v_out, attn_norm_out, weights.attn_v_w[layer]->tensor, weights.attn_v_b[layer]->tensor);

                // 3. Reshape Q, K, V
                auto q = q_out->view({1, meta.nh, meta.dh});
                auto k = k_out->view({1, meta.nkvh, meta.dh});
                auto v = v_out->view({1, meta.nkvh, meta.dh});

                // 4. Apply RoPE
                auto q_rope = llaisys::Tensor::create({1, meta.nh, meta.dh}, meta.dtype, cpp_model->device_type, cpp_model->device_id);
                auto k_rope = llaisys::Tensor::create({1, meta.nkvh, meta.dh}, meta.dtype, cpp_model->device_type, cpp_model->device_id);

                llaisys::ops::rope(q_rope, q, current_pos, meta.theta);
                llaisys::ops::rope(k_rope, k, current_pos, meta.theta);

                // 5. Update KV cache
                // Ensure cache is large enough (allocated in constructor), just slice correct part
                auto k_cache_entry = cpp_model->k_cache[layer]->slice(0, cpp_model->cache_len, cpp_model->cache_len + 1);
                auto v_cache_entry = cpp_model->v_cache[layer]->slice(0, cpp_model->cache_len, cpp_model->cache_len + 1);

                // Copy data to cache
                memcpy(k_cache_entry->data(), k_rope->data(), k_rope->numel() * k_rope->elementSize());
                memcpy(v_cache_entry->data(), v->data(), v->numel() * v->elementSize());

                // 6. Self attention with full KV cache (from 0 to cache_len + 1)
                auto attn_out = llaisys::Tensor::create({1, meta.nh, meta.dh}, meta.dtype, cpp_model->device_type, cpp_model->device_id);
                auto k_cache_full = cpp_model->k_cache[layer]->slice(0, 0, cpp_model->cache_len + 1);
                auto v_cache_full = cpp_model->v_cache[layer]->slice(0, 0, cpp_model->cache_len + 1);

                float scale = 1.0f / std::sqrt(static_cast<float>(meta.dh));
                llaisys::ops::self_attention(attn_out, q_rope, k_cache_full, v_cache_full, scale);

                // 7. Output projection
                auto attn_out_reshaped = attn_out->view({1, meta.nh * meta.dh});
                auto attn_proj_out = llaisys::Tensor::create({1, meta.di}, meta.dtype, cpp_model->device_type, cpp_model->device_id);
                llaisys::ops::linear(attn_proj_out, attn_out_reshaped, weights.attn_o_w[layer]->tensor, nullptr);

                // 8. Residual connection
                auto residual = llaisys::Tensor::create({1, meta.di}, meta.dtype, cpp_model->device_type, cpp_model->device_id);
                llaisys::ops::add(residual, current_hidden, attn_proj_out);

                // 9. MLP RMS norm
                auto mlp_norm_out = llaisys::Tensor::create({1, meta.di}, meta.dtype, cpp_model->device_type, cpp_model->device_id);
                llaisys::ops::rms_norm(mlp_norm_out, residual, weights.mlp_norm_w[layer]->tensor, meta.epsilon);

                // 10. MLP gate and up projections
                auto mlp_gate_out = llaisys::Tensor::create({1, meta.intermediate_size}, meta.dtype, cpp_model->device_type, cpp_model->device_id);
                auto mlp_up_out = llaisys::Tensor::create({1, meta.intermediate_size}, meta.dtype, cpp_model->device_type, cpp_model->device_id);

                llaisys::ops::linear(mlp_gate_out, mlp_norm_out, weights.mlp_gate_w[layer]->tensor, nullptr);
                llaisys::ops::linear(mlp_up_out, mlp_norm_out, weights.mlp_up_w[layer]->tensor, nullptr);

                // 11. SwiGLU activation
                auto mlp_act_out = llaisys::Tensor::create({1, meta.intermediate_size}, meta.dtype, cpp_model->device_type, cpp_model->device_id);
                llaisys::ops::swiglu(mlp_act_out, mlp_gate_out, mlp_up_out);

                // 12. MLP down projection
                auto mlp_down_out = llaisys::Tensor::create({1, meta.di}, meta.dtype, cpp_model->device_type, cpp_model->device_id);
                llaisys::ops::linear(mlp_down_out, mlp_act_out, weights.mlp_down_w[layer]->tensor, nullptr);

                // 13. Second residual connection
                llaisys::ops::add(current_hidden, residual, mlp_down_out);
                
                // Copy result back to hidden_states slice if we need to output all (not strictly needed for just next token, but good for debugging)
                // memcpy(hidden_states->slice(0, step, step+1)->data(), current_hidden->data(), ...);
            }

            // Update cache length for next token/step
            cpp_model->cache_len++;
            
            // Keep the last hidden state for prediction
            if (step == ntoken - 1) {
                // Final RMS norm
                auto final_norm_out = llaisys::Tensor::create({1, meta.di}, meta.dtype, cpp_model->device_type, cpp_model->device_id);
                llaisys::ops::rms_norm(final_norm_out, current_hidden, weights.out_norm_w->tensor, meta.epsilon);

                // Project to vocabulary: [1, voc]
                // out_embed is [voc, di], input is [1, di]. 
                // linear computes X * W^T. [1, di] * [di, voc] = [1, voc].
                auto logits = llaisys::Tensor::create({1, meta.voc}, meta.dtype, cpp_model->device_type, cpp_model->device_id);
                llaisys::ops::linear(logits, final_norm_out, weights.out_embed->tensor, nullptr);

                // Argmax to get next token
                auto max_idx = llaisys::Tensor::create({1}, ::LLAISYS_DTYPE_I64, cpp_model->device_type, cpp_model->device_id);
                auto max_val = llaisys::Tensor::create({1}, meta.dtype, cpp_model->device_type, cpp_model->device_id);
                llaisys::ops::argmax(max_idx, max_val, logits);

                return ((int64_t*)max_idx->data())[0];
            }
        }

        return -1; // Should not reach here
    }

    void llaisysQwen2ModelResetCache(struct LlaisysQwen2Model *model) {
        if (!model) return;
        auto cpp_model = reinterpret_cast<llaisys::models::Qwen2Model*>(model);
        cpp_model->reset_kv_cache();
    }
}