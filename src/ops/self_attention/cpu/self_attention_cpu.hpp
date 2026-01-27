#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t dtype, size_t qlen, size_t kvlen, size_t num_heads, 
                    size_t kv_num_heads, size_t head_dim, size_t v_head_dim, float scale);
}