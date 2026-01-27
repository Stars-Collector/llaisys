#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::cpu {
void rearrange(std::byte *out, const std::byte *in,
               llaisysDataType_t dtype,
               const std::vector<size_t>& in_shape,
               const std::vector<ptrdiff_t>& in_strides,
               const std::vector<size_t>& out_shape,
               const std::vector<ptrdiff_t>& out_strides,
               size_t out_numel);
}
