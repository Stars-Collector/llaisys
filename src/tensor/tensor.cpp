#include "tensor.hpp"
#include "../utils.hpp"
#include <cstring>
#include <numeric>
#include <sstream>
#include <functional>
#include <stdexcept>
#include <iostream>

namespace llaisys {

// ================= 辅助函数 (Internal Helpers) =================

// 1. 解决 C2679: 使用函数重载替代 if constexpr 的流输出
// 这样编译器会精确匹配类型，避免实例化不支持的 << 操作
namespace {
    // 通用打印 (int, float, double 等)
    template <typename T>
    void print_element(const T& val) {
        std::cout << val << " ";
    }

    // 特化打印: bf16_t
    // 注意：假设 utils::cast 存在于你的 utils.hpp 中
    void print_element(const bf16_t& val) {
        std::cout << utils::cast<float>(val) << " ";
    }

    // 特化打印: fp16_t
    void print_element(const fp16_t& val) {
        std::cout << utils::cast<float>(val) << " ";
    }
}

// 递归地将非连续数据复制到连续内存中 (Host端逻辑)
void copy_strided_data(const std::byte* src, std::byte* dst, 
                       const std::vector<size_t>& shape, 
                       const std::vector<ptrdiff_t>& strides, 
                       size_t dim, size_t element_size) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; ++i) {
            std::memcpy(dst + i * element_size, src + i * strides[dim], element_size);
        }
    } else {
        // 计算当前维度之后的一个完整块(block)在 dst 中的大小
        size_t block_elems = 1;
        for(size_t k = dim + 1; k < shape.size(); ++k) block_elems *= shape[k];
        size_t block_bytes = block_elems * element_size;

        for (size_t i = 0; i < shape[dim]; ++i) {
            copy_strided_data(src + i * strides[dim], dst + i * block_bytes, shape, strides, dim + 1, element_size);
        }
    }
}

// ================= Tensor Implementation =================

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    
    // 使用 size_t 避免警告，且注意 ndim_ 可能为 0
    if (ndim_ > 0) {
        for (size_t i = 1; i <= ndim_; i++) {
            strides[ndim_ - i] = stride;
            stride *= shape[ndim_ - i];
        }
    }
    
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    if (ndim_ == 0) total_elems = 1; 

    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    if (_meta.shape.empty()) return 1; 
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;
    ss << "Tensor: " << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype() 
       << " device=" << (this->deviceType() == LLAISYS_DEVICE_CPU ? "CPU" : "GPU");
    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (shape.empty()) { 
         print_element(*data); // 使用重载的 helper
         return;
    }
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            // 这里直接调用 print_element，编译器会自动选择是调用通用模板(int/float)
            // 还是特化版本(bf16/fp16)，从而彻底避开 C2679 错误
            print_element(data[i * strides[dim]]);
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE: return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL: return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:   return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:  return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:  return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:  return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:   return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:  return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:  return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:  return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:  return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:  return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:  return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16: return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto contig_tensor = this->contiguous(); 
        auto host_tensor = create(contig_tensor->shape(), contig_tensor->dtype(), LLAISYS_DEVICE_CPU);
        
        core::context().runtime().api()->memcpy_sync(
            host_tensor->data(),
            contig_tensor->data(),
            contig_tensor->numel() * contig_tensor->elementSize(),
            LLAISYS_MEMCPY_D2H);
            
        debug_print(host_tensor->data(), host_tensor->shape(), host_tensor->strides(), host_tensor->dtype());
    }
}


bool Tensor::isContiguous() const {
    if (numel() == 0) return true;
    size_t z = 1;
    
    // 使用倒序遍历习惯写法 (size_t i = size; i-- > 0; )
    // 这样 i 始终是 size_t，不会触发 int 转换警告，且处理 0 也很安全
    for (size_t i = _meta.shape.size(); i-- > 0; ) {
        if (_meta.shape[i] != 1) {
            // _meta.strides[i] 是 ptrdiff_t, z 是 size_t
            // 比较时最好统一类型：
            if (_meta.strides[i] != static_cast<ptrdiff_t>(z)) {
                return false;
            }
            z *= _meta.shape[i];
        }
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    if (order.size() != ndim()) {
        throw std::runtime_error("Permute dimension mismatch");
    }

    TensorMeta new_meta = _meta;
    new_meta.shape.clear();
    new_meta.strides.clear();

    for (size_t i = 0; i < order.size(); ++i) {
        new_meta.shape.push_back(_meta.shape[order[i]]);
        new_meta.strides.push_back(_meta.strides[order[i]]);
    }

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    size_t new_numel = 1;
    for (auto s : shape) new_numel *= s;
    if (new_numel != numel()) {
        throw std::runtime_error("Shape mismatch in view()");
    }

    if (!isContiguous()) {
        throw std::runtime_error("View called on non-contiguous tensor. Use reshape() or contiguous() first.");
    }

    // 计算新的 strides
    std::vector<ptrdiff_t> new_strides(shape.size());
    size_t stride = 1;
    
    // 3. 解决 C4267: 使用倒序遍历习惯写法
    for (size_t i = shape.size(); i-- > 0; ) {
        new_strides[i] = stride;
        stride *= shape[i];
    }

    TensorMeta new_meta{_meta.dtype, shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    if (dim >= ndim()) throw std::out_of_range("Dimension out of range");
    if (end > _meta.shape[dim] || start >= end) throw std::out_of_range("Invalid slice indices");

    size_t len = end - start;
    size_t added_offset = start * _meta.strides[dim] * elementSize();
    
    TensorMeta new_meta = _meta;
    new_meta.shape[dim] = len;

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset + added_offset));
}

void Tensor::load(const void *src) {
    size_t size_bytes = numel() * elementSize();

    if (isContiguous()) {
        if (deviceType() == LLAISYS_DEVICE_CPU) {
            std::memcpy(data(), src, size_bytes);
        } else {
            core::context().setDevice(deviceType(), deviceId());
            core::context().runtime().api()->memcpy_sync(
                data(), src, size_bytes, LLAISYS_MEMCPY_H2D);
        }
    } else {
        if (deviceType() == LLAISYS_DEVICE_CPU) {
             throw std::runtime_error("Loading into non-contiguous tensor is complex. Use load on a contiguous tensor.");
        } else {
             throw std::runtime_error("Loading into non-contiguous GPU tensor is not supported directly.");
        }
    }
}

tensor_t Tensor::contiguous() const {
    if (isContiguous()) {
        auto new_tensor = Tensor::create(_meta.shape, _meta.dtype, deviceType(), deviceId());
        if (deviceType() == LLAISYS_DEVICE_CPU) {
             std::memcpy(new_tensor->data(), data(), numel() * elementSize());
        } else {
             core::context().runtime().api()->memcpy_sync(
                 new_tensor->data(), data(), numel() * elementSize(), LLAISYS_MEMCPY_D2D);
        }
        return new_tensor;
    }

    auto new_tensor = Tensor::create(_meta.shape, _meta.dtype, deviceType(), deviceId());

    if (deviceType() == LLAISYS_DEVICE_CPU) {
        copy_strided_data(data(), new_tensor->data(), _meta.shape, _meta.strides, 0, elementSize());
    } else {
        size_t total_bytes = _storage->size();
        auto host_storage = core::context().runtime().allocateHostStorage(total_bytes);
        
        core::context().setDevice(deviceType(), deviceId());
        core::context().runtime().api()->memcpy_sync(
            host_storage->memory(), _storage->memory(), total_bytes, LLAISYS_MEMCPY_D2H);
        
        Tensor host_proxy(_meta, host_storage, _offset);
        auto host_contiguous = host_proxy.contiguous();
        
        core::context().runtime().api()->memcpy_sync(
            new_tensor->data(), host_contiguous->data(), 
            new_tensor->numel() * new_tensor->elementSize(), LLAISYS_MEMCPY_H2D);
    }

    return new_tensor;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    size_t new_numel = 1;
    for(auto s : shape) new_numel *= s;
    if (new_numel != numel()) throw std::runtime_error("Shape mismatch");

    if (isContiguous()) {
        return view(shape);
    }
    return contiguous()->view(shape);
}

tensor_t Tensor::to(llaisysDeviceType_t target_device, int device) const {
    if (device == -1) device = 0;

    auto contig_src = this->isContiguous() ? 
        std::shared_ptr<Tensor>(new Tensor(*this)) : 
        this->contiguous();

    auto new_tensor = Tensor::create(contig_src->shape(), contig_src->dtype(), target_device, device);

    auto src_dev = contig_src->deviceType();
    
    if (target_device != LLAISYS_DEVICE_CPU) {
        core::context().setDevice(target_device, device);
    } else if (src_dev != LLAISYS_DEVICE_CPU) {
        core::context().setDevice(src_dev, contig_src->deviceId());
    }

    auto kind = LLAISYS_MEMCPY_H2H;
    if (src_dev == LLAISYS_DEVICE_CPU && target_device != LLAISYS_DEVICE_CPU) kind = LLAISYS_MEMCPY_H2D;
    else if (src_dev != LLAISYS_DEVICE_CPU && target_device == LLAISYS_DEVICE_CPU) kind = LLAISYS_MEMCPY_D2H;
    else if (src_dev != LLAISYS_DEVICE_CPU && target_device != LLAISYS_DEVICE_CPU) kind = LLAISYS_MEMCPY_D2D;

    if (kind == LLAISYS_MEMCPY_H2H) {
        std::memcpy(new_tensor->data(), contig_src->data(), contig_src->numel() * contig_src->elementSize());
    } else {
        core::context().runtime().api()->memcpy_sync(
            new_tensor->data(), contig_src->data(), 
            contig_src->numel() * contig_src->elementSize(), kind);
    }

    return new_tensor;
}

} // namespace llaisys