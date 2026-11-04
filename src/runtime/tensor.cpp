#include "tcr/runtime/tensor.hpp"
#include "tcr/ir/dtype.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/runtime/storage.hpp"

#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tcr::runtime {

using tcr::ir::Dtype;
using tcr::ir::Shape;

Tensor::Tensor(Shape shape, Dtype dtype, std::vector<float> data)
    : shape_(std::move(shape)), dtype_(dtype) {
  if (shape_.num_elements() != data.size()) {
    throw std::invalid_argument(
        "tensor data size does not match shape num_elements");
  }
  // create bytes of the same size
  const std::size_t bytes = data.size() * sizeof(float);
  storage_ = std::make_shared<RuntimeStorage>();
  storage_->bytes.resize(bytes);

  std::memcpy(storage_->bytes.data(), data.data(), bytes);
}

Tensor::Tensor(Shape shape, Dtype dtype,
               std::shared_ptr<RuntimeStorage> storage)
    : shape_(std::move(shape)), dtype_(dtype), storage_(std::move(storage)) {
  // validate storage as per shape and dtype
  const auto required = shape_.num_elements() * ir::size_in_bytes(dtype);

  if (!storage_ || storage_->bytes.size() < required) {
    throw std::invalid_argument("storage is too small for tensor");
  }
}

const Shape &Tensor::shape() const { return shape_; }

Dtype Tensor::dtype() const { return dtype_; }

const std::vector<float> Tensor::data() const {
  const std::size_t elements = shape_.num_elements();
  std::vector<float> result(elements);
  std::memcpy(result.data(), storage_->bytes.data(), elements * sizeof(float));
  return std::move(result);
}

const std::shared_ptr<RuntimeStorage> &Tensor::storage() const {
  return storage_;
}

float Tensor::load(std::size_t offset) const {
  if (offset >= shape_.num_elements()) {
    throw std::out_of_range("offset out of bounds");
  }
  return storage_->load_f32(offset);
}

void Tensor::store(std::size_t offset, float value) {
  if (offset >= shape_.num_elements()) {
    throw std::out_of_range("offset out of bounds");
  }
  storage_->store_f32(offset, value);
}
} // namespace tcr::runtime