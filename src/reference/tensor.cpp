#include "tcr/reference/tensor.hpp"
#include "tcr/ir/dtype.hpp"
#include "tcr/ir/shape.hpp"
#include <stdexcept>
#include <utility>
#include <vector>

namespace tcr::reference {

using tcr::ir::Dtype;
using tcr::ir::Shape;
Tensor::Tensor(Shape shape, Dtype dtype, std::vector<float> data)
    : shape_(std::move(shape)), dtype_(dtype), data_(std::move(data)) {
  if (data_.size() != shape_.num_elements()) {
    throw std::invalid_argument("tensor data size does not match shape");
  }
}

const Shape &Tensor::shape() const { return shape_; }
Dtype Tensor::dtype() const { return dtype_; }
const std::vector<float> &Tensor::data() const { return data_; }
} // namespace tcr::reference