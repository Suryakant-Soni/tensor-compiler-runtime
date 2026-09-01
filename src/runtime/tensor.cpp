#include "tcr/runtime/tensor.hpp"
#include "tcr/ir/dtype.hpp"
#include "tcr/ir/shape.hpp"
#include <stdexcept>
#include <utility>
#include <vector>
namespace tcr::runtime {
using tcr::ir::Dtype;
using tcr::ir::Shape;
Tensor::Tensor(Shape shape, Dtype dtype, std::vector<float> data)
    : shape_(std::move(shape)), dtype_(dtype), data_(std::move(data)) {
  if (shape.num_elements() != data.size()) {
    throw std::invalid_argument(
        "tensor data size does not match shape num_elements");
  }
}

const Shape &Tensor::shape() const { return shape_; }
Dtype Tensor::dtype() const { return dtype_; }
const std::vector<float> &Tensor::data() const { return data_; }
} // namespace tcr::runtime