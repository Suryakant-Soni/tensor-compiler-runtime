#pragma once
#include "tcr/ir/dtype.hpp"
#include "tcr/ir/shape.hpp"
#include <vector>

namespace tcr::reference {

using tcr::ir::Dtype;
using tcr::ir::Shape;
class Tensor {
public:
  // constructor
  Tensor(Shape shape, Dtype dtype, std::vector<float> data);
  // get shape value by reference/const
  const Shape &shape() const;
  // get type's local copy returned by value, since types is a small enum value,
  // it doesnt hit perf
  Dtype dtype() const;

  //constness overloading/redeclaration for fetching data_
  const std::vector<float>& data() const;
  std::vector<float>& data();

private:
  Shape shape_;
  Dtype dtype_;
  std::vector<float> data_;
};
} // namespace tcr::reference