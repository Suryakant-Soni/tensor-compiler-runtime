#pragma once
/**
 * @file tensor.hpp
 * @author your name (you@domain.com)
 * @brief tensor is a class for production tensor not the reference tensors
 * @version 0.1
 * @date 2026-08-31
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "tcr/ir/dtype.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/runtime/storage.hpp"
#include <cstddef>
#include <memory>
#include <vector>
namespace tcr::runtime {
using tcr::ir::Dtype;
using tcr::ir::Shape;
class Tensor {
public:
  Tensor(Shape shape, Dtype dtype, std::vector<float> data);

  Tensor(Shape shape, Dtype dtype, std::shared_ptr<RuntimeStorage> storage);

  const Shape &shape() const;
  Dtype dtype() const;

  const std::vector<float> data() const;
  const std::shared_ptr<RuntimeStorage> &storage() const;

  float load(std::size_t offset) const;
  void store(std::size_t offset, float value);

private:
  Shape shape_;
  Dtype dtype_;
  std::shared_ptr<RuntimeStorage> storage_;
};
} // namespace tcr::runtime