#pragma once
#include "tcr/ir/dtype.hpp"
#include "tcr/ir/op.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/ir/value.hpp"
#include <vector>
namespace tcr::ir {

struct InferredType {
  Shape shape;
  Dtype dtype;
};

InferredType infer_type(Op op, const std::vector<const Value *> &inputs);
} // namespace tcr::ir
