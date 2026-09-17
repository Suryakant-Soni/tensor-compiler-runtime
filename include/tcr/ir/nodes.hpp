#pragma once

#include "tcr/ir/elementwise_expr.hpp"
#include "tcr/ir/ids.hpp"
#include "tcr/ir/op.hpp"
#include <optional>
#include <vector>
namespace tcr::ir {

struct Node {
  Op op;
  std::vector<ValueId> inputs;
  std::optional<ElementwiseExpr> elementwise_expr;
};
} // namespace tcr::ir