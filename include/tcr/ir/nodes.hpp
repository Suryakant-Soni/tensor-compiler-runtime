#pragma once

#include "tcr/ir/ids.hpp"
#include "tcr/ir/op.hpp"
#include <vector>
namespace tcr::ir {

struct Node {
  Op op;
  std::vector<ValueId> inputs;
};
} // namespace tcr::ir