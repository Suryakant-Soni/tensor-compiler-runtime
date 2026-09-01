#pragma once
#include "tcr/ir/shape.hpp"
#include "tcr/ir/dtype.hpp"
#include "tcr/ir/ids.hpp"
#include <optional>
namespace tcr::ir {

struct Value {
  Shape  shape;
  Dtype dtype;
  std::optional<NodeId> producer;
};
}