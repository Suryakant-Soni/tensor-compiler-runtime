#pragma once
#include "tcr/ir/shape.hpp"
#include "tcr/ir/dtype.hpp"
#include "tcr/ir/ids.hpp"
namespace tcr::ir {

struct Value {
  Shape  shape;
  Dtype dtype;
  std::optional<NodeId> producer;
};
}