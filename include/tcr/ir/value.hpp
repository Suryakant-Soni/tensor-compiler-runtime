#pragma once
#include "tcr/ir/shape.hpp"
#include "tcr/ir/dtype.hpp"
#include "tcr/ir/ids.hpp"
namespace tcr::ir {

struct Value {
  tcr::ir::Shape  shape;
  tcr::ir::Dtype dtype;
  std::optional<tcr::ir::NodeId> producer;
};
}