#pragma once
#include "tcr/ir/graph.hpp"
namespace tcr::compiler {
class ElementwiseFusion {
public:
  ir::Graph run(const ir::Graph &graph) const;
};
} // namespace tcr::compiler