#pragma once
#include "tcr/compiler/schedule.hpp"
#include "tcr/ir/graph.hpp"

namespace tcr::compiler {
class Scheduler {
public:
  Schedule schedule(const ir::Graph &graph) const;
};

} // namespace tcr::compiler