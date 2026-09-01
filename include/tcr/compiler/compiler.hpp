#pragma once
#include "tcr/ir/graph.hpp"
#include "tcr/runtime/program.hpp"
namespace tcr::compiler {
using tcr::ir::Graph;
using tcr::runtime::Program;
class Compiler {
public:
  Program compile(const Graph &graph) const;
};
} // namespace tcr::compiler