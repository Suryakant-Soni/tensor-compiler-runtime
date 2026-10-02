#include "tcr/compiler/compiler.hpp"
#include "tcr/compiler/elementwise_fusion.hpp"
#include "tcr/compiler/lowering.hpp"
#include "tcr/compiler/program_emitter.hpp"
#include "tcr/compiler/scheduler.hpp"
#include "tcr/ir/graph.hpp"
#include <cstddef>

namespace tcr::compiler {

runtime::Program Compiler::compile(const ir::Graph &graph) const {
  ElementwiseFusion fusion;
  const auto optimized_graph = fusion.run(graph);

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(optimized_graph);

  // first convert the high IR graph to Low IR graph using lowering api
  Lowering lowering;
  auto lir_program = lowering.lower(optimized_graph, schedule);

  // then convert the lower program to runtime program which can be directly
  // consumed by VM
  ProgramEmitter emitter;
  return emitter.emit(lir_program);
}
} // namespace tcr::compiler