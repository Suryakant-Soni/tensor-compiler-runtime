#include "tcr/compiler/compiler.hpp"
#include "tcr/compiler/lowering.hpp"
#include "tcr/compiler/program_emitter.hpp"
#include "tcr/ir/graph.hpp"
#include <cstddef>
namespace tcr::compiler {

runtime::Program Compiler::compile(const ir::Graph &graph) const {
  // first convert the high IR graph to Low IR graph using lowering api
  Lowering lowering;
  auto lir_program = lowering.lower(graph);

  // then convert the lower program to runtime program which can be directly
  // consumed by VM
  ProgramEmitter emitter;

  return emitter.emit(lir_program);
}
} // namespace tcr::compiler