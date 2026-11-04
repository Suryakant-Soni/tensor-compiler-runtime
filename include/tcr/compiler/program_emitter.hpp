#pragma once
#include "tcr/compiler/memory_plan.hpp"
#include "tcr/lir/program.hpp"
#include "tcr/runtime/program.hpp"
namespace tcr::compiler {
class ProgramEmitter {
public:
  runtime::Program emit(const lir::Program &program,
                        const MemoryPlan &memory_plan);
};
} // namespace tcr::compiler