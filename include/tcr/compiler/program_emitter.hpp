#pragma once
#include "tcr/lir/program.hpp"
#include "tcr/runtime/program.hpp"
namespace tcr::compiler {
class ProgramEmitter {
public:
  runtime::Program emit(const lir::Program &program);
};
} // namespace tcr::compiler