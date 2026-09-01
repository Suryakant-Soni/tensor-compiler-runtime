#pragma once
#include "tcr/runtime/bindings.hpp"
#include "tcr/runtime/program.hpp"
#include <vector>
namespace tcr::runtime {

class VM {
  std::vector<Tensor> run(const Program &program, const InputBindings &inputs);
};
} // namespace tcr::runtime