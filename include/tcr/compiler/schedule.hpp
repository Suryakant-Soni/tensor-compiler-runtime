#pragma once
#include "tcr/ir/ids.hpp"
#include <vector>

namespace tcr::compiler {
struct KernelUnit {
  std::vector<ir::ValueId> members;
  std::vector<ir::ValueId> boundary_inputs;
  std::vector<ir::ValueId> materialized_outputs;
};

struct Schedule {
  // kernel units are executed in the vector order
  std::vector<KernelUnit> kernels;
};
} // namespace tcr::compiler