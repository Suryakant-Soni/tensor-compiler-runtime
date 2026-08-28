#pragma once
#include "tcr/ir/ids.hpp"
#include "tcr/reference/tensor.hpp"
#include <utility>
#include <vector>

namespace tcr::reference {
using tcr::ir::ValueId;

class InputBindings {
public:
  void bind(ValueId id, Tensor tensor);
  const Tensor &get(ValueId id) const;

private:
  std::vector<std::pair<ValueId, Tensor>> bindings_;
};
} // namespace tcr::reference