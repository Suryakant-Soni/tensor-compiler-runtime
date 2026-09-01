#pragma once
#include "tcr/runtime/tensor.hpp"
#include "tcr/runtime/slot_id.hpp"
#include <utility>
#include <vector>
namespace tcr::runtime {

class InputBindings {
public:
  void bind(SlotId slot, Tensor tensor);
  const Tensor &get(SlotId slot) const;

private:
  std::vector<std::pair<SlotId, Tensor>> bindings_;
};
} // namespace tcr::runtime