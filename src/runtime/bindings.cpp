#include "tcr/runtime/bindings.hpp"
#include <stdexcept>
#include <utility>
namespace tcr::runtime {
void InputBindings::bind(SlotId slot, Tensor tensor) {
  for (auto &[existing_slot, existing_tensor] : bindings_) {
    if (slot == existing_slot) {
      existing_tensor = std::move(tensor);
      return;
    }
  }
  bindings_.push_back({slot, std::move(tensor)});
}

const Tensor &InputBindings::get(SlotId slot) const {
  for (const auto &[existing_slot, existing_tensor] : bindings_) {
    if (slot == existing_slot) {
      return existing_tensor;
    }
  }
  throw std::out_of_range("no tensor value bound to input slot_id");
}
} // namespace tcr::runtime