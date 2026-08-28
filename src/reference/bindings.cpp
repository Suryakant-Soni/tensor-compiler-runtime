
#include "tcr/reference/bindings.hpp"
#include "tcr/ir/ids.hpp"
#include "tcr/reference/tensor.hpp"
#include <stdexcept>
namespace tcr::reference {
using tcr::ir::ValueId;

void InputBindings::bind(ValueId id, Tensor tensor) {
  for (auto &[existing_id, existing_tensor] : bindings_) {
    // if id already exists, update the existing tensor
    if (id == existing_id) {
      existing_tensor = std::move(tensor);
      return;
    }
  }
  // if not, push it to the list since this is new id
  bindings_.push_back({
      id,
      std::move(tensor),
  });
}

const Tensor &InputBindings::get(ValueId id) const {
  for (const auto &[existing_id, existing_tensor] : bindings_) {
    if (id == existing_id) {
      return existing_tensor;
    }
  }
  throw std::out_of_range("no Tensor value bound to input ValueId");
}
} // namespace tcr::reference