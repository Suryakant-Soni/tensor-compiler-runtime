#pragma once
// defined a slot id type which is internally value(size_t)
// however value_id and slotid representation would be stored in compiler.hpp

#include <cstddef>
namespace tcr::runtime {

struct SlotId {
  std::size_t value;

  bool operator==(const SlotId &) const = default;
};
} // namespace tcr::runtime