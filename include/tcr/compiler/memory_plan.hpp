#pragma once

#include "tcr/runtime/slot_id.hpp"
#include <cstddef>
#include <optional>
#include <vector>
namespace tcr::compiler {

// index = runtime::slotid
struct StorageSlot {
  std::size_t capacity_bytes{};
  std::size_t alignment{};
};

struct MemoryPlan {
  std::vector<StorageSlot> slots;
  std::vector<std::optional<runtime::SlotId>> buffer_to_slot;
};
} // namespace tcr::compiler
