
#include "tcr/compiler/memory_planner.hpp"
#include "tcr/compiler/lowering.hpp"
#include "tcr/compiler/memory_plan.hpp"
#include "tcr/compiler/schedule.hpp"
#include "tcr/ir/dtype.hpp"
#include "tcr/lir/ids.hpp"
#include "tcr/runtime/slot_id.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace tcr::compiler {
namespace {
struct BufferLifetime {
  lir::BufferId buffer_id;
  std::optional<std::size_t> birth_kernel;
  std::optional<std::size_t> last_use_kernel;
  bool is_input = false;
  bool is_output = false;
};

struct ActiveSlot {
  runtime::SlotId slot_id;
  std::size_t last_use_kernel;
};
} // namespace

MemoryPlan MemoryPlanner::plan(const Schedule &schedule,
                               const LoweringResult &lowered) const {
  // prepare lifetimes
  std::vector<BufferLifetime> lifetimes(lowered.program.buffer_count());

  // -----------------------------------------------------------------------------
  // loop all kernels and collect buffers and their lifetimes
  // ------------------------------------------------------------------------------
  for (std::size_t k = 0; k < schedule.kernels.size(); k++) {
    const auto &kernel = schedule.kernels[k];
    for (const auto &output : kernel.materialized_outputs) {
      const auto &buffer_id = lowered.value_to_buffer.at(output);
      if (lifetimes[buffer_id.value].birth_kernel.has_value()) {
        throw std::logic_error(
            "materialized buffer has multiple producer kernels");
      }
      lifetimes[buffer_id.value].buffer_id = buffer_id;
      lifetimes[buffer_id.value].birth_kernel = k;
    }
    for (const auto &input : kernel.boundary_inputs) {
      const auto &buffer_id = lowered.value_to_buffer.at(input);
      lifetimes[buffer_id.value].buffer_id = buffer_id;
      lifetimes[buffer_id.value].last_use_kernel = k;
    }
  }

  // -----------------------------------------------------------------------------
  // mark inputs and outputs flag in the lifetime
  // ------------------------------------------------------------------------------
  for (const auto &input : lowered.program.inputs()) {
    const auto &buffer_id = lowered.value_to_buffer.at(input.value);
    lifetimes[buffer_id.value].buffer_id = buffer_id;
    lifetimes[buffer_id.value].is_input = true;
  }
  for (const auto &output : lowered.program.outputs()) {
    lifetimes[output.value].is_output = true;
    lifetimes[output.value].buffer_id = output;
    lifetimes[output.value].last_use_kernel = SIZE_MAX;
  }

  // -----------------------------------------------------------------------------
  // remove external inputs from lifetime data and sort by birth kernel
  // ------------------------------------------------------------------------------
  lifetimes.erase(
      std::remove_if(lifetimes.begin(), lifetimes.end(),
                     [](const auto &lifetime) { return lifetime.is_input; }),
      lifetimes.end());
  for (const auto &lt : lifetimes) {
    if (!lt.birth_kernel.has_value() || !lt.last_use_kernel.has_value()) {
      throw std::logic_error("buffer lifetime missing birth/last-use kernel");
    }
    if (lt.last_use_kernel.value() < lt.birth_kernel.value()) {
      throw std::logic_error("buffer is used earlier than its producer kernel");
    }
  }
  std::sort(lifetimes.begin(), lifetimes.end(),
            [](const auto &a, const auto &b) {
              if (a.birth_kernel != b.birth_kernel) {
                return *a.birth_kernel < *b.birth_kernel;
              }
              return a.buffer_id.value < b.buffer_id.value;
            });

  // -----------------------------------------------------------------------------
  // planning slots for buffers in the lifetime list
  // ------------------------------------------------------------------------------
  std::vector<runtime::SlotId> free_slots;
  std::vector<ActiveSlot> active_slots;
  std::vector<StorageSlot> slots;
  std::vector<std::optional<runtime::SlotId>> buffer_to_slot(
      lowered.program.buffer_count());
  for (std::size_t i = 0; i < lifetimes.size(); i++) {
    const auto &lt = lifetimes[i];

    const auto &buffer = lowered.program.buffers()[lt.buffer_id.value];
    const auto req_capacity =
        ir::size_in_bytes(buffer.dtype) * buffer.shape.num_elements();
    const auto req_alignment = alignof(std::max_align_t);

    // -----------------------------------------------------------------------------
    // inspect all active slot and free-up if the slot is idle now
    // ------------------------------------------------------------------------------
    for (std::size_t i = 0; i < active_slots.size();) {
      const auto &slot = active_slots[i];
      if (slot.last_use_kernel < lt.birth_kernel.value()) {
        free_slots.push_back(active_slots[i].slot_id);
        active_slots.erase(active_slots.begin() + i);
      } else {
        i++;
      }
    }

    std::optional<runtime::SlotId> candidate_id;
    std::optional<runtime::SlotId> chosen_slot;
    std::size_t best_cost = SIZE_MAX;
    StorageSlot storage_slot;

    // -----------------------------------------------------------------------------
    // search from free slots if any slot matches capacity,alignment
    // ------------------------------------------------------------------------------
    std::size_t smallest_capacity = SIZE_MAX;
    for (std::size_t j = 0; j < free_slots.size(); j++) {
      const auto &id = free_slots[j];
      const auto &slot = slots[id.value];
      if (slot.capacity_bytes >= req_capacity &&
          slot.alignment >= req_alignment &&
          slot.capacity_bytes < smallest_capacity) {
        // assign buffer to slot
        chosen_slot = id;
        smallest_capacity = slot.capacity_bytes;
      }
      const auto cost =
          std::max(slot.capacity_bytes, req_capacity) - slot.capacity_bytes;

      if (cost < best_cost) {
        best_cost = cost;
        candidate_id = id;
      }
    }

    // -----------------------------------------------------------------------------
    // if a slot is available adjust its capacity, if not create a new slot, and
    // then assign it
    // ------------------------------------------------------------------------------
    if (chosen_slot.has_value()) {
      free_slots.erase(
          std::find(free_slots.begin(), free_slots.end(), *chosen_slot));
      buffer_to_slot[lt.buffer_id.value] = *chosen_slot;
      active_slots.push_back(
          ActiveSlot{*chosen_slot, lt.last_use_kernel.value()});
    } else {
      if (candidate_id.has_value()) {
        storage_slot.capacity_bytes =
            std::max(req_capacity, slots[candidate_id->value].capacity_bytes);
        storage_slot.alignment =
            std::max(req_alignment, slots[candidate_id->value].alignment);
        slots[candidate_id->value] = storage_slot;
        free_slots.erase(
            std::find(free_slots.begin(), free_slots.end(), *candidate_id));
        active_slots.push_back(
            ActiveSlot{*candidate_id, lt.last_use_kernel.value()});
        buffer_to_slot[lt.buffer_id.value] = *candidate_id;
      } else {
        // create new slot if no free slot was available
        const auto new_slot_id = runtime::SlotId{slots.size()};
        storage_slot.capacity_bytes = req_capacity;
        storage_slot.alignment = req_alignment;
        slots.push_back(storage_slot);
        buffer_to_slot[lt.buffer_id.value] = new_slot_id;
        active_slots.push_back(
            ActiveSlot{new_slot_id, lt.last_use_kernel.value()});
      }
    }

  } // lifetime loop

  return MemoryPlan{std::move(slots), std::move(buffer_to_slot)};
} // plan()
} // namespace tcr::compiler