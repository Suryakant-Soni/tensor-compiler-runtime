#include "tcr/runtime/program.hpp"
#include "tcr/runtime/instruction.hpp"
#include "tcr/runtime/slot_id.hpp"
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tcr::runtime {
Program::Program() = default;

Program::Program(std::vector<Instruction> instructions,
                 std::vector<SlotId> inputs, std::vector<SlotId> outputs,
                 std::vector<SlotTensorInfo> slots, std::size_t scalar_count,
                 std::size_t index_count)
    : instructions_(std::move(instructions)), inputs_(std::move(inputs)),
      outputs_(std::move(outputs)), slots_(std::move(slots)),
      scalar_count_(scalar_count), index_count_(index_count) {}

const std::vector<Instruction> &Program::instructions() const {
  return instructions_;
}
std::vector<Instruction> &Program::instructions() { return instructions_; }
void Program::add_instruction(Instruction instruction) {
  instructions_.push_back(std::move(instruction));
}

const std::vector<SlotId> &Program::inputs() const { return inputs_; }
const std::vector<SlotId> &Program::outputs() const { return outputs_; }

std::size_t Program::slots_count() const { return slots_.size(); }
std::size_t Program::scalar_count() const { return scalar_count_; }
std::size_t Program::index_count() const { return index_count_; }

const SlotTensorInfo &Program::tensor_meta_by_slot(SlotId slot_id) const {
  if (slot_id.value >= slots_.size()) {
    throw std::out_of_range(
        "runtime slotid is out of range(greater than slots count)");
  }
  return slots_[slot_id.value];
}

} // namespace tcr::runtime