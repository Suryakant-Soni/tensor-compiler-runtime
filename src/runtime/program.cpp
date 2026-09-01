#include "tcr/runtime/program.hpp"
#include "tcr/runtime/instruction.hpp"
#include "tcr/runtime/slot_id.hpp"
#include <cstddef>
#include <vector>
#include <utility>
namespace tcr::runtime {

Program::Program(std::vector<Instruction> instructions,
                 std::vector<SlotId> inputs, std::vector<SlotId> outputs,
                 std::size_t slots_count)
    : instructions_(std::move(instructions)), inputs_(std::move(inputs)), outputs_(std::move(outputs)),
      slots_count_(slots_count) {}

const std::vector<Instruction> &Program::instructions() const {
  return instructions_;
}
const std::vector<SlotId> &Program::inputs() const { return inputs_; }

const std::vector<SlotId> &Program::outputs() const { return outputs_; }

std::size_t Program::slots_count() const { return slots_count_; }
} // namespace tcr::runtime