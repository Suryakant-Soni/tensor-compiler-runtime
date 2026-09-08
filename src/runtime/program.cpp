#include "tcr/runtime/program.hpp"
#include "tcr/runtime/instruction.hpp"
#include "tcr/runtime/slot_id.hpp"
#include <cstddef>
#include <utility>
#include <vector>

namespace tcr::runtime {
Program::Program() = default;

Program::Program(std::vector<Instruction> instructions,
                 std::vector<SlotId> inputs, std::vector<SlotId> outputs,
                 std::size_t slots_count, std::size_t scalar_count,
                 std::size_t index_count)
    : instructions_(std::move(instructions)), inputs_(std::move(inputs)),
      outputs_(std::move(outputs)), slots_count_(slots_count),
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

std::size_t Program::slots_count() const { return slots_count_; }
std::size_t Program::scalar_count() const { return scalar_count_; }
std::size_t Program::index_count() const { return index_count_; }

void Program::set_slots_count(std::size_t count) { slots_count_ = count; }
void Program::set_scalar_count(std::size_t count) { scalar_count_ = count; }
void Program::set_index_count(std::size_t count) { index_count_ = count; }

} // namespace tcr::runtime