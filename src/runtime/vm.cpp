#include "tcr/runtime/vm.hpp"
#include "tcr/runtime/bindings.hpp"
#include "tcr/runtime/instruction.hpp"
#include "tcr/runtime/ops.hpp"
#include "tcr/runtime/program.hpp"
#include "tcr/runtime/slot_id.hpp"
#include "tcr/runtime/tensor.hpp"
#include <optional>
#include <stdexcept>
#include <vector>
namespace tcr::runtime {
namespace {
const Tensor &get_slot(const std::vector<std::optional<Tensor>> &slots,
                       SlotId slot_id) {
  if (slot_id.value >= slots.size()) {
    throw std::logic_error("runtime slot id out of range");
  }
  if (!slots[slot_id.value].has_value()) {
    throw std::logic_error("attempted to read an empty runtime slot");
  }
  return *slots[slot_id.value];
}
} // namespace

std::vector<Tensor> VM::run(const Program &program,
                            const InputBindings &bindings) {
  // first create a slot table and add all inputs to it
  std::vector<std::optional<Tensor>> slots(program.slots_count());

  // initialize slot table with inputs of the program
  for (const SlotId slot_id : program.inputs()) {
    if (slot_id.value >= slots.size()) {
      throw std::logic_error("runtime slot id out of range");
    }
    slots[slot_id.value] = bindings.get(slot_id);
  }

  // for each instructions in the program
  for (const Instruction &instruction : program.instructions()) {
    // validate instruction output slot id
    if (instruction.output_slot_id.value >= slots.size()) {
      throw std::logic_error("runtime slot id out of range");
    }
    switch (instruction.opcode) {

    case Opcode::ADD:
      if (instruction.inputs.size() != 2) {
        throw std::logic_error("ADD expects exactly 2 inputs");
      }
      slots[instruction.output_slot_id.value] =
          add(get_slot(slots, instruction.inputs[0]),
              get_slot(slots, instruction.inputs[1]));
      break;
    case Opcode::MUL:
      if (instruction.inputs.size() != 2) {
        throw std::logic_error("MUL expects exactly 2 inputs");
      }
      slots[instruction.output_slot_id.value] =
          mul(get_slot(slots, instruction.inputs[0]),
              get_slot(slots, instruction.inputs[1]));
      break;
    case Opcode::MATMUL:
      if (instruction.inputs.size() != 2) {
        throw std::logic_error("MATMUL expects exactly 2 inputs");
      }
      slots[instruction.output_slot_id.value] =
          matmul(get_slot(slots, instruction.inputs[0]),
                 get_slot(slots, instruction.inputs[1]));
      break;
    case Opcode::SUM:
      if (instruction.inputs.size() != 1) {
        throw std::logic_error("SUM expects exactly 1 input");
      }
      slots[instruction.output_slot_id.value] =
          sum(get_slot(slots, instruction.inputs[0]));
      break;
    }
  }

  // now gather output tensor(s) in a vector and return
  std::vector<Tensor> outputs;

  for (const SlotId slot : program.outputs()) {
    outputs.push_back(get_slot(slots, slot));
  }
  return outputs;
}
} // namespace tcr::runtime