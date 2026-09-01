#include "tcr/runtime/vm.hpp"
#include "tcr/runtime/bindings.hpp"
#include "tcr/runtime/instruction.hpp"
#include "tcr/runtime/ops.hpp"
#include "tcr/runtime/program.hpp"
#include "tcr/runtime/slot_id.hpp"
#include "tcr/runtime/tensor.hpp"
#include <optional>
#include <vector>
namespace tcr::runtime {
std::vector<Tensor> VM::run(const Program &program,
                            const InputBindings &bindings) {
  // first create a slot table and add all inputs to it
  std::vector<std::optional<Tensor>> slots(program.slots_count());

  // initialize slot table with inputs of the program
  for (const SlotId slot_id : program.inputs()) {
    slots[slot_id.value] = bindings.get(slot_id);
  }

  // for each instructions in the program
  for (const Instruction &instruction : program.instructions()) {
    switch (instruction.opcode) {

    case Opcode::ADD:
      slots[instruction.output_slot_id.value] =
          add(*slots[instruction.inputs[0].value],
              *slots[instruction.inputs[1].value]);
      break;
    case Opcode::MUL:
      slots[instruction.output_slot_id.value] =
          mul(*slots[instruction.inputs[0].value],
              *slots[instruction.inputs[1].value]);
      break;
    case Opcode::MATMUL:
      slots[instruction.output_slot_id.value] =
          matmul(*slots[instruction.inputs[0].value],
                 *slots[instruction.inputs[1].value]);
      break;
    case Opcode::SUM:
      slots[instruction.output_slot_id.value] =
          sum(*slots[instruction.inputs[0].value]);
      break;
    }
  }

  // now gather output tensor(s) in a vector and return
  std::vector<Tensor> outputs;

  for (const SlotId slot : program.outputs()) {
    outputs.push_back(*slots[slot.value]);
  }
  return outputs;
}
} // namespace tcr::runtime