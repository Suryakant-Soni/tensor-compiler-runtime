#include "tcr/compiler/program_emitter.hpp"
#include "tcr/compiler/memory_plan.hpp"
#include "tcr/ir/dtype.hpp"
#include "tcr/lir/ids.hpp"
#include "tcr/lir/index.hpp"
#include "tcr/lir/instruction.hpp"
#include "tcr/runtime/instruction.hpp"
#include "tcr/runtime/opcode.hpp"
#include "tcr/runtime/program.hpp"
#include "tcr/runtime/slot_id.hpp"
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace tcr::compiler {
namespace {
runtime::IndexId to_runtime_index_id(lir::IndexId id) {
  return runtime::IndexId{id.value};
}
runtime::ScalarId to_runtime_scalar_id(lir::ScalarId id) {
  return runtime::ScalarId{id.value};
}

runtime::IndexOperand to_runtime_index_operand(const lir::IndexExpr &expr) {
  return std::visit(
      [](const auto &value) -> runtime::IndexOperand {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, lir::IndexId>) {
          return runtime::IndexOperand{
              .kind = runtime::IndexOperand::Kind::Register,
              .reg = runtime::IndexId{value.value}};
        } else {
          return runtime::IndexOperand{
              .kind = runtime::IndexOperand::Kind::Constant,
              .constant = value.value,
          };
        }
      },
      expr);
}

void emit_instruction(const lir::Instruction &lir_instruction,
                      std::vector<runtime::Instruction> &rt_instructions,
                      const std::vector<runtime::SlotId> &buffer_to_slot) {
  std::visit(
      [&](const auto &op) {
        using T = std::decay_t<decltype(op)>;
        if constexpr (std::is_same_v<T, lir::Constant>) {
          rt_instructions.push_back(runtime::Instruction{
              .opcode = runtime::Opcode::Constant,
              .scalar_dst = to_runtime_scalar_id(op.dst),
              .constant = op.value,
          });
        } else if constexpr (std::is_same_v<T, lir::Load>) {
          rt_instructions.push_back(runtime::Instruction{
              .opcode = runtime::Opcode::Load,
              .scalar_dst = to_runtime_scalar_id(op.dst),
              .buffer = buffer_to_slot[op.buffer_src.value],
              .offset = to_runtime_index_operand(op.offset),
          });
        } else if constexpr (std::is_same_v<T, lir::Store>) {
          rt_instructions.push_back(runtime::Instruction{
              .opcode = runtime::Opcode::Store,
              .scalar_lhs = to_runtime_scalar_id(op.value),
              .buffer = buffer_to_slot[op.buffer_dst.value],
              .offset = to_runtime_index_operand(op.offset),
          });
        } else if constexpr (std::is_same_v<T, lir::Add>) {
          rt_instructions.push_back(runtime::Instruction{
              .opcode = runtime::Opcode::Add,
              .scalar_dst = to_runtime_scalar_id(op.dst),
              .scalar_lhs = to_runtime_scalar_id(op.lhs),
              .scalar_rhs = to_runtime_scalar_id(op.rhs),
          });
        } else if constexpr (std::is_same_v<T, lir::Mul>) {
          rt_instructions.push_back(runtime::Instruction{
              .opcode = runtime::Opcode::Mul,
              .scalar_dst = to_runtime_scalar_id(op.dst),
              .scalar_lhs = to_runtime_scalar_id(op.lhs),
              .scalar_rhs = to_runtime_scalar_id(op.rhs),
          });
        } else if constexpr (std::is_same_v<T, lir::IndexAdd>) {
          rt_instructions.push_back(runtime::Instruction{
              .opcode = runtime::Opcode::IndexAdd,
              .index_dst = to_runtime_index_id(op.dst),
              .index_lhs = to_runtime_index_operand(op.lhs),
              .index_rhs = to_runtime_index_operand(op.rhs),
          });
        } else if constexpr (std::is_same_v<T, lir::IndexMul>) {
          rt_instructions.push_back(runtime::Instruction{
              .opcode = runtime::Opcode::IndexMul,
              .index_dst = to_runtime_index_id(op.dst),
              .index_lhs = to_runtime_index_operand(op.lhs),
              .index_rhs = to_runtime_index_operand(op.rhs),
          });
        } else if constexpr (std::is_same_v<T, std::unique_ptr<lir::For>>) {
          const auto begin_pc = rt_instructions.size();

          rt_instructions.push_back(runtime::Instruction{
              .opcode = runtime::Opcode::LoopBegin,
              .loop_index = to_runtime_index_id(op->index),
              .loop_begin = to_runtime_index_operand(op->begin),
              .loop_end = to_runtime_index_operand(op->end),
          });
          const auto body_pc = rt_instructions.size();
          for (const auto &child_inst : op->body) {
            emit_instruction(child_inst, rt_instructions, buffer_to_slot);
          }

          rt_instructions.push_back(runtime::Instruction{
              .opcode = runtime::Opcode::LoopEnd,
              .loop_index = to_runtime_index_id(op->index),
              .loop_end = to_runtime_index_operand(op->end),
              .jump = body_pc,
          });
          const auto next_to_loop_pc = rt_instructions.size();
          rt_instructions[begin_pc].jump = next_to_loop_pc;
        };
      },
      lir_instruction.payload);
} // end emit_instructions
} // namespace

// prepare the complete runtime::Program object and emit it
runtime::Program ProgramEmitter::emit(const lir::Program &lir_program,
                                      const MemoryPlan &memory_plan) {

  std::vector<runtime::Instruction> rt_instructions;
  std::vector<runtime::InputInfo> rt_inputs;
  std::vector<runtime::OutputInfo> rt_outputs;

  // index = bufferid and value = slotid
  std::vector<runtime::SlotId> buffer_to_runtime_slot(
      lir_program.buffer_count());
  std::vector<runtime::SlotMetadata> slots(lir_program.inputs().size() +
                                           memory_plan.slots.size());

  // fill new slots for inputs
  for (std::size_t i = 0; i < lir_program.inputs().size(); i++) {
    const auto &lir_input = lir_program.inputs()[i];
    const auto slot_id = runtime::SlotId{i};
    buffer_to_runtime_slot[lir_input.buffer.value] = slot_id;
    const auto &input_buffer = lir_program.buffers()[lir_input.buffer.value];
    slots[i] = runtime::SlotMetadata{ir::size_in_bytes(input_buffer.dtype) *
                                         input_buffer.shape.num_elements(),
                                     alignof(max_align_t)};
    rt_inputs.push_back(runtime::InputInfo{
        lir_input.value, slot_id, input_buffer.shape, input_buffer.dtype});
  }

  // fill buffer to slot for non input slots
  const auto input_count = lir_program.inputs().size();
  for (std::size_t i = 0; i < memory_plan.buffer_to_slot.size(); i++) {
    const auto &planner_slot = memory_plan.buffer_to_slot[i];
    if (!planner_slot.has_value()) {
      continue;
    }
    const auto runtime_slot = planner_slot->value + input_count;
    buffer_to_runtime_slot[i] = runtime::SlotId{runtime_slot};
    const auto &slot_meta = memory_plan.slots[planner_slot->value];
    slots[runtime_slot] =
        runtime::SlotMetadata{slot_meta.capacity_bytes, slot_meta.alignment};
  }

  for (const auto &lir_output : lir_program.outputs()) {
    const auto &buffer = lir_program.buffers()[lir_output.value];
    rt_outputs.push_back(runtime::OutputInfo{
        buffer_to_runtime_slot[lir_output.value], buffer.shape, buffer.dtype});
  }

  // for populating runtime instructions from lir/lowered instructions
  for (const auto &lir_instruc : lir_program.instructions()) {
    emit_instruction(lir_instruc, rt_instructions, buffer_to_runtime_slot);
  }

  return runtime::Program(
      std::move(rt_instructions), std::move(rt_inputs), std::move(rt_outputs),
      slots, lir_program.scalar_count(), lir_program.index_count());
}

} // namespace tcr::compiler