#include "tcr/runtime/vm.hpp"
#include "tcr/runtime/opcode.hpp"
#include "tcr/runtime/slot_id.hpp"
#include "tcr/runtime/tensor.hpp"
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tcr::runtime {
using tcr::runtime::Opcode;
namespace {

const Tensor &get_slot_tensor(const std::vector<std::optional<Tensor>> &slots,
                              SlotId slot_id) {
  if (slot_id.value >= slots.size()) {
    throw std::logic_error("runtime slot id out of range");
  }
  if (!slots[slot_id.value].has_value()) {
    throw std::logic_error("attempted to read an empty runtime slot");
  }
  return *slots[slot_id.value];
}
Tensor &get_slot_tensor(std::vector<std::optional<Tensor>> &slots,
                        SlotId slot_id) {
  if (slot_id.value >= slots.size()) {
    throw std::logic_error("runtime slot id out of range");
  }
  if (!slots[slot_id.value].has_value()) {
    throw std::logic_error("attempted to read an empty runtime slot");
  }
  return *slots[slot_id.value];
}

std::size_t resolve_index_operand_value(const IndexOperand &operand,
                                        const std::vector<size_t> &indices) {
  switch (operand.kind) {
  case IndexOperand::Kind::Register:
    if (operand.reg.value >= indices.size()) {
      throw std::out_of_range(
          "index operand value if out of range in stored indices");
    }
    return indices[operand.reg.value];
  case IndexOperand::Kind::Constant:
    return operand.constant;
  }
  throw std::logic_error("invalid runtime index operand");
}
} // namespace

std::vector<Tensor> VM::run(const runtime::Program &rt_program,
                            const InputBindings &bindings) {
  // for run state, create slots/tensor table for tensors, scalar table for
  // managing float values, index table for managing index and their
  // arithmetic, this willbe needful across execting the incoming instructions
  std::vector<std::optional<Tensor>> slots(rt_program.slots_count());
  std::vector<float> scalars(rt_program.scalar_count());
  std::vector<std::size_t> indices(rt_program.index_count());

  // populate tensor values for input slots
  for (const SlotId slot_id : rt_program.inputs()) {
    if (slot_id.value >= slots.size()) {
      throw std::logic_error("runtime slot id out of range");
    }
    slots[slot_id.value] = bindings.get(slot_id);
  }

  // populate tensor for non-input slots
  for (std::size_t i = 0; i < rt_program.slots_count(); ++i) {
    if (slots[i].has_value()) {
      continue;
    }
    const auto &tensor_info = rt_program.tensor_meta_by_slot(SlotId{i});
    std::vector<float> data(tensor_info.shape.num_elements(), 0.0f);
    slots[i].emplace(tensor_info.shape, tensor_info.dtype, std::move(data));
  }

  std::size_t pc = 0;

  while (pc < rt_program.instructions().size()) {
    const auto &rt_instruc = rt_program.instructions()[pc];
    switch (rt_instruc.opcode) {
    case Opcode::Constant: {
      if (rt_instruc.scalar_dst.value >= scalars.size()) {
        throw std::logic_error(
            "given runtime scalar postion to store value into;is out of range");
      }
      scalars[rt_instruc.scalar_dst.value] = rt_instruc.constant;
      pc++;
      break;
    }
    case Opcode::Load: {
      if (rt_instruc.scalar_dst.value >= scalars.size()) {
        throw std::logic_error("given runtime scalar position to load from "
                               "tensor into;is out of range");
      }
      const auto offset =
          resolve_index_operand_value(rt_instruc.offset, indices);
      const auto &tensor = get_slot_tensor(slots, rt_instruc.buffer);
      if (offset >= tensor.data().size()) {
        throw std::logic_error(
            "given runtime tensor load offset is out of range");
      }
      scalars[rt_instruc.scalar_dst.value] = tensor.load(offset);
      pc++;
      break;
    }
    case Opcode::Store: {
      if (rt_instruc.scalar_lhs.value >= scalars.size()) {
        throw std::logic_error(
            "given runtime scalar value position is out of range");
      }
      const auto offset =
          resolve_index_operand_value(rt_instruc.offset, indices);
      Tensor &tensor = get_slot_tensor(slots, rt_instruc.buffer);
      if (offset >= tensor.data().size()) {
        throw std::logic_error("given runtime scalar position to be stored "
                               "into tensor; is out of range");
      }
      tensor.store(offset, scalars[rt_instruc.scalar_lhs.value]);
      pc++;
      break;
    }
    case Opcode::Add: {
      if (rt_instruc.scalar_dst.value >= scalars.size() ||
          rt_instruc.scalar_lhs.value >= scalars.size() ||
          rt_instruc.scalar_rhs.value >= scalars.size()) {
        throw std::logic_error(
            "given runtime scalar positions for add is/are out of range");
      }
      scalars[rt_instruc.scalar_dst.value] =
          scalars[rt_instruc.scalar_lhs.value] +
          scalars[rt_instruc.scalar_rhs.value];
      pc++;
      break;
    }
    case Opcode::Mul: {
      if (rt_instruc.scalar_dst.value >= scalars.size() ||
          rt_instruc.scalar_lhs.value >= scalars.size() ||
          rt_instruc.scalar_rhs.value >= scalars.size()) {
        throw std::logic_error(
            "given runtime scalar positions for mul is/are out of range");
      }
      scalars[rt_instruc.scalar_dst.value] =
          scalars[rt_instruc.scalar_lhs.value] *
          scalars[rt_instruc.scalar_rhs.value];
      pc++;
      break;
    }
    case Opcode::IndexAdd: {
      if (rt_instruc.index_dst.value >= indices.size()) {
        throw std::logic_error("given runtime index destination value for "
                               "index-add,is out of range");
      }
      indices[rt_instruc.index_dst.value] =
          resolve_index_operand_value(rt_instruc.index_lhs, indices) +
          resolve_index_operand_value(rt_instruc.index_rhs, indices);
      pc++;
      break;
    }
    case Opcode::IndexMul: {
      if (rt_instruc.index_dst.value >= indices.size()) {
        throw std::logic_error("given runtime index destination value for "
                               "index-mul,is out of range");
      }
      indices[rt_instruc.index_dst.value] =
          resolve_index_operand_value(rt_instruc.index_lhs, indices) *
          resolve_index_operand_value(rt_instruc.index_rhs, indices);
      pc++;
      break;
    }
    case Opcode::LoopBegin: {
      if (rt_instruc.loop_index.value >= indices.size()) {
        throw std::logic_error(
            "given runtime loop index is out of range for loop begin");
      }
      const auto begin =
          resolve_index_operand_value(rt_instruc.loop_begin, indices);
      const auto end =
          resolve_index_operand_value(rt_instruc.loop_end, indices);
      indices[rt_instruc.loop_index.value] = begin;
      if (begin >= end) {
        pc = rt_instruc.jump;
      } else {
        ++pc;
      }
      break;
    }
    case Opcode::LoopEnd: {
      if (rt_instruc.loop_index.value >= indices.size()) {
        throw std::logic_error(
            "given runtime loop index is out of range for loop end");
      }
      auto &loop_index = indices[rt_instruc.loop_index.value];
      ++loop_index;
      const std::size_t end =
          resolve_index_operand_value(rt_instruc.loop_end, indices);
      if (loop_index < end) {
        // index is not yet finished run body again
        pc = rt_instruc.jump;
      } else {
        pc++;
      }
      break;
    }
    }
  }
  // for each instructions in the program

  // now gather output tensor(s) in a vector and return
  std::vector<Tensor> outputs;
  outputs.reserve(rt_program.outputs().size());

  for (const SlotId slot : rt_program.outputs()) {
    outputs.push_back(get_slot_tensor(slots, slot));
  }
  return outputs;
}
} // namespace tcr::runtime