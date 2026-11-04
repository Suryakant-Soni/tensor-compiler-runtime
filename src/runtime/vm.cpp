#include "tcr/runtime/vm.hpp"
#include "tcr/ir/dtype.hpp"
#include "tcr/runtime/opcode.hpp"
#include "tcr/runtime/slot_id.hpp"
#include "tcr/runtime/storage.hpp"
#include "tcr/runtime/tensor.hpp"
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace tcr::runtime {
namespace {
std::size_t checked_add(std::size_t lhs, std::size_t rhs) {
  if (lhs > std::numeric_limits<std::size_t>::max() - rhs) {
    throw std::overflow_error("runtime index addition overflow error");
  }
  return lhs + rhs;
}

std::size_t checked_mul(std::size_t lhs, std::size_t rhs) {
  if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
    throw std::overflow_error("runtime index multiplication overflow error");
  }
  return lhs * rhs;
}
}; // namespace
using tcr::runtime::Opcode;
namespace {

const RuntimeStorage &
get_slot_storage(const std::vector<std::shared_ptr<RuntimeStorage>> &slots,
                 SlotId slot_id) {
  if (slot_id.value >= slots.size()) {
    throw std::logic_error("runtime slot id out of range");
  }
  if (!slots[slot_id.value]) {
    throw std::logic_error("attempted to read an unallocated runtime slot");
  }
  return *slots[slot_id.value];
}
RuntimeStorage &
get_slot_storage(std::vector<std::shared_ptr<RuntimeStorage>> &slots,
                 SlotId slot_id) {
  if (slot_id.value >= slots.size()) {
    throw std::logic_error("runtime slot id out of range");
  }
  if (!slots[slot_id.value]) {
    throw std::logic_error("attempted to read an unallocated runtime slot");
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
  // arithmetic, this willbe needful across executing the incoming instructions
  std::vector<std::shared_ptr<RuntimeStorage>> slots(rt_program.slots_count());
  std::vector<float> scalars(rt_program.scalar_count());
  std::vector<std::size_t> indices(rt_program.index_count());

  // populate runtime storage for input slots
  for (const auto &input_info : rt_program.inputs()) {
    if (input_info.slot_id.value >= slots.size()) {
      throw std::logic_error("runtime slot id out of range");
    }
    const auto &actual_tensor = bindings.get(input_info.slot_id);

    if (actual_tensor.shape() != input_info.shape) {
      throw std::invalid_argument(
          "input tensor shape does not match program input");
    }
    if (actual_tensor.dtype() != input_info.dtype) {
      throw std::invalid_argument(
          "input tensor dtype does not match program input");
    }
    if (actual_tensor.storage()->bytes.size() <
        (actual_tensor.shape().num_elements() *
         ir::size_in_bytes(actual_tensor.dtype()))) {
      throw std::logic_error(
          "storage of slot is smaller than what shape defines");
    }
    slots[input_info.slot_id.value] = actual_tensor.storage();
  }

  // populate tensor for non-input slots
  for (std::size_t i = 0; i < rt_program.slots_count(); ++i) {
    // check if the slots shared ptr points to the already allocated memory
    if (slots[i]) {
      continue;
    }
    const auto &storage_meta = rt_program.get_slot_metadata(SlotId{i});
    slots[i] = std::make_shared<RuntimeStorage>();
    slots[i]->bytes.resize(storage_meta.capacity);
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
      const auto &storage = get_slot_storage(slots, rt_instruc.buffer);
      if (offset * sizeof(float) >= storage.bytes.size()) {
        throw std::logic_error(
            "given runtime tensor load offset is out of range");
      }
      scalars[rt_instruc.scalar_dst.value] = storage.load_f32(offset);
      pc++;
      break;
    }
    case Opcode::Store: {
      if (rt_instruc.scalar_lhs.value >= scalars.size()) {
        throw std::logic_error(
            "given runtime scalar value offset is greater than storage length");
      }
      if (rt_program.is_input_slot(rt_instruc.buffer)) {
        throw std::logic_error("Store is trying to mutate input tensor");
      }
      const auto offset =
          resolve_index_operand_value(rt_instruc.offset, indices);
      auto &storage = get_slot_storage(slots, rt_instruc.buffer);
      if (offset * sizeof(float) >= storage.bytes.size()) {
        throw std::logic_error(
            "given runtime scalar value offset is greater than storage length");
      }
      storage.store_f32(offset, scalars[rt_instruc.scalar_lhs.value]);
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
      indices[rt_instruc.index_dst.value] = checked_add(
          resolve_index_operand_value(rt_instruc.index_lhs, indices),
          resolve_index_operand_value(rt_instruc.index_rhs, indices));
      pc++;
      break;
    }
    case Opcode::IndexMul: {
      if (rt_instruc.index_dst.value >= indices.size()) {
        throw std::logic_error("given runtime index destination value for "
                               "index-mul,is out of range");
      }
      indices[rt_instruc.index_dst.value] = checked_mul(
          resolve_index_operand_value(rt_instruc.index_lhs, indices),
          resolve_index_operand_value(rt_instruc.index_rhs, indices));
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
      loop_index = checked_add(loop_index, 1);
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

  for (const auto &output : rt_program.outputs()) {
    if (output.slot_id.value >= slots.size() || !slots[output.slot_id.value]) {
      throw std::logic_error("invalid output slot");
    }

    outputs.emplace_back(output.shape, output.dtype,
                         slots[output.slot_id.value]);
  }
  return outputs;
}
} // namespace tcr::runtime