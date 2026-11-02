#pragma once
/**
 * @file program.hpp
 * @author your name (you@domain.com)
 * @brief program .hpp contains fields - instructions vector, inputs and outputs
 * in forms of slotid
 * @version 0.1
 * @date 2026-08-31
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "tcr/ir/dtype.hpp"
#include "tcr/ir/ids.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/runtime/instruction.hpp"
#include "tcr/runtime/slot_id.hpp"
#include <cstddef>
#include <vector>
namespace tcr::runtime {

struct SlotTensorInfo {
  SlotId slot_id;
  ir::Shape shape;
  ir::Dtype dtype;
};

struct ValueIdSlotIdInput {
  ir::ValueId value_id;
  SlotId slot_id;
};

class Program {
public:
  Program();

  Program(std::vector<Instruction> instructions,
          std::vector<ValueIdSlotIdInput> inputs, std::vector<SlotId> outputs,
          std::vector<SlotTensorInfo> slots, std::size_t scalar_count,
          std::size_t index_count);

  // instructions mutable and immutable getters
  const std::vector<Instruction> &instructions() const;
  std::vector<Instruction> &instructions();
  // add new instruction
  void add_instruction(Instruction instruction);

  const std::vector<ValueIdSlotIdInput> &inputs() const;
  const std::vector<SlotId> &outputs() const;

  std::size_t slots_count() const;
  std::size_t scalar_count() const;
  std::size_t index_count() const;
  const SlotTensorInfo &tensor_meta_by_slot(SlotId slot_id) const;
  SlotId get_slotid_from_valueid(ir::ValueId value_id) const;
  bool is_input_slot(SlotId input) const;

private:
  std::vector<Instruction> instructions_;

  std::vector<ValueIdSlotIdInput> inputs_;
  std::vector<SlotId> outputs_;

  std::vector<SlotTensorInfo> slots_;

  std::size_t scalar_count_{0};
  std::size_t index_count_{0};
};
} // namespace tcr::runtime