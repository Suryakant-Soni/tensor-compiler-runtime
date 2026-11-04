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

struct SlotMetadata {
  std::size_t capacity;
  std::size_t alignment;
};

struct InputInfo {
  ir::ValueId value_id;
  SlotId slot_id;
  ir::Shape shape;
  ir::Dtype dtype;
};

struct OutputInfo {
  SlotId slot_id;
  ir::Shape shape;
  ir::Dtype dtype;
};

class Program {
public:
  Program();

  Program(std::vector<Instruction> instructions, std::vector<InputInfo> inputs,
          std::vector<OutputInfo> outputs, std::vector<SlotMetadata> slots,
          std::size_t scalar_count, std::size_t index_count);

  // instructions mutable and immutable getters
  const std::vector<Instruction> &instructions() const;
  std::vector<Instruction> &instructions();
  // add new instruction
  void add_instruction(Instruction instruction);

  const std::vector<InputInfo> &inputs() const;
  const std::vector<OutputInfo> &outputs() const;

  std::size_t slots_count() const;
  std::size_t scalar_count() const;
  std::size_t index_count() const;
  const SlotMetadata &get_slot_metadata(SlotId slot_id) const;
  SlotId get_slotid_from_valueid(ir::ValueId value_id) const;
  bool is_input_slot(SlotId input) const;

private:
  std::vector<Instruction> instructions_;

  std::vector<InputInfo> inputs_;
  std::vector<OutputInfo> outputs_;

  std::vector<SlotMetadata> slots_;

  std::size_t scalar_count_{0};
  std::size_t index_count_{0};
};
} // namespace tcr::runtime