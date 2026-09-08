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

#include "tcr/runtime/instruction.hpp"
#include "tcr/runtime/slot_id.hpp"
#include <cstddef>
#include <vector>
namespace tcr::runtime {

class Program {
public:
  Program();

  Program(std::vector<Instruction> instructions, std::vector<SlotId> inputs,
          std::vector<SlotId> outputs, std::size_t slots_count,
          std::size_t scalar_count, std::size_t index_count);

  // instructions mutable and immutable getters
  const std::vector<Instruction> &instructions() const;
  std::vector<Instruction> &instructions();
  // add new instruction
  void add_instruction(Instruction instruction);

  const std::vector<SlotId> &inputs() const;
  const std::vector<SlotId> &outputs() const;

  std::size_t slots_count() const;
  std::size_t scalar_count() const;
  std::size_t index_count() const;

  void set_slots_count(std::size_t count);
  void set_scalar_count(std::size_t count);
  void set_index_count(std::size_t count);

private:
  std::vector<Instruction> instructions_;

  std::vector<SlotId> inputs_;
  std::vector<SlotId> outputs_;

  std::size_t slots_count_{0};
  std::size_t scalar_count_{0};
  std::size_t index_count_{0};
};
} // namespace tcr::runtime