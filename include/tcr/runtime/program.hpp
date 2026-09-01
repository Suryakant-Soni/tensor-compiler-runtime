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
  Program(std::vector<Instruction> instructions, std::vector<SlotId> inputs,
          std::vector<SlotId> outputs, std::size_t num_slots);
  const std::vector<Instruction> &instructions() const;
  const std::vector<SlotId> &inputs() const;
  const std::vector<SlotId> &outputs() const;
  std::size_t slots_count() const;

private:
  std::vector<Instruction> instructions_;
  std::vector<SlotId> inputs_;
  std::vector<SlotId> outputs_;
  std::size_t slots_count_;
};
} // namespace tcr::runtime