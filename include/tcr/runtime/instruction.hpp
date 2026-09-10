#pragma once
/**
 * @file instructions.hpp
 * @author your name (you@domain.com)
 * @brief instructions to be prepared by compiler for VM, it will contain fields
like inputs - input slotids for inputs to instructions
opcode - operation to be performed  and outputs - the operation result will be
moved to those slotids
 * @version 0.1
 * @date 2026-08-30
 *
 * @copyright Copyright (c) 2026
 */

#include "tcr/runtime/opcode.hpp"
#include "tcr/runtime/slot_id.hpp"
#include <cstddef>
namespace tcr::runtime {
struct ScalarId {
  std::size_t value{};
};

struct IndexId {
  std::size_t value{};
};

struct IndexOperand {
  enum class Kind {
    Register,
    Constant,
  };
  // default value is Register
  Kind kind{Kind::Register};
  IndexId reg{};
  std::size_t constant{0};
};

struct Instruction {
  Opcode opcode;

  // scalar/F32 arithmetic
  ScalarId scalar_dst{};
  ScalarId scalar_lhs{};
  ScalarId scalar_rhs{};

  // index/integer arithmetic
  IndexId index_dst{};
  IndexOperand index_lhs{};
  IndexOperand index_rhs{};

  // memory access
  SlotId buffer{};
  IndexOperand offset{};

  // scalar constant
  float constant{0.0F};

  // Loop Constrol
  IndexId loop_index{};
  IndexOperand loop_begin{};
  IndexOperand loop_end{};

  // instruction index used for control-flow jumps
  std::size_t jump{};
};

} // namespace tcr::runtime