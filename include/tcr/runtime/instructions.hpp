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
#include <vector>
namespace tcr::runtime {

struct Instructions {
      Opcode opcode;
  std::vector<SlotId> inputs;
  SlotId output_slot_id;
};

} // namespace tcr::runtime