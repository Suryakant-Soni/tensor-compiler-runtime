#include "tcr/lir/instruction.hpp"

#include <utility>

namespace tcr::lir {

Instruction::Instruction(Add value) : payload(std::move(value)) {}

Instruction::Instruction(Load value) : payload(std::move(value)) {}

Instruction::Instruction(Store value) : payload(std::move(value)) {}

Instruction::Instruction(Mul value) : payload(std::move(value)) {}

Instruction::Instruction(Constant value) : payload(std::move(value)) {}

Instruction::Instruction(IndexAdd value) : payload(std::move(value)) {}

Instruction::Instruction(IndexMul value) : payload(std::move(value)) {}

Instruction::Instruction(std::unique_ptr<For> value)
    : payload(std::move(value)) {}

Instruction::~Instruction() = default;

} // namespace tcr::lir