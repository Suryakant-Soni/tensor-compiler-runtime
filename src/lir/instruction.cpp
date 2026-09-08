#include "tcr/lir/instruction.hpp"
#include <memory>
#include <utility>
namespace tcr::lir {

Instruction::Instruction(Constant value) : payload(std::move(value)) {}
Instruction::Instruction(Load value) : payload(std::move(value)) {}
Instruction::Instruction(Store value) : payload(std::move(value)) {}
Instruction::Instruction(Add value) : payload(std::move(value)) {}
Instruction::Instruction(Mul value) : payload(std::move(value)) {}
Instruction::Instruction(std::unique_ptr<For> value)
    : payload(std::move(value)) {}
Instruction::~Instruction() = default;
} // namespace tcr::lir