#include "tcr/ir/op.hpp"
#include <cassert>
#include <string_view>
namespace tcr::ir {

std::string_view to_string(Op operation) {
  switch (operation) {
  case Op::Input:
    return "Input";
  case Op::Add:
    return "Add";
  case Op::Mul:
    return "Mul";
  case Op::MatMul:
    return "MatMul";
  case Op::Sum:
    return "Sum";
  default:
    assert(false && "unhandled Operation in to_string");
    return "Unknown";
  }
}
} // namespace tcr::ir