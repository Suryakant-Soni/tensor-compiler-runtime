#pragma once
#include <string_view>
namespace tcr::ir {
enum class Op {
  Input,
  Add,
  Mul,
  Sum,
  MatMul,

  FusedElementwise
};
std::string_view to_string(Op);
}