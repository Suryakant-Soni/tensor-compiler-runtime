
#include <string_view>
namespace tcr::ir {
enum class Op {
  Input,
  Add,
  Mul,
  Sum,
  MatMul,
};
std::string_view to_string(Op);
}