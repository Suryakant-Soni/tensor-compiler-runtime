#include <cstddef>
#include <memory>
namespace tcr::ir {

enum class ElementwiseExprKind {
  Input,
  Add,
  Mul,
};

struct ElementwiseExpr {
  ElementwiseExprKind kind;
  std::size_t input_index{};
  std::unique_ptr<ElementwiseExpr> lhs;
  std::unique_ptr<ElementwiseExpr> rhs;
};
} // namespace tcr::ir