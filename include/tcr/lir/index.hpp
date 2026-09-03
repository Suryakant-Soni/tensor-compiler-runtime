#pragma once
#include "tcr/lir/ids.hpp"
#include <cstddef>
#include <variant>
namespace tcr::lir {
using trc::lir::IndexId;
// indexexpression is either a constant index mentioned in code or the index
// variable

struct ConstantIndex {
  std::size_t value;
};

using IndexExpr = std::variant<IndexId, ConstantIndex>;
} // namespace tcr::lir