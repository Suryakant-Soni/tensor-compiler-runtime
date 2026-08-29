#pragma once

#include "tcr/reference/tensor.hpp"
#include <cstddef>
#include <vector>
namespace tcr::reference {

/**
 * @brief elementwise binary operation with generic template 
 * 
 * @tparam F can be a lambda function
 * @param lhs left tensor operand
 * @param rhs  right tensor operand
 * @param fn  lamda function for element wise operation
 * @return Tensor  output of the elementwise operation
 */
template <typename F>
Tensor elementwise_binary(const Tensor &lhs, const Tensor &rhs, F fn) {
  if (lhs.shape() != rhs.shape()) {
    throw std::invalid_argument("elementwise op requires matching shapes");
  }

  if (lhs.dtype() != rhs.dtype()) {
    throw std::invalid_argument("elementwise op requires matching dtypes");
  }
    // validations are completed when graph is prepared
  const auto &a = lhs.data();
  const auto &b = rhs.data();

  std::vector<float> out(a.size());

  for (std::size_t i = 0; i < a.size(); i++) {
    out[i] = fn(a[i], b[i]);
  }
  return Tensor{
      lhs.shape(),
      lhs.dtype(),
      std::move(out),
  };
}

Tensor add(const Tensor &lhs, const Tensor &rhs);
Tensor mul(const Tensor &lhs, const Tensor &rhs);
Tensor matmul(const Tensor &lhs, const Tensor &rhs);
Tensor sum(const Tensor &input);

} // namespace tcr::reference