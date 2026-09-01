#pragma once

#include "tcr/runtime/tensor.hpp"
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>
namespace tcr::runtime {

template <typename F>
Tensor elementwise_binary(const Tensor &lhs, const Tensor &rhs, F fn) {
  if (lhs.shape() != rhs.shape()) {
    throw std::invalid_argument("elementwise ops require matching shape");
  }

  if (lhs.dtype() != rhs.dtype()) {
    throw std::invalid_argument("elementwise ops require matching dtypes");
  }

  const auto &a = lhs.data();
  const auto &b = rhs.data();

  std::vector<float> out(a.size());

  for (std::size_t i = 0; i < a.size(); ++i) {
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
} // namespace tcr::runtime