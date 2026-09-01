#include "tcr/reference/ops.hpp"
#include "tcr/reference/tensor.hpp"
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>
namespace tcr::reference {

Tensor add(const Tensor &lhs, const Tensor &rhs) {
  return elementwise_binary(lhs, rhs, [](float a, float b) { return a + b; });
}

Tensor mul(const Tensor &lhs, const Tensor &rhs) {
  return elementwise_binary(lhs, rhs, [](float a, float b) { return a * b; });
}

Tensor sum(const Tensor &input) {
  float result = 0.0f;
  for (float x : input.data()) {
    result += x;
  }
  return Tensor{
      Shape{},
      input.dtype(),
      std::vector<float>{result},
  };
}

Tensor matmul(const Tensor &lhs, const Tensor &rhs) {
  if (lhs.shape().rank() != 2 || rhs.shape().rank() != 2) {
    throw std::invalid_argument("matmul requires rank-2 tensors");
  }

  if (lhs.shape().dim(1) != rhs.shape().dim(0)) {
    throw std::invalid_argument("matmul inner dimensions must match");
  }

  if (lhs.dtype() != rhs.dtype()) {
    throw std::invalid_argument("matmul requires matching dtypes");
  }
  const auto &a = lhs.data();
  const auto &b = rhs.data();
  const auto M = lhs.shape().dim(0);
  const auto K = lhs.shape().dim(1);
  const auto N = rhs.shape().dim(1);
  std::vector<float> c(static_cast<size_t>(M * N), 0.0f);

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t k = 0; k < K; ++k) {
      const float a_ik = a[i * K + k];

      for (std::size_t j = 0; j < N; ++j) {
        c[i * N + j] += a_ik * b[k * N + j];
      }
    }
  }
  return Tensor{
      Shape{M, N},
      lhs.dtype(),
      std::move(c),
  };
}
} // namespace tcr::reference