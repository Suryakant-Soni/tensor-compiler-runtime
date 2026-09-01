#include <catch2/catch_test_macros.hpp>

#include "tcr/ir/dtype.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/runtime/ops.hpp"
#include "tcr/runtime/tensor.hpp"

#include <stdexcept>
#include <vector>

using tcr::ir::Dtype;
using tcr::ir::Shape;
using tcr::runtime::add;
using tcr::runtime::matmul;
using tcr::runtime::mul;
using tcr::runtime::sum;
using tcr::runtime::Tensor;

// -----------------------------------------------------------------------------
// Add
// -----------------------------------------------------------------------------

TEST_CASE("runtime add adds two vectors elementwise") {
  const Tensor lhs{
      Shape{3},
      Dtype::F32,
      std::vector<float>{1.0F, 2.0F, 3.0F},
  };

  const Tensor rhs{
      Shape{3},
      Dtype::F32,
      std::vector<float>{4.0F, 5.0F, 6.0F},
  };

  const Tensor result = add(lhs, rhs);

  REQUIRE(result.shape() == Shape{3});
  REQUIRE(result.dtype() == Dtype::F32);
  REQUIRE(result.data() == std::vector<float>{5.0F, 7.0F, 9.0F});
}

TEST_CASE("runtime add adds matrices elementwise") {
  const Tensor lhs{
      Shape{2, 2},
      Dtype::F32,
      std::vector<float>{
          1.0F,
          2.0F,
          3.0F,
          4.0F,
      },
  };

  const Tensor rhs{
      Shape{2, 2},
      Dtype::F32,
      std::vector<float>{
          5.0F,
          6.0F,
          7.0F,
          8.0F,
      },
  };

  const Tensor result = add(lhs, rhs);

  REQUIRE(result.shape() == Shape{2, 2});
  REQUIRE(result.dtype() == Dtype::F32);

  REQUIRE(result.data() == std::vector<float>{
                               6.0F,
                               8.0F,
                               10.0F,
                               12.0F,
                           });
}

TEST_CASE("runtime add supports scalar tensors") {
  const Tensor lhs{
      Shape{},
      Dtype::F32,
      std::vector<float>{2.0F},
  };

  const Tensor rhs{
      Shape{},
      Dtype::F32,
      std::vector<float>{3.0F},
  };

  const Tensor result = add(lhs, rhs);

  REQUIRE(result.shape() == Shape{});
  REQUIRE(result.dtype() == Dtype::F32);
  REQUIRE(result.data() == std::vector<float>{5.0F});
}

TEST_CASE("runtime add handles negative values") {
  const Tensor lhs{
      Shape{3},
      Dtype::F32,
      std::vector<float>{-1.0F, 2.0F, -3.0F},
  };

  const Tensor rhs{
      Shape{3},
      Dtype::F32,
      std::vector<float>{4.0F, -5.0F, 6.0F},
  };

  const Tensor result = add(lhs, rhs);

  REQUIRE(result.data() == std::vector<float>{3.0F, -3.0F, 3.0F});
}

TEST_CASE("runtime add rejects mismatching shapes") {
  const Tensor lhs{
      Shape{2},
      Dtype::F32,
      std::vector<float>{1.0F, 2.0F},
  };

  const Tensor rhs{
      Shape{1, 2},
      Dtype::F32,
      std::vector<float>{3.0F, 4.0F},
  };

  REQUIRE_THROWS_AS(add(lhs, rhs), std::invalid_argument);
}

// -----------------------------------------------------------------------------
// Mul
// -----------------------------------------------------------------------------

TEST_CASE("runtime mul multiplies two vectors elementwise") {
  const Tensor lhs{
      Shape{3},
      Dtype::F32,
      std::vector<float>{1.0F, 2.0F, 3.0F},
  };

  const Tensor rhs{
      Shape{3},
      Dtype::F32,
      std::vector<float>{4.0F, 5.0F, 6.0F},
  };

  const Tensor result = mul(lhs, rhs);

  REQUIRE(result.shape() == Shape{3});
  REQUIRE(result.dtype() == Dtype::F32);
  REQUIRE(result.data() == std::vector<float>{4.0F, 10.0F, 18.0F});
}

TEST_CASE("runtime mul multiplies matrices elementwise") {
  const Tensor lhs{
      Shape{2, 2},
      Dtype::F32,
      std::vector<float>{
          1.0F,
          2.0F,
          3.0F,
          4.0F,
      },
  };

  const Tensor rhs{
      Shape{2, 2},
      Dtype::F32,
      std::vector<float>{
          5.0F,
          6.0F,
          7.0F,
          8.0F,
      },
  };

  const Tensor result = mul(lhs, rhs);

  REQUIRE(result.shape() == Shape{2, 2});

  REQUIRE(result.data() == std::vector<float>{
                               5.0F,
                               12.0F,
                               21.0F,
                               32.0F,
                           });
}

TEST_CASE("runtime mul supports scalar tensors") {
  const Tensor lhs{
      Shape{},
      Dtype::F32,
      std::vector<float>{4.0F},
  };

  const Tensor rhs{
      Shape{},
      Dtype::F32,
      std::vector<float>{5.0F},
  };

  const Tensor result = mul(lhs, rhs);

  REQUIRE(result.shape() == Shape{});
  REQUIRE(result.data() == std::vector<float>{20.0F});
}

TEST_CASE("runtime mul handles zero and negative values") {
  const Tensor lhs{
      Shape{3},
      Dtype::F32,
      std::vector<float>{0.0F, -2.0F, -3.0F},
  };

  const Tensor rhs{
      Shape{3},
      Dtype::F32,
      std::vector<float>{10.0F, 4.0F, -5.0F},
  };

  const Tensor result = mul(lhs, rhs);

  REQUIRE(result.data() == std::vector<float>{0.0F, -8.0F, 15.0F});
}

TEST_CASE("runtime mul rejects mismatching shapes") {
  const Tensor lhs{
      Shape{2},
      Dtype::F32,
      std::vector<float>{1.0F, 2.0F},
  };

  const Tensor rhs{
      Shape{1, 2},
      Dtype::F32,
      std::vector<float>{3.0F, 4.0F},
  };

  REQUIRE_THROWS_AS(mul(lhs, rhs), std::invalid_argument);
}

// -----------------------------------------------------------------------------
// Sum
// -----------------------------------------------------------------------------

TEST_CASE("runtime sum reduces vector to scalar") {
  const Tensor input{
      Shape{4},
      Dtype::F32,
      std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F},
  };

  const Tensor result = sum(input);

  REQUIRE(result.shape() == Shape{});
  REQUIRE(result.shape().rank() == 0);
  REQUIRE(result.dtype() == Dtype::F32);
  REQUIRE(result.data() == std::vector<float>{10.0F});
}

TEST_CASE("runtime sum reduces matrix to scalar") {
  const Tensor input{
      Shape{2, 3},
      Dtype::F32,
      std::vector<float>{
          1.0F,
          2.0F,
          3.0F,
          4.0F,
          5.0F,
          6.0F,
      },
  };

  const Tensor result = sum(input);

  REQUIRE(result.shape() == Shape{});
  REQUIRE(result.data() == std::vector<float>{21.0F});
}

TEST_CASE("runtime sum handles negative values") {
  const Tensor input{
      Shape{4},
      Dtype::F32,
      std::vector<float>{
          -1.0F,
          2.0F,
          -3.0F,
          4.0F,
      },
  };

  const Tensor result = sum(input);

  REQUIRE(result.data() == std::vector<float>{2.0F});
}

TEST_CASE("runtime sum handles all zero values") {
  const Tensor input{
      Shape{3},
      Dtype::F32,
      std::vector<float>{0.0F, 0.0F, 0.0F},
  };

  const Tensor result = sum(input);

  REQUIRE(result.shape() == Shape{});
  REQUIRE(result.data() == std::vector<float>{0.0F});
}

TEST_CASE("runtime sum of scalar remains scalar") {
  const Tensor input{
      Shape{},
      Dtype::F32,
      std::vector<float>{7.0F},
  };

  const Tensor result = sum(input);

  REQUIRE(result.shape() == Shape{});
  REQUIRE(result.data() == std::vector<float>{7.0F});
}

// -----------------------------------------------------------------------------
// MatMul
// -----------------------------------------------------------------------------

TEST_CASE("runtime matmul multiplies two 2x2 matrices") {
  const Tensor lhs{
      Shape{2, 2},
      Dtype::F32,
      std::vector<float>{
          1.0F,
          2.0F,
          3.0F,
          4.0F,
      },
  };

  const Tensor rhs{
      Shape{2, 2},
      Dtype::F32,
      std::vector<float>{
          5.0F,
          6.0F,
          7.0F,
          8.0F,
      },
  };

  const Tensor result = matmul(lhs, rhs);

  REQUIRE(result.shape() == Shape{2, 2});
  REQUIRE(result.dtype() == Dtype::F32);

  REQUIRE(result.data() == std::vector<float>{
                               19.0F,
                               22.0F,
                               43.0F,
                               50.0F,
                           });
}

TEST_CASE("runtime matmul supports rectangular matrices") {
  // 2x3
  const Tensor lhs{
      Shape{2, 3},
      Dtype::F32,
      std::vector<float>{
          1.0F,
          2.0F,
          3.0F,
          4.0F,
          5.0F,
          6.0F,
      },
  };

  // 3x2
  const Tensor rhs{
      Shape{3, 2},
      Dtype::F32,
      std::vector<float>{
          7.0F,
          8.0F,
          9.0F,
          10.0F,
          11.0F,
          12.0F,
      },
  };

  const Tensor result = matmul(lhs, rhs);

  REQUIRE(result.shape() == Shape{2, 2});

  REQUIRE(result.data() == std::vector<float>{
                               58.0F,
                               64.0F,
                               139.0F,
                               154.0F,
                           });
}

TEST_CASE("runtime matmul supports 1x1 matrices") {
  const Tensor lhs{
      Shape{1, 1},
      Dtype::F32,
      std::vector<float>{3.0F},
  };

  const Tensor rhs{
      Shape{1, 1},
      Dtype::F32,
      std::vector<float>{4.0F},
  };

  const Tensor result = matmul(lhs, rhs);

  REQUIRE(result.shape() == Shape{1, 1});
  REQUIRE(result.data() == std::vector<float>{12.0F});
}

TEST_CASE("runtime matmul handles zero values") {
  const Tensor lhs{
      Shape{2, 2},
      Dtype::F32,
      std::vector<float>{
          0.0F,
          0.0F,
          0.0F,
          0.0F,
      },
  };

  const Tensor rhs{
      Shape{2, 2},
      Dtype::F32,
      std::vector<float>{
          1.0F,
          2.0F,
          3.0F,
          4.0F,
      },
  };

  const Tensor result = matmul(lhs, rhs);

  REQUIRE(result.data() == std::vector<float>{
                               0.0F,
                               0.0F,
                               0.0F,
                               0.0F,
                           });
}

TEST_CASE("runtime matmul handles negative values") {
  const Tensor lhs{
      Shape{1, 2},
      Dtype::F32,
      std::vector<float>{-1.0F, 2.0F},
  };

  const Tensor rhs{
      Shape{2, 1},
      Dtype::F32,
      std::vector<float>{3.0F, -4.0F},
  };

  const Tensor result = matmul(lhs, rhs);

  // (-1 * 3) + (2 * -4) = -11
  REQUIRE(result.shape() == Shape{1, 1});
  REQUIRE(result.data() == std::vector<float>{-11.0F});
}

// -----------------------------------------------------------------------------
// MatMul negative cases
// -----------------------------------------------------------------------------

TEST_CASE("runtime matmul rejects rank-1 lhs") {
  const Tensor lhs{
      Shape{2},
      Dtype::F32,
      std::vector<float>{1.0F, 2.0F},
  };

  const Tensor rhs{
      Shape{2, 2},
      Dtype::F32,
      std::vector<float>{
          1.0F,
          2.0F,
          3.0F,
          4.0F,
      },
  };

  REQUIRE_THROWS_AS(matmul(lhs, rhs), std::invalid_argument);
}

TEST_CASE("runtime matmul rejects rank-1 rhs") {
  const Tensor lhs{
      Shape{2, 2},
      Dtype::F32,
      std::vector<float>{
          1.0F,
          2.0F,
          3.0F,
          4.0F,
      },
  };

  const Tensor rhs{
      Shape{2},
      Dtype::F32,
      std::vector<float>{1.0F, 2.0F},
  };

  REQUIRE_THROWS_AS(matmul(lhs, rhs), std::invalid_argument);
}

TEST_CASE("runtime matmul rejects scalar lhs") {
  const Tensor lhs{
      Shape{},
      Dtype::F32,
      std::vector<float>{1.0F},
  };

  const Tensor rhs{
      Shape{1, 1},
      Dtype::F32,
      std::vector<float>{2.0F},
  };

  REQUIRE_THROWS_AS(matmul(lhs, rhs), std::invalid_argument);
}

TEST_CASE("runtime matmul rejects scalar rhs") {
  const Tensor lhs{
      Shape{1, 1},
      Dtype::F32,
      std::vector<float>{1.0F},
  };

  const Tensor rhs{
      Shape{},
      Dtype::F32,
      std::vector<float>{2.0F},
  };

  REQUIRE_THROWS_AS(matmul(lhs, rhs), std::invalid_argument);
}

TEST_CASE("runtime matmul rejects incompatible inner dimensions") {
  const Tensor lhs{
      Shape{2, 3},
      Dtype::F32,
      std::vector<float>{
          1.0F,
          2.0F,
          3.0F,
          4.0F,
          5.0F,
          6.0F,
      },
  };

  const Tensor rhs{
      Shape{2, 2},
      Dtype::F32,
      std::vector<float>{
          1.0F,
          2.0F,
          3.0F,
          4.0F,
      },
  };

  REQUIRE_THROWS_AS(matmul(lhs, rhs), std::invalid_argument);
}