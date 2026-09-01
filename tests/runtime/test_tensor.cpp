#include <catch2/catch_test_macros.hpp>

#include "tcr/ir/dtype.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/runtime/tensor.hpp"

#include <stdexcept>
#include <vector>

using tcr::ir::Dtype;
using tcr::ir::Shape;
using tcr::runtime::Tensor;

TEST_CASE("runtime Tensor stores vector shape dtype and data") {
  const Tensor tensor{
      Shape{3},
      Dtype::F32,
      std::vector<float>{1.0F, 2.0F, 3.0F},
  };

  REQUIRE(tensor.shape() == Shape{3});
  REQUIRE(tensor.dtype() == Dtype::F32);
  REQUIRE(tensor.data() == std::vector<float>{1.0F, 2.0F, 3.0F});
}

TEST_CASE("runtime Tensor stores matrix shape dtype and data") {
  const Tensor tensor{
      Shape{2, 3},
      Dtype::F32,
      std::vector<float>{
          1.0F, 2.0F, 3.0F,
          4.0F, 5.0F, 6.0F,
      },
  };

  REQUIRE(tensor.shape() == Shape{2, 3});
  REQUIRE(tensor.shape().rank() == 2);
  REQUIRE(tensor.dtype() == Dtype::F32);

  REQUIRE(tensor.data() ==
          std::vector<float>{
              1.0F, 2.0F, 3.0F,
              4.0F, 5.0F, 6.0F,
          });
}

TEST_CASE("runtime Tensor supports scalar shape") {
  const Tensor tensor{
      Shape{},
      Dtype::F32,
      std::vector<float>{42.0F},
  };

  REQUIRE(tensor.shape() == Shape{});
  REQUIRE(tensor.shape().rank() == 0);
  REQUIRE(tensor.shape().num_elements() == 1);
  REQUIRE(tensor.dtype() == Dtype::F32);
  REQUIRE(tensor.data() == std::vector<float>{42.0F});
}

TEST_CASE("runtime Tensor supports single-element vector") {
  const Tensor tensor{
      Shape{1},
      Dtype::F32,
      std::vector<float>{7.0F},
  };

  REQUIRE(tensor.shape() == Shape{1});
  REQUIRE(tensor.dtype() == Dtype::F32);
  REQUIRE(tensor.data() == std::vector<float>{7.0F});
}

TEST_CASE("runtime Tensor preserves zero values") {
  const Tensor tensor{
      Shape{3},
      Dtype::F32,
      std::vector<float>{0.0F, 0.0F, 0.0F},
  };

  REQUIRE(tensor.data() ==
          std::vector<float>{0.0F, 0.0F, 0.0F});
}

TEST_CASE("runtime Tensor preserves negative values") {
  const Tensor tensor{
      Shape{3},
      Dtype::F32,
      std::vector<float>{-1.0F, -2.0F, -3.0F},
  };

  REQUIRE(tensor.data() ==
          std::vector<float>{-1.0F, -2.0F, -3.0F});
}

TEST_CASE("runtime Tensor rejects too little data for shape") {
  REQUIRE_THROWS_AS(
      (Tensor{
          Shape{3},
          Dtype::F32,
          std::vector<float>{1.0F, 2.0F},
      }),
      std::invalid_argument);
}

TEST_CASE("runtime Tensor rejects too much data for shape") {
  REQUIRE_THROWS_AS(
      (Tensor{
          Shape{2},
          Dtype::F32,
          std::vector<float>{1.0F, 2.0F, 3.0F},
      }),
      std::invalid_argument);
}

TEST_CASE("runtime Tensor rejects empty data for scalar") {
  REQUIRE_THROWS_AS(
      (Tensor{
          Shape{},
          Dtype::F32,
          std::vector<float>{},
      }),
      std::invalid_argument);
}

TEST_CASE("runtime Tensor rejects multiple values for scalar") {
  REQUIRE_THROWS_AS(
      (Tensor{
          Shape{},
          Dtype::F32,
          std::vector<float>{1.0F, 2.0F},
      }),
      std::invalid_argument);
}

TEST_CASE("runtime Tensor rejects incorrect matrix data size") {
  REQUIRE_THROWS_AS(
      (Tensor{
          Shape{2, 3},
          Dtype::F32,
          std::vector<float>{
              1.0F,
              2.0F,
              3.0F,
              4.0F,
              5.0F,
          },
      }),
      std::invalid_argument);
}