#include "tcr/reference/tensor.hpp"
#include "tcr/ir/dtype.hpp"
#include "tcr/ir/shape.hpp"

#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <vector>

using namespace tcr::ir;
using namespace tcr::reference;

TEST_CASE("Tensor stores shape dtype and data", "[tensor]") {
  Tensor tensor{
      Shape{2, 3},
      Dtype::F32,
      std::vector<float>{
          1.0f, 2.0f, 3.0f,
          4.0f, 5.0f, 6.0f,
      },
  };

  REQUIRE(tensor.shape() == Shape{2, 3});
  REQUIRE(tensor.dtype() == Dtype::F32);
  REQUIRE(tensor.data() ==
          std::vector<float>{
              1.0f, 2.0f, 3.0f,
              4.0f, 5.0f, 6.0f,
          });
}

TEST_CASE("Tensor supports rank-1 shape", "[tensor]") {
  Tensor tensor{
      Shape{4},
      Dtype::F32,
      std::vector<float>{
          1.0f, 2.0f, 3.0f, 4.0f,
      },
  };

  REQUIRE(tensor.shape() == Shape{4});
  REQUIRE(tensor.shape().rank() == 1);
  REQUIRE(tensor.shape().num_elements() == 4);
  REQUIRE(tensor.data().size() == 4);
}

TEST_CASE("Tensor supports rank-2 shape", "[tensor]") {
  Tensor tensor{
      Shape{2, 2},
      Dtype::F32,
      std::vector<float>{
          1.0f, 2.0f,
          3.0f, 4.0f,
      },
  };

  REQUIRE(tensor.shape() == Shape{2, 2});
  REQUIRE(tensor.shape().rank() == 2);
  REQUIRE(tensor.shape().num_elements() == 4);
  REQUIRE(tensor.data().size() == 4);
}

TEST_CASE("Tensor supports higher-rank shape", "[tensor]") {
  Tensor tensor{
      Shape{2, 2, 2},
      Dtype::F32,
      std::vector<float>{
          1.0f, 2.0f, 3.0f, 4.0f,
          5.0f, 6.0f, 7.0f, 8.0f,
      },
  };

  REQUIRE(tensor.shape() == Shape{2, 2, 2});
  REQUIRE(tensor.shape().rank() == 3);
  REQUIRE(tensor.shape().num_elements() == 8);
  REQUIRE(tensor.data().size() == 8);
}

TEST_CASE("Tensor supports scalar shape", "[tensor]") {
  Tensor tensor{
      Shape{},
      Dtype::F32,
      std::vector<float>{42.0f},
  };

  REQUIRE(tensor.shape() == Shape{});
  REQUIRE(tensor.shape().rank() == 0);
  REQUIRE(tensor.shape().num_elements() == 1);
  REQUIRE(tensor.dtype() == Dtype::F32);
  REQUIRE(tensor.data() == std::vector<float>{42.0f});
}

TEST_CASE("Tensor preserves zero values", "[tensor]") {
  Tensor tensor{
      Shape{3},
      Dtype::F32,
      std::vector<float>{
          0.0f, 0.0f, 0.0f,
      },
  };

  REQUIRE(tensor.data() ==
          std::vector<float>{
              0.0f, 0.0f, 0.0f,
          });
}

TEST_CASE("Tensor preserves negative values", "[tensor]") {
  Tensor tensor{
      Shape{3},
      Dtype::F32,
      std::vector<float>{
          -1.0f, -2.5f, -3.0f,
      },
  };

  REQUIRE(tensor.data() ==
          std::vector<float>{
              -1.0f, -2.5f, -3.0f,
          });
}

TEST_CASE("Tensor preserves mixed positive negative and zero values",
          "[tensor]") {
  Tensor tensor{
      Shape{5},
      Dtype::F32,
      std::vector<float>{
          -2.0f,
          0.0f,
          3.5f,
          -7.25f,
          10.0f,
      },
  };

  REQUIRE(tensor.data() ==
          std::vector<float>{
              -2.0f,
              0.0f,
              3.5f,
              -7.25f,
              10.0f,
          });
}

TEST_CASE("Tensor constructor rejects too little data", "[tensor]") {
  REQUIRE_THROWS_AS(
      (Tensor{
          Shape{2, 3},
          Dtype::F32,
          std::vector<float>{
              1.0f,
              2.0f,
          },
      }),
      std::invalid_argument);
}

TEST_CASE("Tensor constructor rejects too much data", "[tensor]") {
  REQUIRE_THROWS_AS(
      (Tensor{
          Shape{2},
          Dtype::F32,
          std::vector<float>{
              1.0f,
              2.0f,
              3.0f,
          },
      }),
      std::invalid_argument);
}

TEST_CASE("Tensor scalar requires exactly one data element", "[tensor]") {
  SECTION("empty scalar data is rejected") {
    REQUIRE_THROWS_AS(
        (Tensor{
            Shape{},
            Dtype::F32,
            std::vector<float>{},
        }),
        std::invalid_argument);
  }

  SECTION("multiple scalar elements are rejected") {
    REQUIRE_THROWS_AS(
        (Tensor{
            Shape{},
            Dtype::F32,
            std::vector<float>{
                1.0f,
                2.0f,
            },
        }),
        std::invalid_argument);
  }
}

TEST_CASE("Tensor rank-1 shape rejects empty data", "[tensor]") {
  REQUIRE_THROWS_AS(
      (Tensor{
          Shape{1},
          Dtype::F32,
          std::vector<float>{},
      }),
      std::invalid_argument);
}

TEST_CASE("Tensor accessor returns correct dtype", "[tensor]") {
  Tensor tensor{
      Shape{1},
      Dtype::F32,
      std::vector<float>{5.0f},
  };

  REQUIRE(tensor.dtype() == Dtype::F32);
}

TEST_CASE("Tensor data accessor works on const Tensor", "[tensor]") {
  const Tensor tensor{
      Shape{2},
      Dtype::F32,
      std::vector<float>{
          10.0f,
          20.0f,
      },
  };

  const auto& data = tensor.data();

  REQUIRE(data.size() == 2);
  REQUIRE(data[0] == 10.0f);
  REQUIRE(data[1] == 20.0f);
}

TEST_CASE("Tensor shape accessor works on const Tensor", "[tensor]") {
  const Tensor tensor{
      Shape{2, 2},
      Dtype::F32,
      std::vector<float>{
          1.0f, 2.0f,
          3.0f, 4.0f,
      },
  };

  const Shape& shape = tensor.shape();

  REQUIRE(shape == Shape{2, 2});
  REQUIRE(shape.rank() == 2);
  REQUIRE(shape.dim(0) == 2);
  REQUIRE(shape.dim(1) == 2);
}

TEST_CASE("Tensor data size always matches shape element count", "[tensor]") {
  Tensor tensor{
      Shape{2, 3, 2},
      Dtype::F32,
      std::vector<float>{
          1.0f, 2.0f, 3.0f, 4.0f,
          5.0f, 6.0f, 7.0f, 8.0f,
          9.0f, 10.0f, 11.0f, 12.0f,
      },
  };

  REQUIRE(tensor.data().size() == tensor.shape().num_elements());
}