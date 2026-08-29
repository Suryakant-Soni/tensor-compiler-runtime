#include "tcr/ir/dtype.hpp"
#include "tcr/ir/ids.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/reference/bindings.hpp"
#include "tcr/reference/tensor.hpp"

#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <vector>

using namespace tcr::ir;
using namespace tcr::reference;

TEST_CASE("bind stores a tensor for a ValueId", "[bindings]") {
  InputBindings bindings;

  ValueId id{0};

  Tensor tensor{
      Shape{2},
      Dtype::F32,
      std::vector<float>{1.0f, 2.0f},
  };

  bindings.bind(id, tensor);

  const Tensor &result = bindings.get(id);

  REQUIRE(result.shape() == Shape{2});
  REQUIRE(result.dtype() == Dtype::F32);
  REQUIRE(result.data() == std::vector<float>{1.0f, 2.0f});
}

TEST_CASE("multiple ValueIds can have independent bindings", "[bindings]") {
  InputBindings bindings;

  ValueId a{0};
  ValueId b{1};

  bindings.bind(a, Tensor{
                       Shape{2},
                       Dtype::F32,
                       std::vector<float>{1.0f, 2.0f},
                   });

  bindings.bind(b, Tensor{
                       Shape{2},
                       Dtype::F32,
                       std::vector<float>{3.0f, 4.0f},
                   });

  REQUIRE(bindings.get(a).data() == std::vector<float>{1.0f, 2.0f});

  REQUIRE(bindings.get(b).data() == std::vector<float>{3.0f, 4.0f});
}

TEST_CASE("binding the same ValueId replaces the previous tensor",
          "[bindings]") {
  InputBindings bindings;

  ValueId id{0};

  bindings.bind(id, Tensor{
                        Shape{2},
                        Dtype::F32,
                        std::vector<float>{1.0f, 2.0f},
                    });

  bindings.bind(id, Tensor{
                        Shape{3},
                        Dtype::F32,
                        std::vector<float>{10.0f, 20.0f, 30.0f},
                    });

  const Tensor &result = bindings.get(id);

  REQUIRE(result.shape() == Shape{3});
  REQUIRE(result.dtype() == Dtype::F32);
  REQUIRE(result.data() == std::vector<float>{10.0f, 20.0f, 30.0f});
}

TEST_CASE("replacing one binding does not modify other bindings",
          "[bindings]") {
  InputBindings bindings;

  ValueId a{0};
  ValueId b{1};

  bindings.bind(a, Tensor{
                       Shape{2},
                       Dtype::F32,
                       std::vector<float>{1.0f, 2.0f},
                   });

  bindings.bind(b, Tensor{
                       Shape{2},
                       Dtype::F32,
                       std::vector<float>{3.0f, 4.0f},
                   });

  bindings.bind(a, Tensor{
                       Shape{2},
                       Dtype::F32,
                       std::vector<float>{10.0f, 20.0f},
                   });

  REQUIRE(bindings.get(a).data() == std::vector<float>{10.0f, 20.0f});

  REQUIRE(bindings.get(b).data() == std::vector<float>{3.0f, 4.0f});
}

TEST_CASE("get throws when ValueId has no binding", "[bindings]") {
  InputBindings bindings;

  REQUIRE_THROWS_AS(bindings.get(ValueId{42}), std::out_of_range);
}

TEST_CASE("get throws on an empty InputBindings", "[bindings]") {
  InputBindings bindings;

  REQUIRE_THROWS_AS(bindings.get(ValueId{0}), std::out_of_range);
}

TEST_CASE("negative ValueId is treated as an unbound ValueId", "[bindings]") {
  InputBindings bindings;

  REQUIRE_THROWS_AS(bindings.get(ValueId{-1}), std::out_of_range);
}

TEST_CASE("binding negative ValueId can currently be retrieved", "[bindings]") {
  InputBindings bindings;

  ValueId id{-1};

  bindings.bind(id, Tensor{
                        Shape{1},
                        Dtype::F32,
                        std::vector<float>{5.0f},
                    });

  REQUIRE(bindings.get(id).data() == std::vector<float>{5.0f});
}