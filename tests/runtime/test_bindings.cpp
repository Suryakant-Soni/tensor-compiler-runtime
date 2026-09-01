#include <catch2/catch_test_macros.hpp>

#include "tcr/ir/dtype.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/runtime/bindings.hpp"
#include "tcr/runtime/slot_id.hpp"
#include "tcr/runtime/tensor.hpp"

#include <stdexcept>
#include <vector>

using tcr::ir::Dtype;
using tcr::ir::Shape;
using tcr::runtime::InputBindings;
using tcr::runtime::SlotId;
using tcr::runtime::Tensor;

TEST_CASE("InputBindings binds and retrieves a tensor") {
  InputBindings bindings;

  const SlotId slot{0};

  bindings.bind(
      slot,
      Tensor{
          Shape{2},
          Dtype::F32,
          std::vector<float>{1.0F, 2.0F},
      });

  const Tensor &tensor = bindings.get(slot);

  REQUIRE(tensor.shape() == Shape{2});
  REQUIRE(tensor.dtype() == Dtype::F32);
  REQUIRE(tensor.data() == std::vector<float>{1.0F, 2.0F});
}

TEST_CASE("InputBindings supports multiple independent slots") {
  InputBindings bindings;

  const SlotId slot0{0};
  const SlotId slot1{1};

  bindings.bind(
      slot0,
      Tensor{
          Shape{2},
          Dtype::F32,
          std::vector<float>{1.0F, 2.0F},
      });

  bindings.bind(
      slot1,
      Tensor{
          Shape{2},
          Dtype::F32,
          std::vector<float>{3.0F, 4.0F},
      });

  const Tensor &tensor0 = bindings.get(slot0);
  const Tensor &tensor1 = bindings.get(slot1);

  REQUIRE(tensor0.data() == std::vector<float>{1.0F, 2.0F});
  REQUIRE(tensor1.data() == std::vector<float>{3.0F, 4.0F});
}

TEST_CASE("InputBindings replaces an existing binding") {
  InputBindings bindings;

  const SlotId slot{3};

  bindings.bind(
      slot,
      Tensor{
          Shape{2},
          Dtype::F32,
          std::vector<float>{1.0F, 2.0F},
      });

  bindings.bind(
      slot,
      Tensor{
          Shape{2},
          Dtype::F32,
          std::vector<float>{10.0F, 20.0F},
      });

  const Tensor &tensor = bindings.get(slot);

  REQUIRE(tensor.shape() == Shape{2});
  REQUIRE(tensor.dtype() == Dtype::F32);
  REQUIRE(tensor.data() == std::vector<float>{10.0F, 20.0F});
}

TEST_CASE("Replacing one InputBindings slot does not modify other slots") {
  InputBindings bindings;

  const SlotId slot0{0};
  const SlotId slot1{1};

  bindings.bind(
      slot0,
      Tensor{
          Shape{2},
          Dtype::F32,
          std::vector<float>{1.0F, 2.0F},
      });

  bindings.bind(
      slot1,
      Tensor{
          Shape{2},
          Dtype::F32,
          std::vector<float>{3.0F, 4.0F},
      });

  bindings.bind(
      slot0,
      Tensor{
          Shape{2},
          Dtype::F32,
          std::vector<float>{5.0F, 6.0F},
      });

  REQUIRE(bindings.get(slot0).data() ==
          std::vector<float>{5.0F, 6.0F});

  REQUIRE(bindings.get(slot1).data() ==
          std::vector<float>{3.0F, 4.0F});
}

TEST_CASE("InputBindings can store a scalar tensor") {
  InputBindings bindings;

  const SlotId slot{0};

  bindings.bind(
      slot,
      Tensor{
          Shape{},
          Dtype::F32,
          std::vector<float>{42.0F},
      });

  const Tensor &tensor = bindings.get(slot);

  REQUIRE(tensor.shape() == Shape{});
  REQUIRE(tensor.shape().rank() == 0);
  REQUIRE(tensor.data() == std::vector<float>{42.0F});
}

TEST_CASE("InputBindings can store tensors with different shapes") {
  InputBindings bindings;

  const SlotId vector_slot{0};
  const SlotId matrix_slot{1};

  bindings.bind(
      vector_slot,
      Tensor{
          Shape{2},
          Dtype::F32,
          std::vector<float>{1.0F, 2.0F},
      });

  bindings.bind(
      matrix_slot,
      Tensor{
          Shape{2, 2},
          Dtype::F32,
          std::vector<float>{
              1.0F,
              2.0F,
              3.0F,
              4.0F,
          },
      });

  REQUIRE(bindings.get(vector_slot).shape() == Shape{2});
  REQUIRE(bindings.get(matrix_slot).shape() == Shape{2, 2});

  REQUIRE(bindings.get(vector_slot).data() ==
          std::vector<float>{1.0F, 2.0F});

  REQUIRE(bindings.get(matrix_slot).data() ==
          std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F});
}

TEST_CASE("InputBindings get throws for an unbound slot") {
  const InputBindings bindings;

  REQUIRE_THROWS_AS(
      bindings.get(SlotId{0}),
      std::out_of_range);
}

TEST_CASE("InputBindings get throws for a different unbound slot") {
  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      Tensor{
          Shape{2},
          Dtype::F32,
          std::vector<float>{1.0F, 2.0F},
      });

  REQUIRE_THROWS_AS(
      bindings.get(SlotId{1}),
      std::out_of_range);
}

TEST_CASE("InputBindings reports expected error message for missing slot") {
  const InputBindings bindings;

  REQUIRE_THROWS_AS(bindings.get(SlotId{7}), std::out_of_range);
}

TEST_CASE("InputBindings preserves binding after failed lookup") {
  InputBindings bindings;

  const SlotId bound_slot{0};

  bindings.bind(
      bound_slot,
      Tensor{
          Shape{2},
          Dtype::F32,
          std::vector<float>{1.0F, 2.0F},
      });

  REQUIRE_THROWS_AS(
      bindings.get(SlotId{10}),
      std::out_of_range);

  REQUIRE(bindings.get(bound_slot).data() ==
          std::vector<float>{1.0F, 2.0F});
}

TEST_CASE("InputBindings supports non-consecutive SlotIds") {
  InputBindings bindings;

  const SlotId slot2{2};
  const SlotId slot100{100};

  bindings.bind(
      slot2,
      Tensor{
          Shape{1},
          Dtype::F32,
          std::vector<float>{2.0F},
      });

  bindings.bind(
      slot100,
      Tensor{
          Shape{1},
          Dtype::F32,
          std::vector<float>{100.0F},
      });

  REQUIRE(bindings.get(slot2).data() ==
          std::vector<float>{2.0F});

  REQUIRE(bindings.get(slot100).data() ==
          std::vector<float>{100.0F});
}