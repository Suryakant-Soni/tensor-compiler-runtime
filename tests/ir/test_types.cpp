#include "tcr/ir/dtype.hpp"
#include "tcr/ir/ids.hpp"
#include "tcr/ir/nodes.hpp"
#include "tcr/ir/op.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/ir/value.hpp"

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace tcr::ir;

// -----------------------------------------------------------------------------
// ValueId
// -----------------------------------------------------------------------------

TEST_CASE("ValueId stores integer identity", "[ids]") {
  ValueId id{42};

  REQUIRE(id.value == 42);
}

TEST_CASE("ValueId equality compares underlying identity", "[ids]") {
  REQUIRE(ValueId{3} == ValueId{3});
  REQUIRE_FALSE(ValueId{3} == ValueId{4});
}

TEST_CASE("ValueId supports negative sentinel values as a raw type", "[ids]") {
  ValueId id{-1};

  REQUIRE(id.value == -1);
}

// -----------------------------------------------------------------------------
// NodeId
// -----------------------------------------------------------------------------

TEST_CASE("NodeId stores integer identity", "[ids]") {
  NodeId id{7};

  REQUIRE(id.value == 7);
}

TEST_CASE("NodeId equality compares underlying identity", "[ids]") {
  REQUIRE(NodeId{5} == NodeId{5});
  REQUIRE_FALSE(NodeId{5} == NodeId{6});
}

// -----------------------------------------------------------------------------
// ID printing
// -----------------------------------------------------------------------------

TEST_CASE("ValueId prints useful debug representation", "[ids]") {
  std::ostringstream os;

  os << ValueId{12};

  REQUIRE(os.str() == "ValueId(12)");
}

TEST_CASE("NodeId prints useful debug representation", "[ids]") {
  std::ostringstream os;

  os << NodeId{8};

  REQUIRE(os.str() == "NodeId(8)");
}

// -----------------------------------------------------------------------------
// Dtype
// -----------------------------------------------------------------------------

TEST_CASE("F32 has four-byte element size", "[dtype]") {
  REQUIRE(size_in_bytes(Dtype::F32) == 4);
}

TEST_CASE("F32 has expected string representation", "[dtype]") {
  REQUIRE(to_string(Dtype::F32) == "f32");
}

TEST_CASE("FusedElementwise has expected string representation",
          "[op][fusion]") {
  REQUIRE(to_string(Op::FusedElementwise) == "FusedElementwise");
}

// -----------------------------------------------------------------------------
// Node
// -----------------------------------------------------------------------------

TEST_CASE("Node stores operation and input ValueIds", "[node]") {
  ValueId a{0};
  ValueId b{1};

  Node node{
      .op = Op::Add,
      .inputs = {a, b},
      .elementwise_expr = std::nullopt,
  };

  REQUIRE(node.op == Op::Add);
  REQUIRE(node.inputs == std::vector<ValueId>{a, b});
  REQUIRE_FALSE(node.elementwise_expr.has_value());
}

TEST_CASE("FusedElementwise node stores expression payload", "[node][fusion]") {
  ValueId a{0};
  ValueId b{1};

  ElementwiseExpr expr{
      .kind = ElementwiseExprKind::Add,
      .input_index = 0,
      .lhs = std::make_unique<ElementwiseExpr>(ElementwiseExpr{
          .kind = ElementwiseExprKind::Input,
          .input_index = 0,
          .lhs = nullptr,
          .rhs = nullptr,
      }),
      .rhs = std::make_unique<ElementwiseExpr>(ElementwiseExpr{
          .kind = ElementwiseExprKind::Input,
          .input_index = 1,
          .lhs = nullptr,
          .rhs = nullptr,
      }),
  };

  Node node{
      .op = Op::FusedElementwise,
      .inputs = {a, b},
      .elementwise_expr = std::optional<ElementwiseExpr>{std::move(expr)},
  };

  REQUIRE(node.op == Op::FusedElementwise);
  REQUIRE(node.inputs == std::vector<ValueId>{a, b});
  REQUIRE(node.elementwise_expr.has_value());

  const auto &stored = *node.elementwise_expr;

  REQUIRE(stored.kind == ElementwiseExprKind::Add);
  REQUIRE(stored.lhs != nullptr);
  REQUIRE(stored.rhs != nullptr);

  REQUIRE(stored.lhs->kind == ElementwiseExprKind::Input);
  REQUIRE(stored.lhs->input_index == 0);

  REQUIRE(stored.rhs->kind == ElementwiseExprKind::Input);
  REQUIRE(stored.rhs->input_index == 1);
}

TEST_CASE("Unary node stores one input", "[node]") {
  ValueId input{5};

  Node node{
      .op = Op::Sum,
      .inputs = {input},
      .elementwise_expr = std::nullopt,
  };

  REQUIRE(node.op == Op::Sum);
  REQUIRE(node.inputs.size() == 1);
  REQUIRE(node.inputs[0] == input);
}

// -----------------------------------------------------------------------------
// Value
// -----------------------------------------------------------------------------

TEST_CASE("graph input Value can have no producer", "[value]") {
  Value value{
      .shape = Shape{2, 3},
      .dtype = Dtype::F32,
      .producer = std::nullopt,
  };

  REQUIRE(value.shape == Shape{2, 3});
  REQUIRE(value.dtype == Dtype::F32);
  REQUIRE_FALSE(value.producer.has_value());
}

TEST_CASE("computed Value records producer NodeId", "[value]") {
  Value value{
      .shape = Shape{2, 3},
      .dtype = Dtype::F32,
      .producer = NodeId{4},
  };

  REQUIRE(value.shape == Shape{2, 3});
  REQUIRE(value.dtype == Dtype::F32);

  REQUIRE(value.producer.has_value());
  REQUIRE(value.producer.value() == NodeId{4});
}

TEST_CASE("scalar Value can use rank-zero Shape", "[value]") {
  Value value{
      .shape = Shape{},
      .dtype = Dtype::F32,
      .producer = NodeId{2},
  };

  REQUIRE(value.shape.rank() == 0);
  REQUIRE(value.shape.num_elements() == 1);
}