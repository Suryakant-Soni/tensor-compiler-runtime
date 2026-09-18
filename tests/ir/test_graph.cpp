#include "tcr/ir/dtype.hpp"
#include "tcr/ir/graph.hpp"
#include "tcr/ir/ids.hpp"
#include "tcr/ir/nodes.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/ir/value.hpp"

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace tcr::ir;

namespace {

ElementwiseExpr input_expr(std::size_t index) {
  return ElementwiseExpr{
      .kind = ElementwiseExprKind::Input,
      .input_index = index,
      .lhs = nullptr,
      .rhs = nullptr,
  };
}

ElementwiseExpr binary_expr(ElementwiseExprKind kind, ElementwiseExpr lhs,
                            ElementwiseExpr rhs) {
  return ElementwiseExpr{
      .kind = kind,
      .input_index = 0,
      .lhs = std::make_unique<ElementwiseExpr>(std::move(lhs)),
      .rhs = std::make_unique<ElementwiseExpr>(std::move(rhs)),
  };
}

} // namespace

TEST_CASE("input creates a graph input value", "[graph]") {
  Graph g;

  ValueId id = g.input(Shape{2, 3}, Dtype::F32);

  const Value &v = g.value(id);

  REQUIRE(v.shape == Shape{2, 3});
  REQUIRE(v.dtype == Dtype::F32);
  REQUIRE(v.producer == std::nullopt);

  const auto &ins = g.inputs();

  REQUIRE(ins.size() == 1);
  REQUIRE(ins[0] == id);
}

TEST_CASE("add records correct node and output metadata", "[graph]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{2, 3}, Dtype::F32);

  ValueId out = g.add(a, b);

  const Node &n = g.node(NodeId{0});

  REQUIRE(n.op == Op::Add);
  REQUIRE(n.inputs == std::vector<ValueId>{a, b});

  const Value &v = g.value(out);

  REQUIRE(v.shape == Shape{2, 3});
  REQUIRE(v.dtype == Dtype::F32);
  REQUIRE(v.producer == NodeId{0});
}

TEST_CASE("add rejects mismatched shapes", "[graph]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{3, 2}, Dtype::F32);

  REQUIRE_THROWS_AS(g.add(a, b), std::invalid_argument);
}

TEST_CASE("mul records correct node and output metadata", "[graph]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{2, 3}, Dtype::F32);

  ValueId out = g.mul(a, b);

  const Node &n = g.node(NodeId{0});

  REQUIRE(n.op == Op::Mul);
  REQUIRE(n.inputs == std::vector<ValueId>{a, b});

  const Value &v = g.value(out);

  REQUIRE(v.shape == Shape{2, 3});
  REQUIRE(v.dtype == Dtype::F32);
  REQUIRE(v.producer == NodeId{0});
}

TEST_CASE("mul rejects mismatched shapes", "[graph]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{3, 2}, Dtype::F32);

  REQUIRE_THROWS_AS(g.mul(a, b), std::invalid_argument);
}

TEST_CASE("matmul records correct node and output metadata", "[graph]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{3, 4}, Dtype::F32);

  ValueId out = g.matmul(a, b);

  const Node &n = g.node(NodeId{0});

  REQUIRE(n.op == Op::MatMul);
  REQUIRE(n.inputs == std::vector<ValueId>{a, b});

  const Value &v = g.value(out);

  REQUIRE(v.shape == Shape{2, 4});
  REQUIRE(v.dtype == Dtype::F32);
  REQUIRE(v.producer == NodeId{0});
}

TEST_CASE("matmul rejects invalid operand shapes", "[graph]") {
  Graph g;

  SECTION("operand rank is not 2") {
    ValueId a = g.input(Shape{3}, Dtype::F32);
    ValueId b = g.input(Shape{3, 4}, Dtype::F32);

    REQUIRE_THROWS_AS(g.matmul(a, b), std::invalid_argument);
  }

  SECTION("inner dimensions do not match") {
    ValueId a = g.input(Shape{2, 3}, Dtype::F32);
    ValueId b = g.input(Shape{5, 4}, Dtype::F32);

    REQUIRE_THROWS_AS(g.matmul(a, b), std::invalid_argument);
  }
}

TEST_CASE("sum records correct node and scalar output metadata", "[graph]") {
  Graph g;

  ValueId input = g.input(Shape{2, 3}, Dtype::F32);
  ValueId out = g.sum(input);

  const Node &n = g.node(NodeId{0});

  REQUIRE(n.op == Op::Sum);
  REQUIRE(n.inputs == std::vector<ValueId>{input});

  const Value &v = g.value(out);

  REQUIRE(v.shape == Shape{});
  REQUIRE(v.shape.rank() == 0);
  REQUIRE(v.dtype == Dtype::F32);
  REQUIRE(v.producer == NodeId{0});
}

TEST_CASE("sum of higher-rank value creates scalar metadata", "[graph]") {
  Graph g;

  ValueId input = g.input(Shape{2, 3, 4, 5}, Dtype::F32);
  ValueId out = g.sum(input);

  const Value &v = g.value(out);

  REQUIRE(v.shape == Shape{});
  REQUIRE(v.dtype == Dtype::F32);
}

TEST_CASE("sum rejects invalid ValueId", "[graph]") {
  Graph g;

  REQUIRE_THROWS_AS(g.sum(ValueId{42}), std::out_of_range);
}

TEST_CASE("fused_elementwise records node expression and output metadata",
          "[graph][fused_elementwise]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{2, 3}, Dtype::F32);
  ValueId c = g.input(Shape{2, 3}, Dtype::F32);

  auto expr = binary_expr(
      ElementwiseExprKind::Add,
      binary_expr(ElementwiseExprKind::Mul, input_expr(0), input_expr(1)),
      input_expr(2));

  ValueId out = g.fused_elementwise({a, b, c}, std::move(expr));

  const Node &n = g.node(NodeId{0});
  REQUIRE(n.op == Op::FusedElementwise);
  REQUIRE(n.inputs == std::vector<ValueId>{a, b, c});
  REQUIRE(n.elementwise_expr.has_value());

  const ElementwiseExpr &root = *n.elementwise_expr;
  REQUIRE(root.kind == ElementwiseExprKind::Add);
  REQUIRE(root.lhs != nullptr);
  REQUIRE(root.rhs != nullptr);

  REQUIRE(root.lhs->kind == ElementwiseExprKind::Mul);
  REQUIRE(root.lhs->lhs != nullptr);
  REQUIRE(root.lhs->rhs != nullptr);
  REQUIRE(root.lhs->lhs->kind == ElementwiseExprKind::Input);
  REQUIRE(root.lhs->lhs->input_index == 0);
  REQUIRE(root.lhs->rhs->kind == ElementwiseExprKind::Input);
  REQUIRE(root.lhs->rhs->input_index == 1);

  REQUIRE(root.rhs->kind == ElementwiseExprKind::Input);
  REQUIRE(root.rhs->input_index == 2);

  const Value &v = g.value(out);
  REQUIRE(v.shape == Shape{2, 3});
  REQUIRE(v.dtype == Dtype::F32);
  REQUIRE(v.producer == NodeId{0});
}

TEST_CASE("fused_elementwise preserves duplicate boundary inputs",
          "[graph][fused_elementwise]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);

  auto expr =
      binary_expr(ElementwiseExprKind::Add, input_expr(0), input_expr(1));
  ValueId out = g.fused_elementwise({a, a}, std::move(expr));

  const Node &n = g.node(*g.value(out).producer);
  REQUIRE(n.inputs == std::vector<ValueId>{a, a});
  REQUIRE(n.elementwise_expr.has_value());
  REQUIRE(n.elementwise_expr->lhs->input_index == 0);
  REQUIRE(n.elementwise_expr->rhs->input_index == 1);
}

TEST_CASE("fused_elementwise accepts computed values as boundary inputs",
          "[graph][fused_elementwise]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{3, 4}, Dtype::F32);
  ValueId bias = g.input(Shape{2, 4}, Dtype::F32);

  ValueId product = g.matmul(a, b);

  auto expr =
      binary_expr(ElementwiseExprKind::Add, input_expr(0), input_expr(1));
  ValueId out = g.fused_elementwise({product, bias}, std::move(expr));

  const Node &n = g.node(*g.value(out).producer);
  REQUIRE(n.op == Op::FusedElementwise);
  REQUIRE(n.inputs == std::vector<ValueId>{product, bias});

  const Value &v = g.value(out);
  REQUIRE(v.shape == Shape{2, 4});
  REQUIRE(v.dtype == Dtype::F32);
}

TEST_CASE("fused_elementwise rejects Input as the root expression",
          "[graph][fused_elementwise]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);

  REQUIRE_THROWS_AS(g.fused_elementwise({a}, input_expr(0)), std::logic_error);
}

TEST_CASE("fused_elementwise rejects an empty boundary input list",
          "[graph][fused_elementwise]") {
  Graph g;

  auto expr =
      binary_expr(ElementwiseExprKind::Add, input_expr(0), input_expr(0));

  REQUIRE_THROWS_AS(g.fused_elementwise({}, std::move(expr)), std::logic_error);
}

TEST_CASE("fused_elementwise rejects invalid boundary ValueIds",
          "[graph][fused_elementwise]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);

  SECTION("positive out-of-range ValueId") {
    auto expr =
        binary_expr(ElementwiseExprKind::Add, input_expr(0), input_expr(1));
    REQUIRE_THROWS_AS(g.fused_elementwise({a, ValueId{99}}, std::move(expr)),
                      std::out_of_range);
  }

  SECTION("negative ValueId") {
    auto expr =
        binary_expr(ElementwiseExprKind::Add, input_expr(0), input_expr(1));
    REQUIRE_THROWS_AS(g.fused_elementwise({a, ValueId{-1}}, std::move(expr)),
                      std::out_of_range);
  }
}

TEST_CASE("fused_elementwise rejects mismatched boundary shapes",
          "[graph][fused_elementwise]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{3, 2}, Dtype::F32);

  auto expr =
      binary_expr(ElementwiseExprKind::Add, input_expr(0), input_expr(1));

  REQUIRE_THROWS_AS(g.fused_elementwise({a, b}, std::move(expr)),
                    std::invalid_argument);
}

TEST_CASE("metadata propagates through multiple graph operations",
          "[graph][type_inference]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{3, 4}, Dtype::F32);
  ValueId bias = g.input(Shape{2, 4}, Dtype::F32);

  ValueId product = g.matmul(a, b);

  REQUIRE(g.value(product).shape == Shape{2, 4});
  REQUIRE(g.value(product).dtype == Dtype::F32);

  ValueId added = g.add(product, bias);

  REQUIRE(g.value(added).shape == Shape{2, 4});
  REQUIRE(g.value(added).dtype == Dtype::F32);

  ValueId reduced = g.sum(added);

  REQUIRE(g.value(reduced).shape == Shape{});
  REQUIRE(g.value(reduced).dtype == Dtype::F32);
}

TEST_CASE("out-of-range ValueId throws in value and operations", "[graph]") {
  Graph g;

  ValueId bad{99};

  SECTION("value with positive out-of-range id") {
    REQUIRE_THROWS_AS(g.value(bad), std::out_of_range);
  }

  SECTION("value with negative id") {
    REQUIRE_THROWS_AS(g.value(ValueId{-1}), std::out_of_range);
  }

  SECTION("add with invalid operand") {
    ValueId a = g.input(Shape{2, 3}, Dtype::F32);

    REQUIRE_THROWS_AS(g.add(a, bad), std::out_of_range);
  }
}

TEST_CASE("node rejects invalid NodeId", "[graph]") {
  Graph g;

  SECTION("positive out-of-range NodeId") {
    REQUIRE_THROWS_AS(g.node(NodeId{99}), std::out_of_range);
  }

  SECTION("negative NodeId") {
    REQUIRE_THROWS_AS(g.node(NodeId{-1}), std::out_of_range);
  }
}

TEST_CASE("empty graph has empty inputs and outputs", "[graph]") {
  Graph g;

  REQUIRE(g.inputs().empty());
  REQUIRE(g.outputs().empty());
}

TEST_CASE("add_output appends without clobbering existing outputs", "[graph]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{2, 3}, Dtype::F32);

  g.add_output(a);
  g.add_output(b);

  REQUIRE(g.outputs() == std::vector<ValueId>{a, b});
}

TEST_CASE("add_output rejects invalid ValueId", "[graph]") {
  Graph g;

  REQUIRE_THROWS_AS(g.add_output(ValueId{99}), std::out_of_range);
  REQUIRE(g.outputs().empty());
}

TEST_CASE("set_outputs stores graph outputs", "[graph]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{2, 3}, Dtype::F32);
  ValueId c = g.add(a, b);

  g.set_outputs({a, c});

  REQUIRE(g.outputs() == std::vector<ValueId>{a, c});
}

TEST_CASE("set_outputs replaces previous outputs", "[graph]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{2, 3}, Dtype::F32);
  ValueId c = g.add(a, b);

  g.set_outputs({a, b});

  REQUIRE(g.outputs() == std::vector<ValueId>{a, b});

  g.set_outputs({c});

  REQUIRE(g.outputs() == std::vector<ValueId>{c});
}

TEST_CASE("set_outputs validates every ValueId", "[graph]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId invalid{999};

  REQUIRE_THROWS_AS(g.set_outputs({a, invalid}), std::out_of_range);
}

TEST_CASE("failed set_outputs preserves existing outputs", "[graph]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{2, 3}, Dtype::F32);

  g.set_outputs({a});

  REQUIRE_THROWS_AS(g.set_outputs({b, ValueId{999}}), std::out_of_range);

  REQUIRE(g.outputs() == std::vector<ValueId>{a});
}

TEST_CASE("operation results are not automatically graph inputs", "[graph]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{2, 3}, Dtype::F32);

  ValueId add_out = g.add(a, b);
  auto expr =
      binary_expr(ElementwiseExprKind::Mul, input_expr(0), input_expr(1));
  ValueId fused_out = g.fused_elementwise({a, b}, std::move(expr));

  REQUIRE(g.inputs() == std::vector<ValueId>{a, b});

  for (ValueId id : g.inputs()) {
    REQUIRE(id != add_out);
    REQUIRE(id != fused_out);
  }
}

TEST_CASE("ValueIds increase as values are created", "[graph]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{2, 3}, Dtype::F32);
  ValueId c = g.add(a, b);
  ValueId d = g.mul(a, b);

  auto expr =
      binary_expr(ElementwiseExprKind::Add, input_expr(0), input_expr(1));
  ValueId e = g.fused_elementwise({a, b}, std::move(expr));

  REQUIRE(a == ValueId{0});
  REQUIRE(b == ValueId{1});
  REQUIRE(c == ValueId{2});
  REQUIRE(d == ValueId{3});
  REQUIRE(e == ValueId{4});
}

TEST_CASE("NodeIds increase as operations are created", "[graph]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{2, 3}, Dtype::F32);

  ValueId add_out = g.add(a, b);
  ValueId mul_out = g.mul(a, b);

  auto expr =
      binary_expr(ElementwiseExprKind::Add, input_expr(0), input_expr(1));
  ValueId fused_out = g.fused_elementwise({a, b}, std::move(expr));

  REQUIRE(g.value(add_out).producer == NodeId{0});
  REQUIRE(g.value(mul_out).producer == NodeId{1});
  REQUIRE(g.value(fused_out).producer == NodeId{2});

  REQUIRE(g.node(NodeId{0}).op == Op::Add);
  REQUIRE(g.node(NodeId{1}).op == Op::Mul);
  REQUIRE(g.node(NodeId{2}).op == Op::FusedElementwise);
}

TEST_CASE("failed add does not create a node or value", "[graph]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{3, 2}, Dtype::F32);

  const std::size_t values_before = g.values_count();
  const std::size_t nodes_before = g.nodes_count();

  REQUIRE_THROWS_AS(g.add(a, b), std::invalid_argument);

  REQUIRE(g.values_count() == values_before);
  REQUIRE(g.nodes_count() == nodes_before);
}

TEST_CASE("failed matmul does not create a node or value", "[graph]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{5, 4}, Dtype::F32);

  const std::size_t values_before = g.values_count();
  const std::size_t nodes_before = g.nodes_count();

  REQUIRE_THROWS_AS(g.matmul(a, b), std::invalid_argument);

  REQUIRE(g.values_count() == values_before);
  REQUIRE(g.nodes_count() == nodes_before);
}

TEST_CASE("failed fused_elementwise does not create a node or value",
          "[graph][fused_elementwise]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{3, 2}, Dtype::F32);

  const std::size_t values_before = g.values_count();
  const std::size_t nodes_before = g.nodes_count();

  auto expr =
      binary_expr(ElementwiseExprKind::Add, input_expr(0), input_expr(1));

  REQUIRE_THROWS_AS(g.fused_elementwise({a, b}, std::move(expr)),
                    std::invalid_argument);

  REQUIRE(g.values_count() == values_before);
  REQUIRE(g.nodes_count() == nodes_before);
}
