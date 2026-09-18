#include <catch2/catch_test_macros.hpp>

#include "tcr/compiler/elementwise_fusion.hpp"
#include "tcr/ir/dtype.hpp"
#include "tcr/ir/elementwise_expr.hpp"
#include "tcr/ir/graph.hpp"
#include "tcr/ir/ids.hpp"
#include "tcr/ir/nodes.hpp"
#include "tcr/ir/op.hpp"
#include "tcr/ir/shape.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using tcr::compiler::ElementwiseFusion;

using tcr::ir::Dtype;
using tcr::ir::ElementwiseExpr;
using tcr::ir::ElementwiseExprKind;
using tcr::ir::Graph;
using tcr::ir::Node;
using tcr::ir::NodeId;
using tcr::ir::Op;
using tcr::ir::Shape;
using tcr::ir::ValueId;

namespace {

const Node &require_producer_node(const Graph &graph, ValueId value_id) {
  const auto &value = graph.value(value_id);
  REQUIRE(value.producer.has_value());
  return graph.node(*value.producer);
}

const Node &require_output_node(const Graph &graph,
                                std::size_t output_index = 0) {
  REQUIRE(output_index < graph.outputs().size());
  return require_producer_node(graph, graph.outputs()[output_index]);
}

const ElementwiseExpr &require_fused_expr(const Node &node) {
  REQUIRE(node.op == Op::FusedElementwise);
  REQUIRE(node.elementwise_expr.has_value());
  return *node.elementwise_expr;
}

void require_input_expr(const ElementwiseExpr &expr, std::size_t input_index) {
  REQUIRE(expr.kind == ElementwiseExprKind::Input);
  REQUIRE(expr.input_index == input_index);
  REQUIRE(expr.lhs == nullptr);
  REQUIRE(expr.rhs == nullptr);
}

void require_binary_expr(const ElementwiseExpr &expr,
                         ElementwiseExprKind kind) {
  REQUIRE(expr.kind == kind);
  REQUIRE(expr.lhs != nullptr);
  REQUIRE(expr.rhs != nullptr);
}

ValueId require_resolved_input_value(const Node &fused_node,
                                     const ElementwiseExpr &input_expr) {
  REQUIRE(fused_node.op == Op::FusedElementwise);
  REQUIRE(input_expr.kind == ElementwiseExprKind::Input);
  REQUIRE(input_expr.input_index < fused_node.inputs.size());
  return fused_node.inputs[input_expr.input_index];
}

std::size_t count_nodes_with_op(const Graph &graph, Op op) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < graph.nodes_count(); ++i) {
    const auto &node = graph.node(NodeId{static_cast<std::int32_t>(i)});
    if (node.op == op) {
      ++count;
    }
  }
  return count;
}

ElementwiseExpr make_binary_expr(ElementwiseExprKind kind,
                                 std::size_t lhs_index, std::size_t rhs_index) {
  auto lhs = std::make_unique<ElementwiseExpr>(ElementwiseExpr{
      .kind = ElementwiseExprKind::Input,
      .input_index = lhs_index,
      .lhs = nullptr,
      .rhs = nullptr,
  });
  auto rhs = std::make_unique<ElementwiseExpr>(ElementwiseExpr{
      .kind = ElementwiseExprKind::Input,
      .input_index = rhs_index,
      .lhs = nullptr,
      .rhs = nullptr,
  });

  return ElementwiseExpr{
      .kind = kind,
      .input_index = 0,
      .lhs = std::move(lhs),
      .rhs = std::move(rhs),
  };
}

} // namespace

// =============================================================================
// Empty / inputs / reachability
// =============================================================================

TEST_CASE("ElementwiseFusion preserves an empty graph",
          "[compiler][elementwise_fusion]") {
  Graph graph;

  ElementwiseFusion fusion;
  const auto optimized = fusion.run(graph);

  REQUIRE(optimized.inputs().empty());
  REQUIRE(optimized.outputs().empty());
  REQUIRE(optimized.nodes_count() == 0);
  REQUIRE(optimized.values_count() == 0);
}

TEST_CASE("ElementwiseFusion preserves direct input outputs and their order",
          "[compiler][elementwise_fusion]") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);
  const auto y = graph.input(Shape{4}, Dtype::F32);
  graph.set_outputs({y, x});

  ElementwiseFusion fusion;
  const auto optimized = fusion.run(graph);

  REQUIRE(optimized.inputs().size() == 2);
  REQUIRE(optimized.outputs().size() == 2);
  REQUIRE(optimized.nodes_count() == 0);

  REQUIRE(optimized.inputs()[0] == ValueId{0});
  REQUIRE(optimized.inputs()[1] == ValueId{1});
  REQUIRE(optimized.outputs()[0] == optimized.inputs()[1]);
  REQUIRE(optimized.outputs()[1] == optimized.inputs()[0]);

  REQUIRE(optimized.value(optimized.inputs()[0]).shape == Shape{2});
  REQUIRE(optimized.value(optimized.inputs()[1]).shape == Shape{4});
}

TEST_CASE("ElementwiseFusion keeps graph inputs even when there are no outputs",
          "[compiler][elementwise_fusion]") {
  Graph graph;

  graph.input(Shape{2}, Dtype::F32);
  graph.input(Shape{3}, Dtype::F32);
  graph.mul(ValueId{0}, ValueId{0}); // unreachable computed value

  ElementwiseFusion fusion;
  const auto optimized = fusion.run(graph);

  REQUIRE(optimized.inputs().size() == 2);
  REQUIRE(optimized.outputs().empty());
  REQUIRE(optimized.nodes_count() == 0);
  REQUIRE(optimized.values_count() == 2);
}

TEST_CASE("ElementwiseFusion drops unreachable computed nodes",
          "[compiler][elementwise_fusion]") {
  Graph graph;

  const auto a = graph.input(Shape{2}, Dtype::F32);
  const auto b = graph.input(Shape{2}, Dtype::F32);
  const auto c = graph.input(Shape{2}, Dtype::F32);
  graph.mul(a, b); // dead
  graph.set_outputs({c});

  ElementwiseFusion fusion;
  const auto optimized = fusion.run(graph);

  REQUIRE(optimized.inputs().size() == 3);
  REQUIRE(optimized.outputs().size() == 1);
  REQUIRE(optimized.outputs()[0] == optimized.inputs()[2]);
  REQUIRE(optimized.nodes_count() == 0);
}

// =============================================================================
// No-fusion cases
// =============================================================================

TEST_CASE("ElementwiseFusion leaves a single Add unfused",
          "[compiler][elementwise_fusion]") {
  Graph graph;

  const auto a = graph.input(Shape{4}, Dtype::F32);
  const auto b = graph.input(Shape{4}, Dtype::F32);
  const auto out = graph.add(a, b);
  graph.set_outputs({out});

  ElementwiseFusion fusion;
  const auto optimized = fusion.run(graph);

  REQUIRE(optimized.nodes_count() == 1);
  REQUIRE(count_nodes_with_op(optimized, Op::FusedElementwise) == 0);

  const auto &node = require_output_node(optimized);
  REQUIRE(node.op == Op::Add);
  REQUIRE(node.inputs.size() == 2);
  REQUIRE(node.inputs[0] == optimized.inputs()[0]);
  REQUIRE(node.inputs[1] == optimized.inputs()[1]);
}

TEST_CASE("ElementwiseFusion leaves a single Mul unfused",
          "[compiler][elementwise_fusion]") {
  Graph graph;

  const auto a = graph.input(Shape{4}, Dtype::F32);
  const auto b = graph.input(Shape{4}, Dtype::F32);
  const auto out = graph.mul(a, b);
  graph.set_outputs({out});

  ElementwiseFusion fusion;
  const auto optimized = fusion.run(graph);

  REQUIRE(optimized.nodes_count() == 1);
  REQUIRE(count_nodes_with_op(optimized, Op::FusedElementwise) == 0);
  REQUIRE(require_output_node(optimized).op == Op::Mul);
}

TEST_CASE("ElementwiseFusion does not absorb a MatMul boundary by itself",
          "[compiler][elementwise_fusion]") {
  Graph graph;

  const auto a = graph.input(Shape{2, 3}, Dtype::F32);
  const auto b = graph.input(Shape{3, 2}, Dtype::F32);
  const auto c = graph.input(Shape{2, 2}, Dtype::F32);

  const auto mm = graph.matmul(a, b);
  const auto out = graph.add(mm, c);
  graph.set_outputs({out});

  ElementwiseFusion fusion;
  const auto optimized = fusion.run(graph);

  REQUIRE(optimized.nodes_count() == 2);
  REQUIRE(count_nodes_with_op(optimized, Op::MatMul) == 1);
  REQUIRE(count_nodes_with_op(optimized, Op::Add) == 1);
  REQUIRE(count_nodes_with_op(optimized, Op::FusedElementwise) == 0);
  REQUIRE(require_output_node(optimized).op == Op::Add);
}

// =============================================================================
// Basic fusion
// =============================================================================

TEST_CASE("ElementwiseFusion fuses Mul feeding Add into one fused node",
          "[compiler][elementwise_fusion]") {
  Graph graph;

  const auto a = graph.input(Shape{2, 3}, Dtype::F32);
  const auto b = graph.input(Shape{2, 3}, Dtype::F32);
  const auto c = graph.input(Shape{2, 3}, Dtype::F32);

  const auto product = graph.mul(a, b);
  const auto out = graph.add(product, c);
  graph.set_outputs({out});

  ElementwiseFusion fusion;
  const auto optimized = fusion.run(graph);

  REQUIRE(optimized.nodes_count() == 1);
  REQUIRE(count_nodes_with_op(optimized, Op::FusedElementwise) == 1);
  REQUIRE(count_nodes_with_op(optimized, Op::Mul) == 0);
  REQUIRE(count_nodes_with_op(optimized, Op::Add) == 0);

  const auto optimized_out = optimized.outputs()[0];
  REQUIRE(optimized.value(optimized_out).shape == Shape{2, 3});
  REQUIRE(optimized.value(optimized_out).dtype == Dtype::F32);

  const auto &fused = require_output_node(optimized);
  REQUIRE(fused.inputs.size() == 3);
  REQUIRE(fused.inputs[0] == optimized.inputs()[0]);
  REQUIRE(fused.inputs[1] == optimized.inputs()[1]);
  REQUIRE(fused.inputs[2] == optimized.inputs()[2]);

  const auto &expr = require_fused_expr(fused);
  require_binary_expr(expr, ElementwiseExprKind::Add);
  require_binary_expr(*expr.lhs, ElementwiseExprKind::Mul);

  require_input_expr(*expr.lhs->lhs, 0);
  require_input_expr(*expr.lhs->rhs, 1);
  require_input_expr(*expr.rhs, 2);
}

TEST_CASE(
    "ElementwiseFusion fuses Add feeding Mul and preserves operation kinds",
    "[compiler][elementwise_fusion]") {
  Graph graph;

  const auto a = graph.input(Shape{4}, Dtype::F32);
  const auto b = graph.input(Shape{4}, Dtype::F32);
  const auto c = graph.input(Shape{4}, Dtype::F32);

  const auto sum = graph.add(a, b);
  const auto out = graph.mul(sum, c);
  graph.set_outputs({out});

  ElementwiseFusion fusion;
  const auto optimized = fusion.run(graph);

  const auto &fused = require_output_node(optimized);
  const auto &expr = require_fused_expr(fused);

  require_binary_expr(expr, ElementwiseExprKind::Mul);
  require_binary_expr(*expr.lhs, ElementwiseExprKind::Add);
  require_input_expr(*expr.lhs->lhs, 0);
  require_input_expr(*expr.lhs->rhs, 1);
  require_input_expr(*expr.rhs, 2);
}

TEST_CASE(
    "ElementwiseFusion preserves operand side when fusible producer is rhs",
    "[compiler][elementwise_fusion]") {
  Graph graph;

  const auto a = graph.input(Shape{4}, Dtype::F32);
  const auto b = graph.input(Shape{4}, Dtype::F32);
  const auto c = graph.input(Shape{4}, Dtype::F32);

  const auto product = graph.mul(a, b);
  const auto out = graph.add(c, product);
  graph.set_outputs({out});

  ElementwiseFusion fusion;
  const auto optimized = fusion.run(graph);

  const auto &fused = require_output_node(optimized);
  const auto &expr = require_fused_expr(fused);

  require_binary_expr(expr, ElementwiseExprKind::Add);
  require_input_expr(*expr.lhs, 0);
  require_binary_expr(*expr.rhs, ElementwiseExprKind::Mul);
  require_input_expr(*expr.rhs->lhs, 1);
  require_input_expr(*expr.rhs->rhs, 2);

  REQUIRE(fused.inputs[0] == optimized.inputs()[2]); // c
  REQUIRE(fused.inputs[1] == optimized.inputs()[0]); // a
  REQUIRE(fused.inputs[2] == optimized.inputs()[1]); // b
}

TEST_CASE("ElementwiseFusion fuses a deep linear elementwise chain",
          "[compiler][elementwise_fusion]") {
  Graph graph;

  const auto a = graph.input(Shape{5}, Dtype::F32);
  const auto b = graph.input(Shape{5}, Dtype::F32);
  const auto c = graph.input(Shape{5}, Dtype::F32);
  const auto d = graph.input(Shape{5}, Dtype::F32);
  const auto e = graph.input(Shape{5}, Dtype::F32);

  const auto v1 = graph.mul(a, b);
  const auto v2 = graph.add(v1, c);
  const auto v3 = graph.mul(v2, d);
  const auto out = graph.add(v3, e);
  graph.set_outputs({out});

  ElementwiseFusion fusion;
  const auto optimized = fusion.run(graph);

  REQUIRE(optimized.nodes_count() == 1);

  const auto &fused = require_output_node(optimized);
  const auto &expr = require_fused_expr(fused);

  require_binary_expr(expr, ElementwiseExprKind::Add);
  require_binary_expr(*expr.lhs, ElementwiseExprKind::Mul);
  require_binary_expr(*expr.lhs->lhs, ElementwiseExprKind::Add);
  require_binary_expr(*expr.lhs->lhs->lhs, ElementwiseExprKind::Mul);

  require_input_expr(*expr.lhs->lhs->lhs->lhs, 0);
  require_input_expr(*expr.lhs->lhs->lhs->rhs, 1);
  require_input_expr(*expr.lhs->lhs->rhs, 2);
  require_input_expr(*expr.lhs->rhs, 3);
  require_input_expr(*expr.rhs, 4);
}

TEST_CASE("ElementwiseFusion fuses both elementwise branches into one tree",
          "[compiler][elementwise_fusion]") {
  Graph graph;

  const auto a = graph.input(Shape{3}, Dtype::F32);
  const auto b = graph.input(Shape{3}, Dtype::F32);
  const auto c = graph.input(Shape{3}, Dtype::F32);
  const auto d = graph.input(Shape{3}, Dtype::F32);

  const auto left = graph.mul(a, b);
  const auto right = graph.add(c, d);
  const auto out = graph.add(left, right);
  graph.set_outputs({out});

  ElementwiseFusion fusion;
  const auto optimized = fusion.run(graph);

  REQUIRE(optimized.nodes_count() == 1);

  const auto &fused = require_output_node(optimized);
  const auto &expr = require_fused_expr(fused);

  require_binary_expr(expr, ElementwiseExprKind::Add);
  require_binary_expr(*expr.lhs, ElementwiseExprKind::Mul);
  require_binary_expr(*expr.rhs, ElementwiseExprKind::Add);

  require_input_expr(*expr.lhs->lhs, 0);
  require_input_expr(*expr.lhs->rhs, 1);
  require_input_expr(*expr.rhs->lhs, 2);
  require_input_expr(*expr.rhs->rhs, 3);
}

TEST_CASE("ElementwiseFusion supports scalar elementwise chains",
          "[compiler][elementwise_fusion]") {
  Graph graph;

  const auto a = graph.input(Shape{}, Dtype::F32);
  const auto b = graph.input(Shape{}, Dtype::F32);
  const auto c = graph.input(Shape{}, Dtype::F32);

  const auto t = graph.add(a, b);
  const auto out = graph.mul(t, c);
  graph.set_outputs({out});

  ElementwiseFusion fusion;
  const auto optimized = fusion.run(graph);

  REQUIRE(optimized.nodes_count() == 1);
  REQUIRE(optimized.value(optimized.outputs()[0]).shape == Shape{});
  REQUIRE(optimized.value(optimized.outputs()[0]).dtype == Dtype::F32);
  REQUIRE(require_output_node(optimized).op == Op::FusedElementwise);
}

// =============================================================================
// Fusion boundaries
// =============================================================================

TEST_CASE("ElementwiseFusion keeps non-elementwise producer as fused boundary",
          "[compiler][elementwise_fusion]") {
  Graph graph;

  const auto a = graph.input(Shape{2, 3}, Dtype::F32);
  const auto b = graph.input(Shape{3, 2}, Dtype::F32);
  const auto c = graph.input(Shape{2, 2}, Dtype::F32);
  const auto d = graph.input(Shape{2, 2}, Dtype::F32);

  const auto mm = graph.matmul(a, b);
  const auto t = graph.add(mm, c);
  const auto out = graph.mul(t, d);
  graph.set_outputs({out});

  ElementwiseFusion fusion;
  const auto optimized = fusion.run(graph);

  REQUIRE(optimized.nodes_count() == 2);
  REQUIRE(count_nodes_with_op(optimized, Op::MatMul) == 1);
  REQUIRE(count_nodes_with_op(optimized, Op::FusedElementwise) == 1);

  const auto &fused = require_output_node(optimized);
  const auto &expr = require_fused_expr(fused);

  require_binary_expr(expr, ElementwiseExprKind::Mul);
  require_binary_expr(*expr.lhs, ElementwiseExprKind::Add);
  require_input_expr(*expr.lhs->lhs, 0); // matmul boundary
  require_input_expr(*expr.lhs->rhs, 1); // c
  require_input_expr(*expr.rhs, 2);      // d

  const auto mm_boundary = fused.inputs[0];
  REQUIRE(require_producer_node(optimized, mm_boundary).op == Op::MatMul);
}

TEST_CASE("ElementwiseFusion can fuse below a non-elementwise Sum consumer",
          "[compiler][elementwise_fusion]") {
  Graph graph;

  const auto a = graph.input(Shape{4}, Dtype::F32);
  const auto b = graph.input(Shape{4}, Dtype::F32);
  const auto c = graph.input(Shape{4}, Dtype::F32);

  const auto t = graph.mul(a, b);
  const auto u = graph.add(t, c);
  const auto out = graph.sum(u);
  graph.set_outputs({out});

  ElementwiseFusion fusion;
  const auto optimized = fusion.run(graph);

  REQUIRE(optimized.nodes_count() == 2);
  REQUIRE(count_nodes_with_op(optimized, Op::FusedElementwise) == 1);
  REQUIRE(count_nodes_with_op(optimized, Op::Sum) == 1);

  const auto &sum_node = require_output_node(optimized);
  REQUIRE(sum_node.op == Op::Sum);
  REQUIRE(sum_node.inputs.size() == 1);
  REQUIRE(require_producer_node(optimized, sum_node.inputs[0]).op ==
          Op::FusedElementwise);
}

TEST_CASE(
    "ElementwiseFusion does not absorb a value that is also a graph output",
    "[compiler][elementwise_fusion]") {
  Graph graph;

  const auto a = graph.input(Shape{4}, Dtype::F32);
  const auto b = graph.input(Shape{4}, Dtype::F32);
  const auto c = graph.input(Shape{4}, Dtype::F32);

  const auto t = graph.mul(a, b);
  const auto out = graph.add(t, c);
  graph.set_outputs({t, out});

  ElementwiseFusion fusion;
  const auto optimized = fusion.run(graph);

  REQUIRE(optimized.nodes_count() == 2);
  REQUIRE(count_nodes_with_op(optimized, Op::Mul) == 1);
  REQUIRE(count_nodes_with_op(optimized, Op::Add) == 1);
  REQUIRE(count_nodes_with_op(optimized, Op::FusedElementwise) == 0);

  REQUIRE(require_output_node(optimized, 0).op == Op::Mul);
  REQUIRE(require_output_node(optimized, 1).op == Op::Add);
}

TEST_CASE(
    "ElementwiseFusion can fuse above an output boundary without crossing it",
    "[compiler][elementwise_fusion]") {
  Graph graph;

  const auto a = graph.input(Shape{4}, Dtype::F32);
  const auto b = graph.input(Shape{4}, Dtype::F32);
  const auto c = graph.input(Shape{4}, Dtype::F32);
  const auto d = graph.input(Shape{4}, Dtype::F32);

  const auto published = graph.mul(a, b);
  const auto internal = graph.add(published, c);
  const auto out = graph.mul(internal, d);
  graph.set_outputs({published, out});

  ElementwiseFusion fusion;
  const auto optimized = fusion.run(graph);

  REQUIRE(optimized.nodes_count() == 2);
  REQUIRE(count_nodes_with_op(optimized, Op::Mul) == 1);
  REQUIRE(count_nodes_with_op(optimized, Op::FusedElementwise) == 1);

  const auto published_new = optimized.outputs()[0];
  REQUIRE(require_producer_node(optimized, published_new).op == Op::Mul);

  const auto &fused = require_output_node(optimized, 1);
  const auto &expr = require_fused_expr(fused);
  require_binary_expr(expr, ElementwiseExprKind::Mul);
  require_binary_expr(*expr.lhs, ElementwiseExprKind::Add);

  const auto published_leaf =
      require_resolved_input_value(fused, *expr.lhs->lhs);
  REQUIRE(published_leaf == published_new);
}

TEST_CASE("ElementwiseFusion does not absorb a value with multiple uses",
          "[compiler][elementwise_fusion]") {
  Graph graph;

  const auto a = graph.input(Shape{4}, Dtype::F32);
  const auto b = graph.input(Shape{4}, Dtype::F32);
  const auto c = graph.input(Shape{4}, Dtype::F32);
  const auto d = graph.input(Shape{4}, Dtype::F32);

  const auto shared = graph.mul(a, b);
  const auto left = graph.add(shared, c);
  const auto right = graph.mul(shared, d);
  graph.set_outputs({left, right});

  ElementwiseFusion fusion;
  const auto optimized = fusion.run(graph);

  REQUIRE(optimized.nodes_count() == 3);
  REQUIRE(count_nodes_with_op(optimized, Op::Mul) == 2);
  REQUIRE(count_nodes_with_op(optimized, Op::Add) == 1);
  REQUIRE(count_nodes_with_op(optimized, Op::FusedElementwise) == 0);

  const auto &left_node = require_output_node(optimized, 0);
  const auto &right_node = require_output_node(optimized, 1);
  REQUIRE(left_node.inputs[0] == right_node.inputs[0]);
}

TEST_CASE(
    "ElementwiseFusion counts repeated use in one consumer as multiple uses",
    "[compiler][elementwise_fusion]") {
  Graph graph;

  const auto a = graph.input(Shape{4}, Dtype::F32);
  const auto b = graph.input(Shape{4}, Dtype::F32);

  const auto t = graph.mul(a, b);
  const auto out = graph.add(t, t);
  graph.set_outputs({out});

  ElementwiseFusion fusion;
  const auto optimized = fusion.run(graph);

  REQUIRE(optimized.nodes_count() == 2);
  REQUIRE(count_nodes_with_op(optimized, Op::Mul) == 1);
  REQUIRE(count_nodes_with_op(optimized, Op::Add) == 1);
  REQUIRE(count_nodes_with_op(optimized, Op::FusedElementwise) == 0);

  const auto &add = require_output_node(optimized);
  REQUIRE(add.inputs[0] == add.inputs[1]);
}

// =============================================================================
// Branching / remapping / multiple outputs
// =============================================================================

TEST_CASE("ElementwiseFusion fuses around a shared boundary without "
          "duplicating its producer",
          "[compiler][elementwise_fusion]") {
  Graph graph;

  const auto a = graph.input(Shape{4}, Dtype::F32);
  const auto b = graph.input(Shape{4}, Dtype::F32);
  const auto c = graph.input(Shape{4}, Dtype::F32);
  const auto d = graph.input(Shape{4}, Dtype::F32);

  const auto shared = graph.mul(a, b);
  const auto left = graph.add(shared, c);
  const auto right = graph.mul(shared, d);
  const auto out = graph.add(left, right);
  graph.set_outputs({out});

  ElementwiseFusion fusion;
  const auto optimized = fusion.run(graph);

  REQUIRE(optimized.nodes_count() == 2);
  REQUIRE(count_nodes_with_op(optimized, Op::Mul) == 1); // shared only
  REQUIRE(count_nodes_with_op(optimized, Op::FusedElementwise) == 1);

  const auto &fused = require_output_node(optimized);
  const auto &expr = require_fused_expr(fused);
  require_binary_expr(expr, ElementwiseExprKind::Add);
  require_binary_expr(*expr.lhs, ElementwiseExprKind::Add);
  require_binary_expr(*expr.rhs, ElementwiseExprKind::Mul);

  const auto shared_from_left =
      require_resolved_input_value(fused, *expr.lhs->lhs);
  const auto shared_from_right =
      require_resolved_input_value(fused, *expr.rhs->lhs);

  REQUIRE(shared_from_left == shared_from_right);
  REQUIRE(require_producer_node(optimized, shared_from_left).op == Op::Mul);
}

TEST_CASE("ElementwiseFusion creates independent fused islands for independent "
          "outputs",
          "[compiler][elementwise_fusion]") {
  Graph graph;

  const auto a = graph.input(Shape{3}, Dtype::F32);
  const auto b = graph.input(Shape{3}, Dtype::F32);
  const auto c = graph.input(Shape{3}, Dtype::F32);
  const auto d = graph.input(Shape{3}, Dtype::F32);
  const auto e = graph.input(Shape{3}, Dtype::F32);
  const auto f = graph.input(Shape{3}, Dtype::F32);

  const auto t1 = graph.mul(a, b);
  const auto out1 = graph.add(t1, c);

  const auto t2 = graph.add(d, e);
  const auto out2 = graph.mul(t2, f);

  graph.set_outputs({out2, out1});

  ElementwiseFusion fusion;
  const auto optimized = fusion.run(graph);

  REQUIRE(optimized.nodes_count() == 2);
  REQUIRE(count_nodes_with_op(optimized, Op::FusedElementwise) == 2);
  REQUIRE(optimized.outputs().size() == 2);

  REQUIRE(require_output_node(optimized, 0).op == Op::FusedElementwise);
  REQUIRE(require_output_node(optimized, 1).op == Op::FusedElementwise);

  const auto &expr0 = require_fused_expr(require_output_node(optimized, 0));
  const auto &expr1 = require_fused_expr(require_output_node(optimized, 1));
  REQUIRE(expr0.kind == ElementwiseExprKind::Mul);
  REQUIRE(expr1.kind == ElementwiseExprKind::Add);
}

TEST_CASE(
    "ElementwiseFusion preserves duplicate graph outputs through remapping",
    "[compiler][elementwise_fusion]") {
  Graph graph;

  const auto a = graph.input(Shape{3}, Dtype::F32);
  const auto b = graph.input(Shape{3}, Dtype::F32);
  const auto c = graph.input(Shape{3}, Dtype::F32);

  const auto t = graph.mul(a, b);
  const auto out = graph.add(t, c);
  graph.set_outputs({out, out});

  ElementwiseFusion fusion;
  const auto optimized = fusion.run(graph);

  REQUIRE(optimized.outputs().size() == 2);
  REQUIRE(optimized.outputs()[0] == optimized.outputs()[1]);
  REQUIRE(optimized.nodes_count() == 1);
  REQUIRE(require_output_node(optimized, 0).op == Op::FusedElementwise);
}

TEST_CASE(
    "ElementwiseFusion reuses one rewritten shared non-elementwise boundary",
    "[compiler][elementwise_fusion]") {
  Graph graph;

  const auto a = graph.input(Shape{2, 3}, Dtype::F32);
  const auto b = graph.input(Shape{3, 2}, Dtype::F32);
  const auto c = graph.input(Shape{2, 2}, Dtype::F32);
  const auto d = graph.input(Shape{2, 2}, Dtype::F32);

  const auto shared = graph.matmul(a, b);
  const auto out1 = graph.add(shared, c);
  const auto out2 = graph.mul(shared, d);
  graph.set_outputs({out1, out2});

  ElementwiseFusion fusion;
  const auto optimized = fusion.run(graph);

  REQUIRE(optimized.nodes_count() == 3);
  REQUIRE(count_nodes_with_op(optimized, Op::MatMul) == 1);

  const auto &left = require_output_node(optimized, 0);
  const auto &right = require_output_node(optimized, 1);
  REQUIRE(left.inputs[0] == right.inputs[0]);
  REQUIRE(require_producer_node(optimized, left.inputs[0]).op == Op::MatMul);
}

// =============================================================================
// Repeated boundary values / metadata / source graph immutability
// =============================================================================

TEST_CASE("ElementwiseFusion correctly represents repeated occurrences of the "
          "same boundary value",
          "[compiler][elementwise_fusion]") {
  Graph graph;

  const auto a = graph.input(Shape{4}, Dtype::F32);
  const auto t = graph.mul(a, a);
  const auto out = graph.add(t, a);
  graph.set_outputs({out});

  ElementwiseFusion fusion;
  const auto optimized = fusion.run(graph);

  const auto &fused = require_output_node(optimized);
  const auto &expr = require_fused_expr(fused);

  require_binary_expr(expr, ElementwiseExprKind::Add);
  require_binary_expr(*expr.lhs, ElementwiseExprKind::Mul);

  const auto lhs_a = require_resolved_input_value(fused, *expr.lhs->lhs);
  const auto rhs_a = require_resolved_input_value(fused, *expr.lhs->rhs);
  const auto outer_a = require_resolved_input_value(fused, *expr.rhs);

  REQUIRE(lhs_a == optimized.inputs()[0]);
  REQUIRE(rhs_a == optimized.inputs()[0]);
  REQUIRE(outer_a == optimized.inputs()[0]);
}

TEST_CASE(
    "ElementwiseFusion preserves input metadata while rebuilding the graph",
    "[compiler][elementwise_fusion]") {
  Graph graph;

  const auto a = graph.input(Shape{2, 3}, Dtype::F32);
  const auto b = graph.input(Shape{2, 3}, Dtype::F32);
  const auto c = graph.input(Shape{2, 3}, Dtype::F32);

  const auto t = graph.mul(a, b);
  const auto out = graph.add(t, c);
  graph.set_outputs({out});

  ElementwiseFusion fusion;
  const auto optimized = fusion.run(graph);

  REQUIRE(optimized.inputs().size() == 3);
  for (const auto input : optimized.inputs()) {
    REQUIRE(optimized.value(input).shape == Shape{2, 3});
    REQUIRE(optimized.value(input).dtype == Dtype::F32);
    REQUIRE_FALSE(optimized.value(input).producer.has_value());
  }
}

TEST_CASE("ElementwiseFusion does not mutate the source graph",
          "[compiler][elementwise_fusion]") {
  Graph graph;

  const auto a = graph.input(Shape{4}, Dtype::F32);
  const auto b = graph.input(Shape{4}, Dtype::F32);
  const auto c = graph.input(Shape{4}, Dtype::F32);
  const auto t = graph.mul(a, b);
  const auto out = graph.add(t, c);
  graph.set_outputs({out});

  ElementwiseFusion fusion;
  const auto optimized = fusion.run(graph);

  REQUIRE(optimized.nodes_count() == 1);

  REQUIRE(graph.inputs().size() == 3);
  REQUIRE(graph.outputs().size() == 1);
  REQUIRE(graph.outputs()[0] == out);
  REQUIRE(graph.nodes_count() == 2);
  REQUIRE(graph.values_count() == 5);
  REQUIRE(require_producer_node(graph, t).op == Op::Mul);
  REQUIRE(require_producer_node(graph, out).op == Op::Add);
}

// =============================================================================
// Negative / pass preconditions
// =============================================================================

TEST_CASE(
    "ElementwiseFusion rejects a graph that already contains FusedElementwise",
    "[compiler][elementwise_fusion]") {
  Graph graph;

  const auto a = graph.input(Shape{4}, Dtype::F32);
  const auto b = graph.input(Shape{4}, Dtype::F32);

  auto expr = make_binary_expr(ElementwiseExprKind::Add, 0, 1);
  const auto fused = graph.fused_elementwise({a, b}, std::move(expr));
  graph.set_outputs({fused});

  ElementwiseFusion fusion;
  REQUIRE_THROWS_AS(fusion.run(graph), std::logic_error);
}

// =============================================================================
// Current-legality nuance
// =============================================================================

TEST_CASE(
    "ElementwiseFusion dead consumer currently prevents producer absorption",
    "[compiler][elementwise_fusion][current-behavior]") {
  Graph graph;

  const auto a = graph.input(Shape{4}, Dtype::F32);
  const auto b = graph.input(Shape{4}, Dtype::F32);
  const auto c = graph.input(Shape{4}, Dtype::F32);
  const auto d = graph.input(Shape{4}, Dtype::F32);

  const auto t = graph.mul(a, b);
  const auto live_out = graph.add(t, c);
  graph.mul(t, d); // dead consumer still contributes to global use_count
  graph.set_outputs({live_out});

  ElementwiseFusion fusion;
  const auto optimized = fusion.run(graph);

  // Current implementation computes use_count over every old node before
  // reachability rewriting. Therefore t has use_count == 2 and is not absorbed,
  // even though one consumer is dead and disappears from the rebuilt graph.
  REQUIRE(optimized.nodes_count() == 2);
  REQUIRE(count_nodes_with_op(optimized, Op::Mul) == 1);
  REQUIRE(count_nodes_with_op(optimized, Op::Add) == 1);
  REQUIRE(count_nodes_with_op(optimized, Op::FusedElementwise) == 0);
}
