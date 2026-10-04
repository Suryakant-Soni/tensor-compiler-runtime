#include <catch2/catch_test_macros.hpp>

#include "tcr/compiler/lowering.hpp"
#include "tcr/compiler/schedule.hpp"
#include "tcr/compiler/scheduler.hpp"
#include "tcr/ir/dtype.hpp"
#include "tcr/ir/elementwise_expr.hpp"
#include "tcr/ir/graph.hpp"
#include "tcr/ir/ids.hpp"
#include "tcr/ir/shape.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using tcr::compiler::KernelUnit;
using tcr::compiler::Schedule;
using tcr::compiler::Scheduler;

using tcr::ir::Dtype;
using tcr::ir::ElementwiseExpr;
using tcr::ir::ElementwiseExprKind;
using tcr::ir::Graph;
using tcr::ir::Shape;
using tcr::ir::ValueId;

namespace {

// =============================================================================
// Helpers
// =============================================================================

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

bool contains_value(const std::vector<ValueId> &values, ValueId value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

std::size_t kernel_index_containing(const Schedule &schedule, ValueId value) {
  for (std::size_t k = 0; k < schedule.kernels.size(); ++k) {
    if (contains_value(schedule.kernels[k].members, value)) {
      return k;
    }
  }

  throw std::logic_error("expected value is not a member of any kernel");
}

const KernelUnit &kernel_containing(const Schedule &schedule, ValueId value) {
  return schedule.kernels[kernel_index_containing(schedule, value)];
}

std::size_t member_occurrences(const Schedule &schedule, ValueId value) {
  std::size_t count = 0;

  for (const auto &kernel : schedule.kernels) {
    count += static_cast<std::size_t>(
        std::count(kernel.members.begin(), kernel.members.end(), value));
  }

  return count;
}

} // namespace

// =============================================================================
// Empty / input-only graphs
// =============================================================================

TEST_CASE("Scheduler produces empty schedule for empty graph",
          "[compiler][scheduler]") {
  Graph graph;

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.empty());
}

TEST_CASE(
    "Scheduler produces empty schedule when graph has inputs but no outputs",
    "[compiler][scheduler]") {
  Graph graph;

  graph.input(Shape{4}, Dtype::F32);
  graph.input(Shape{4}, Dtype::F32);

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.empty());
}

TEST_CASE(
    "Scheduler does not create kernel for graph input used directly as output",
    "[compiler][scheduler]") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);
  graph.set_outputs({x});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  // Inputs do not have producer operations and therefore are not kernel
  // members.
  REQUIRE(schedule.kernels.empty());
}

// =============================================================================
// Basic singleton kernels
// =============================================================================

TEST_CASE("Scheduler creates singleton kernel for Add",
          "[compiler][scheduler]") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);
  const auto y = graph.input(Shape{4}, Dtype::F32);
  const auto out = graph.add(x, y);

  graph.set_outputs({out});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.size() == 1);

  const auto &kernel = schedule.kernels[0];

  const std::vector<ValueId> expected_members{out};
  const std::vector<ValueId> expected_inputs{x, y};
  const std::vector<ValueId> expected_outputs{out};

  REQUIRE(kernel.members == expected_members);
  REQUIRE(kernel.boundary_inputs == expected_inputs);
  REQUIRE(kernel.materialized_outputs == expected_outputs);
}

TEST_CASE("Scheduler creates singleton kernel for Mul",
          "[compiler][scheduler]") {
  Graph graph;

  const auto x = graph.input(Shape{3}, Dtype::F32);
  const auto y = graph.input(Shape{3}, Dtype::F32);
  const auto out = graph.mul(x, y);

  graph.set_outputs({out});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.size() == 1);

  const auto &kernel = schedule.kernels[0];

  REQUIRE(kernel.members == std::vector<ValueId>{out});
  REQUIRE(kernel.boundary_inputs == std::vector<ValueId>{x, y});
  REQUIRE(kernel.materialized_outputs == std::vector<ValueId>{out});
}

TEST_CASE("Scheduler creates singleton kernel for Sum of graph input",
          "[compiler][scheduler]") {
  Graph graph;

  const auto x = graph.input(Shape{2, 3}, Dtype::F32);
  const auto out = graph.sum(x);

  graph.set_outputs({out});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.size() == 1);

  const auto &kernel = schedule.kernels[0];

  REQUIRE(kernel.members == std::vector<ValueId>{out});
  REQUIRE(kernel.boundary_inputs == std::vector<ValueId>{x});
  REQUIRE(kernel.materialized_outputs == std::vector<ValueId>{out});
}

TEST_CASE("Scheduler creates singleton kernel for MatMul",
          "[compiler][scheduler]") {
  Graph graph;

  const auto a = graph.input(Shape{2, 3}, Dtype::F32);
  const auto b = graph.input(Shape{3, 4}, Dtype::F32);
  const auto out = graph.matmul(a, b);

  graph.set_outputs({out});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.size() == 1);

  const auto &kernel = schedule.kernels[0];

  REQUIRE(kernel.members == std::vector<ValueId>{out});
  REQUIRE(kernel.boundary_inputs == std::vector<ValueId>{a, b});
  REQUIRE(kernel.materialized_outputs == std::vector<ValueId>{out});
}

// =============================================================================
// Dependency ordering / cross-kernel materialization
// =============================================================================

TEST_CASE("Scheduler orders producer kernel before consumer kernel",
          "[compiler][scheduler]") {
  Graph graph;

  const auto a = graph.input(Shape{2, 3}, Dtype::F32);
  const auto b = graph.input(Shape{3, 4}, Dtype::F32);
  const auto bias = graph.input(Shape{2, 4}, Dtype::F32);

  const auto product = graph.matmul(a, b);
  const auto out = graph.add(product, bias);

  graph.set_outputs({out});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.size() == 2);

  const auto product_kernel = kernel_index_containing(schedule, product);
  const auto out_kernel = kernel_index_containing(schedule, out);

  REQUIRE(product_kernel < out_kernel);

  REQUIRE(schedule.kernels[product_kernel].materialized_outputs ==
          std::vector<ValueId>{product});

  REQUIRE(schedule.kernels[out_kernel].boundary_inputs ==
          std::vector<ValueId>{product, bias});

  REQUIRE(schedule.kernels[out_kernel].materialized_outputs ==
          std::vector<ValueId>{out});
}

TEST_CASE("Scheduler materializes value crossing a kernel boundary",
          "[compiler][scheduler]") {
  Graph graph;

  const auto a = graph.input(Shape{2}, Dtype::F32);
  const auto b = graph.input(Shape{2}, Dtype::F32);
  const auto c = graph.input(Shape{2}, Dtype::F32);

  const auto first = graph.add(a, b);
  const auto out = graph.mul(first, c);

  graph.set_outputs({out});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.size() == 2);

  const auto &first_kernel = kernel_containing(schedule, first);
  const auto &out_kernel = kernel_containing(schedule, out);

  REQUIRE(first_kernel.materialized_outputs == std::vector<ValueId>{first});

  REQUIRE(out_kernel.boundary_inputs == std::vector<ValueId>{first, c});
}

// =============================================================================
// Boundary-input deduplication
// =============================================================================

TEST_CASE("Scheduler deduplicates repeated boundary input",
          "[compiler][scheduler]") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);
  const auto out = graph.add(x, x);

  graph.set_outputs({out});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.size() == 1);

  const auto &kernel = schedule.kernels[0];

  REQUIRE(kernel.boundary_inputs == std::vector<ValueId>{x});
}

TEST_CASE("Scheduler deduplicates boundary value used twice by grouped member",
          "[compiler][scheduler]") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);

  const auto doubled = graph.add(x, x);
  const auto out = graph.sum(doubled);

  graph.set_outputs({out});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.size() == 1);

  const auto &kernel = schedule.kernels[0];

  REQUIRE(kernel.members == std::vector<ValueId>{doubled, out});

  REQUIRE(kernel.boundary_inputs == std::vector<ValueId>{x});

  REQUIRE(kernel.materialized_outputs == std::vector<ValueId>{out});
}

// =============================================================================
// Reachability
// =============================================================================

TEST_CASE("Scheduler excludes unreachable computed operations",
          "[compiler][scheduler]") {
  Graph graph;

  const auto x = graph.input(Shape{3}, Dtype::F32);
  const auto y = graph.input(Shape{3}, Dtype::F32);
  const auto z = graph.input(Shape{3}, Dtype::F32);

  const auto live = graph.add(x, y);

  const auto dead_a = graph.mul(x, z);
  const auto dead_b = graph.add(dead_a, z);

  (void)dead_b;

  graph.set_outputs({live});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.size() == 1);
  REQUIRE(schedule.kernels[0].members == std::vector<ValueId>{live});

  REQUIRE(member_occurrences(schedule, dead_a) == 0);
  REQUIRE(member_occurrences(schedule, dead_b) == 0);

  // z belongs only to the unreachable subtree.
  REQUIRE_FALSE(contains_value(schedule.kernels[0].boundary_inputs, z));
}

TEST_CASE(
    "Unreachable consumer does not prevent reachable producer Sum grouping",
    "[compiler][scheduler][reachability]") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);
  const auto y = graph.input(Shape{4}, Dtype::F32);
  const auto z = graph.input(Shape{4}, Dtype::F32);

  const auto mapped = graph.add(x, y);
  const auto out = graph.sum(mapped);

  // Dead consumer of mapped.
  const auto dead = graph.mul(mapped, z);
  (void)dead;

  graph.set_outputs({out});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  // Consumer analysis is reachability-scoped, so the dead consumer must not
  // block grouping.
  REQUIRE(schedule.kernels.size() == 1);

  REQUIRE(schedule.kernels[0].members == std::vector<ValueId>{mapped, out});

  REQUIRE_FALSE(contains_value(schedule.kernels[0].boundary_inputs, z));
}

// =============================================================================
// Add -> Sum grouping
// =============================================================================

TEST_CASE("Scheduler groups Add followed by Sum",
          "[compiler][scheduler][grouping]") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);
  const auto y = graph.input(Shape{4}, Dtype::F32);

  const auto added = graph.add(x, y);
  const auto out = graph.sum(added);

  graph.set_outputs({out});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.size() == 1);

  const auto &kernel = schedule.kernels[0];

  REQUIRE(kernel.members == std::vector<ValueId>{added, out});

  REQUIRE(kernel.boundary_inputs == std::vector<ValueId>{x, y});

  // added stays internal; only Sum survives the kernel boundary.
  REQUIRE(kernel.materialized_outputs == std::vector<ValueId>{out});

  REQUIRE_FALSE(contains_value(kernel.materialized_outputs, added));
}

// =============================================================================
// Mul -> Sum grouping
// =============================================================================

TEST_CASE("Scheduler groups Mul followed by Sum",
          "[compiler][scheduler][grouping]") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);
  const auto y = graph.input(Shape{4}, Dtype::F32);

  const auto product = graph.mul(x, y);
  const auto out = graph.sum(product);

  graph.set_outputs({out});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.size() == 1);

  const auto &kernel = schedule.kernels[0];

  REQUIRE(kernel.members == std::vector<ValueId>{product, out});

  REQUIRE(kernel.boundary_inputs == std::vector<ValueId>{x, y});

  REQUIRE(kernel.materialized_outputs == std::vector<ValueId>{out});

  REQUIRE_FALSE(contains_value(kernel.materialized_outputs, product));
}

// =============================================================================
// FusedElementwise -> Sum grouping
// =============================================================================

TEST_CASE("Scheduler groups FusedElementwise followed by Sum",
          "[compiler][scheduler][grouping]") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);
  const auto y = graph.input(Shape{4}, Dtype::F32);
  const auto z = graph.input(Shape{4}, Dtype::F32);

  auto expr = binary_expr(
      ElementwiseExprKind::Add,
      binary_expr(ElementwiseExprKind::Mul, input_expr(0), input_expr(1)),
      input_expr(2));

  const auto fused = graph.fused_elementwise({x, y, z}, std::move(expr));

  const auto out = graph.sum(fused);

  graph.set_outputs({out});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.size() == 1);

  const auto &kernel = schedule.kernels[0];

  REQUIRE(kernel.members == std::vector<ValueId>{fused, out});

  REQUIRE(kernel.boundary_inputs == std::vector<ValueId>{x, y, z});

  REQUIRE(kernel.materialized_outputs == std::vector<ValueId>{out});

  REQUIRE_FALSE(contains_value(kernel.materialized_outputs, fused));
}

TEST_CASE("Scheduler deduplicates duplicate FusedElementwise boundary values",
          "[compiler][scheduler][grouping]") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);

  auto expr =
      binary_expr(ElementwiseExprKind::Add, input_expr(0), input_expr(1));

  const auto fused = graph.fused_elementwise({x, x}, std::move(expr));

  const auto out = graph.sum(fused);

  graph.set_outputs({out});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.size() == 1);

  REQUIRE(schedule.kernels[0].members == std::vector<ValueId>{fused, out});

  REQUIRE(schedule.kernels[0].boundary_inputs == std::vector<ValueId>{x});
}

// =============================================================================
// Negative grouping cases
// =============================================================================

TEST_CASE("Scheduler does not group elementwise producer when producer is "
          "graph output",
          "[compiler][scheduler][no-group]") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);
  const auto y = graph.input(Shape{4}, Dtype::F32);

  const auto added = graph.add(x, y);
  const auto reduced = graph.sum(added);

  // added must survive as an externally visible graph output.
  graph.set_outputs({added, reduced});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.size() == 2);

  const auto added_k = kernel_index_containing(schedule, added);
  const auto reduced_k = kernel_index_containing(schedule, reduced);

  REQUIRE(added_k < reduced_k);

  REQUIRE(schedule.kernels[added_k].members == std::vector<ValueId>{added});

  REQUIRE(schedule.kernels[added_k].materialized_outputs ==
          std::vector<ValueId>{added});

  REQUIRE(schedule.kernels[reduced_k].members == std::vector<ValueId>{reduced});

  REQUIRE(schedule.kernels[reduced_k].boundary_inputs ==
          std::vector<ValueId>{added});

  // Graph output + later consumer must still produce only one materialized
  // output entry.
  REQUIRE(schedule.kernels[added_k].materialized_outputs.size() == 1);
}

TEST_CASE("Scheduler does not group elementwise producer with Sum when "
          "producer has multiple consumers",
          "[compiler][scheduler][no-group]") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);
  const auto y = graph.input(Shape{4}, Dtype::F32);
  const auto z = graph.input(Shape{4}, Dtype::F32);

  const auto shared = graph.add(x, y);

  const auto reduced = graph.sum(shared);
  const auto other = graph.mul(shared, z);

  graph.set_outputs({reduced, other});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.size() == 3);

  const auto &shared_kernel = kernel_containing(schedule, shared);
  const auto &reduced_kernel = kernel_containing(schedule, reduced);
  const auto &other_kernel = kernel_containing(schedule, other);

  REQUIRE(shared_kernel.members == std::vector<ValueId>{shared});

  REQUIRE(reduced_kernel.members == std::vector<ValueId>{reduced});

  REQUIRE(other_kernel.members == std::vector<ValueId>{other});

  // shared crosses into two later kernels.
  REQUIRE(shared_kernel.materialized_outputs == std::vector<ValueId>{shared});

  REQUIRE(reduced_kernel.boundary_inputs == std::vector<ValueId>{shared});

  REQUIRE(contains_value(other_kernel.boundary_inputs, shared));
}

TEST_CASE("Scheduler does not group MatMul followed by Sum",
          "[compiler][scheduler][no-group]") {
  Graph graph;

  const auto a = graph.input(Shape{2, 3}, Dtype::F32);
  const auto b = graph.input(Shape{3, 2}, Dtype::F32);

  const auto product = graph.matmul(a, b);
  const auto out = graph.sum(product);

  graph.set_outputs({out});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.size() == 2);

  const auto product_k = kernel_index_containing(schedule, product);
  const auto out_k = kernel_index_containing(schedule, out);

  REQUIRE(product_k < out_k);

  REQUIRE(schedule.kernels[product_k].members == std::vector<ValueId>{product});

  REQUIRE(schedule.kernels[product_k].materialized_outputs ==
          std::vector<ValueId>{product});

  REQUIRE(schedule.kernels[out_k].members == std::vector<ValueId>{out});

  REQUIRE(schedule.kernels[out_k].boundary_inputs ==
          std::vector<ValueId>{product});
}

TEST_CASE("Scheduler does not group ordinary elementwise chain without Sum",
          "[compiler][scheduler][no-group]") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);
  const auto y = graph.input(Shape{4}, Dtype::F32);
  const auto z = graph.input(Shape{4}, Dtype::F32);

  const auto first = graph.add(x, y);
  const auto out = graph.mul(first, z);

  graph.set_outputs({out});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.size() == 2);

  REQUIRE(kernel_containing(schedule, first).members ==
          std::vector<ValueId>{first});

  REQUIRE(kernel_containing(schedule, out).members ==
          std::vector<ValueId>{out});
}

// =============================================================================
// Immediate producer grouping only
// =============================================================================

TEST_CASE("Scheduler groups only immediate elementwise producer with Sum",
          "[compiler][scheduler][grouping]") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);
  const auto y = graph.input(Shape{4}, Dtype::F32);
  const auto z = graph.input(Shape{4}, Dtype::F32);

  const auto first = graph.add(x, y);
  const auto second = graph.mul(first, z);
  const auto out = graph.sum(second);

  graph.set_outputs({out});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.size() == 2);

  const auto first_k = kernel_index_containing(schedule, first);
  const auto grouped_k = kernel_index_containing(schedule, second);

  REQUIRE(first_k < grouped_k);

  REQUIRE(schedule.kernels[first_k].members == std::vector<ValueId>{first});

  REQUIRE(schedule.kernels[first_k].materialized_outputs ==
          std::vector<ValueId>{first});

  REQUIRE(schedule.kernels[grouped_k].members ==
          std::vector<ValueId>{second, out});

  REQUIRE(schedule.kernels[grouped_k].boundary_inputs ==
          std::vector<ValueId>{first, z});

  REQUIRE(schedule.kernels[grouped_k].materialized_outputs ==
          std::vector<ValueId>{out});
}

// =============================================================================
// Owner rebuild / empty-kernel removal
// =============================================================================

TEST_CASE("Scheduler correctly handles two independent Elementwise Sum groups",
          "[compiler][scheduler][grouping]") {
  Graph graph;

  const auto a = graph.input(Shape{4}, Dtype::F32);
  const auto b = graph.input(Shape{4}, Dtype::F32);
  const auto c = graph.input(Shape{4}, Dtype::F32);
  const auto d = graph.input(Shape{4}, Dtype::F32);

  const auto first_map = graph.add(a, b);
  const auto first_sum = graph.sum(first_map);

  const auto second_map = graph.mul(c, d);
  const auto second_sum = graph.sum(second_map);

  graph.set_outputs({first_sum, second_sum});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  // Initial four singleton kernels should become two grouped kernels.
  REQUIRE(schedule.kernels.size() == 2);

  const auto &first_group = kernel_containing(schedule, first_map);

  const auto &second_group = kernel_containing(schedule, second_map);

  REQUIRE(first_group.members == std::vector<ValueId>{first_map, first_sum});

  REQUIRE(first_group.boundary_inputs == std::vector<ValueId>{a, b});

  REQUIRE(first_group.materialized_outputs == std::vector<ValueId>{first_sum});

  REQUIRE(second_group.members == std::vector<ValueId>{second_map, second_sum});

  REQUIRE(second_group.boundary_inputs == std::vector<ValueId>{c, d});

  REQUIRE(second_group.materialized_outputs ==
          std::vector<ValueId>{second_sum});

  REQUIRE(member_occurrences(schedule, first_sum) == 1);
  REQUIRE(member_occurrences(schedule, second_sum) == 1);
}

// This is particularly important because the Elementwise->Sum merge removes an
// empty kernel and shifts later kernel indices. recompute_boundaries() must use
// the rebuilt owner map rather than the pre-erasure owner indices.
TEST_CASE("Scheduler rebuilds ownership correctly after grouped kernel removal",
          "[compiler][scheduler][owner]") {
  Graph graph;

  const auto lhs = graph.input(Shape{2, 3}, Dtype::F32);
  const auto rhs = graph.input(Shape{3, 4}, Dtype::F32);
  const auto bias = graph.input(Shape{2, 4}, Dtype::F32);

  const auto product = graph.matmul(lhs, rhs);
  const auto mapped = graph.add(product, bias);
  const auto out = graph.sum(mapped);

  graph.set_outputs({out});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.size() == 2);

  const auto product_k = kernel_index_containing(schedule, product);

  const auto grouped_k = kernel_index_containing(schedule, mapped);

  REQUIRE(product_k < grouped_k);

  const auto &product_kernel = schedule.kernels[product_k];
  const auto &grouped_kernel = schedule.kernels[grouped_k];

  REQUIRE(product_kernel.members == std::vector<ValueId>{product});

  // product is consumed by the later grouped kernel, so it must still be
  // materialized after kernel indices have shifted.
  REQUIRE(product_kernel.materialized_outputs == std::vector<ValueId>{product});

  REQUIRE(grouped_kernel.members == std::vector<ValueId>{mapped, out});

  REQUIRE(grouped_kernel.boundary_inputs ==
          std::vector<ValueId>{product, bias});

  REQUIRE(grouped_kernel.materialized_outputs == std::vector<ValueId>{out});
}

// =============================================================================
// Duplicate uses / distinct consumer semantics
// =============================================================================

TEST_CASE("Scheduler handles value used twice by same consumer without "
          "duplicate boundary",
          "[compiler][scheduler][dedup]") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);
  const auto y = graph.input(Shape{4}, Dtype::F32);

  const auto first = graph.add(x, y);
  const auto doubled = graph.mul(first, first);
  const auto out = graph.sum(doubled);

  graph.set_outputs({out});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.size() == 2);

  const auto &first_kernel = kernel_containing(schedule, first);

  const auto &grouped_kernel = kernel_containing(schedule, doubled);

  REQUIRE(first_kernel.materialized_outputs == std::vector<ValueId>{first});

  REQUIRE(grouped_kernel.members == std::vector<ValueId>{doubled, out});

  // doubled uses first twice, but the kernel interface should contain first
  // once.
  REQUIRE(grouped_kernel.boundary_inputs == std::vector<ValueId>{first});

  REQUIRE(grouped_kernel.materialized_outputs == std::vector<ValueId>{out});
}

// =============================================================================
// Duplicate graph outputs
// =============================================================================

TEST_CASE("Scheduler does not duplicate kernel or materialization for "
          "duplicate graph outputs",
          "[compiler][scheduler]") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);
  const auto y = graph.input(Shape{2}, Dtype::F32);

  const auto out = graph.add(x, y);

  graph.set_outputs({out, out});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.size() == 1);

  REQUIRE(schedule.kernels[0].members == std::vector<ValueId>{out});

  REQUIRE(schedule.kernels[0].materialized_outputs ==
          std::vector<ValueId>{out});

  REQUIRE(member_occurrences(schedule, out) == 1);
}

// =============================================================================
// Graph input + computed output
// =============================================================================

TEST_CASE("Scheduler represents computed output but not direct input output as "
          "kernel",
          "[compiler][scheduler]") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);
  const auto y = graph.input(Shape{2}, Dtype::F32);

  const auto added = graph.add(x, y);

  graph.set_outputs({x, added});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.size() == 1);

  REQUIRE(schedule.kernels[0].members == std::vector<ValueId>{added});

  REQUIRE(schedule.kernels[0].boundary_inputs == std::vector<ValueId>{x, y});

  REQUIRE(schedule.kernels[0].materialized_outputs ==
          std::vector<ValueId>{added});

  REQUIRE(member_occurrences(schedule, x) == 0);
}

// =============================================================================
// Independent graph outputs
// =============================================================================

TEST_CASE(
    "Scheduler schedules each independent computed graph output exactly once",
    "[compiler][scheduler]") {
  Graph graph;

  const auto x = graph.input(Shape{3}, Dtype::F32);
  const auto y = graph.input(Shape{3}, Dtype::F32);

  const auto added = graph.add(x, y);
  const auto multiplied = graph.mul(x, y);

  graph.set_outputs({multiplied, added});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.size() == 2);

  // Do not over-constrain relative order of independent computations.
  REQUIRE(member_occurrences(schedule, added) == 1);
  REQUIRE(member_occurrences(schedule, multiplied) == 1);

  const auto &add_kernel = kernel_containing(schedule, added);
  const auto &mul_kernel = kernel_containing(schedule, multiplied);

  REQUIRE(add_kernel.boundary_inputs == std::vector<ValueId>{x, y});

  REQUIRE(mul_kernel.boundary_inputs == std::vector<ValueId>{x, y});

  REQUIRE(add_kernel.materialized_outputs == std::vector<ValueId>{added});

  REQUIRE(mul_kernel.materialized_outputs == std::vector<ValueId>{multiplied});
}

// =============================================================================
// Partition invariant / diamond graph
// =============================================================================

TEST_CASE(
    "Scheduler places every reachable computed value in exactly one kernel",
    "[compiler][scheduler][invariant]") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);
  const auto y = graph.input(Shape{2}, Dtype::F32);
  const auto z = graph.input(Shape{2}, Dtype::F32);

  const auto shared = graph.mul(x, y);
  const auto left = graph.add(shared, z);
  const auto right = graph.mul(shared, z);
  const auto out = graph.add(left, right);

  graph.set_outputs({out});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  // No Sum exists, therefore no Phase-5 grouping applies.
  REQUIRE(schedule.kernels.size() == 4);

  REQUIRE(member_occurrences(schedule, shared) == 1);
  REQUIRE(member_occurrences(schedule, left) == 1);
  REQUIRE(member_occurrences(schedule, right) == 1);
  REQUIRE(member_occurrences(schedule, out) == 1);

  REQUIRE(member_occurrences(schedule, x) == 0);
  REQUIRE(member_occurrences(schedule, y) == 0);
  REQUIRE(member_occurrences(schedule, z) == 0);

  REQUIRE(kernel_containing(schedule, shared).materialized_outputs ==
          std::vector<ValueId>{shared});

  REQUIRE(kernel_containing(schedule, left).materialized_outputs ==
          std::vector<ValueId>{left});

  REQUIRE(kernel_containing(schedule, right).materialized_outputs ==
          std::vector<ValueId>{right});

  REQUIRE(kernel_containing(schedule, out).materialized_outputs ==
          std::vector<ValueId>{out});
}

// =============================================================================
// Topological ordering on longer dependency chain
// =============================================================================

TEST_CASE("Scheduler preserves dependency order across longer kernel chain",
          "[compiler][scheduler][ordering]") {
  Graph graph;

  const auto a = graph.input(Shape{2, 3}, Dtype::F32);
  const auto b = graph.input(Shape{3, 4}, Dtype::F32);
  const auto bias = graph.input(Shape{2, 4}, Dtype::F32);
  const auto scale = graph.input(Shape{2, 4}, Dtype::F32);

  const auto matmul = graph.matmul(a, b);
  const auto added = graph.add(matmul, bias);
  const auto out = graph.mul(added, scale);

  graph.set_outputs({out});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.size() == 3);

  const auto matmul_k = kernel_index_containing(schedule, matmul);

  const auto added_k = kernel_index_containing(schedule, added);

  const auto out_k = kernel_index_containing(schedule, out);

  REQUIRE(matmul_k < added_k);
  REQUIRE(added_k < out_k);

  REQUIRE(kernel_containing(schedule, matmul).materialized_outputs ==
          std::vector<ValueId>{matmul});

  REQUIRE(kernel_containing(schedule, added).materialized_outputs ==
          std::vector<ValueId>{added});

  REQUIRE(kernel_containing(schedule, out).materialized_outputs ==
          std::vector<ValueId>{out});
}

TEST_CASE("Lowering rejects schedule where consumer executes before producer",
          "[compiler][lowering][schedule]") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);
  const auto y = graph.input(Shape{4}, Dtype::F32);

  const auto added = graph.add(x, y);
  const auto out = graph.sum(added);

  graph.set_outputs({out});

  Schedule bad_schedule;

  // Wrong: Sum executes before Add.
  bad_schedule.kernels.push_back(KernelUnit{
      .members = {out},
      .boundary_inputs = {added},
      .materialized_outputs = {out},
  });

  bad_schedule.kernels.push_back(KernelUnit{
      .members = {added},
      .boundary_inputs = {x, y},
      .materialized_outputs = {added},
  });

  tcr::compiler::Lowering lowering;

  REQUIRE_THROWS_AS(lowering.lower(graph, bad_schedule), std::logic_error);
}
