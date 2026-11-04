#include <catch2/catch_test_macros.hpp>

#include "tcr/compiler/lowering.hpp"
#include "tcr/compiler/schedule.hpp"
#include "tcr/compiler/scheduler.hpp"
#include "tcr/ir/dtype.hpp"
#include "tcr/ir/elementwise_expr.hpp"
#include "tcr/ir/graph.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/lir/index.hpp"
#include "tcr/lir/instruction.hpp"
#include "tcr/lir/program.hpp"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

using tcr::compiler::KernelUnit;
using tcr::compiler::Lowering;
using tcr::compiler::Schedule;
using tcr::compiler::Scheduler;

using tcr::ir::Dtype;
using tcr::ir::ElementwiseExpr;
using tcr::ir::ElementwiseExprKind;
using tcr::ir::Graph;
using tcr::ir::Shape;
using tcr::ir::ValueId;

namespace lir = tcr::lir;

namespace {

// -----------------------------------------------------------------------------
// LIR inspection helpers
// -----------------------------------------------------------------------------

const lir::For &require_for(const lir::Instruction &instruction) {
  const auto *loop =
      std::get_if<std::unique_ptr<lir::For>>(&instruction.payload);

  REQUIRE(loop != nullptr);
  REQUIRE(*loop != nullptr);

  return **loop;
}

template <typename T>
const T &require_instruction(const lir::Instruction &instruction) {

  const auto *op = std::get_if<T>(&instruction.payload);

  REQUIRE(op != nullptr);

  return *op;
}

std::size_t require_constant_index(const lir::IndexExpr &index) {

  const auto *constant = std::get_if<lir::ConstantIndex>(&index);

  REQUIRE(constant != nullptr);

  return constant->value;
}

lir::IndexId require_index_register(const lir::IndexExpr &index) {

  const auto *reg = std::get_if<lir::IndexId>(&index);

  REQUIRE(reg != nullptr);

  return *reg;
}

ElementwiseExpr input_expr(std::size_t input_index) {
  return ElementwiseExpr{
      .kind = ElementwiseExprKind::Input,
      .input_index = input_index,
      .lhs = nullptr,
      .rhs = nullptr,
  };
}

ElementwiseExpr add_expr(ElementwiseExpr lhs, ElementwiseExpr rhs) {
  return ElementwiseExpr{
      .kind = ElementwiseExprKind::Add,
      .input_index = 0,
      .lhs = std::make_unique<ElementwiseExpr>(std::move(lhs)),
      .rhs = std::make_unique<ElementwiseExpr>(std::move(rhs)),
  };
}

ElementwiseExpr mul_expr(ElementwiseExpr lhs, ElementwiseExpr rhs) {
  return ElementwiseExpr{
      .kind = ElementwiseExprKind::Mul,
      .input_index = 0,
      .lhs = std::make_unique<ElementwiseExpr>(std::move(lhs)),
      .rhs = std::make_unique<ElementwiseExpr>(std::move(rhs)),
  };
}

lir::Program lower_with_schedule(const Graph &graph) {
  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  Lowering lowering;
  auto result = lowering.lower(graph, schedule);
  return std::move(result.program);
}

std::size_t count_buffers_with_shape(const lir::Program &program,
                                     const Shape &shape) {
  std::size_t count = 0;
  for (const auto &buffer : program.buffers()) {
    if (buffer.shape == shape) {
      ++count;
    }
  }
  return count;
}

} // namespace

// -----------------------------------------------------------------------------
// Empty graph
// -----------------------------------------------------------------------------

TEST_CASE("Lowering empty graph produces empty LIR program") {
  Graph graph;

  const auto program = lower_with_schedule(graph);

  REQUIRE(program.buffers().empty());
  REQUIRE(program.inputs().empty());
  REQUIRE(program.outputs().empty());
  REQUIRE(program.instructions().empty());

  REQUIRE(program.buffer_count() == 0);
  REQUIRE(program.scalar_count() == 0);
  REQUIRE(program.index_count() == 0);
}

// -----------------------------------------------------------------------------
// Input
// -----------------------------------------------------------------------------

TEST_CASE("Lowering graph input creates one LIR input buffer") {
  Graph graph;

  const auto x = graph.input(Shape{3}, Dtype::F32);

  graph.set_outputs({x});

  const auto program = lower_with_schedule(graph);

  REQUIRE(program.buffer_count() == 1);
  REQUIRE(program.buffers().size() == 1);

  REQUIRE(program.inputs().size() == 1);
  REQUIRE(program.outputs().size() == 1);

  REQUIRE(program.instructions().empty());

  REQUIRE(program.scalar_count() == 0);
  REQUIRE(program.index_count() == 0);

  REQUIRE(program.buffers()[0].id.value == 0);
  REQUIRE(program.buffers()[0].shape == Shape{3});
  REQUIRE(program.buffers()[0].dtype == Dtype::F32);

  REQUIRE(program.inputs()[0].value == x);
  REQUIRE(program.inputs()[0].buffer.value == 0);

  REQUIRE(program.outputs()[0].value == 0);
}

TEST_CASE("Lowering scalar input preserves scalar metadata") {
  Graph graph;

  const auto x = graph.input(Shape{}, Dtype::F32);

  graph.set_outputs({x});

  const auto program = lower_with_schedule(graph);

  REQUIRE(program.buffer_count() == 1);

  REQUIRE(program.buffers()[0].shape == Shape{});
  REQUIRE(program.buffers()[0].dtype == Dtype::F32);

  REQUIRE(program.inputs().size() == 1);
  REQUIRE(program.outputs().size() == 1);
}

TEST_CASE("Lowering multiple input outputs preserves output ordering") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);

  const auto y = graph.input(Shape{3}, Dtype::F32);

  graph.set_outputs({
      y,
      x,
  });

  const auto program = lower_with_schedule(graph);

  REQUIRE(program.buffer_count() == 2);
  REQUIRE(program.inputs().size() == 2);
  REQUIRE(program.outputs().size() == 2);

  // y is encountered first because it is the first requested output.
  REQUIRE(program.outputs()[0].value == 0);
  REQUIRE(program.outputs()[1].value == 1);

  REQUIRE(program.buffers()[0].shape == Shape{3});
  REQUIRE(program.buffers()[1].shape == Shape{2});
}

// -----------------------------------------------------------------------------
// Add
// -----------------------------------------------------------------------------

TEST_CASE("Lowering Add creates input input output buffers") {
  Graph graph;

  const auto a = graph.input(Shape{4}, Dtype::F32);

  const auto b = graph.input(Shape{4}, Dtype::F32);

  const auto out = graph.add(a, b);

  graph.set_outputs({out});

  const auto program = lower_with_schedule(graph);

  REQUIRE(program.buffer_count() == 3);
  REQUIRE(program.inputs().size() == 2);
  REQUIRE(program.outputs().size() == 1);

  REQUIRE(program.buffers()[0].shape == Shape{4});
  REQUIRE(program.buffers()[1].shape == Shape{4});
  REQUIRE(program.buffers()[2].shape == Shape{4});

  REQUIRE(program.buffers()[0].dtype == Dtype::F32);
  REQUIRE(program.buffers()[1].dtype == Dtype::F32);
  REQUIRE(program.buffers()[2].dtype == Dtype::F32);

  REQUIRE(program.outputs()[0].value == 2);
}

TEST_CASE("Lowering Add allocates expected scalar and index registers") {
  Graph graph;

  const auto a = graph.input(Shape{4}, Dtype::F32);

  const auto b = graph.input(Shape{4}, Dtype::F32);

  const auto out = graph.add(a, b);

  graph.set_outputs({out});

  const auto program = lower_with_schedule(graph);

  REQUIRE(program.scalar_count() == 3);
  REQUIRE(program.index_count() == 1);
}

TEST_CASE("Lowering Add creates one elementwise loop") {
  Graph graph;

  const auto a = graph.input(Shape{4}, Dtype::F32);

  const auto b = graph.input(Shape{4}, Dtype::F32);

  const auto out = graph.add(a, b);

  graph.set_outputs({out});

  const auto program = lower_with_schedule(graph);

  REQUIRE(program.instructions().size() == 1);

  const auto &loop = require_for(program.instructions()[0]);

  REQUIRE(loop.index.value == 0);

  REQUIRE(require_constant_index(loop.begin) == 0);

  REQUIRE(require_constant_index(loop.end) == 4);

  REQUIRE(loop.body.size() == 4);

  require_instruction<lir::Load>(loop.body[0]);

  require_instruction<lir::Load>(loop.body[1]);

  require_instruction<lir::Add>(loop.body[2]);

  require_instruction<lir::Store>(loop.body[3]);
}

TEST_CASE("Lowering Add loop range uses total number of elements") {
  Graph graph;

  const auto a = graph.input(Shape{2, 3}, Dtype::F32);

  const auto b = graph.input(Shape{2, 3}, Dtype::F32);

  const auto out = graph.add(a, b);

  graph.set_outputs({out});

  const auto program = lower_with_schedule(graph);

  const auto &loop = require_for(program.instructions()[0]);

  REQUIRE(require_constant_index(loop.begin) == 0);

  REQUIRE(require_constant_index(loop.end) == 6);
}

// -----------------------------------------------------------------------------
// Mul
// -----------------------------------------------------------------------------

TEST_CASE("Lowering Mul creates elementwise multiplication loop") {
  Graph graph;

  const auto a = graph.input(Shape{5}, Dtype::F32);

  const auto b = graph.input(Shape{5}, Dtype::F32);

  const auto out = graph.mul(a, b);

  graph.set_outputs({out});

  const auto program = lower_with_schedule(graph);

  REQUIRE(program.buffer_count() == 3);
  REQUIRE(program.inputs().size() == 2);
  REQUIRE(program.outputs().size() == 1);

  REQUIRE(program.scalar_count() == 3);
  REQUIRE(program.index_count() == 1);

  REQUIRE(program.instructions().size() == 1);

  const auto &loop = require_for(program.instructions()[0]);

  REQUIRE(require_constant_index(loop.begin) == 0);

  REQUIRE(require_constant_index(loop.end) == 5);

  REQUIRE(loop.body.size() == 4);

  require_instruction<lir::Load>(loop.body[0]);

  require_instruction<lir::Load>(loop.body[1]);

  require_instruction<lir::Mul>(loop.body[2]);

  require_instruction<lir::Store>(loop.body[3]);
}

TEST_CASE("Lowering Add and Mul use different arithmetic LIR instructions") {
  SECTION("Add") {
    Graph graph;

    const auto a = graph.input(Shape{2}, Dtype::F32);

    const auto b = graph.input(Shape{2}, Dtype::F32);

    const auto out = graph.add(a, b);

    graph.set_outputs({out});

    const auto program = lower_with_schedule(graph);

    const auto &loop = require_for(program.instructions()[0]);

    REQUIRE(std::holds_alternative<lir::Add>(loop.body[2].payload));

    REQUIRE_FALSE(std::holds_alternative<lir::Mul>(loop.body[2].payload));
  }

  SECTION("Mul") {
    Graph graph;

    const auto a = graph.input(Shape{2}, Dtype::F32);

    const auto b = graph.input(Shape{2}, Dtype::F32);

    const auto out = graph.mul(a, b);

    graph.set_outputs({out});

    const auto program = lower_with_schedule(graph);

    const auto &loop = require_for(program.instructions()[0]);

    REQUIRE(std::holds_alternative<lir::Mul>(loop.body[2].payload));

    REQUIRE_FALSE(std::holds_alternative<lir::Add>(loop.body[2].payload));
  }
}

// -----------------------------------------------------------------------------
// Fused elementwise
// -----------------------------------------------------------------------------

TEST_CASE(
    "Lowering FusedElementwise creates boundary input and output buffers") {
  Graph graph;

  const auto a = graph.input(Shape{2, 3}, Dtype::F32);
  const auto b = graph.input(Shape{2, 3}, Dtype::F32);
  const auto c = graph.input(Shape{2, 3}, Dtype::F32);

  auto expr = add_expr(mul_expr(input_expr(0), input_expr(1)), input_expr(2));

  const auto out = graph.fused_elementwise({a, b, c}, std::move(expr));

  graph.set_outputs({out});

  const auto program = lower_with_schedule(graph);

  REQUIRE(program.buffer_count() == 4);
  REQUIRE(program.inputs().size() == 3);
  REQUIRE(program.outputs().size() == 1);

  const auto output_buffer = program.outputs()[0];
  REQUIRE(program.buffers()[output_buffer.value].shape == Shape{2, 3});
  REQUIRE(program.buffers()[output_buffer.value].dtype == Dtype::F32);
}

TEST_CASE(
    "Lowering FusedElementwise allocates one index and expression scalars") {
  Graph graph;

  const auto a = graph.input(Shape{4}, Dtype::F32);
  const auto b = graph.input(Shape{4}, Dtype::F32);
  const auto c = graph.input(Shape{4}, Dtype::F32);

  auto expr = add_expr(mul_expr(input_expr(0), input_expr(1)), input_expr(2));

  const auto out = graph.fused_elementwise({a, b, c}, std::move(expr));

  graph.set_outputs({out});

  const auto program = lower_with_schedule(graph);

  // Three boundary-load scalars + Mul result + Add result.
  REQUIRE(program.scalar_count() == 5);
  REQUIRE(program.index_count() == 1);
}

TEST_CASE("Lowering FusedElementwise emits one loop with fused arithmetic") {
  Graph graph;

  const auto a = graph.input(Shape{2, 3}, Dtype::F32);
  const auto b = graph.input(Shape{2, 3}, Dtype::F32);
  const auto c = graph.input(Shape{2, 3}, Dtype::F32);

  auto expr = add_expr(mul_expr(input_expr(0), input_expr(1)), input_expr(2));

  const auto out = graph.fused_elementwise({a, b, c}, std::move(expr));

  graph.set_outputs({out});

  const auto program = lower_with_schedule(graph);

  REQUIRE(program.instructions().size() == 1);

  const auto &loop = require_for(program.instructions()[0]);

  REQUIRE(require_constant_index(loop.begin) == 0);
  REQUIRE(require_constant_index(loop.end) == 6);

  // All boundary inputs are loaded first, then the expression is emitted
  // recursively in post-order, followed by one final store.
  REQUIRE(loop.body.size() == 6);

  require_instruction<lir::Load>(loop.body[0]);
  require_instruction<lir::Load>(loop.body[1]);
  require_instruction<lir::Load>(loop.body[2]);
  require_instruction<lir::Mul>(loop.body[3]);
  require_instruction<lir::Add>(loop.body[4]);
  require_instruction<lir::Store>(loop.body[5]);
}

TEST_CASE("Lowering FusedElementwise preserves expression scalar wiring") {
  Graph graph;

  const auto a = graph.input(Shape{3}, Dtype::F32);
  const auto b = graph.input(Shape{3}, Dtype::F32);
  const auto c = graph.input(Shape{3}, Dtype::F32);

  auto expr = add_expr(mul_expr(input_expr(0), input_expr(1)), input_expr(2));

  const auto out = graph.fused_elementwise({a, b, c}, std::move(expr));

  graph.set_outputs({out});

  const auto program = lower_with_schedule(graph);

  const auto &loop = require_for(program.instructions()[0]);

  const auto &load_a = require_instruction<lir::Load>(loop.body[0]);
  const auto &load_b = require_instruction<lir::Load>(loop.body[1]);
  const auto &load_c = require_instruction<lir::Load>(loop.body[2]);
  const auto &mul = require_instruction<lir::Mul>(loop.body[3]);
  const auto &add = require_instruction<lir::Add>(loop.body[4]);
  const auto &store = require_instruction<lir::Store>(loop.body[5]);

  REQUIRE(mul.lhs == load_a.dst);
  REQUIRE(mul.rhs == load_b.dst);

  REQUIRE(add.lhs == mul.dst);
  REQUIRE(add.rhs == load_c.dst);

  REQUIRE(store.value == add.dst);
}

TEST_CASE(
    "Lowering FusedElementwise lowers deep expression tree in post-order") {
  Graph graph;

  const auto a = graph.input(Shape{2}, Dtype::F32);
  const auto b = graph.input(Shape{2}, Dtype::F32);
  const auto c = graph.input(Shape{2}, Dtype::F32);
  const auto d = graph.input(Shape{2}, Dtype::F32);

  auto expr = mul_expr(add_expr(input_expr(0), input_expr(1)),
                       add_expr(input_expr(2), input_expr(3)));

  const auto out = graph.fused_elementwise({a, b, c, d}, std::move(expr));

  graph.set_outputs({out});

  const auto program = lower_with_schedule(graph);

  REQUIRE(program.scalar_count() == 7);
  REQUIRE(program.index_count() == 1);
  REQUIRE(program.instructions().size() == 1);

  const auto &loop = require_for(program.instructions()[0]);

  REQUIRE(loop.body.size() == 8);

  require_instruction<lir::Load>(loop.body[0]);
  require_instruction<lir::Load>(loop.body[1]);
  require_instruction<lir::Load>(loop.body[2]);
  require_instruction<lir::Load>(loop.body[3]);
  require_instruction<lir::Add>(loop.body[4]);
  require_instruction<lir::Add>(loop.body[5]);
  require_instruction<lir::Mul>(loop.body[6]);
  require_instruction<lir::Store>(loop.body[7]);
}

TEST_CASE("Lowering FusedElementwise scalar executes over one element") {
  Graph graph;

  const auto a = graph.input(Shape{}, Dtype::F32);
  const auto b = graph.input(Shape{}, Dtype::F32);

  auto expr = add_expr(input_expr(0), input_expr(1));

  const auto out = graph.fused_elementwise({a, b}, std::move(expr));

  graph.set_outputs({out});

  const auto program = lower_with_schedule(graph);

  const auto &loop = require_for(program.instructions()[0]);

  REQUIRE(require_constant_index(loop.begin) == 0);
  REQUIRE(require_constant_index(loop.end) == 1);
}

TEST_CASE(
    "Lowering FusedElementwise zero-sized tensor creates zero-trip loop") {
  Graph graph;

  const auto a = graph.input(Shape{0, 3}, Dtype::F32);
  const auto b = graph.input(Shape{0, 3}, Dtype::F32);

  auto expr = mul_expr(input_expr(0), input_expr(1));

  const auto out = graph.fused_elementwise({a, b}, std::move(expr));

  graph.set_outputs({out});

  const auto program = lower_with_schedule(graph);

  const auto &loop = require_for(program.instructions()[0]);

  REQUIRE(require_constant_index(loop.begin) == 0);
  REQUIRE(require_constant_index(loop.end) == 0);
}

TEST_CASE(
    "Lowering FusedElementwise may reuse one boundary scalar multiple times") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);

  auto expr = add_expr(input_expr(0), input_expr(0));

  const auto out = graph.fused_elementwise({x}, std::move(expr));

  graph.set_outputs({out});

  const auto program = lower_with_schedule(graph);

  REQUIRE(program.buffer_count() == 2);
  REQUIRE(program.inputs().size() == 1);
  REQUIRE(program.scalar_count() == 2);

  const auto &loop = require_for(program.instructions()[0]);

  REQUIRE(loop.body.size() == 3);

  const auto &load = require_instruction<lir::Load>(loop.body[0]);
  const auto &add = require_instruction<lir::Add>(loop.body[1]);
  require_instruction<lir::Store>(loop.body[2]);

  REQUIRE(add.lhs == load.dst);
  REQUIRE(add.rhs == load.dst);
}

TEST_CASE("Lowering duplicate fused boundary entries reuse buffer but emit "
          "separate loads") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);

  auto expr = add_expr(input_expr(0), input_expr(1));

  const auto out = graph.fused_elementwise({x, x}, std::move(expr));

  graph.set_outputs({out});

  const auto program = lower_with_schedule(graph);

  // The HIR boundary contains x twice, but require_buffer reuses x's buffer.
  REQUIRE(program.buffer_count() == 2);
  REQUIRE(program.inputs().size() == 1);
  REQUIRE(program.scalar_count() == 3);

  const auto &loop = require_for(program.instructions()[0]);

  REQUIRE(loop.body.size() == 4);

  const auto &first_load = require_instruction<lir::Load>(loop.body[0]);
  const auto &second_load = require_instruction<lir::Load>(loop.body[1]);
  require_instruction<lir::Add>(loop.body[2]);
  require_instruction<lir::Store>(loop.body[3]);

  REQUIRE(first_load.buffer_src == second_load.buffer_src);
}

TEST_CASE("Lowering FusedElementwise consumes materialized non-elementwise "
          "boundary") {
  Graph graph;

  const auto a = graph.input(Shape{2, 3}, Dtype::F32);
  const auto b = graph.input(Shape{3, 4}, Dtype::F32);
  const auto bias = graph.input(Shape{2, 4}, Dtype::F32);

  const auto product = graph.matmul(a, b);

  auto expr = add_expr(input_expr(0), input_expr(1));

  const auto out = graph.fused_elementwise({product, bias}, std::move(expr));

  graph.set_outputs({out});

  const auto program = lower_with_schedule(graph);

  REQUIRE(program.inputs().size() == 3);
  REQUIRE(program.outputs().size() == 1);

  // One MatMul nest followed by the one fused elementwise loop.
  REQUIRE(program.instructions().size() == 2);

  const auto &matmul_loop = require_for(program.instructions()[0]);
  const auto &fused_loop = require_for(program.instructions()[1]);

  REQUIRE(require_constant_index(matmul_loop.end) == 2);
  REQUIRE(require_constant_index(fused_loop.end) == 8);

  REQUIRE(fused_loop.body.size() == 4);
  require_instruction<lir::Load>(fused_loop.body[0]);
  require_instruction<lir::Load>(fused_loop.body[1]);
  require_instruction<lir::Add>(fused_loop.body[2]);
  require_instruction<lir::Store>(fused_loop.body[3]);
}

TEST_CASE(
    "Lowering duplicate FusedElementwise graph outputs lower fused node once") {
  Graph graph;

  const auto a = graph.input(Shape{2}, Dtype::F32);
  const auto b = graph.input(Shape{2}, Dtype::F32);

  auto expr = mul_expr(input_expr(0), input_expr(1));

  const auto out = graph.fused_elementwise({a, b}, std::move(expr));

  graph.set_outputs({out, out});

  const auto program = lower_with_schedule(graph);

  REQUIRE(program.instructions().size() == 1);
  REQUIRE(program.outputs().size() == 2);
  REQUIRE(program.outputs()[0] == program.outputs()[1]);
}

TEST_CASE(
    "Lowering FusedElementwise rejects expression input index out of range") {
  Graph graph;

  const auto a = graph.input(Shape{2}, Dtype::F32);
  const auto b = graph.input(Shape{2}, Dtype::F32);

  auto expr = add_expr(input_expr(0), input_expr(2));

  const auto out = graph.fused_elementwise({a, b}, std::move(expr));

  graph.set_outputs({out});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  Lowering lowering;

  REQUIRE_THROWS_AS(lowering.lower(graph, schedule), std::out_of_range);
}

TEST_CASE(
    "Lowering FusedElementwise rejects binary expression missing operand") {
  Graph graph;

  const auto a = graph.input(Shape{2}, Dtype::F32);
  const auto b = graph.input(Shape{2}, Dtype::F32);

  ElementwiseExpr malformed{
      .kind = ElementwiseExprKind::Add,
      .input_index = 0,
      .lhs = std::make_unique<ElementwiseExpr>(input_expr(0)),
      .rhs = nullptr,
  };

  const auto out = graph.fused_elementwise({a, b}, std::move(malformed));

  graph.set_outputs({out});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  Lowering lowering;

  REQUIRE_THROWS_AS(lowering.lower(graph, schedule), std::logic_error);
}

// -----------------------------------------------------------------------------
// Phase 5 grouped Elementwise -> Sum lowering
// -----------------------------------------------------------------------------

TEST_CASE("Lowering grouped Add Sum does not materialize Add intermediate",
          "[compiler][lowering][schedule][grouping]") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);
  const auto y = graph.input(Shape{4}, Dtype::F32);

  const auto added = graph.add(x, y);
  const auto out = graph.sum(added);

  graph.set_outputs({out});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.size() == 1);
  REQUIRE(schedule.kernels[0].members == std::vector<ValueId>{added, out});

  Lowering lowering;
  const auto lowering_result = lowering.lower(graph, schedule);
  const auto &program = lowering_result.program;

  // Only x, y, and the scalar Sum output need tensor buffers.
  // There is no Shape{4} buffer for `added`.
  REQUIRE(program.buffer_count() == 3);
  REQUIRE(program.inputs().size() == 2);
  REQUIRE(program.outputs().size() == 1);

  REQUIRE(count_buffers_with_shape(program, Shape{4}) == 2);
  REQUIRE(count_buffers_with_shape(program, Shape{}) == 1);

  // LoweringResult exposes mappings only for values that actually have
  // persistent LIR buffers. The grouped Add intermediate stays internal.
  REQUIRE(lowering_result.value_to_buffer.find(x) !=
          lowering_result.value_to_buffer.end());
  REQUIRE(lowering_result.value_to_buffer.find(y) !=
          lowering_result.value_to_buffer.end());
  REQUIRE(lowering_result.value_to_buffer.find(out) !=
          lowering_result.value_to_buffer.end());
  REQUIRE(lowering_result.value_to_buffer.find(added) ==
          lowering_result.value_to_buffer.end());

  // Grouped lowering is one reduction:
  // Constant accumulator
  // loop
  // final Store
  REQUIRE(program.instructions().size() == 3);

  require_instruction<lir::Constant>(program.instructions()[0]);
  const auto &loop = require_for(program.instructions()[1]);
  require_instruction<lir::Store>(program.instructions()[2]);

  REQUIRE(loop.body.size() == 4);
  require_instruction<lir::Load>(loop.body[0]);
  require_instruction<lir::Load>(loop.body[1]);
  require_instruction<lir::Add>(loop.body[2]); // mapped Add
  require_instruction<lir::Add>(loop.body[3]); // reduction Add

  // The grouped intermediate is never stored as a tensor.
  for (const auto &instruction : loop.body) {
    REQUIRE_FALSE(std::holds_alternative<lir::Store>(instruction.payload));
  }
}

TEST_CASE("Lowering grouped Mul Sum does not materialize Mul intermediate",
          "[compiler][lowering][schedule][grouping]") {
  Graph graph;

  const auto x = graph.input(Shape{5}, Dtype::F32);
  const auto y = graph.input(Shape{5}, Dtype::F32);

  const auto product = graph.mul(x, y);
  const auto out = graph.sum(product);

  graph.set_outputs({out});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.size() == 1);
  REQUIRE(schedule.kernels[0].members == std::vector<ValueId>{product, out});

  Lowering lowering;
  const auto lowering_result = lowering.lower(graph, schedule);
  const auto &program = lowering_result.program;

  REQUIRE(program.buffer_count() == 3);
  REQUIRE(count_buffers_with_shape(program, Shape{5}) == 2);
  REQUIRE(count_buffers_with_shape(program, Shape{}) == 1);

  REQUIRE(program.instructions().size() == 3);

  const auto &loop = require_for(program.instructions()[1]);

  REQUIRE(require_constant_index(loop.begin) == 0);
  REQUIRE(require_constant_index(loop.end) == 5);

  REQUIRE(loop.body.size() == 4);
  require_instruction<lir::Load>(loop.body[0]);
  require_instruction<lir::Load>(loop.body[1]);
  require_instruction<lir::Mul>(loop.body[2]);
  require_instruction<lir::Add>(loop.body[3]);

  for (const auto &instruction : loop.body) {
    REQUIRE_FALSE(std::holds_alternative<lir::Store>(instruction.payload));
  }
}

TEST_CASE("Lowering grouped FusedElementwise Sum does not materialize fused "
          "intermediate",
          "[compiler][lowering][schedule][grouping]") {
  Graph graph;

  const auto a = graph.input(Shape{4}, Dtype::F32);
  const auto b = graph.input(Shape{4}, Dtype::F32);
  const auto c = graph.input(Shape{4}, Dtype::F32);

  auto expr = add_expr(mul_expr(input_expr(0), input_expr(1)), input_expr(2));

  const auto fused = graph.fused_elementwise({a, b, c}, std::move(expr));
  const auto out = graph.sum(fused);

  graph.set_outputs({out});

  Scheduler scheduler;
  const auto schedule = scheduler.schedule(graph);

  REQUIRE(schedule.kernels.size() == 1);
  REQUIRE(schedule.kernels[0].members == std::vector<ValueId>{fused, out});

  Lowering lowering;
  const auto lowering_result = lowering.lower(graph, schedule);
  const auto &program = lowering_result.program;

  // Three vector inputs plus one scalar Sum output.
  // The FusedElementwise result has no tensor buffer.
  REQUIRE(program.buffer_count() == 4);
  REQUIRE(count_buffers_with_shape(program, Shape{4}) == 3);
  REQUIRE(count_buffers_with_shape(program, Shape{}) == 1);

  REQUIRE(program.instructions().size() == 3);

  const auto &loop = require_for(program.instructions()[1]);

  REQUIRE(loop.body.size() == 6);

  require_instruction<lir::Load>(loop.body[0]);
  require_instruction<lir::Load>(loop.body[1]);
  require_instruction<lir::Load>(loop.body[2]);
  require_instruction<lir::Mul>(loop.body[3]);
  require_instruction<lir::Add>(loop.body[4]);
  require_instruction<lir::Add>(loop.body[5]); // reduction

  for (const auto &instruction : loop.body) {
    REQUIRE_FALSE(std::holds_alternative<lir::Store>(instruction.payload));
  }
}

// -----------------------------------------------------------------------------
// Sum
// -----------------------------------------------------------------------------

TEST_CASE("Lowering Sum creates input and scalar output buffer") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);

  const auto out = graph.sum(x);

  graph.set_outputs({out});

  const auto program = lower_with_schedule(graph);

  REQUIRE(program.buffer_count() == 2);
  REQUIRE(program.inputs().size() == 1);
  REQUIRE(program.outputs().size() == 1);

  REQUIRE(program.buffers()[0].shape == Shape{4});
  REQUIRE(program.buffers()[1].shape == Shape{});

  REQUIRE(program.buffers()[0].dtype == Dtype::F32);
  REQUIRE(program.buffers()[1].dtype == Dtype::F32);

  REQUIRE(program.outputs()[0].value == 1);
}

TEST_CASE("Lowering Sum allocates expected scalar and index registers") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);

  const auto out = graph.sum(x);

  graph.set_outputs({out});

  const auto program = lower_with_schedule(graph);

  REQUIRE(program.scalar_count() == 2);
  REQUIRE(program.index_count() == 1);
}

TEST_CASE("Lowering Sum creates initialize loop and final store") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);

  const auto out = graph.sum(x);

  graph.set_outputs({out});

  const auto program = lower_with_schedule(graph);

  REQUIRE(program.instructions().size() == 3);

  require_instruction<lir::Constant>(program.instructions()[0]);

  const auto &loop = require_for(program.instructions()[1]);

  require_instruction<lir::Store>(program.instructions()[2]);

  REQUIRE(require_constant_index(loop.begin) == 0);

  REQUIRE(require_constant_index(loop.end) == 4);

  REQUIRE(loop.body.size() == 2);

  require_instruction<lir::Load>(loop.body[0]);

  require_instruction<lir::Add>(loop.body[1]);
}

TEST_CASE("Lowering Sum reduction loop uses all matrix elements") {
  Graph graph;

  const auto x = graph.input(Shape{2, 3}, Dtype::F32);

  const auto out = graph.sum(x);

  graph.set_outputs({out});

  const auto program = lower_with_schedule(graph);

  const auto &loop = require_for(program.instructions()[1]);

  REQUIRE(require_constant_index(loop.end) == 6);
}

// -----------------------------------------------------------------------------
// MatMul
// -----------------------------------------------------------------------------

TEST_CASE("Lowering MatMul creates two input buffers and output buffer") {
  Graph graph;

  const auto a = graph.input(Shape{2, 3}, Dtype::F32);

  const auto b = graph.input(Shape{3, 4}, Dtype::F32);

  const auto out = graph.matmul(a, b);

  graph.set_outputs({out});

  const auto program = lower_with_schedule(graph);

  REQUIRE(program.buffer_count() == 3);
  REQUIRE(program.inputs().size() == 2);
  REQUIRE(program.outputs().size() == 1);

  REQUIRE(program.buffers()[0].shape == Shape{2, 3});
  REQUIRE(program.buffers()[1].shape == Shape{3, 4});
  REQUIRE(program.buffers()[2].shape == Shape{2, 4});

  REQUIRE(program.outputs()[0].value == 2);
}

TEST_CASE("Lowering MatMul allocates expected scalar and index registers") {
  Graph graph;

  const auto a = graph.input(Shape{2, 3}, Dtype::F32);

  const auto b = graph.input(Shape{3, 4}, Dtype::F32);

  const auto out = graph.matmul(a, b);

  graph.set_outputs({out});

  const auto program = lower_with_schedule(graph);

  // i, j, k
  // a_row, b_row, a_off, b_off, c_row, c_off
  REQUIRE(program.index_count() == 9);

  // a_scalar, b_scalar, product, acc
  REQUIRE(program.scalar_count() == 4);
}

TEST_CASE("Lowering MatMul creates three nested loops") {
  Graph graph;

  const auto a = graph.input(Shape{2, 3}, Dtype::F32);

  const auto b = graph.input(Shape{3, 4}, Dtype::F32);

  const auto out = graph.matmul(a, b);

  graph.set_outputs({out});

  const auto program = lower_with_schedule(graph);

  REQUIRE(program.instructions().size() == 1);

  const auto &i_loop = require_for(program.instructions()[0]);

  REQUIRE(require_constant_index(i_loop.begin) == 0);

  // M = 2
  REQUIRE(require_constant_index(i_loop.end) == 2);

  REQUIRE(i_loop.body.size() == 3);

  require_instruction<lir::IndexMul>(i_loop.body[0]);

  require_instruction<lir::IndexMul>(i_loop.body[1]);

  const auto &j_loop = require_for(i_loop.body[2]);

  REQUIRE(require_constant_index(j_loop.begin) == 0);

  // N = 4
  REQUIRE(require_constant_index(j_loop.end) == 4);

  REQUIRE(j_loop.body.size() == 4);

  require_instruction<lir::Constant>(j_loop.body[0]);

  const auto &k_loop = require_for(j_loop.body[1]);

  // K = 3
  REQUIRE(require_constant_index(k_loop.begin) == 0);

  REQUIRE(require_constant_index(k_loop.end) == 3);

  require_instruction<lir::IndexAdd>(j_loop.body[2]);

  require_instruction<lir::Store>(j_loop.body[3]);
}

TEST_CASE("Lowering MatMul inner K loop contains expected operations") {
  Graph graph;

  const auto a = graph.input(Shape{2, 3}, Dtype::F32);

  const auto b = graph.input(Shape{3, 4}, Dtype::F32);

  const auto out = graph.matmul(a, b);

  graph.set_outputs({out});

  const auto program = lower_with_schedule(graph);

  const auto &i_loop = require_for(program.instructions()[0]);

  const auto &j_loop = require_for(i_loop.body[2]);

  const auto &k_loop = require_for(j_loop.body[1]);

  REQUIRE(k_loop.body.size() == 7);

  require_instruction<lir::IndexAdd>(k_loop.body[0]);

  require_instruction<lir::IndexMul>(k_loop.body[1]);

  require_instruction<lir::IndexAdd>(k_loop.body[2]);

  require_instruction<lir::Load>(k_loop.body[3]);

  require_instruction<lir::Load>(k_loop.body[4]);

  require_instruction<lir::Mul>(k_loop.body[5]);

  require_instruction<lir::Add>(k_loop.body[6]);
}

TEST_CASE("Lowering rectangular MatMul uses M K and N as loop bounds") {
  Graph graph;

  const auto a = graph.input(Shape{5, 7}, Dtype::F32);

  const auto b = graph.input(Shape{7, 11}, Dtype::F32);

  const auto out = graph.matmul(a, b);

  graph.set_outputs({out});

  const auto program = lower_with_schedule(graph);

  const auto &i_loop = require_for(program.instructions()[0]);

  const auto &j_loop = require_for(i_loop.body[2]);

  const auto &k_loop = require_for(j_loop.body[1]);

  REQUIRE(require_constant_index(i_loop.end) == 5);

  REQUIRE(require_constant_index(j_loop.end) == 11);

  REQUIRE(require_constant_index(k_loop.end) == 7);

  REQUIRE(program.buffers()[2].shape == Shape{5, 11});
}

// -----------------------------------------------------------------------------
// Phase 5 schedule authority / negative schedules
// -----------------------------------------------------------------------------

TEST_CASE("Lowering rejects schedule where consumer executes before producer",
          "[compiler][lowering][schedule][negative]") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);
  const auto y = graph.input(Shape{4}, Dtype::F32);

  const auto added = graph.add(x, y);
  const auto out = graph.sum(added);

  graph.set_outputs({out});

  Schedule bad_schedule;
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

  Lowering lowering;

  // Sum asks require_buffer() for `added`, but the Add kernel has not executed.
  // Lowering must reject the schedule instead of recursively repairing it.
  REQUIRE_THROWS_AS(lowering.lower(graph, bad_schedule), std::logic_error);
}

TEST_CASE("Lowering rejects missing previously computed boundary value",
          "[compiler][lowering][schedule][negative]") {
  Graph graph;

  const auto x = graph.input(Shape{3}, Dtype::F32);
  const auto y = graph.input(Shape{3}, Dtype::F32);
  const auto z = graph.input(Shape{3}, Dtype::F32);

  const auto product = graph.mul(x, y);
  const auto out = graph.add(product, z);

  graph.set_outputs({out});

  // Deliberately omit the kernel that computes `product`.
  Schedule bad_schedule;
  bad_schedule.kernels.push_back(KernelUnit{
      .members = {out},
      .boundary_inputs = {product, z},
      .materialized_outputs = {out},
  });

  Lowering lowering;

  REQUIRE_THROWS_AS(lowering.lower(graph, bad_schedule), std::logic_error);
}

TEST_CASE("Lowering rejects same computed value in two kernels",
          "[compiler][lowering][schedule][negative]") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);
  const auto y = graph.input(Shape{2}, Dtype::F32);
  const auto out = graph.add(x, y);

  graph.set_outputs({out});

  Schedule bad_schedule;
  bad_schedule.kernels.push_back(KernelUnit{
      .members = {out},
      .boundary_inputs = {x, y},
      .materialized_outputs = {out},
  });
  bad_schedule.kernels.push_back(KernelUnit{
      .members = {out},
      .boundary_inputs = {x, y},
      .materialized_outputs = {out},
  });

  Lowering lowering;

  // A Schedule is a partition: one computed value may belong to only one
  // KernelUnit. Lowering should reject a malformed hand-built Schedule.
  REQUIRE_THROWS_AS(lowering.lower(graph, bad_schedule), std::logic_error);
}

TEST_CASE("Lowering rejects graph input as kernel member",
          "[compiler][lowering][schedule][negative]") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);
  graph.set_outputs({x});

  Schedule bad_schedule;
  bad_schedule.kernels.push_back(KernelUnit{
      .members = {x},
      .boundary_inputs = {},
      .materialized_outputs = {x},
  });

  Lowering lowering;

  REQUIRE_THROWS_AS(lowering.lower(graph, bad_schedule), std::logic_error);
}

TEST_CASE("Lowering rejects unsupported two-member kernel",
          "[compiler][lowering][schedule][negative]") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);
  const auto y = graph.input(Shape{4}, Dtype::F32);
  const auto z = graph.input(Shape{4}, Dtype::F32);

  const auto added = graph.add(x, y);
  const auto multiplied = graph.mul(added, z);

  graph.set_outputs({multiplied});

  // Current Phase 5 grouped lowering supports only
  // Add/Mul/FusedElementwise -> Sum.
  Schedule bad_schedule;
  bad_schedule.kernels.push_back(KernelUnit{
      .members = {added, multiplied},
      .boundary_inputs = {x, y, z},
      .materialized_outputs = {multiplied},
  });

  Lowering lowering;

  REQUIRE_THROWS_AS(lowering.lower(graph, bad_schedule), std::logic_error);
}

// -----------------------------------------------------------------------------
// Dependency lowering
// -----------------------------------------------------------------------------

TEST_CASE("Lowering follows scheduled producer before consumer") {
  Graph graph;

  const auto x = graph.input(Shape{3}, Dtype::F32);

  const auto y = graph.input(Shape{3}, Dtype::F32);

  const auto z = graph.input(Shape{3}, Dtype::F32);

  const auto product = graph.mul(x, y);

  const auto out = graph.add(product, z);

  graph.set_outputs({out});

  const auto program = lower_with_schedule(graph);

  // x, y, product, z, out
  REQUIRE(program.buffer_count() == 5);

  REQUIRE(program.inputs().size() == 3);
  REQUIRE(program.outputs().size() == 1);

  REQUIRE(program.instructions().size() == 2);

  const auto &mul_loop = require_for(program.instructions()[0]);

  const auto &add_loop = require_for(program.instructions()[1]);

  REQUIRE(std::holds_alternative<lir::Mul>(mul_loop.body[2].payload));

  REQUIRE(std::holds_alternative<lir::Add>(add_loop.body[2].payload));
}

// -----------------------------------------------------------------------------
// Shared subexpressions / scheduled reuse
// -----------------------------------------------------------------------------

TEST_CASE("Lowering scheduled shared producer executes only once") {
  Graph graph;

  const auto x = graph.input(Shape{3}, Dtype::F32);

  const auto y = graph.input(Shape{3}, Dtype::F32);

  const auto shared = graph.mul(x, y);

  const auto out = graph.add(shared, shared);

  graph.set_outputs({out});

  const auto program = lower_with_schedule(graph);

  // x + y + shared + out
  REQUIRE(program.buffer_count() == 4);

  REQUIRE(program.inputs().size() == 2);

  // One Mul loop + one Add loop.
  REQUIRE(program.instructions().size() == 2);

  REQUIRE(program.outputs().size() == 1);
  REQUIRE(program.outputs()[0].value == 3);
}

TEST_CASE("Lowering duplicate graph outputs reuse same buffer") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);

  const auto y = graph.input(Shape{2}, Dtype::F32);

  const auto out = graph.add(x, y);

  graph.set_outputs({
      out,
      out,
  });

  const auto program = lower_with_schedule(graph);

  REQUIRE(program.buffer_count() == 3);

  REQUIRE(program.instructions().size() == 1);

  REQUIRE(program.outputs().size() == 2);

  REQUIRE(program.outputs()[0].value == program.outputs()[1].value);

  REQUIRE(program.outputs()[0].value == 2);
}

// -----------------------------------------------------------------------------
// Multiple outputs
// -----------------------------------------------------------------------------

TEST_CASE("Lowering supports multiple computed graph outputs") {
  Graph graph;

  const auto x = graph.input(Shape{3}, Dtype::F32);

  const auto y = graph.input(Shape{3}, Dtype::F32);

  const auto add_out = graph.add(x, y);

  const auto mul_out = graph.mul(x, y);

  graph.set_outputs({
      add_out,
      mul_out,
  });

  const auto program = lower_with_schedule(graph);

  // x, y, add_out, mul_out
  REQUIRE(program.buffer_count() == 4);

  REQUIRE(program.inputs().size() == 2);
  REQUIRE(program.outputs().size() == 2);

  REQUIRE(program.instructions().size() == 2);

  REQUIRE(program.outputs()[0].value == 2);
  REQUIRE(program.outputs()[1].value == 3);

  const auto &add_loop = require_for(program.instructions()[0]);

  const auto &mul_loop = require_for(program.instructions()[1]);

  REQUIRE(std::holds_alternative<lir::Add>(add_loop.body[2].payload));

  REQUIRE(std::holds_alternative<lir::Mul>(mul_loop.body[2].payload));
}

TEST_CASE("Lowering supports input and computed value as outputs") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);

  const auto y = graph.input(Shape{2}, Dtype::F32);

  const auto sum = graph.add(x, y);

  graph.set_outputs({
      x,
      sum,
  });

  const auto program = lower_with_schedule(graph);

  REQUIRE(program.outputs().size() == 2);

  // x is lowered first because it appears first in outputs.
  REQUIRE(program.outputs()[0].value == 0);

  // The scheduled Add resolves x and y, then creates the computed output.
  REQUIRE(program.outputs()[1].value == 2);

  REQUIRE(program.buffer_count() == 3);
  REQUIRE(program.inputs().size() == 2);
  REQUIRE(program.instructions().size() == 1);
}

// -----------------------------------------------------------------------------
// Reachability
// -----------------------------------------------------------------------------

TEST_CASE("Lowering does not lower unreachable computed value") {
  Graph graph;

  const auto x = graph.input(Shape{3}, Dtype::F32);

  const auto y = graph.input(Shape{3}, Dtype::F32);

  const auto live = graph.add(x, y);

  const auto dead = graph.mul(x, y);

  (void)dead;

  graph.set_outputs({live});

  const auto program = lower_with_schedule(graph);

  REQUIRE(program.buffer_count() == 3);
  REQUIRE(program.inputs().size() == 2);

  // Only the live Add loop is lowered.
  REQUIRE(program.instructions().size() == 1);

  const auto &loop = require_for(program.instructions()[0]);

  REQUIRE(std::holds_alternative<lir::Add>(loop.body[2].payload));
}

TEST_CASE("Lowering excludes unreachable dependency subtree") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);

  const auto y = graph.input(Shape{2}, Dtype::F32);

  const auto z = graph.input(Shape{2}, Dtype::F32);

  const auto live = graph.add(x, y);

  const auto dead_mul = graph.mul(y, z);

  const auto dead_add = graph.add(dead_mul, z);

  (void)dead_add;

  graph.set_outputs({live});

  const auto program = lower_with_schedule(graph);

  // Only x, y and live are reachable.
  REQUIRE(program.buffer_count() == 3);

  // Schedule-driven lowering preserves reachability:
  // unreachable z is not lowered as an LIR input either.
  REQUIRE(program.inputs().size() == 2);

  REQUIRE(program.instructions().size() == 1);

  REQUIRE(program.outputs().size() == 1);
}

// -----------------------------------------------------------------------------
// Lowering object reuse
// -----------------------------------------------------------------------------

TEST_CASE("Lowering clears value to buffer state between lower calls") {
  Lowering lowering;

  Graph first_graph;

  const auto first_x = first_graph.input(Shape{2}, Dtype::F32);

  const auto first_y = first_graph.input(Shape{2}, Dtype::F32);

  const auto first_out = first_graph.add(first_x, first_y);

  first_graph.set_outputs({
      first_out,
  });

  Scheduler scheduler;
  const auto first_schedule = scheduler.schedule(first_graph);
  const auto first_result = lowering.lower(first_graph, first_schedule);
  const auto &first_program = first_result.program;

  REQUIRE(first_program.buffer_count() == 3);
  REQUIRE(first_program.inputs().size() == 2);
  REQUIRE(first_result.value_to_buffer.size() == 3);

  Graph second_graph;

  const auto second_x = second_graph.input(Shape{5}, Dtype::F32);

  second_graph.set_outputs({
      second_x,
  });

  const auto second_schedule = scheduler.schedule(second_graph);
  const auto second_result = lowering.lower(second_graph, second_schedule);
  const auto &second_program = second_result.program;

  REQUIRE(second_program.buffer_count() == 1);
  REQUIRE(second_program.inputs().size() == 1);
  REQUIRE(second_program.outputs().size() == 1);
  REQUIRE(second_result.value_to_buffer.size() == 1);
  REQUIRE(second_result.value_to_buffer.find(second_x) !=
          second_result.value_to_buffer.end());

  REQUIRE(second_program.buffers()[0].shape == Shape{5});

  REQUIRE(second_program.outputs()[0].value == 0);
}
