#include <catch2/catch_test_macros.hpp>

#include "tcr/compiler/compiler.hpp"
#include "tcr/ir/dtype.hpp"
#include "tcr/ir/graph.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/runtime/opcode.hpp"
#include "tcr/runtime/slot_id.hpp"

using tcr::compiler::Compiler;
using tcr::ir::Dtype;
using tcr::ir::Graph;
using tcr::ir::Shape;
using tcr::runtime::Opcode;
using tcr::runtime::SlotId;

// -----------------------------------------------------------------------------
// Empty graph
// -----------------------------------------------------------------------------

TEST_CASE("Compiler compiles empty graph to empty program") {
  Graph graph;

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.instructions().empty());
  REQUIRE(program.inputs().empty());
  REQUIRE(program.outputs().empty());
  REQUIRE(program.slots_count() == 0);
}

// -----------------------------------------------------------------------------
// Graph input handling
// -----------------------------------------------------------------------------

TEST_CASE("Compiler maps graph inputs to runtime slots") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);
  const auto y = graph.input(Shape{2}, Dtype::F32);

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.inputs().size() == 2);

  REQUIRE(program.inputs()[0] == SlotId{0});
  REQUIRE(program.inputs()[1] == SlotId{1});

  REQUIRE(program.slots_count() == 2);

  REQUIRE(program.instructions().empty());
  REQUIRE(program.outputs().empty());
}

TEST_CASE("Compiler preserves graph input ordering") {
  Graph graph;

  graph.input(Shape{1}, Dtype::F32);
  graph.input(Shape{2}, Dtype::F32);
  graph.input(Shape{3}, Dtype::F32);

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.inputs().size() == 3);

  REQUIRE(program.inputs()[0] == SlotId{0});
  REQUIRE(program.inputs()[1] == SlotId{1});
  REQUIRE(program.inputs()[2] == SlotId{2});
}

// -----------------------------------------------------------------------------
// Input directly used as graph output
// -----------------------------------------------------------------------------

TEST_CASE("Compiler maps graph input directly to program output") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);

  graph.set_outputs({x});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.inputs().size() == 1);
  REQUIRE(program.outputs().size() == 1);

  REQUIRE(program.inputs()[0] == SlotId{0});
  REQUIRE(program.outputs()[0] == SlotId{0});

  REQUIRE(program.instructions().empty());
  REQUIRE(program.slots_count() == 1);
}

// -----------------------------------------------------------------------------
// ADD
// -----------------------------------------------------------------------------

TEST_CASE("Compiler lowers Add node to ADD instruction") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);
  const auto y = graph.input(Shape{2}, Dtype::F32);

  const auto out = graph.add(x, y);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.inputs().size() == 2);
  REQUIRE(program.instructions().size() == 1);
  REQUIRE(program.outputs().size() == 1);
  REQUIRE(program.slots_count() == 3);

  REQUIRE(program.inputs()[0] == SlotId{0});
  REQUIRE(program.inputs()[1] == SlotId{1});

  const auto &instruction = program.instructions()[0];

  REQUIRE(instruction.opcode == Opcode::ADD);
  REQUIRE(instruction.inputs.size() == 2);
  REQUIRE(instruction.inputs[0] == SlotId{0});
  REQUIRE(instruction.inputs[1] == SlotId{1});
  REQUIRE(instruction.output_slot_id == SlotId{2});

  REQUIRE(program.outputs()[0] == SlotId{2});
}

// -----------------------------------------------------------------------------
// MUL
// -----------------------------------------------------------------------------

TEST_CASE("Compiler lowers Mul node to MUL instruction") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);
  const auto y = graph.input(Shape{2}, Dtype::F32);

  const auto out = graph.mul(x, y);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.instructions().size() == 1);

  const auto &instruction = program.instructions()[0];

  REQUIRE(instruction.opcode == Opcode::MUL);
  REQUIRE(instruction.inputs.size() == 2);

  REQUIRE(instruction.inputs[0] == SlotId{0});
  REQUIRE(instruction.inputs[1] == SlotId{1});

  REQUIRE(instruction.output_slot_id == SlotId{2});

  REQUIRE(program.outputs().size() == 1);
  REQUIRE(program.outputs()[0] == SlotId{2});

  REQUIRE(program.slots_count() == 3);
}

// -----------------------------------------------------------------------------
// SUM
// -----------------------------------------------------------------------------

TEST_CASE("Compiler lowers Sum node to SUM instruction") {
  Graph graph;

  const auto x = graph.input(Shape{3}, Dtype::F32);
  const auto out = graph.sum(x);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.instructions().size() == 1);

  const auto &instruction = program.instructions()[0];

  REQUIRE(instruction.opcode == Opcode::SUM);
  REQUIRE(instruction.inputs.size() == 1);
  REQUIRE(instruction.inputs[0] == SlotId{0});
  REQUIRE(instruction.output_slot_id == SlotId{1});

  REQUIRE(program.outputs().size() == 1);
  REQUIRE(program.outputs()[0] == SlotId{1});

  REQUIRE(program.slots_count() == 2);
}

// -----------------------------------------------------------------------------
// MATMUL
// -----------------------------------------------------------------------------

TEST_CASE("Compiler lowers MatMul node to MATMUL instruction") {
  Graph graph;

  const auto lhs = graph.input(Shape{2, 3}, Dtype::F32);
  const auto rhs = graph.input(Shape{3, 4}, Dtype::F32);

  const auto out = graph.matmul(lhs, rhs);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.instructions().size() == 1);

  const auto &instruction = program.instructions()[0];

  REQUIRE(instruction.opcode == Opcode::MATMUL);
  REQUIRE(instruction.inputs.size() == 2);

  REQUIRE(instruction.inputs[0] == SlotId{0});
  REQUIRE(instruction.inputs[1] == SlotId{1});

  REQUIRE(instruction.output_slot_id == SlotId{2});

  REQUIRE(program.outputs()[0] == SlotId{2});
  REQUIRE(program.slots_count() == 3);
}

// -----------------------------------------------------------------------------
// Dependency ordering
// -----------------------------------------------------------------------------

TEST_CASE("Compiler emits dependent instructions in topological order") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);
  const auto y = graph.input(Shape{2}, Dtype::F32);
  const auto z = graph.input(Shape{2}, Dtype::F32);

  // t = x * y
  // out = t + z
  const auto t = graph.mul(x, y);
  const auto out = graph.add(t, z);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.instructions().size() == 2);
  REQUIRE(program.slots_count() == 5);

  const auto &mul_instruction = program.instructions()[0];
  const auto &add_instruction = program.instructions()[1];

  REQUIRE(mul_instruction.opcode == Opcode::MUL);
  REQUIRE(mul_instruction.inputs.size() == 2);
  REQUIRE(mul_instruction.inputs[0] == SlotId{0});
  REQUIRE(mul_instruction.inputs[1] == SlotId{1});
  REQUIRE(mul_instruction.output_slot_id == SlotId{3});

  REQUIRE(add_instruction.opcode == Opcode::ADD);
  REQUIRE(add_instruction.inputs.size() == 2);

  // ADD consumes result of MUL.
  REQUIRE(add_instruction.inputs[0] == SlotId{3});
  REQUIRE(add_instruction.inputs[1] == SlotId{2});

  REQUIRE(add_instruction.output_slot_id == SlotId{4});

  REQUIRE(program.outputs().size() == 1);
  REQUIRE(program.outputs()[0] == SlotId{4});
}

// -----------------------------------------------------------------------------
// Longer dependency chain
// -----------------------------------------------------------------------------

TEST_CASE("Compiler emits longer dependency chain in topological order") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);
  const auto y = graph.input(Shape{2}, Dtype::F32);

  const auto a = graph.mul(x, y);
  const auto b = graph.add(a, x);
  const auto c = graph.mul(b, y);
  const auto out = graph.sum(c);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.instructions().size() == 4);

  REQUIRE(program.instructions()[0].opcode == Opcode::MUL);
  REQUIRE(program.instructions()[1].opcode == Opcode::ADD);
  REQUIRE(program.instructions()[2].opcode == Opcode::MUL);
  REQUIRE(program.instructions()[3].opcode == Opcode::SUM);

  REQUIRE(program.instructions()[0].output_slot_id == SlotId{2});
  REQUIRE(program.instructions()[1].output_slot_id == SlotId{3});
  REQUIRE(program.instructions()[2].output_slot_id == SlotId{4});
  REQUIRE(program.instructions()[3].output_slot_id == SlotId{5});

  REQUIRE(program.outputs()[0] == SlotId{5});
  REQUIRE(program.slots_count() == 6);
}

// -----------------------------------------------------------------------------
// Diamond/shared dependency
// -----------------------------------------------------------------------------

TEST_CASE("Compiler emits shared producer exactly once") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);
  const auto y = graph.input(Shape{2}, Dtype::F32);
  const auto z = graph.input(Shape{2}, Dtype::F32);

  //             shared = x * y
  //                 /       \
  //          left = +z      right = *z
  //                 \       /
  //                  output = left + right

  const auto shared = graph.mul(x, y);
  const auto left = graph.add(shared, z);
  const auto right = graph.mul(shared, z);
  const auto out = graph.add(left, right);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  // shared MUL
  // left ADD
  // right MUL
  // final ADD
  //
  // Shared producer must NOT be emitted twice.
  REQUIRE(program.instructions().size() == 4);

  REQUIRE(program.instructions()[0].opcode == Opcode::MUL);
  REQUIRE(program.instructions()[1].opcode == Opcode::ADD);
  REQUIRE(program.instructions()[2].opcode == Opcode::MUL);
  REQUIRE(program.instructions()[3].opcode == Opcode::ADD);

  // Inputs occupy slots 0,1,2.
  // Shared producer gets slot 3.
  const SlotId shared_slot{3};

  REQUIRE(program.instructions()[0].output_slot_id == shared_slot);

  // Both consumers must reference exactly the same shared slot.
  REQUIRE(program.instructions()[1].inputs[0] == shared_slot);
  REQUIRE(program.instructions()[2].inputs[0] == shared_slot);

  REQUIRE(program.outputs().size() == 1);
  REQUIRE(program.outputs()[0] == SlotId{6});

  REQUIRE(program.slots_count() == 7);
}

// -----------------------------------------------------------------------------
// Multiple outputs
// -----------------------------------------------------------------------------

TEST_CASE("Compiler preserves graph output ordering") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);
  const auto y = graph.input(Shape{2}, Dtype::F32);

  const auto add_result = graph.add(x, y);
  const auto mul_result = graph.mul(x, y);

  graph.set_outputs({
      mul_result,
      add_result,
  });

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.outputs().size() == 2);

  // Traversal starts from mul_result because it is the first graph output.
  REQUIRE(program.instructions().size() == 2);

  REQUIRE(program.instructions()[0].opcode == Opcode::MUL);
  REQUIRE(program.instructions()[1].opcode == Opcode::ADD);

  REQUIRE(program.instructions()[0].output_slot_id == SlotId{2});
  REQUIRE(program.instructions()[1].output_slot_id == SlotId{3});

  // Output ordering must match Graph::outputs().
  REQUIRE(program.outputs()[0] == SlotId{2});
  REQUIRE(program.outputs()[1] == SlotId{3});
}

// -----------------------------------------------------------------------------
// Shared result appears as multiple outputs
// -----------------------------------------------------------------------------

TEST_CASE("Compiler does not emit same node twice for duplicate graph outputs") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);
  const auto y = graph.input(Shape{2}, Dtype::F32);

  const auto out = graph.add(x, y);

  graph.set_outputs({
      out,
      out,
  });

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.instructions().size() == 1);

  REQUIRE(program.outputs().size() == 2);

  REQUIRE(program.outputs()[0] == SlotId{2});
  REQUIRE(program.outputs()[1] == SlotId{2});

  REQUIRE(program.slots_count() == 3);
}

// -----------------------------------------------------------------------------
// Graph input plus computed output
// -----------------------------------------------------------------------------

TEST_CASE("Compiler supports graph input and computed value as outputs") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);
  const auto y = graph.input(Shape{2}, Dtype::F32);

  const auto sum_value = graph.add(x, y);

  graph.set_outputs({
      x,
      sum_value,
  });

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.instructions().size() == 1);

  REQUIRE(program.outputs().size() == 2);

  REQUIRE(program.outputs()[0] == SlotId{0});
  REQUIRE(program.outputs()[1] == SlotId{2});

  REQUIRE(program.slots_count() == 3);
}

// -----------------------------------------------------------------------------
// Dead computed node
// -----------------------------------------------------------------------------

TEST_CASE("Compiler does not emit unreachable computed nodes") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);
  const auto y = graph.input(Shape{2}, Dtype::F32);

  const auto live = graph.add(x, y);

  // This node exists in the graph but is not reachable from an output.
  const auto dead = graph.mul(x, y);

  (void)dead;

  graph.set_outputs({live});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.instructions().size() == 1);
  REQUIRE(program.instructions()[0].opcode == Opcode::ADD);

  REQUIRE(program.outputs().size() == 1);
  REQUIRE(program.outputs()[0] == SlotId{2});

  // Two graph inputs + one emitted ADD result.
  REQUIRE(program.slots_count() == 3);
}

// -----------------------------------------------------------------------------
// Dead dependency subtree
// -----------------------------------------------------------------------------

TEST_CASE("Compiler excludes unreachable dependency subtree") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);
  const auto y = graph.input(Shape{2}, Dtype::F32);
  const auto z = graph.input(Shape{2}, Dtype::F32);

  const auto live = graph.add(x, y);

  const auto dead_a = graph.mul(y, z);
  const auto dead_b = graph.add(dead_a, z);

  (void)dead_b;

  graph.set_outputs({live});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.instructions().size() == 1);
  REQUIRE(program.instructions()[0].opcode == Opcode::ADD);

  // All graph inputs currently receive runtime slots.
  REQUIRE(program.inputs().size() == 3);

  REQUIRE(program.outputs().size() == 1);

  // x,y,z get 0,1,2.
  // live ADD gets 3.
  REQUIRE(program.outputs()[0] == SlotId{3});

  REQUIRE(program.slots_count() == 4);
}

// -----------------------------------------------------------------------------
// Typed metadata propagation through compiler input graph
// -----------------------------------------------------------------------------

TEST_CASE(
    "Compiler lowers graph whose operation depends on inferred MatMul type") {
  Graph graph;

  const auto lhs = graph.input(Shape{2, 3}, Dtype::F32);
  const auto rhs = graph.input(Shape{3, 4}, Dtype::F32);
  const auto bias = graph.input(Shape{2, 4}, Dtype::F32);

  // matmul_result is inferred as Shape{2, 4}, F32.
  const auto matmul_result = graph.matmul(lhs, rhs);

  // This ADD is only legal because the MatMul result metadata
  // propagated correctly through the graph.
  const auto out = graph.add(matmul_result, bias);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.instructions().size() == 2);
  REQUIRE(program.slots_count() == 5);

  const auto &matmul_instruction = program.instructions()[0];
  const auto &add_instruction = program.instructions()[1];

  REQUIRE(matmul_instruction.opcode == Opcode::MATMUL);
  REQUIRE(matmul_instruction.inputs.size() == 2);
  REQUIRE(matmul_instruction.inputs[0] == SlotId{0});
  REQUIRE(matmul_instruction.inputs[1] == SlotId{1});
  REQUIRE(matmul_instruction.output_slot_id == SlotId{3});

  REQUIRE(add_instruction.opcode == Opcode::ADD);
  REQUIRE(add_instruction.inputs.size() == 2);

  // ADD consumes the result produced by MATMUL.
  REQUIRE(add_instruction.inputs[0] == SlotId{3});
  REQUIRE(add_instruction.inputs[1] == SlotId{2});
  REQUIRE(add_instruction.output_slot_id == SlotId{4});

  REQUIRE(program.outputs().size() == 1);
  REQUIRE(program.outputs()[0] == SlotId{4});
}