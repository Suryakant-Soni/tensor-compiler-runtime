#include <catch2/catch_test_macros.hpp>

#include "tcr/compiler/compiler.hpp"

#include "tcr/ir/dtype.hpp"
#include "tcr/ir/graph.hpp"
#include "tcr/ir/shape.hpp"

#include "tcr/reference/bindings.hpp"
#include "tcr/reference/interpreter.hpp"
#include "tcr/reference/tensor.hpp"

#include "tcr/runtime/bindings.hpp"
#include "tcr/runtime/slot_id.hpp"
#include "tcr/runtime/tensor.hpp"
#include "tcr/runtime/vm.hpp"

#include <cstddef>
#include <utility>
#include <vector>

using tcr::compiler::Compiler;

using tcr::ir::Dtype;
using tcr::ir::Graph;
using tcr::ir::Shape;
using tcr::ir::ValueId;

namespace reference = tcr::reference;
namespace runtime = tcr::runtime;

namespace {

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

void require_same_tensor(
    const reference::Tensor &expected,
    const runtime::Tensor &actual) {

  REQUIRE(actual.shape() == expected.shape());
  REQUIRE(actual.dtype() == expected.dtype());
  REQUIRE(actual.data() == expected.data());
}

void bind_input(
    reference::InputBindings &reference_inputs,
    runtime::InputBindings &runtime_inputs,
    ValueId value_id,
    runtime::SlotId slot_id,
    Shape shape,
    std::vector<float> data) {

  reference_inputs.bind(
      value_id,
      reference::Tensor{
          shape,
          Dtype::F32,
          data,
      });

  runtime_inputs.bind(
      slot_id,
      runtime::Tensor{
          std::move(shape),
          Dtype::F32,
          std::move(data),
      });
}

std::vector<runtime::Tensor>
compile_and_run(
    const Graph &graph,
    const runtime::InputBindings &runtime_inputs) {

  Compiler compiler;
  const auto program =
      compiler.compile(graph);

  runtime::VM vm;

  return vm.run(
      program,
      runtime_inputs);
}

} // namespace

// =============================================================================
// Empty graph
// =============================================================================

TEST_CASE("Compiler and VM execute empty graph with no outputs") {
  Graph graph;

  const runtime::InputBindings runtime_inputs;

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  REQUIRE(actual_outputs.empty());
}

// =============================================================================
// Direct graph input
// =============================================================================

TEST_CASE(
    "Compiler and VM match reference interpreter when vector input is output") {

  Graph graph;

  const auto x =
      graph.input(
          Shape{3},
          Dtype::F32);

  graph.set_outputs({x});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      x,
      runtime::SlotId{0},
      Shape{3},
      {
          10.0F,
          20.0F,
          30.0F,
      });

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          x,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(
      expected,
      actual_outputs[0]);
}

TEST_CASE(
    "Compiler and VM match reference interpreter when scalar input is output") {

  Graph graph;

  const auto x =
      graph.input(
          Shape{},
          Dtype::F32);

  graph.set_outputs({x});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      x,
      runtime::SlotId{0},
      Shape{},
      {17.5F});

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          x,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(
      expected,
      actual_outputs[0]);
}

// =============================================================================
// ADD
// =============================================================================

TEST_CASE("Compiler and VM match reference interpreter for vector Add") {
  Graph graph;

  const auto x =
      graph.input(
          Shape{4},
          Dtype::F32);

  const auto y =
      graph.input(
          Shape{4},
          Dtype::F32);

  const auto out =
      graph.add(x, y);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      x,
      runtime::SlotId{0},
      Shape{4},
      {
          1.0F,
          2.0F,
          3.0F,
          4.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      y,
      runtime::SlotId{1},
      Shape{4},
      {
          10.0F,
          20.0F,
          30.0F,
          40.0F,
      });

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          out,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(
      expected,
      actual_outputs[0]);

  REQUIRE(
      actual_outputs[0].data() ==
      std::vector<float>{
          11.0F,
          22.0F,
          33.0F,
          44.0F,
      });
}

TEST_CASE("Compiler and VM match reference interpreter for matrix Add") {
  Graph graph;

  const auto x =
      graph.input(
          Shape{2, 3},
          Dtype::F32);

  const auto y =
      graph.input(
          Shape{2, 3},
          Dtype::F32);

  const auto out =
      graph.add(x, y);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      x,
      runtime::SlotId{0},
      Shape{2, 3},
      {
          1.0F, 2.0F, 3.0F,
          4.0F, 5.0F, 6.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      y,
      runtime::SlotId{1},
      Shape{2, 3},
      {
          10.0F, 20.0F, 30.0F,
          40.0F, 50.0F, 60.0F,
      });

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          out,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(
      expected,
      actual_outputs[0]);

  REQUIRE(actual_outputs[0].shape() == Shape{2, 3});
}

TEST_CASE("Compiler and VM match reference interpreter for scalar Add") {
  Graph graph;

  const auto x =
      graph.input(
          Shape{},
          Dtype::F32);

  const auto y =
      graph.input(
          Shape{},
          Dtype::F32);

  const auto out =
      graph.add(x, y);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      x,
      runtime::SlotId{0},
      Shape{},
      {-2.5F});

  bind_input(
      reference_inputs,
      runtime_inputs,
      y,
      runtime::SlotId{1},
      Shape{},
      {7.0F});

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          out,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(
      expected,
      actual_outputs[0]);

  REQUIRE(
      actual_outputs[0].data() ==
      std::vector<float>{4.5F});
}

// =============================================================================
// MUL
// =============================================================================

TEST_CASE("Compiler and VM match reference interpreter for vector Mul") {
  Graph graph;

  const auto x =
      graph.input(
          Shape{4},
          Dtype::F32);

  const auto y =
      graph.input(
          Shape{4},
          Dtype::F32);

  const auto out =
      graph.mul(x, y);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      x,
      runtime::SlotId{0},
      Shape{4},
      {
          2.0F,
          3.0F,
          4.0F,
          5.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      y,
      runtime::SlotId{1},
      Shape{4},
      {
          5.0F,
          6.0F,
          7.0F,
          8.0F,
      });

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          out,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(
      expected,
      actual_outputs[0]);

  REQUIRE(
      actual_outputs[0].data() ==
      std::vector<float>{
          10.0F,
          18.0F,
          28.0F,
          40.0F,
      });
}

TEST_CASE(
    "Compiler and VM preserve negative and zero values through Mul") {

  Graph graph;

  const auto x =
      graph.input(
          Shape{4},
          Dtype::F32);

  const auto y =
      graph.input(
          Shape{4},
          Dtype::F32);

  const auto out =
      graph.mul(x, y);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      x,
      runtime::SlotId{0},
      Shape{4},
      {
          -2.0F,
          0.0F,
          3.0F,
          -4.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      y,
      runtime::SlotId{1},
      Shape{4},
      {
          5.0F,
          -10.0F,
          -2.0F,
          -3.0F,
      });

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          out,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  require_same_tensor(
      expected,
      actual_outputs[0]);

  REQUIRE(
      actual_outputs[0].data() ==
      std::vector<float>{
          -10.0F,
          0.0F,
          -6.0F,
          12.0F,
      });
}

TEST_CASE("Compiler and VM match reference interpreter for scalar Mul") {
  Graph graph;

  const auto x =
      graph.input(
          Shape{},
          Dtype::F32);

  const auto y =
      graph.input(
          Shape{},
          Dtype::F32);

  const auto out =
      graph.mul(x, y);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      x,
      runtime::SlotId{0},
      Shape{},
      {3.5F});

  bind_input(
      reference_inputs,
      runtime_inputs,
      y,
      runtime::SlotId{1},
      Shape{},
      {2.0F});

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          out,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  require_same_tensor(
      expected,
      actual_outputs[0]);

  REQUIRE(actual_outputs[0].load(0) == 7.0F);
}

// =============================================================================
// SUM
// =============================================================================

TEST_CASE("Compiler and VM match reference interpreter for vector Sum") {
  Graph graph;

  const auto x =
      graph.input(
          Shape{5},
          Dtype::F32);

  const auto out =
      graph.sum(x);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      x,
      runtime::SlotId{0},
      Shape{5},
      {
          1.0F,
          2.0F,
          3.0F,
          4.0F,
          5.0F,
      });

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          out,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(
      expected,
      actual_outputs[0]);

  REQUIRE(actual_outputs[0].shape() == Shape{});
  REQUIRE(actual_outputs[0].load(0) == 15.0F);
}

TEST_CASE("Compiler and VM match reference interpreter for matrix Sum") {
  Graph graph;

  const auto x =
      graph.input(
          Shape{2, 3},
          Dtype::F32);

  const auto out =
      graph.sum(x);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      x,
      runtime::SlotId{0},
      Shape{2, 3},
      {
          1.0F, 2.0F, 3.0F,
          4.0F, 5.0F, 6.0F,
      });

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          out,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  require_same_tensor(
      expected,
      actual_outputs[0]);

  REQUIRE(actual_outputs[0].shape() == Shape{});
  REQUIRE(actual_outputs[0].load(0) == 21.0F);
}

TEST_CASE("Compiler and VM match reference interpreter for scalar Sum") {
  Graph graph;

  const auto x =
      graph.input(
          Shape{},
          Dtype::F32);

  const auto out =
      graph.sum(x);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      x,
      runtime::SlotId{0},
      Shape{},
      {9.25F});

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          out,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  require_same_tensor(
      expected,
      actual_outputs[0]);

  REQUIRE(actual_outputs[0].load(0) == 9.25F);
}

TEST_CASE(
    "Compiler and VM match reference interpreter for Sum containing mixed signs") {

  Graph graph;

  const auto x =
      graph.input(
          Shape{5},
          Dtype::F32);

  const auto out =
      graph.sum(x);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      x,
      runtime::SlotId{0},
      Shape{5},
      {
          -5.0F,
          2.0F,
          0.0F,
          10.0F,
          -3.0F,
      });

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          out,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  require_same_tensor(
      expected,
      actual_outputs[0]);

  REQUIRE(actual_outputs[0].load(0) == 4.0F);
}

// =============================================================================
// MATMUL
// =============================================================================

TEST_CASE("Compiler and VM match reference interpreter for 1x1 MatMul") {
  Graph graph;

  const auto lhs =
      graph.input(
          Shape{1, 1},
          Dtype::F32);

  const auto rhs =
      graph.input(
          Shape{1, 1},
          Dtype::F32);

  const auto out =
      graph.matmul(lhs, rhs);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      lhs,
      runtime::SlotId{0},
      Shape{1, 1},
      {6.0F});

  bind_input(
      reference_inputs,
      runtime_inputs,
      rhs,
      runtime::SlotId{1},
      Shape{1, 1},
      {7.0F});

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          out,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  require_same_tensor(
      expected,
      actual_outputs[0]);

  REQUIRE(actual_outputs[0].shape() == Shape{1, 1});
  REQUIRE(actual_outputs[0].load(0) == 42.0F);
}

TEST_CASE("Compiler and VM match reference interpreter for square MatMul") {
  Graph graph;

  const auto lhs =
      graph.input(
          Shape{2, 2},
          Dtype::F32);

  const auto rhs =
      graph.input(
          Shape{2, 2},
          Dtype::F32);

  const auto out =
      graph.matmul(lhs, rhs);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      lhs,
      runtime::SlotId{0},
      Shape{2, 2},
      {
          1.0F, 2.0F,
          3.0F, 4.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      rhs,
      runtime::SlotId{1},
      Shape{2, 2},
      {
          5.0F, 6.0F,
          7.0F, 8.0F,
      });

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          out,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  require_same_tensor(
      expected,
      actual_outputs[0]);

  REQUIRE(
      actual_outputs[0].data() ==
      std::vector<float>{
          19.0F, 22.0F,
          43.0F, 50.0F,
      });
}

TEST_CASE("Compiler and VM match reference interpreter for rectangular MatMul") {
  Graph graph;

  const auto lhs =
      graph.input(
          Shape{2, 3},
          Dtype::F32);

  const auto rhs =
      graph.input(
          Shape{3, 2},
          Dtype::F32);

  const auto out =
      graph.matmul(lhs, rhs);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      lhs,
      runtime::SlotId{0},
      Shape{2, 3},
      {
          1.0F, 2.0F, 3.0F,
          4.0F, 5.0F, 6.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      rhs,
      runtime::SlotId{1},
      Shape{3, 2},
      {
          7.0F, 8.0F,
          9.0F, 10.0F,
          11.0F, 12.0F,
      });

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          out,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  require_same_tensor(
      expected,
      actual_outputs[0]);

  REQUIRE(actual_outputs[0].shape() == Shape{2, 2});

  REQUIRE(
      actual_outputs[0].data() ==
      std::vector<float>{
          58.0F, 64.0F,
          139.0F, 154.0F,
      });
}

TEST_CASE(
    "Compiler and VM match reference interpreter for one-row rectangular MatMul") {

  Graph graph;

  const auto lhs =
      graph.input(
          Shape{1, 3},
          Dtype::F32);

  const auto rhs =
      graph.input(
          Shape{3, 4},
          Dtype::F32);

  const auto out =
      graph.matmul(lhs, rhs);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      lhs,
      runtime::SlotId{0},
      Shape{1, 3},
      {
          1.0F,
          2.0F,
          3.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      rhs,
      runtime::SlotId{1},
      Shape{3, 4},
      {
          1.0F, 2.0F, 3.0F, 4.0F,
          5.0F, 6.0F, 7.0F, 8.0F,
          9.0F, 10.0F, 11.0F, 12.0F,
      });

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          out,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  require_same_tensor(
      expected,
      actual_outputs[0]);

  REQUIRE(actual_outputs[0].shape() == Shape{1, 4});

  REQUIRE(
      actual_outputs[0].data() ==
      std::vector<float>{
          38.0F,
          44.0F,
          50.0F,
          56.0F,
      });
}

// =============================================================================
// Elementwise dependency chains
// =============================================================================

TEST_CASE(
    "Compiler and VM match reference interpreter for chained Mul then Add") {

  Graph graph;

  const auto x =
      graph.input(
          Shape{3},
          Dtype::F32);

  const auto y =
      graph.input(
          Shape{3},
          Dtype::F32);

  const auto z =
      graph.input(
          Shape{3},
          Dtype::F32);

  // product is allocated before z because lowering recursively
  // visits the left Add operand first:
  //
  // slot 0 = x
  // slot 1 = y
  // slot 2 = product
  // slot 3 = z
  // slot 4 = out

  const auto product =
      graph.mul(x, y);

  const auto out =
      graph.add(
          product,
          z);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      x,
      runtime::SlotId{0},
      Shape{3},
      {
          1.0F,
          2.0F,
          3.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      y,
      runtime::SlotId{1},
      Shape{3},
      {
          4.0F,
          5.0F,
          6.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      z,
      runtime::SlotId{3},
      Shape{3},
      {
          10.0F,
          20.0F,
          30.0F,
      });

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          out,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(
      expected,
      actual_outputs[0]);

  REQUIRE(
      actual_outputs[0].data() ==
      std::vector<float>{
          14.0F,
          30.0F,
          48.0F,
      });
}

TEST_CASE(
    "Compiler and VM match reference interpreter for longer elementwise chain") {

  Graph graph;

  const auto x =
      graph.input(
          Shape{3},
          Dtype::F32);

  const auto y =
      graph.input(
          Shape{3},
          Dtype::F32);

  const auto a =
      graph.mul(x, y);

  const auto b =
      graph.add(a, x);

  const auto out =
      graph.mul(b, y);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      x,
      runtime::SlotId{0},
      Shape{3},
      {
          1.0F,
          2.0F,
          3.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      y,
      runtime::SlotId{1},
      Shape{3},
      {
          4.0F,
          5.0F,
          6.0F,
      });

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          out,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  require_same_tensor(
      expected,
      actual_outputs[0]);
}

// =============================================================================
// Elementwise -> reduction
// =============================================================================

TEST_CASE(
    "Compiler and VM match reference interpreter for Mul followed by Sum") {

  Graph graph;

  const auto x =
      graph.input(
          Shape{4},
          Dtype::F32);

  const auto y =
      graph.input(
          Shape{4},
          Dtype::F32);

  const auto product =
      graph.mul(x, y);

  const auto out =
      graph.sum(product);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      x,
      runtime::SlotId{0},
      Shape{4},
      {
          1.0F,
          2.0F,
          3.0F,
          4.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      y,
      runtime::SlotId{1},
      Shape{4},
      {
          5.0F,
          6.0F,
          7.0F,
          8.0F,
      });

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          out,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  require_same_tensor(
      expected,
      actual_outputs[0]);

  REQUIRE(actual_outputs[0].shape() == Shape{});
  REQUIRE(actual_outputs[0].load(0) == 70.0F);
}

TEST_CASE(
    "Compiler and VM match reference interpreter for Add followed by Sum") {

  Graph graph;

  const auto x =
      graph.input(
          Shape{3},
          Dtype::F32);

  const auto y =
      graph.input(
          Shape{3},
          Dtype::F32);

  const auto added =
      graph.add(x, y);

  const auto out =
      graph.sum(added);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      x,
      runtime::SlotId{0},
      Shape{3},
      {
          1.0F,
          2.0F,
          3.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      y,
      runtime::SlotId{1},
      Shape{3},
      {
          4.0F,
          5.0F,
          6.0F,
      });

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          out,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  require_same_tensor(
      expected,
      actual_outputs[0]);

  REQUIRE(actual_outputs[0].load(0) == 21.0F);
}

// =============================================================================
// Longer mixed graph
// =============================================================================

TEST_CASE(
    "Compiler and VM match reference interpreter for mixed operation chain") {

  Graph graph;

  const auto x =
      graph.input(
          Shape{3},
          Dtype::F32);

  const auto y =
      graph.input(
          Shape{3},
          Dtype::F32);

  // a = x * y
  // b = a + x
  // c = b * y
  // out = sum(c)

  const auto a =
      graph.mul(x, y);

  const auto b =
      graph.add(a, x);

  const auto c =
      graph.mul(b, y);

  const auto out =
      graph.sum(c);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      x,
      runtime::SlotId{0},
      Shape{3},
      {
          1.0F,
          2.0F,
          3.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      y,
      runtime::SlotId{1},
      Shape{3},
      {
          4.0F,
          5.0F,
          6.0F,
      });

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          out,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(
      expected,
      actual_outputs[0]);

  REQUIRE(actual_outputs[0].shape() == Shape{});
}

// =============================================================================
// Shared producer
// =============================================================================

TEST_CASE(
    "Compiler and VM preserve semantics when one intermediate is used twice") {

  Graph graph;

  const auto x =
      graph.input(
          Shape{3},
          Dtype::F32);

  const auto y =
      graph.input(
          Shape{3},
          Dtype::F32);

  const auto shared =
      graph.mul(x, y);

  const auto out =
      graph.add(
          shared,
          shared);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      x,
      runtime::SlotId{0},
      Shape{3},
      {
          1.0F,
          2.0F,
          3.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      y,
      runtime::SlotId{1},
      Shape{3},
      {
          4.0F,
          5.0F,
          6.0F,
      });

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          out,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  require_same_tensor(
      expected,
      actual_outputs[0]);

  REQUIRE(
      actual_outputs[0].data() ==
      std::vector<float>{
          8.0F,
          20.0F,
          36.0F,
      });
}

TEST_CASE(
    "Compiler and VM preserve semantics for shared producer diamond") {

  Graph graph;

  const auto x =
      graph.input(
          Shape{2},
          Dtype::F32);

  const auto y =
      graph.input(
          Shape{2},
          Dtype::F32);

  const auto z =
      graph.input(
          Shape{2},
          Dtype::F32);

  /*
   *              shared = x * y
   *                 /         \
   *       left = shared + z   right = shared * z
   *                 \         /
   *              out = left + right
   *
   * Reachable lowering order:
   *
   * slot 0 = x
   * slot 1 = y
   * slot 2 = shared
   * slot 3 = z
   * slot 4 = left
   * slot 5 = right
   * slot 6 = out
   */

  const auto shared =
      graph.mul(x, y);

  const auto left =
      graph.add(shared, z);

  const auto right =
      graph.mul(shared, z);

  const auto out =
      graph.add(left, right);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      x,
      runtime::SlotId{0},
      Shape{2},
      {
          1.0F,
          2.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      y,
      runtime::SlotId{1},
      Shape{2},
      {
          3.0F,
          4.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      z,
      runtime::SlotId{3},
      Shape{2},
      {
          5.0F,
          6.0F,
      });

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          out,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(
      expected,
      actual_outputs[0]);
}

// =============================================================================
// Multiple outputs
// =============================================================================

TEST_CASE(
    "Compiler and VM match reference interpreter for multiple computed outputs") {

  Graph graph;

  const auto x =
      graph.input(
          Shape{3},
          Dtype::F32);

  const auto y =
      graph.input(
          Shape{3},
          Dtype::F32);

  const auto add_result =
      graph.add(x, y);

  const auto mul_result =
      graph.mul(x, y);

  graph.set_outputs({
      mul_result,
      add_result,
  });

  /*
   * mul_result is visited first:
   *
   * slot 0 = x
   * slot 1 = y
   * slot 2 = mul_result
   * slot 3 = add_result
   */

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      x,
      runtime::SlotId{0},
      Shape{3},
      {
          2.0F,
          3.0F,
          4.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      y,
      runtime::SlotId{1},
      Shape{3},
      {
          5.0F,
          6.0F,
          7.0F,
      });

  reference::Interpreter interpreter;

  const auto expected_mul =
      interpreter.run(
          graph,
          mul_result,
          reference_inputs);

  const auto expected_add =
      interpreter.run(
          graph,
          add_result,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  REQUIRE(actual_outputs.size() == 2);

  require_same_tensor(
      expected_mul,
      actual_outputs[0]);

  require_same_tensor(
      expected_add,
      actual_outputs[1]);
}

TEST_CASE("Compiler and VM preserve duplicate graph outputs") {
  Graph graph;

  const auto x =
      graph.input(
          Shape{2},
          Dtype::F32);

  const auto y =
      graph.input(
          Shape{2},
          Dtype::F32);

  const auto out =
      graph.add(x, y);

  graph.set_outputs({
      out,
      out,
  });

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      x,
      runtime::SlotId{0},
      Shape{2},
      {
          1.0F,
          2.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      y,
      runtime::SlotId{1},
      Shape{2},
      {
          3.0F,
          4.0F,
      });

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          out,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  REQUIRE(actual_outputs.size() == 2);

  require_same_tensor(
      expected,
      actual_outputs[0]);

  require_same_tensor(
      expected,
      actual_outputs[1]);
}

TEST_CASE(
    "Compiler and VM support input and computed value as outputs") {

  Graph graph;

  const auto x =
      graph.input(
          Shape{2},
          Dtype::F32);

  const auto y =
      graph.input(
          Shape{2},
          Dtype::F32);

  const auto added =
      graph.add(x, y);

  graph.set_outputs({
      x,
      added,
  });

  /*
   * x is first output:
   *
   * slot 0 = x
   *
   * lowering added next discovers:
   *
   * slot 1 = y
   * slot 2 = added
   */

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      x,
      runtime::SlotId{0},
      Shape{2},
      {
          1.0F,
          2.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      y,
      runtime::SlotId{1},
      Shape{2},
      {
          10.0F,
          20.0F,
      });

  reference::Interpreter interpreter;

  const auto expected_x =
      interpreter.run(
          graph,
          x,
          reference_inputs);

  const auto expected_added =
      interpreter.run(
          graph,
          added,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  REQUIRE(actual_outputs.size() == 2);

  require_same_tensor(
      expected_x,
      actual_outputs[0]);

  require_same_tensor(
      expected_added,
      actual_outputs[1]);
}

// =============================================================================
// Reachability
// =============================================================================

TEST_CASE(
    "Compiler and VM ignore unreachable computed operation") {

  Graph graph;

  const auto x =
      graph.input(
          Shape{2},
          Dtype::F32);

  const auto y =
      graph.input(
          Shape{2},
          Dtype::F32);

  const auto live =
      graph.add(x, y);

  const auto dead =
      graph.mul(x, y);

  (void)dead;

  graph.set_outputs({live});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      x,
      runtime::SlotId{0},
      Shape{2},
      {
          1.0F,
          2.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      y,
      runtime::SlotId{1},
      Shape{2},
      {
          3.0F,
          4.0F,
      });

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          live,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(
      expected,
      actual_outputs[0]);
}

TEST_CASE(
    "Compiler and VM do not require binding for unreachable graph input") {

  Graph graph;

  const auto x =
      graph.input(
          Shape{2},
          Dtype::F32);

  const auto y =
      graph.input(
          Shape{2},
          Dtype::F32);

  const auto unused =
      graph.input(
          Shape{2},
          Dtype::F32);

  const auto out =
      graph.add(x, y);

  // Make unused participate in a dead subtree too.
  const auto dead =
      graph.mul(y, unused);

  (void)dead;

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  // Intentionally do NOT bind unused.

  bind_input(
      reference_inputs,
      runtime_inputs,
      x,
      runtime::SlotId{0},
      Shape{2},
      {
          5.0F,
          6.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      y,
      runtime::SlotId{1},
      Shape{2},
      {
          7.0F,
          8.0F,
      });

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          out,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(
      expected,
      actual_outputs[0]);
}

// =============================================================================
// MatMul followed by elementwise operation
// =============================================================================

TEST_CASE(
    "Compiler and VM match reference interpreter when Add consumes MatMul result") {

  Graph graph;

  const auto lhs =
      graph.input(
          Shape{2, 3},
          Dtype::F32);

  const auto rhs =
      graph.input(
          Shape{3, 2},
          Dtype::F32);

  const auto bias =
      graph.input(
          Shape{2, 2},
          Dtype::F32);

  const auto matmul_result =
      graph.matmul(lhs, rhs);

  const auto out =
      graph.add(
          matmul_result,
          bias);

  graph.set_outputs({out});

  /*
   * Reachable lowering order:
   *
   * slot 0 = lhs
   * slot 1 = rhs
   * slot 2 = matmul_result
   * slot 3 = bias
   * slot 4 = out
   */

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      lhs,
      runtime::SlotId{0},
      Shape{2, 3},
      {
          1.0F, 2.0F, 3.0F,
          4.0F, 5.0F, 6.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      rhs,
      runtime::SlotId{1},
      Shape{3, 2},
      {
          7.0F, 8.0F,
          9.0F, 10.0F,
          11.0F, 12.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      bias,
      runtime::SlotId{3},
      Shape{2, 2},
      {
          1.0F, 2.0F,
          3.0F, 4.0F,
      });

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          out,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(
      expected,
      actual_outputs[0]);

  REQUIRE(actual_outputs[0].shape() == Shape{2, 2});
}

TEST_CASE(
    "Compiler and VM match reference interpreter when Mul consumes MatMul result") {

  Graph graph;

  const auto lhs =
      graph.input(
          Shape{2, 2},
          Dtype::F32);

  const auto rhs =
      graph.input(
          Shape{2, 2},
          Dtype::F32);

  const auto scale =
      graph.input(
          Shape{2, 2},
          Dtype::F32);

  const auto matmul_result =
      graph.matmul(lhs, rhs);

  const auto out =
      graph.mul(
          matmul_result,
          scale);

  graph.set_outputs({out});

  // lhs=0, rhs=1, matmul=2, scale=3, out=4.

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      lhs,
      runtime::SlotId{0},
      Shape{2, 2},
      {
          1.0F, 2.0F,
          3.0F, 4.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      rhs,
      runtime::SlotId{1},
      Shape{2, 2},
      {
          5.0F, 6.0F,
          7.0F, 8.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      scale,
      runtime::SlotId{3},
      Shape{2, 2},
      {
          2.0F, 2.0F,
          3.0F, 3.0F,
      });

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          out,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  require_same_tensor(
      expected,
      actual_outputs[0]);
}

// =============================================================================
// MatMul followed by reduction
// =============================================================================

TEST_CASE(
    "Compiler and VM match reference interpreter for Sum of MatMul") {

  Graph graph;

  const auto lhs =
      graph.input(
          Shape{2, 3},
          Dtype::F32);

  const auto rhs =
      graph.input(
          Shape{3, 2},
          Dtype::F32);

  const auto product =
      graph.matmul(lhs, rhs);

  const auto out =
      graph.sum(product);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      lhs,
      runtime::SlotId{0},
      Shape{2, 3},
      {
          1.0F, 2.0F, 3.0F,
          4.0F, 5.0F, 6.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      rhs,
      runtime::SlotId{1},
      Shape{3, 2},
      {
          7.0F, 8.0F,
          9.0F, 10.0F,
          11.0F, 12.0F,
      });

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          out,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  require_same_tensor(
      expected,
      actual_outputs[0]);

  REQUIRE(actual_outputs[0].shape() == Shape{});

  // MatMul =
  // [58, 64]
  // [139,154]
  //
  // total = 415
  REQUIRE(actual_outputs[0].load(0) == 415.0F);
}

// =============================================================================
// Two MatMuls
// =============================================================================

TEST_CASE(
    "Compiler and VM match reference interpreter for chained MatMul operations") {

  Graph graph;

  const auto a =
      graph.input(
          Shape{2, 3},
          Dtype::F32);

  const auto b =
      graph.input(
          Shape{3, 2},
          Dtype::F32);

  const auto c =
      graph.input(
          Shape{2, 2},
          Dtype::F32);

  const auto first =
      graph.matmul(a, b);

  const auto out =
      graph.matmul(first, c);

  graph.set_outputs({out});

  /*
   * a     -> slot 0
   * b     -> slot 1
   * first -> slot 2
   * c     -> slot 3
   * out   -> slot 4
   */

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      a,
      runtime::SlotId{0},
      Shape{2, 3},
      {
          1.0F, 2.0F, 3.0F,
          4.0F, 5.0F, 6.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      b,
      runtime::SlotId{1},
      Shape{3, 2},
      {
          1.0F, 2.0F,
          3.0F, 4.0F,
          5.0F, 6.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      c,
      runtime::SlotId{3},
      Shape{2, 2},
      {
          2.0F, 0.0F,
          0.0F, 3.0F,
      });

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          out,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  require_same_tensor(
      expected,
      actual_outputs[0]);

  REQUIRE(actual_outputs[0].shape() == Shape{2, 2});
}

// =============================================================================
// Shared MatMul producer
// =============================================================================

TEST_CASE(
    "Compiler and VM preserve shared MatMul intermediate used by two consumers") {

  Graph graph;

  const auto lhs =
      graph.input(
          Shape{2, 2},
          Dtype::F32);

  const auto rhs =
      graph.input(
          Shape{2, 2},
          Dtype::F32);

  const auto shared =
      graph.matmul(lhs, rhs);

  const auto out =
      graph.add(
          shared,
          shared);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      lhs,
      runtime::SlotId{0},
      Shape{2, 2},
      {
          1.0F, 2.0F,
          3.0F, 4.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      rhs,
      runtime::SlotId{1},
      Shape{2, 2},
      {
          5.0F, 6.0F,
          7.0F, 8.0F,
      });

  reference::Interpreter interpreter;

  const auto expected =
      interpreter.run(
          graph,
          out,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  require_same_tensor(
      expected,
      actual_outputs[0]);

  REQUIRE(
      actual_outputs[0].data() ==
      std::vector<float>{
          38.0F, 44.0F,
          86.0F, 100.0F,
      });
}

// =============================================================================
// Independent outputs of different shapes
// =============================================================================

TEST_CASE(
    "Compiler and VM support tensor and scalar outputs in same program") {

  Graph graph;

  const auto x =
      graph.input(
          Shape{3},
          Dtype::F32);

  const auto y =
      graph.input(
          Shape{3},
          Dtype::F32);

  const auto added =
      graph.add(x, y);

  const auto reduced =
      graph.sum(added);

  graph.set_outputs({
      added,
      reduced,
  });

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      x,
      runtime::SlotId{0},
      Shape{3},
      {
          1.0F,
          2.0F,
          3.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      y,
      runtime::SlotId{1},
      Shape{3},
      {
          4.0F,
          5.0F,
          6.0F,
      });

  reference::Interpreter interpreter;

  const auto expected_added =
      interpreter.run(
          graph,
          added,
          reference_inputs);

  const auto expected_reduced =
      interpreter.run(
          graph,
          reduced,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  REQUIRE(actual_outputs.size() == 2);

  require_same_tensor(
      expected_added,
      actual_outputs[0]);

  require_same_tensor(
      expected_reduced,
      actual_outputs[1]);

  REQUIRE(actual_outputs[0].shape() == Shape{3});
  REQUIRE(actual_outputs[1].shape() == Shape{});
}

// =============================================================================
// Output ordering changes traversal order
// =============================================================================

TEST_CASE(
    "Compiler and VM preserve output ordering when output order drives lowering") {

  Graph graph;

  const auto x =
      graph.input(
          Shape{2},
          Dtype::F32);

  const auto y =
      graph.input(
          Shape{2},
          Dtype::F32);

  const auto sum =
      graph.add(x, y);

  const auto product =
      graph.mul(x, y);

  // product is deliberately requested first.
  graph.set_outputs({
      product,
      sum,
  });

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(
      reference_inputs,
      runtime_inputs,
      x,
      runtime::SlotId{0},
      Shape{2},
      {
          2.0F,
          4.0F,
      });

  bind_input(
      reference_inputs,
      runtime_inputs,
      y,
      runtime::SlotId{1},
      Shape{2},
      {
          3.0F,
          5.0F,
      });

  reference::Interpreter interpreter;

  const auto expected_product =
      interpreter.run(
          graph,
          product,
          reference_inputs);

  const auto expected_sum =
      interpreter.run(
          graph,
          sum,
          reference_inputs);

  const auto actual_outputs =
      compile_and_run(
          graph,
          runtime_inputs);

  REQUIRE(actual_outputs.size() == 2);

  require_same_tensor(
      expected_product,
      actual_outputs[0]);

  require_same_tensor(
      expected_sum,
      actual_outputs[1]);

  REQUIRE(
      actual_outputs[0].data() ==
      std::vector<float>{
          6.0F,
          20.0F,
      });

  REQUIRE(
      actual_outputs[1].data() ==
      std::vector<float>{
          5.0F,
          9.0F,
      });
}