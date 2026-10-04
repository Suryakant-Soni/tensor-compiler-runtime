#include <catch2/catch_test_macros.hpp>

#include "tcr/compiler/compiler.hpp"

#include "tcr/ir/dtype.hpp"
#include "tcr/ir/graph.hpp"
#include "tcr/ir/shape.hpp"

#include "tcr/reference/bindings.hpp"
#include "tcr/reference/interpreter.hpp"
#include "tcr/reference/tensor.hpp"

#include "tcr/runtime/bindings.hpp"
#include "tcr/runtime/tensor.hpp"
#include "tcr/runtime/vm.hpp"

#include <cstddef>
#include <stdexcept>
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

struct RuntimeInput {
  ValueId value_id;
  runtime::Tensor tensor;
};

using RuntimeInputs = std::vector<RuntimeInput>;

void require_same_tensor(const reference::Tensor &expected,
                         const runtime::Tensor &actual) {

  REQUIRE(actual.shape() == expected.shape());
  REQUIRE(actual.dtype() == expected.dtype());
  REQUIRE(actual.data() == expected.data());
}

void bind_input(reference::InputBindings &reference_inputs,
                RuntimeInputs &runtime_inputs, ValueId value_id, Shape shape,
                std::vector<float> data) {

  reference_inputs.bind(value_id, reference::Tensor{
                                      shape,
                                      Dtype::F32,
                                      data,
                                  });

  runtime_inputs.push_back(RuntimeInput{
      value_id,
      runtime::Tensor{
          std::move(shape),
          Dtype::F32,
          std::move(data),
      },
  });
}

std::vector<runtime::Tensor> compile_and_run(const Graph &graph,
                                             const RuntimeInputs &inputs) {

  Compiler compiler;
  const auto program = compiler.compile(graph);

  runtime::InputBindings bindings;

  for (const auto &input : inputs) {
    bindings.bind(program.get_slotid_from_valueid(input.value_id),
                  input.tensor);
  }

  runtime::VM vm;

  return vm.run(program, bindings);
}

} // namespace

// =============================================================================
// Empty graph
// =============================================================================

TEST_CASE("Compiler and VM execute empty graph with no outputs") {
  Graph graph;

  const RuntimeInputs runtime_inputs;

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  REQUIRE(actual_outputs.empty());
}

// =============================================================================
// Direct graph input
// =============================================================================

TEST_CASE(
    "Compiler and VM match reference interpreter when vector input is output") {

  Graph graph;

  const auto x = graph.input(Shape{3}, Dtype::F32);

  graph.set_outputs({x});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{3},
             {
                 10.0F,
                 20.0F,
                 30.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, x, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(expected, actual_outputs[0]);
}

TEST_CASE(
    "Compiler and VM match reference interpreter when scalar input is output") {

  Graph graph;

  const auto x = graph.input(Shape{}, Dtype::F32);

  graph.set_outputs({x});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{}, {17.5F});

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, x, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(expected, actual_outputs[0]);
}

// =============================================================================
// ADD
// =============================================================================

TEST_CASE("Compiler and VM match reference interpreter for vector Add") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);

  const auto y = graph.input(Shape{4}, Dtype::F32);

  const auto out = graph.add(x, y);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{4},
             {
                 1.0F,
                 2.0F,
                 3.0F,
                 4.0F,
             });

  bind_input(reference_inputs, runtime_inputs, y, Shape{4},
             {
                 10.0F,
                 20.0F,
                 30.0F,
                 40.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].data() == std::vector<float>{
                                          11.0F,
                                          22.0F,
                                          33.0F,
                                          44.0F,
                                      });
}

TEST_CASE("Compiler and VM match reference interpreter for matrix Add") {
  Graph graph;

  const auto x = graph.input(Shape{2, 3}, Dtype::F32);

  const auto y = graph.input(Shape{2, 3}, Dtype::F32);

  const auto out = graph.add(x, y);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{2, 3},
             {
                 1.0F,
                 2.0F,
                 3.0F,
                 4.0F,
                 5.0F,
                 6.0F,
             });

  bind_input(reference_inputs, runtime_inputs, y, Shape{2, 3},
             {
                 10.0F,
                 20.0F,
                 30.0F,
                 40.0F,
                 50.0F,
                 60.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].shape() == Shape{2, 3});
}

TEST_CASE("Compiler and VM match reference interpreter for scalar Add") {
  Graph graph;

  const auto x = graph.input(Shape{}, Dtype::F32);

  const auto y = graph.input(Shape{}, Dtype::F32);

  const auto out = graph.add(x, y);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{}, {-2.5F});

  bind_input(reference_inputs, runtime_inputs, y, Shape{}, {7.0F});

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].data() == std::vector<float>{4.5F});
}

// =============================================================================
// MUL
// =============================================================================

TEST_CASE("Compiler and VM match reference interpreter for vector Mul") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);

  const auto y = graph.input(Shape{4}, Dtype::F32);

  const auto out = graph.mul(x, y);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{4},
             {
                 2.0F,
                 3.0F,
                 4.0F,
                 5.0F,
             });

  bind_input(reference_inputs, runtime_inputs, y, Shape{4},
             {
                 5.0F,
                 6.0F,
                 7.0F,
                 8.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].data() == std::vector<float>{
                                          10.0F,
                                          18.0F,
                                          28.0F,
                                          40.0F,
                                      });
}

TEST_CASE("Compiler and VM preserve negative and zero values through Mul") {

  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);

  const auto y = graph.input(Shape{4}, Dtype::F32);

  const auto out = graph.mul(x, y);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{4},
             {
                 -2.0F,
                 0.0F,
                 3.0F,
                 -4.0F,
             });

  bind_input(reference_inputs, runtime_inputs, y, Shape{4},
             {
                 5.0F,
                 -10.0F,
                 -2.0F,
                 -3.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].data() == std::vector<float>{
                                          -10.0F,
                                          0.0F,
                                          -6.0F,
                                          12.0F,
                                      });
}

TEST_CASE("Compiler and VM match reference interpreter for scalar Mul") {
  Graph graph;

  const auto x = graph.input(Shape{}, Dtype::F32);

  const auto y = graph.input(Shape{}, Dtype::F32);

  const auto out = graph.mul(x, y);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{}, {3.5F});

  bind_input(reference_inputs, runtime_inputs, y, Shape{}, {2.0F});

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].load(0) == 7.0F);
}

// =============================================================================
// SUM
// =============================================================================

TEST_CASE("Compiler and VM match reference interpreter for vector Sum") {
  Graph graph;

  const auto x = graph.input(Shape{5}, Dtype::F32);

  const auto out = graph.sum(x);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{5},
             {
                 1.0F,
                 2.0F,
                 3.0F,
                 4.0F,
                 5.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].shape() == Shape{});
  REQUIRE(actual_outputs[0].load(0) == 15.0F);
}

TEST_CASE("Compiler and VM match reference interpreter for matrix Sum") {
  Graph graph;

  const auto x = graph.input(Shape{2, 3}, Dtype::F32);

  const auto out = graph.sum(x);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{2, 3},
             {
                 1.0F,
                 2.0F,
                 3.0F,
                 4.0F,
                 5.0F,
                 6.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].shape() == Shape{});
  REQUIRE(actual_outputs[0].load(0) == 21.0F);
}

TEST_CASE("Compiler and VM match reference interpreter for scalar Sum") {
  Graph graph;

  const auto x = graph.input(Shape{}, Dtype::F32);

  const auto out = graph.sum(x);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{}, {9.25F});

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].load(0) == 9.25F);
}

TEST_CASE("Compiler and VM match reference interpreter for Sum containing "
          "mixed signs") {

  Graph graph;

  const auto x = graph.input(Shape{5}, Dtype::F32);

  const auto out = graph.sum(x);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{5},
             {
                 -5.0F,
                 2.0F,
                 0.0F,
                 10.0F,
                 -3.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].load(0) == 4.0F);
}

// =============================================================================
// MATMUL
// =============================================================================

TEST_CASE("Compiler and VM match reference interpreter for 1x1 MatMul") {
  Graph graph;

  const auto lhs = graph.input(Shape{1, 1}, Dtype::F32);

  const auto rhs = graph.input(Shape{1, 1}, Dtype::F32);

  const auto out = graph.matmul(lhs, rhs);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, lhs, Shape{1, 1}, {6.0F});

  bind_input(reference_inputs, runtime_inputs, rhs, Shape{1, 1}, {7.0F});

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].shape() == Shape{1, 1});
  REQUIRE(actual_outputs[0].load(0) == 42.0F);
}

TEST_CASE("Compiler and VM match reference interpreter for square MatMul") {
  Graph graph;

  const auto lhs = graph.input(Shape{2, 2}, Dtype::F32);

  const auto rhs = graph.input(Shape{2, 2}, Dtype::F32);

  const auto out = graph.matmul(lhs, rhs);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, lhs, Shape{2, 2},
             {
                 1.0F,
                 2.0F,
                 3.0F,
                 4.0F,
             });

  bind_input(reference_inputs, runtime_inputs, rhs, Shape{2, 2},
             {
                 5.0F,
                 6.0F,
                 7.0F,
                 8.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].data() == std::vector<float>{
                                          19.0F,
                                          22.0F,
                                          43.0F,
                                          50.0F,
                                      });
}

TEST_CASE(
    "Compiler and VM match reference interpreter for rectangular MatMul") {
  Graph graph;

  const auto lhs = graph.input(Shape{2, 3}, Dtype::F32);

  const auto rhs = graph.input(Shape{3, 2}, Dtype::F32);

  const auto out = graph.matmul(lhs, rhs);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, lhs, Shape{2, 3},
             {
                 1.0F,
                 2.0F,
                 3.0F,
                 4.0F,
                 5.0F,
                 6.0F,
             });

  bind_input(reference_inputs, runtime_inputs, rhs, Shape{3, 2},
             {
                 7.0F,
                 8.0F,
                 9.0F,
                 10.0F,
                 11.0F,
                 12.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].shape() == Shape{2, 2});

  REQUIRE(actual_outputs[0].data() == std::vector<float>{
                                          58.0F,
                                          64.0F,
                                          139.0F,
                                          154.0F,
                                      });
}

TEST_CASE("Compiler and VM match reference interpreter for one-row rectangular "
          "MatMul") {

  Graph graph;

  const auto lhs = graph.input(Shape{1, 3}, Dtype::F32);

  const auto rhs = graph.input(Shape{3, 4}, Dtype::F32);

  const auto out = graph.matmul(lhs, rhs);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, lhs, Shape{1, 3},
             {
                 1.0F,
                 2.0F,
                 3.0F,
             });

  bind_input(reference_inputs, runtime_inputs, rhs, Shape{3, 4},
             {
                 1.0F,
                 2.0F,
                 3.0F,
                 4.0F,
                 5.0F,
                 6.0F,
                 7.0F,
                 8.0F,
                 9.0F,
                 10.0F,
                 11.0F,
                 12.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].shape() == Shape{1, 4});

  REQUIRE(actual_outputs[0].data() == std::vector<float>{
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

  const auto x = graph.input(Shape{3}, Dtype::F32);

  const auto y = graph.input(Shape{3}, Dtype::F32);

  const auto z = graph.input(Shape{3}, Dtype::F32);

  const auto product = graph.mul(x, y);

  const auto out = graph.add(product, z);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{3},
             {
                 1.0F,
                 2.0F,
                 3.0F,
             });

  bind_input(reference_inputs, runtime_inputs, y, Shape{3},
             {
                 4.0F,
                 5.0F,
                 6.0F,
             });

  bind_input(reference_inputs, runtime_inputs, z, Shape{3},
             {
                 10.0F,
                 20.0F,
                 30.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].data() == std::vector<float>{
                                          14.0F,
                                          30.0F,
                                          48.0F,
                                      });
}

TEST_CASE("Compiler and VM match reference interpreter for longer elementwise "
          "chain") {

  Graph graph;

  const auto x = graph.input(Shape{3}, Dtype::F32);

  const auto y = graph.input(Shape{3}, Dtype::F32);

  const auto a = graph.mul(x, y);

  const auto b = graph.add(a, x);

  const auto out = graph.mul(b, y);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{3},
             {
                 1.0F,
                 2.0F,
                 3.0F,
             });

  bind_input(reference_inputs, runtime_inputs, y, Shape{3},
             {
                 4.0F,
                 5.0F,
                 6.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  require_same_tensor(expected, actual_outputs[0]);
}

// =============================================================================
// Fusion-specific end-to-end coverage
// =============================================================================

TEST_CASE("Compiler and VM preserve semantics when both elementwise branches "
          "are fusible") {

  Graph graph;

  const auto a = graph.input(Shape{3}, Dtype::F32);
  const auto b = graph.input(Shape{3}, Dtype::F32);
  const auto c = graph.input(Shape{3}, Dtype::F32);
  const auto d = graph.input(Shape{3}, Dtype::F32);

  const auto left = graph.mul(a, b);
  const auto right = graph.add(c, d);
  const auto out = graph.add(left, right);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, a, Shape{3},
             {
                 1.0F,
                 2.0F,
                 3.0F,
             });

  bind_input(reference_inputs, runtime_inputs, b, Shape{3},
             {
                 4.0F,
                 5.0F,
                 6.0F,
             });

  bind_input(reference_inputs, runtime_inputs, c, Shape{3},
             {
                 10.0F,
                 20.0F,
                 30.0F,
             });

  bind_input(reference_inputs, runtime_inputs, d, Shape{3},
             {
                 1.0F,
                 2.0F,
                 3.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].data() == std::vector<float>{
                                          15.0F,
                                          32.0F,
                                          51.0F,
                                      });
}

TEST_CASE("Compiler and VM preserve semantics when fused expression reuses "
          "same boundary input") {

  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);
  const auto y = graph.input(Shape{2}, Dtype::F32);

  const auto product = graph.mul(x, y);
  const auto out = graph.add(product, x);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{2},
             {
                 2.0F,
                 3.0F,
             });

  bind_input(reference_inputs, runtime_inputs, y, Shape{2},
             {
                 4.0F,
                 5.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].data() == std::vector<float>{
                                          10.0F,
                                          18.0F,
                                      });
}

TEST_CASE("Compiler and VM preserve semantics for fused scalar chain") {

  Graph graph;

  const auto x = graph.input(Shape{}, Dtype::F32);
  const auto y = graph.input(Shape{}, Dtype::F32);
  const auto z = graph.input(Shape{}, Dtype::F32);

  const auto product = graph.mul(x, y);
  const auto added = graph.add(product, z);
  const auto out = graph.mul(added, x);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{}, {2.0F});
  bind_input(reference_inputs, runtime_inputs, y, Shape{}, {3.0F});
  bind_input(reference_inputs, runtime_inputs, z, Shape{}, {4.0F});

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].shape() == Shape{});
  REQUIRE(actual_outputs[0].load(0) == 20.0F);
}

TEST_CASE(
    "Compiler and VM preserve graph output boundary while fusing above it") {

  Graph graph;

  const auto x = graph.input(Shape{3}, Dtype::F32);
  const auto y = graph.input(Shape{3}, Dtype::F32);
  const auto z = graph.input(Shape{3}, Dtype::F32);
  const auto w = graph.input(Shape{3}, Dtype::F32);

  const auto exposed = graph.mul(x, y);
  const auto added = graph.add(exposed, z);
  const auto out = graph.mul(added, w);

  graph.set_outputs({
      exposed,
      out,
  });

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{3},
             {
                 1.0F,
                 2.0F,
                 3.0F,
             });

  bind_input(reference_inputs, runtime_inputs, y, Shape{3},
             {
                 4.0F,
                 5.0F,
                 6.0F,
             });

  bind_input(reference_inputs, runtime_inputs, z, Shape{3},
             {
                 10.0F,
                 20.0F,
                 30.0F,
             });

  bind_input(reference_inputs, runtime_inputs, w, Shape{3},
             {
                 2.0F,
                 3.0F,
                 4.0F,
             });

  reference::Interpreter interpreter;

  const auto expected_exposed =
      interpreter.run(graph, exposed, reference_inputs);

  const auto expected_out = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  REQUIRE(actual_outputs.size() == 2);

  require_same_tensor(expected_exposed, actual_outputs[0]);
  require_same_tensor(expected_out, actual_outputs[1]);

  REQUIRE(actual_outputs[0].data() == std::vector<float>{
                                          4.0F,
                                          10.0F,
                                          18.0F,
                                      });

  REQUIRE(actual_outputs[1].data() == std::vector<float>{
                                          28.0F,
                                          90.0F,
                                          192.0F,
                                      });
}

TEST_CASE(
    "Compiler and VM preserve semantics when fusion stops at MatMul boundary") {

  Graph graph;

  const auto lhs = graph.input(Shape{2, 2}, Dtype::F32);
  const auto rhs = graph.input(Shape{2, 2}, Dtype::F32);
  const auto bias = graph.input(Shape{2, 2}, Dtype::F32);
  const auto scale = graph.input(Shape{2, 2}, Dtype::F32);

  const auto product = graph.matmul(lhs, rhs);
  const auto added = graph.add(product, bias);
  const auto out = graph.mul(added, scale);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, lhs, Shape{2, 2},
             {
                 1.0F,
                 2.0F,
                 3.0F,
                 4.0F,
             });

  bind_input(reference_inputs, runtime_inputs, rhs, Shape{2, 2},
             {
                 5.0F,
                 6.0F,
                 7.0F,
                 8.0F,
             });

  bind_input(reference_inputs, runtime_inputs, bias, Shape{2, 2},
             {
                 1.0F,
                 2.0F,
                 3.0F,
                 4.0F,
             });

  bind_input(reference_inputs, runtime_inputs, scale, Shape{2, 2},
             {
                 2.0F,
                 3.0F,
                 4.0F,
                 5.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].data() == std::vector<float>{
                                          40.0F,
                                          72.0F,
                                          184.0F,
                                          270.0F,
                                      });
}

TEST_CASE(
    "Compiler and VM preserve semantics for independent fused output islands") {

  Graph graph;

  const auto a = graph.input(Shape{2}, Dtype::F32);
  const auto b = graph.input(Shape{2}, Dtype::F32);
  const auto c = graph.input(Shape{2}, Dtype::F32);
  const auto d = graph.input(Shape{2}, Dtype::F32);
  const auto e = graph.input(Shape{2}, Dtype::F32);
  const auto f = graph.input(Shape{2}, Dtype::F32);

  const auto first_product = graph.mul(a, b);
  const auto first_out = graph.add(first_product, c);

  const auto second_sum = graph.add(d, e);
  const auto second_out = graph.mul(second_sum, f);

  graph.set_outputs({
      first_out,
      second_out,
  });

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, a, Shape{2}, {1.0F, 2.0F});
  bind_input(reference_inputs, runtime_inputs, b, Shape{2}, {3.0F, 4.0F});
  bind_input(reference_inputs, runtime_inputs, c, Shape{2}, {5.0F, 6.0F});

  bind_input(reference_inputs, runtime_inputs, d, Shape{2}, {7.0F, 8.0F});
  bind_input(reference_inputs, runtime_inputs, e, Shape{2}, {1.0F, 2.0F});
  bind_input(reference_inputs, runtime_inputs, f, Shape{2}, {2.0F, 3.0F});

  reference::Interpreter interpreter;

  const auto expected_first =
      interpreter.run(graph, first_out, reference_inputs);

  const auto expected_second =
      interpreter.run(graph, second_out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  REQUIRE(actual_outputs.size() == 2);

  require_same_tensor(expected_first, actual_outputs[0]);
  require_same_tensor(expected_second, actual_outputs[1]);

  REQUIRE(actual_outputs[0].data() == std::vector<float>{
                                          8.0F,
                                          14.0F,
                                      });

  REQUIRE(actual_outputs[1].data() == std::vector<float>{
                                          16.0F,
                                          30.0F,
                                      });
}

// =============================================================================
// Elementwise -> reduction
// =============================================================================

TEST_CASE(
    "Compiler and VM match reference interpreter for Mul followed by Sum") {

  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);

  const auto y = graph.input(Shape{4}, Dtype::F32);

  const auto product = graph.mul(x, y);

  const auto out = graph.sum(product);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{4},
             {
                 1.0F,
                 2.0F,
                 3.0F,
                 4.0F,
             });

  bind_input(reference_inputs, runtime_inputs, y, Shape{4},
             {
                 5.0F,
                 6.0F,
                 7.0F,
                 8.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].shape() == Shape{});
  REQUIRE(actual_outputs[0].load(0) == 70.0F);
}

TEST_CASE(
    "Compiler and VM match reference interpreter for Add followed by Sum") {

  Graph graph;

  const auto x = graph.input(Shape{3}, Dtype::F32);

  const auto y = graph.input(Shape{3}, Dtype::F32);

  const auto added = graph.add(x, y);

  const auto out = graph.sum(added);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{3},
             {
                 1.0F,
                 2.0F,
                 3.0F,
             });

  bind_input(reference_inputs, runtime_inputs, y, Shape{3},
             {
                 4.0F,
                 5.0F,
                 6.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].shape() == Shape{});
  REQUIRE(actual_outputs[0].load(0) == 21.0F);
}

TEST_CASE(
    "Compiler and VM preserve elementwise producer shared by Sum and another "
    "consumer") {

  Graph graph;

  const auto x = graph.input(Shape{3}, Dtype::F32);
  const auto y = graph.input(Shape{3}, Dtype::F32);
  const auto z = graph.input(Shape{3}, Dtype::F32);

  const auto shared = graph.add(x, y);
  const auto reduced = graph.sum(shared);
  const auto other = graph.mul(shared, z);

  // `shared` has two reachable consumers. Under the current conservative
  // Phase-5 policy it must not be absorbed only into the Sum kernel.
  graph.set_outputs({
      reduced,
      other,
  });

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{3},
             {
                 1.0F,
                 2.0F,
                 3.0F,
             });

  bind_input(reference_inputs, runtime_inputs, y, Shape{3},
             {
                 4.0F,
                 5.0F,
                 6.0F,
             });

  bind_input(reference_inputs, runtime_inputs, z, Shape{3},
             {
                 2.0F,
                 3.0F,
                 4.0F,
             });

  reference::Interpreter interpreter;

  const auto expected_reduced =
      interpreter.run(graph, reduced, reference_inputs);

  const auto expected_other = interpreter.run(graph, other, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  REQUIRE(actual_outputs.size() == 2);

  require_same_tensor(expected_reduced, actual_outputs[0]);
  require_same_tensor(expected_other, actual_outputs[1]);

  REQUIRE(actual_outputs[0].shape() == Shape{});
  REQUIRE(actual_outputs[0].load(0) == 21.0F);

  REQUIRE(actual_outputs[1].data() == std::vector<float>{
                                          10.0F,
                                          21.0F,
                                          36.0F,
                                      });
}

// =============================================================================
// Longer mixed graph
// =============================================================================

TEST_CASE(
    "Compiler and VM match reference interpreter for fused elementwise chain "
    "followed by Sum") {

  Graph graph;

  const auto x = graph.input(Shape{3}, Dtype::F32);

  const auto y = graph.input(Shape{3}, Dtype::F32);

  // a = x * y
  // b = a + x
  // c = b * y
  // out = sum(c)

  const auto a = graph.mul(x, y);

  const auto b = graph.add(a, x);

  const auto c = graph.mul(b, y);

  const auto out = graph.sum(c);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{3},
             {
                 1.0F,
                 2.0F,
                 3.0F,
             });

  bind_input(reference_inputs, runtime_inputs, y, Shape{3},
             {
                 4.0F,
                 5.0F,
                 6.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].shape() == Shape{});
  REQUIRE(actual_outputs[0].load(0) == 206.0F);
}

// =============================================================================
// Shared producer
// =============================================================================

TEST_CASE(
    "Compiler and VM preserve semantics when one intermediate is used twice") {

  Graph graph;

  const auto x = graph.input(Shape{3}, Dtype::F32);

  const auto y = graph.input(Shape{3}, Dtype::F32);

  const auto shared = graph.mul(x, y);

  const auto out = graph.add(shared, shared);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{3},
             {
                 1.0F,
                 2.0F,
                 3.0F,
             });

  bind_input(reference_inputs, runtime_inputs, y, Shape{3},
             {
                 4.0F,
                 5.0F,
                 6.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].data() == std::vector<float>{
                                          8.0F,
                                          20.0F,
                                          36.0F,
                                      });
}

TEST_CASE("Compiler and VM preserve semantics for shared producer diamond") {

  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);

  const auto y = graph.input(Shape{2}, Dtype::F32);

  const auto z = graph.input(Shape{2}, Dtype::F32);
  const auto shared = graph.mul(x, y);

  const auto left = graph.add(shared, z);

  const auto right = graph.mul(shared, z);

  const auto out = graph.add(left, right);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{2},
             {
                 1.0F,
                 2.0F,
             });

  bind_input(reference_inputs, runtime_inputs, y, Shape{2},
             {
                 3.0F,
                 4.0F,
             });

  bind_input(reference_inputs, runtime_inputs, z, Shape{2},
             {
                 5.0F,
                 6.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(expected, actual_outputs[0]);
}

// =============================================================================
// Multiple outputs
// =============================================================================

TEST_CASE("Compiler and VM match reference interpreter for multiple computed "
          "outputs") {

  Graph graph;

  const auto x = graph.input(Shape{3}, Dtype::F32);

  const auto y = graph.input(Shape{3}, Dtype::F32);

  const auto add_result = graph.add(x, y);

  const auto mul_result = graph.mul(x, y);

  graph.set_outputs({
      mul_result,
      add_result,
  });
  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{3},
             {
                 2.0F,
                 3.0F,
                 4.0F,
             });

  bind_input(reference_inputs, runtime_inputs, y, Shape{3},
             {
                 5.0F,
                 6.0F,
                 7.0F,
             });

  reference::Interpreter interpreter;

  const auto expected_mul =
      interpreter.run(graph, mul_result, reference_inputs);

  const auto expected_add =
      interpreter.run(graph, add_result, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  REQUIRE(actual_outputs.size() == 2);

  require_same_tensor(expected_mul, actual_outputs[0]);

  require_same_tensor(expected_add, actual_outputs[1]);
}

TEST_CASE("Compiler and VM preserve duplicate graph outputs") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);

  const auto y = graph.input(Shape{2}, Dtype::F32);

  const auto out = graph.add(x, y);

  graph.set_outputs({
      out,
      out,
  });

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{2},
             {
                 1.0F,
                 2.0F,
             });

  bind_input(reference_inputs, runtime_inputs, y, Shape{2},
             {
                 3.0F,
                 4.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  REQUIRE(actual_outputs.size() == 2);

  require_same_tensor(expected, actual_outputs[0]);

  require_same_tensor(expected, actual_outputs[1]);
}

TEST_CASE("Compiler and VM support input and computed value as outputs") {

  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);

  const auto y = graph.input(Shape{2}, Dtype::F32);

  const auto added = graph.add(x, y);

  graph.set_outputs({
      x,
      added,
  });
  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{2},
             {
                 1.0F,
                 2.0F,
             });

  bind_input(reference_inputs, runtime_inputs, y, Shape{2},
             {
                 10.0F,
                 20.0F,
             });

  reference::Interpreter interpreter;

  const auto expected_x = interpreter.run(graph, x, reference_inputs);

  const auto expected_added = interpreter.run(graph, added, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  REQUIRE(actual_outputs.size() == 2);

  require_same_tensor(expected_x, actual_outputs[0]);

  require_same_tensor(expected_added, actual_outputs[1]);
}

// =============================================================================
// Reachability
// =============================================================================

TEST_CASE("Compiler and VM ignore unreachable computed operation") {

  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);

  const auto y = graph.input(Shape{2}, Dtype::F32);

  const auto live = graph.add(x, y);

  const auto dead = graph.mul(x, y);

  (void)dead;

  graph.set_outputs({live});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{2},
             {
                 1.0F,
                 2.0F,
             });

  bind_input(reference_inputs, runtime_inputs, y, Shape{2},
             {
                 3.0F,
                 4.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, live, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(expected, actual_outputs[0]);
}

TEST_CASE(
    "Compiler and VM do not require binding for unreachable graph input") {

  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);

  const auto y = graph.input(Shape{2}, Dtype::F32);

  const auto unused = graph.input(Shape{2}, Dtype::F32);

  const auto out = graph.add(x, y);

  // Make unused participate in a dead subtree too.
  const auto dead = graph.mul(y, unused);

  (void)dead;

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  // Intentionally do NOT bind unused.

  bind_input(reference_inputs, runtime_inputs, x, Shape{2},
             {
                 5.0F,
                 6.0F,
             });

  bind_input(reference_inputs, runtime_inputs, y, Shape{2},
             {
                 7.0F,
                 8.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(expected, actual_outputs[0]);
}

// =============================================================================
// MatMul followed by elementwise operation
// =============================================================================

TEST_CASE("Compiler and VM match reference interpreter when Add consumes "
          "MatMul result") {

  Graph graph;

  const auto lhs = graph.input(Shape{2, 3}, Dtype::F32);

  const auto rhs = graph.input(Shape{3, 2}, Dtype::F32);

  const auto bias = graph.input(Shape{2, 2}, Dtype::F32);

  const auto matmul_result = graph.matmul(lhs, rhs);

  const auto out = graph.add(matmul_result, bias);

  graph.set_outputs({out});
  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, lhs, Shape{2, 3},
             {
                 1.0F,
                 2.0F,
                 3.0F,
                 4.0F,
                 5.0F,
                 6.0F,
             });

  bind_input(reference_inputs, runtime_inputs, rhs, Shape{3, 2},
             {
                 7.0F,
                 8.0F,
                 9.0F,
                 10.0F,
                 11.0F,
                 12.0F,
             });

  bind_input(reference_inputs, runtime_inputs, bias, Shape{2, 2},
             {
                 1.0F,
                 2.0F,
                 3.0F,
                 4.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].shape() == Shape{2, 2});
}

TEST_CASE("Compiler and VM match reference interpreter when Mul consumes "
          "MatMul result") {

  Graph graph;

  const auto lhs = graph.input(Shape{2, 2}, Dtype::F32);

  const auto rhs = graph.input(Shape{2, 2}, Dtype::F32);

  const auto scale = graph.input(Shape{2, 2}, Dtype::F32);

  const auto matmul_result = graph.matmul(lhs, rhs);

  const auto out = graph.mul(matmul_result, scale);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, lhs, Shape{2, 2},
             {
                 1.0F,
                 2.0F,
                 3.0F,
                 4.0F,
             });

  bind_input(reference_inputs, runtime_inputs, rhs, Shape{2, 2},
             {
                 5.0F,
                 6.0F,
                 7.0F,
                 8.0F,
             });

  bind_input(reference_inputs, runtime_inputs, scale, Shape{2, 2},
             {
                 2.0F,
                 2.0F,
                 3.0F,
                 3.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  require_same_tensor(expected, actual_outputs[0]);
}

// =============================================================================
// MatMul followed by reduction
// =============================================================================

TEST_CASE("Compiler and VM match reference interpreter for Sum of MatMul") {

  Graph graph;

  const auto lhs = graph.input(Shape{2, 3}, Dtype::F32);

  const auto rhs = graph.input(Shape{3, 2}, Dtype::F32);

  const auto product = graph.matmul(lhs, rhs);

  const auto out = graph.sum(product);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, lhs, Shape{2, 3},
             {
                 1.0F,
                 2.0F,
                 3.0F,
                 4.0F,
                 5.0F,
                 6.0F,
             });

  bind_input(reference_inputs, runtime_inputs, rhs, Shape{3, 2},
             {
                 7.0F,
                 8.0F,
                 9.0F,
                 10.0F,
                 11.0F,
                 12.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  require_same_tensor(expected, actual_outputs[0]);

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

TEST_CASE("Compiler and VM match reference interpreter for chained MatMul "
          "operations") {

  Graph graph;

  const auto a = graph.input(Shape{2, 3}, Dtype::F32);

  const auto b = graph.input(Shape{3, 2}, Dtype::F32);

  const auto c = graph.input(Shape{2, 2}, Dtype::F32);

  const auto first = graph.matmul(a, b);

  const auto out = graph.matmul(first, c);

  graph.set_outputs({out});
  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, a, Shape{2, 3},
             {
                 1.0F,
                 2.0F,
                 3.0F,
                 4.0F,
                 5.0F,
                 6.0F,
             });

  bind_input(reference_inputs, runtime_inputs, b, Shape{3, 2},
             {
                 1.0F,
                 2.0F,
                 3.0F,
                 4.0F,
                 5.0F,
                 6.0F,
             });

  bind_input(reference_inputs, runtime_inputs, c, Shape{2, 2},
             {
                 2.0F,
                 0.0F,
                 0.0F,
                 3.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].shape() == Shape{2, 2});
}

// =============================================================================
// Shared MatMul producer
// =============================================================================

TEST_CASE("Compiler and VM preserve shared MatMul intermediate used by two "
          "consumers") {

  Graph graph;

  const auto lhs = graph.input(Shape{2, 2}, Dtype::F32);

  const auto rhs = graph.input(Shape{2, 2}, Dtype::F32);

  const auto shared = graph.matmul(lhs, rhs);

  const auto out = graph.add(shared, shared);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, lhs, Shape{2, 2},
             {
                 1.0F,
                 2.0F,
                 3.0F,
                 4.0F,
             });

  bind_input(reference_inputs, runtime_inputs, rhs, Shape{2, 2},
             {
                 5.0F,
                 6.0F,
                 7.0F,
                 8.0F,
             });

  reference::Interpreter interpreter;

  const auto expected = interpreter.run(graph, out, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].data() == std::vector<float>{
                                          38.0F,
                                          44.0F,
                                          86.0F,
                                          100.0F,
                                      });
}

// =============================================================================
// Independent outputs of different shapes
// =============================================================================

TEST_CASE("Compiler and VM support tensor and scalar outputs in same program") {

  Graph graph;

  const auto x = graph.input(Shape{3}, Dtype::F32);

  const auto y = graph.input(Shape{3}, Dtype::F32);

  const auto added = graph.add(x, y);

  const auto reduced = graph.sum(added);

  // `added` is itself a graph output as well as the input to Sum. Under the
  // current Phase-5 policy it remains materialized rather than being grouped
  // away as an internal-only intermediate.
  graph.set_outputs({
      added,
      reduced,
  });

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{3},
             {
                 1.0F,
                 2.0F,
                 3.0F,
             });

  bind_input(reference_inputs, runtime_inputs, y, Shape{3},
             {
                 4.0F,
                 5.0F,
                 6.0F,
             });

  reference::Interpreter interpreter;

  const auto expected_added = interpreter.run(graph, added, reference_inputs);

  const auto expected_reduced =
      interpreter.run(graph, reduced, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  REQUIRE(actual_outputs.size() == 2);

  require_same_tensor(expected_added, actual_outputs[0]);

  require_same_tensor(expected_reduced, actual_outputs[1]);

  REQUIRE(actual_outputs[0].shape() == Shape{3});
  REQUIRE(actual_outputs[1].shape() == Shape{});
}

// =============================================================================
// Output ordering
// =============================================================================

TEST_CASE("Compiler and VM preserve requested graph output ordering through "
          "scheduling and lowering") {

  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);

  const auto y = graph.input(Shape{2}, Dtype::F32);

  const auto sum = graph.add(x, y);

  const auto product = graph.mul(x, y);

  // product is deliberately requested first.
  graph.set_outputs({
      product,
      sum,
  });

  reference::InputBindings reference_inputs;
  RuntimeInputs runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, Shape{2},
             {
                 2.0F,
                 4.0F,
             });

  bind_input(reference_inputs, runtime_inputs, y, Shape{2},
             {
                 3.0F,
                 5.0F,
             });

  reference::Interpreter interpreter;

  const auto expected_product =
      interpreter.run(graph, product, reference_inputs);

  const auto expected_sum = interpreter.run(graph, sum, reference_inputs);

  const auto actual_outputs = compile_and_run(graph, runtime_inputs);

  REQUIRE(actual_outputs.size() == 2);

  require_same_tensor(expected_product, actual_outputs[0]);

  require_same_tensor(expected_sum, actual_outputs[1]);

  REQUIRE(actual_outputs[0].data() == std::vector<float>{
                                          6.0F,
                                          20.0F,
                                      });

  REQUIRE(actual_outputs[1].data() == std::vector<float>{
                                          5.0F,
                                          9.0F,
                                      });
}

// =============================================================================
// Runtime input contract validation
// =============================================================================

TEST_CASE("Compiler and VM reject a bound tensor whose shape does not match "
          "graph input") {

  Graph graph;

  const auto x = graph.input(Shape{2, 3}, Dtype::F32);
  graph.set_outputs({x});

  RuntimeInputs runtime_inputs;

  // The graph declares x as Shape{2, 3}, but the actual bound tensor is
  // Shape{6}.
  runtime_inputs.push_back(RuntimeInput{
      x,
      runtime::Tensor{
          Shape{6},
          Dtype::F32,
          {
              1.0F,
              2.0F,
              3.0F,
              4.0F,
              5.0F,
              6.0F,
          },
      },
  });

  REQUIRE_THROWS_AS(compile_and_run(graph, runtime_inputs),
                    std::invalid_argument);
}

TEST_CASE("Compiler and VM reject wrong input shape even when element count "
          "matches") {

  Graph graph;

  const auto x = graph.input(Shape{2, 3}, Dtype::F32);
  const auto y = graph.input(Shape{2, 3}, Dtype::F32);

  const auto out = graph.add(x, y);
  graph.set_outputs({out});

  RuntimeInputs runtime_inputs;

  runtime_inputs.push_back(RuntimeInput{
      x,
      runtime::Tensor{
          Shape{2, 3},
          Dtype::F32,
          {
              1.0F,
              2.0F,
              3.0F,
              4.0F,
              5.0F,
              6.0F,
          },
      },
  });

  // Same element count as Shape{2, 3}, but a different shape.
  runtime_inputs.push_back(RuntimeInput{
      y,
      runtime::Tensor{
          Shape{3, 2},
          Dtype::F32,
          {
              10.0F,
              20.0F,
              30.0F,
              40.0F,
              50.0F,
              60.0F,
          },
      },
  });

  REQUIRE_THROWS_AS(compile_and_run(graph, runtime_inputs),
                    std::invalid_argument);
}

TEST_CASE(
    "Compiler and VM validate input shape after ValueId to SlotId mapping") {

  Graph graph;

  const auto x = graph.input(Shape{3}, Dtype::F32);
  const auto y = graph.input(Shape{3}, Dtype::F32);
  const auto z = graph.input(Shape{3}, Dtype::F32);

  const auto product = graph.mul(x, y);
  const auto out = graph.add(product, z);

  graph.set_outputs({out});

  RuntimeInputs runtime_inputs;

  runtime_inputs.push_back(RuntimeInput{
      x,
      runtime::Tensor{
          Shape{3},
          Dtype::F32,
          {1.0F, 2.0F, 3.0F},
      },
  });

  runtime_inputs.push_back(RuntimeInput{
      y,
      runtime::Tensor{
          Shape{3},
          Dtype::F32,
          {4.0F, 5.0F, 6.0F},
      },
  });

  // z's runtime slot is determined only after compilation. The caller still
  // binds by ValueId, and VM validation must reject the mismatched shape.
  runtime_inputs.push_back(RuntimeInput{
      z,
      runtime::Tensor{
          Shape{1, 3},
          Dtype::F32,
          {10.0F, 20.0F, 30.0F},
      },
  });

  REQUIRE_THROWS_AS(compile_and_run(graph, runtime_inputs),
                    std::invalid_argument);
}
