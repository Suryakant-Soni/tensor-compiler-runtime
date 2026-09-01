#include <catch2/catch_test_macros.hpp>

#include "tcr/compiler/compiler.hpp"
#include "tcr/ir/dtype.hpp"
#include "tcr/ir/graph.hpp"
#include "tcr/ir/shape.hpp"

#include "tcr/reference/bindings.hpp"
#include "tcr/reference/interpreter.hpp"
#include "tcr/reference/tensor.hpp"

#include "tcr/runtime/bindings.hpp"
#include "tcr/runtime/program.hpp"
#include "tcr/runtime/slot_id.hpp"
#include "tcr/runtime/tensor.hpp"
#include "tcr/runtime/vm.hpp"

#include <cstddef>
#include <vector>

using tcr::compiler::Compiler;

using tcr::ir::Dtype;
using tcr::ir::Graph;
using tcr::ir::Shape;
using tcr::ir::ValueId;

namespace reference = tcr::reference;
namespace runtime = tcr::runtime;

namespace {

void require_same_tensor(const reference::Tensor &expected,
                         const runtime::Tensor &actual) {
  REQUIRE(actual.shape() == expected.shape());
  REQUIRE(actual.dtype() == expected.dtype());
  REQUIRE(actual.data() == expected.data());
}

void bind_input(reference::InputBindings &reference_inputs,
                runtime::InputBindings &runtime_inputs, ValueId value_id,
                runtime::SlotId slot_id, Shape shape,
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
          shape,
          Dtype::F32,
          std::move(data),
      });
}

} // namespace

// -----------------------------------------------------------------------------
// ADD
// -----------------------------------------------------------------------------

TEST_CASE("Compiler and VM match reference interpreter for Add") {
  Graph graph;

  const auto x = graph.input(Shape{3}, Dtype::F32);
  const auto y = graph.input(Shape{3}, Dtype::F32);

  const auto out = graph.add(x, y);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, runtime::SlotId{0}, Shape{3},
             {1.0F, 2.0F, 3.0F});

  bind_input(reference_inputs, runtime_inputs, y, runtime::SlotId{1}, Shape{3},
             {4.0F, 5.0F, 6.0F});

  reference::Interpreter interpreter;
  const auto expected = interpreter.run(graph, out, reference_inputs);

  Compiler compiler;
  const auto program = compiler.compile(graph);

  runtime::VM vm;
  const auto actual_outputs = vm.run(program, runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(expected, actual_outputs[0]);
}

// -----------------------------------------------------------------------------
// MUL
// -----------------------------------------------------------------------------

TEST_CASE("Compiler and VM match reference interpreter for Mul") {
  Graph graph;

  const auto x = graph.input(Shape{3}, Dtype::F32);
  const auto y = graph.input(Shape{3}, Dtype::F32);

  const auto out = graph.mul(x, y);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, runtime::SlotId{0}, Shape{3},
             {2.0F, 3.0F, 4.0F});

  bind_input(reference_inputs, runtime_inputs, y, runtime::SlotId{1}, Shape{3},
             {5.0F, 6.0F, 7.0F});

  reference::Interpreter interpreter;
  const auto expected = interpreter.run(graph, out, reference_inputs);

  Compiler compiler;
  const auto program = compiler.compile(graph);

  runtime::VM vm;
  const auto actual_outputs = vm.run(program, runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(expected, actual_outputs[0]);
}

// -----------------------------------------------------------------------------
// SUM
// -----------------------------------------------------------------------------

TEST_CASE("Compiler and VM match reference interpreter for Sum") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);
  const auto out = graph.sum(x);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, runtime::SlotId{0}, Shape{4},
             {1.0F, 2.0F, 3.0F, 4.0F});

  reference::Interpreter interpreter;
  const auto expected = interpreter.run(graph, out, reference_inputs);

  Compiler compiler;
  const auto program = compiler.compile(graph);

  runtime::VM vm;
  const auto actual_outputs = vm.run(program, runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].shape() == Shape{});
}

// -----------------------------------------------------------------------------
// MATMUL
// -----------------------------------------------------------------------------

TEST_CASE("Compiler and VM match reference interpreter for MatMul") {
  Graph graph;

  const auto lhs = graph.input(Shape{2, 3}, Dtype::F32);
  const auto rhs = graph.input(Shape{3, 2}, Dtype::F32);

  const auto out = graph.matmul(lhs, rhs);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, lhs, runtime::SlotId{0},
             Shape{2, 3},
             {
                 1.0F, 2.0F, 3.0F,
                 4.0F, 5.0F, 6.0F,
             });

  bind_input(reference_inputs, runtime_inputs, rhs, runtime::SlotId{1},
             Shape{3, 2},
             {
                 7.0F, 8.0F,
                 9.0F, 10.0F,
                 11.0F, 12.0F,
             });

  reference::Interpreter interpreter;
  const auto expected = interpreter.run(graph, out, reference_inputs);

  Compiler compiler;
  const auto program = compiler.compile(graph);

  runtime::VM vm;
  const auto actual_outputs = vm.run(program, runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].shape() == Shape{2, 2});
}

// -----------------------------------------------------------------------------
// CHAIN
// -----------------------------------------------------------------------------

TEST_CASE("Compiler and VM match reference interpreter for chained graph") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);
  const auto y = graph.input(Shape{2}, Dtype::F32);
  const auto z = graph.input(Shape{2}, Dtype::F32);

  // t = x * y
  // out = t + z

  const auto t = graph.mul(x, y);
  const auto out = graph.add(t, z);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, runtime::SlotId{0}, Shape{2},
             {1.0F, 2.0F});

  bind_input(reference_inputs, runtime_inputs, y, runtime::SlotId{1}, Shape{2},
             {3.0F, 4.0F});

  bind_input(reference_inputs, runtime_inputs, z, runtime::SlotId{2}, Shape{2},
             {5.0F, 6.0F});

  reference::Interpreter interpreter;
  const auto expected = interpreter.run(graph, out, reference_inputs);

  Compiler compiler;
  const auto program = compiler.compile(graph);

  runtime::VM vm;
  const auto actual_outputs = vm.run(program, runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].data() ==
          std::vector<float>{8.0F, 14.0F});
}

// -----------------------------------------------------------------------------
// LONGER MIXED GRAPH
// -----------------------------------------------------------------------------

TEST_CASE("Compiler and VM match reference interpreter for mixed operation chain") {
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
  runtime::InputBindings runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, runtime::SlotId{0}, Shape{3},
             {1.0F, 2.0F, 3.0F});

  bind_input(reference_inputs, runtime_inputs, y, runtime::SlotId{1}, Shape{3},
             {4.0F, 5.0F, 6.0F});

  reference::Interpreter interpreter;
  const auto expected = interpreter.run(graph, out, reference_inputs);

  Compiler compiler;
  const auto program = compiler.compile(graph);

  runtime::VM vm;
  const auto actual_outputs = vm.run(program, runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(expected, actual_outputs[0]);

  REQUIRE(actual_outputs[0].shape() == Shape{});
}

// -----------------------------------------------------------------------------
// DIAMOND / SHARED PRODUCER
// -----------------------------------------------------------------------------

TEST_CASE("Compiler and VM preserve semantics for shared producer diamond") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);
  const auto y = graph.input(Shape{2}, Dtype::F32);
  const auto z = graph.input(Shape{2}, Dtype::F32);

  //              shared = x * y
  //                 /         \
  //       left = shared + z   right = shared * z
  //                 \         /
  //             out = left + right

  const auto shared = graph.mul(x, y);
  const auto left = graph.add(shared, z);
  const auto right = graph.mul(shared, z);
  const auto out = graph.add(left, right);

  graph.set_outputs({out});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, runtime::SlotId{0}, Shape{2},
             {1.0F, 2.0F});

  bind_input(reference_inputs, runtime_inputs, y, runtime::SlotId{1}, Shape{2},
             {3.0F, 4.0F});

  bind_input(reference_inputs, runtime_inputs, z, runtime::SlotId{2}, Shape{2},
             {5.0F, 6.0F});

  reference::Interpreter interpreter;
  const auto expected = interpreter.run(graph, out, reference_inputs);

  Compiler compiler;
  const auto program = compiler.compile(graph);

  // This is also a scheduling invariant:
  // shared MUL must only be emitted once.
  REQUIRE(program.instructions().size() == 4);

  runtime::VM vm;
  const auto actual_outputs = vm.run(program, runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(expected, actual_outputs[0]);
}

// -----------------------------------------------------------------------------
// MULTIPLE OUTPUTS
// -----------------------------------------------------------------------------

TEST_CASE("Compiler and VM match reference interpreter for multiple outputs") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);
  const auto y = graph.input(Shape{2}, Dtype::F32);

  const auto add_result = graph.add(x, y);
  const auto mul_result = graph.mul(x, y);

  graph.set_outputs({
      mul_result,
      add_result,
  });

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, runtime::SlotId{0}, Shape{2},
             {2.0F, 3.0F});

  bind_input(reference_inputs, runtime_inputs, y, runtime::SlotId{1}, Shape{2},
             {4.0F, 5.0F});

  reference::Interpreter interpreter;

  const auto expected_mul =
      interpreter.run(graph, mul_result, reference_inputs);

  const auto expected_add =
      interpreter.run(graph, add_result, reference_inputs);

  Compiler compiler;
  const auto program = compiler.compile(graph);

  runtime::VM vm;
  const auto actual_outputs = vm.run(program, runtime_inputs);

  REQUIRE(actual_outputs.size() == 2);

  // Program output ordering must correspond to graph output ordering.
  require_same_tensor(expected_mul, actual_outputs[0]);
  require_same_tensor(expected_add, actual_outputs[1]);
}

// -----------------------------------------------------------------------------
// GRAPH INPUT DIRECTLY AS OUTPUT
// -----------------------------------------------------------------------------

TEST_CASE("Compiler and VM match reference interpreter when graph input is output") {
  Graph graph;

  const auto x = graph.input(Shape{3}, Dtype::F32);

  graph.set_outputs({x});

  reference::InputBindings reference_inputs;
  runtime::InputBindings runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, runtime::SlotId{0}, Shape{3},
             {10.0F, 20.0F, 30.0F});

  reference::Interpreter interpreter;
  const auto expected = interpreter.run(graph, x, reference_inputs);

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.instructions().empty());

  runtime::VM vm;
  const auto actual_outputs = vm.run(program, runtime_inputs);

  REQUIRE(actual_outputs.size() == 1);

  require_same_tensor(expected, actual_outputs[0]);
}

// -----------------------------------------------------------------------------
// DUPLICATE OUTPUT
// -----------------------------------------------------------------------------

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
  runtime::InputBindings runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, runtime::SlotId{0}, Shape{2},
             {1.0F, 2.0F});

  bind_input(reference_inputs, runtime_inputs, y, runtime::SlotId{1}, Shape{2},
             {3.0F, 4.0F});

  reference::Interpreter interpreter;
  const auto expected = interpreter.run(graph, out, reference_inputs);

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.instructions().size() == 1);
  REQUIRE(program.outputs().size() == 2);
  REQUIRE(program.outputs()[0] == program.outputs()[1]);

  runtime::VM vm;
  const auto actual_outputs = vm.run(program, runtime_inputs);

  REQUIRE(actual_outputs.size() == 2);

  require_same_tensor(expected, actual_outputs[0]);
  require_same_tensor(expected, actual_outputs[1]);
}

// -----------------------------------------------------------------------------
// INPUT AND COMPUTED VALUE BOTH AS OUTPUTS
// -----------------------------------------------------------------------------

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
  runtime::InputBindings runtime_inputs;

  bind_input(reference_inputs, runtime_inputs, x, runtime::SlotId{0}, Shape{2},
             {1.0F, 2.0F});

  bind_input(reference_inputs, runtime_inputs, y, runtime::SlotId{1}, Shape{2},
             {10.0F, 20.0F});

  reference::Interpreter interpreter;

  const auto expected_x =
      interpreter.run(graph, x, reference_inputs);

  const auto expected_added =
      interpreter.run(graph, added, reference_inputs);

  Compiler compiler;
  const auto program = compiler.compile(graph);

  runtime::VM vm;
  const auto actual_outputs = vm.run(program, runtime_inputs);

  REQUIRE(actual_outputs.size() == 2);

  require_same_tensor(expected_x, actual_outputs[0]);
  require_same_tensor(expected_added, actual_outputs[1]);
}