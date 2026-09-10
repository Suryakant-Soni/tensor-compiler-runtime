#include <catch2/catch_test_macros.hpp>

#include "tcr/compiler/lowering.hpp"
#include "tcr/ir/dtype.hpp"
#include "tcr/ir/graph.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/lir/index.hpp"
#include "tcr/lir/instruction.hpp"
#include "tcr/lir/program.hpp"

#include <cstddef>
#include <memory>
#include <variant>
#include <vector>

using tcr::compiler::Lowering;

using tcr::ir::Dtype;
using tcr::ir::Graph;
using tcr::ir::Shape;

namespace lir = tcr::lir;

namespace {

// -----------------------------------------------------------------------------
// LIR inspection helpers
// -----------------------------------------------------------------------------

const lir::For &require_for(const lir::Instruction &instruction) {
  const auto *loop =
      std::get_if<std::unique_ptr<lir::For>>(
          &instruction.payload);

  REQUIRE(loop != nullptr);
  REQUIRE(*loop != nullptr);

  return **loop;
}

template <typename T>
const T &require_instruction(
    const lir::Instruction &instruction) {

  const auto *op =
      std::get_if<T>(&instruction.payload);

  REQUIRE(op != nullptr);

  return *op;
}

std::size_t require_constant_index(
    const lir::IndexExpr &index) {

  const auto *constant =
      std::get_if<lir::ConstantIndex>(&index);

  REQUIRE(constant != nullptr);

  return constant->value;
}

lir::IndexId require_index_register(
    const lir::IndexExpr &index) {

  const auto *reg =
      std::get_if<lir::IndexId>(&index);

  REQUIRE(reg != nullptr);

  return *reg;
}

} // namespace

// -----------------------------------------------------------------------------
// Empty graph
// -----------------------------------------------------------------------------

TEST_CASE("Lowering empty graph produces empty LIR program") {
  Graph graph;

  Lowering lowering;
  const auto program =
      lowering.lower(graph);

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

  const auto x =
      graph.input(
          Shape{3},
          Dtype::F32);

  graph.set_outputs({x});

  Lowering lowering;
  const auto program =
      lowering.lower(graph);

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

  const auto x =
      graph.input(
          Shape{},
          Dtype::F32);

  graph.set_outputs({x});

  Lowering lowering;
  const auto program =
      lowering.lower(graph);

  REQUIRE(program.buffer_count() == 1);

  REQUIRE(program.buffers()[0].shape == Shape{});
  REQUIRE(program.buffers()[0].dtype == Dtype::F32);

  REQUIRE(program.inputs().size() == 1);
  REQUIRE(program.outputs().size() == 1);
}

TEST_CASE("Lowering multiple input outputs preserves output ordering") {
  Graph graph;

  const auto x =
      graph.input(
          Shape{2},
          Dtype::F32);

  const auto y =
      graph.input(
          Shape{3},
          Dtype::F32);

  graph.set_outputs({
      y,
      x,
  });

  Lowering lowering;
  const auto program =
      lowering.lower(graph);

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

  const auto a =
      graph.input(
          Shape{4},
          Dtype::F32);

  const auto b =
      graph.input(
          Shape{4},
          Dtype::F32);

  const auto out =
      graph.add(a, b);

  graph.set_outputs({out});

  Lowering lowering;
  const auto program =
      lowering.lower(graph);

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

  const auto a =
      graph.input(
          Shape{4},
          Dtype::F32);

  const auto b =
      graph.input(
          Shape{4},
          Dtype::F32);

  const auto out =
      graph.add(a, b);

  graph.set_outputs({out});

  Lowering lowering;
  const auto program =
      lowering.lower(graph);

  REQUIRE(program.scalar_count() == 3);
  REQUIRE(program.index_count() == 1);
}

TEST_CASE("Lowering Add creates one elementwise loop") {
  Graph graph;

  const auto a =
      graph.input(
          Shape{4},
          Dtype::F32);

  const auto b =
      graph.input(
          Shape{4},
          Dtype::F32);

  const auto out =
      graph.add(a, b);

  graph.set_outputs({out});

  Lowering lowering;
  const auto program =
      lowering.lower(graph);

  REQUIRE(program.instructions().size() == 1);

  const auto &loop =
      require_for(program.instructions()[0]);

  REQUIRE(loop.index.value == 0);

  REQUIRE(
      require_constant_index(loop.begin) ==
      0);

  REQUIRE(
      require_constant_index(loop.end) ==
      4);

  REQUIRE(loop.body.size() == 4);

  require_instruction<lir::Load>(
      loop.body[0]);

  require_instruction<lir::Load>(
      loop.body[1]);

  require_instruction<lir::Add>(
      loop.body[2]);

  require_instruction<lir::Store>(
      loop.body[3]);
}

TEST_CASE("Lowering Add loop range uses total number of elements") {
  Graph graph;

  const auto a =
      graph.input(
          Shape{2, 3},
          Dtype::F32);

  const auto b =
      graph.input(
          Shape{2, 3},
          Dtype::F32);

  const auto out =
      graph.add(a, b);

  graph.set_outputs({out});

  Lowering lowering;
  const auto program =
      lowering.lower(graph);

  const auto &loop =
      require_for(program.instructions()[0]);

  REQUIRE(
      require_constant_index(loop.begin) ==
      0);

  REQUIRE(
      require_constant_index(loop.end) ==
      6);
}

// -----------------------------------------------------------------------------
// Mul
// -----------------------------------------------------------------------------

TEST_CASE("Lowering Mul creates elementwise multiplication loop") {
  Graph graph;

  const auto a =
      graph.input(
          Shape{5},
          Dtype::F32);

  const auto b =
      graph.input(
          Shape{5},
          Dtype::F32);

  const auto out =
      graph.mul(a, b);

  graph.set_outputs({out});

  Lowering lowering;
  const auto program =
      lowering.lower(graph);

  REQUIRE(program.buffer_count() == 3);
  REQUIRE(program.inputs().size() == 2);
  REQUIRE(program.outputs().size() == 1);

  REQUIRE(program.scalar_count() == 3);
  REQUIRE(program.index_count() == 1);

  REQUIRE(program.instructions().size() == 1);

  const auto &loop =
      require_for(program.instructions()[0]);

  REQUIRE(
      require_constant_index(loop.begin) ==
      0);

  REQUIRE(
      require_constant_index(loop.end) ==
      5);

  REQUIRE(loop.body.size() == 4);

  require_instruction<lir::Load>(
      loop.body[0]);

  require_instruction<lir::Load>(
      loop.body[1]);

  require_instruction<lir::Mul>(
      loop.body[2]);

  require_instruction<lir::Store>(
      loop.body[3]);
}

TEST_CASE("Lowering Add and Mul use different arithmetic LIR instructions") {
  SECTION("Add") {
    Graph graph;

    const auto a =
        graph.input(
            Shape{2},
            Dtype::F32);

    const auto b =
        graph.input(
            Shape{2},
            Dtype::F32);

    const auto out =
        graph.add(a, b);

    graph.set_outputs({out});

    Lowering lowering;
    const auto program =
        lowering.lower(graph);

    const auto &loop =
        require_for(program.instructions()[0]);

    REQUIRE(
        std::holds_alternative<lir::Add>(
            loop.body[2].payload));

    REQUIRE_FALSE(
        std::holds_alternative<lir::Mul>(
            loop.body[2].payload));
  }

  SECTION("Mul") {
    Graph graph;

    const auto a =
        graph.input(
            Shape{2},
            Dtype::F32);

    const auto b =
        graph.input(
            Shape{2},
            Dtype::F32);

    const auto out =
        graph.mul(a, b);

    graph.set_outputs({out});

    Lowering lowering;
    const auto program =
        lowering.lower(graph);

    const auto &loop =
        require_for(program.instructions()[0]);

    REQUIRE(
        std::holds_alternative<lir::Mul>(
            loop.body[2].payload));

    REQUIRE_FALSE(
        std::holds_alternative<lir::Add>(
            loop.body[2].payload));
  }
}

// -----------------------------------------------------------------------------
// Sum
// -----------------------------------------------------------------------------

TEST_CASE("Lowering Sum creates input and scalar output buffer") {
  Graph graph;

  const auto x =
      graph.input(
          Shape{4},
          Dtype::F32);

  const auto out =
      graph.sum(x);

  graph.set_outputs({out});

  Lowering lowering;
  const auto program =
      lowering.lower(graph);

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

  const auto x =
      graph.input(
          Shape{4},
          Dtype::F32);

  const auto out =
      graph.sum(x);

  graph.set_outputs({out});

  Lowering lowering;
  const auto program =
      lowering.lower(graph);

  REQUIRE(program.scalar_count() == 2);
  REQUIRE(program.index_count() == 1);
}

TEST_CASE("Lowering Sum creates initialize loop and final store") {
  Graph graph;

  const auto x =
      graph.input(
          Shape{4},
          Dtype::F32);

  const auto out =
      graph.sum(x);

  graph.set_outputs({out});

  Lowering lowering;
  const auto program =
      lowering.lower(graph);

  REQUIRE(program.instructions().size() == 3);

  require_instruction<lir::Constant>(
      program.instructions()[0]);

  const auto &loop =
      require_for(program.instructions()[1]);

  require_instruction<lir::Store>(
      program.instructions()[2]);

  REQUIRE(
      require_constant_index(loop.begin) ==
      0);

  REQUIRE(
      require_constant_index(loop.end) ==
      4);

  REQUIRE(loop.body.size() == 2);

  require_instruction<lir::Load>(
      loop.body[0]);

  require_instruction<lir::Add>(
      loop.body[1]);
}

TEST_CASE("Lowering Sum reduction loop uses all matrix elements") {
  Graph graph;

  const auto x =
      graph.input(
          Shape{2, 3},
          Dtype::F32);

  const auto out =
      graph.sum(x);

  graph.set_outputs({out});

  Lowering lowering;
  const auto program =
      lowering.lower(graph);

  const auto &loop =
      require_for(program.instructions()[1]);

  REQUIRE(
      require_constant_index(loop.end) ==
      6);
}

// -----------------------------------------------------------------------------
// MatMul
// -----------------------------------------------------------------------------

TEST_CASE("Lowering MatMul creates two input buffers and output buffer") {
  Graph graph;

  const auto a =
      graph.input(
          Shape{2, 3},
          Dtype::F32);

  const auto b =
      graph.input(
          Shape{3, 4},
          Dtype::F32);

  const auto out =
      graph.matmul(a, b);

  graph.set_outputs({out});

  Lowering lowering;
  const auto program =
      lowering.lower(graph);

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

  const auto a =
      graph.input(
          Shape{2, 3},
          Dtype::F32);

  const auto b =
      graph.input(
          Shape{3, 4},
          Dtype::F32);

  const auto out =
      graph.matmul(a, b);

  graph.set_outputs({out});

  Lowering lowering;
  const auto program =
      lowering.lower(graph);

  // i, j, k
  // a_row, b_row, a_off, b_off, c_row, c_off
  REQUIRE(program.index_count() == 9);

  // a_scalar, b_scalar, product, acc
  REQUIRE(program.scalar_count() == 4);
}

TEST_CASE("Lowering MatMul creates three nested loops") {
  Graph graph;

  const auto a =
      graph.input(
          Shape{2, 3},
          Dtype::F32);

  const auto b =
      graph.input(
          Shape{3, 4},
          Dtype::F32);

  const auto out =
      graph.matmul(a, b);

  graph.set_outputs({out});

  Lowering lowering;
  const auto program =
      lowering.lower(graph);

  REQUIRE(program.instructions().size() == 1);

  const auto &i_loop =
      require_for(program.instructions()[0]);

  REQUIRE(
      require_constant_index(i_loop.begin) ==
      0);

  // M = 2
  REQUIRE(
      require_constant_index(i_loop.end) ==
      2);

  REQUIRE(i_loop.body.size() == 3);

  require_instruction<lir::IndexMul>(
      i_loop.body[0]);

  require_instruction<lir::IndexMul>(
      i_loop.body[1]);

  const auto &j_loop =
      require_for(i_loop.body[2]);

  REQUIRE(
      require_constant_index(j_loop.begin) ==
      0);

  // N = 4
  REQUIRE(
      require_constant_index(j_loop.end) ==
      4);

  REQUIRE(j_loop.body.size() == 4);

  require_instruction<lir::Constant>(
      j_loop.body[0]);

  const auto &k_loop =
      require_for(j_loop.body[1]);

  // K = 3
  REQUIRE(
      require_constant_index(k_loop.begin) ==
      0);

  REQUIRE(
      require_constant_index(k_loop.end) ==
      3);

  require_instruction<lir::IndexAdd>(
      j_loop.body[2]);

  require_instruction<lir::Store>(
      j_loop.body[3]);
}

TEST_CASE("Lowering MatMul inner K loop contains expected operations") {
  Graph graph;

  const auto a =
      graph.input(
          Shape{2, 3},
          Dtype::F32);

  const auto b =
      graph.input(
          Shape{3, 4},
          Dtype::F32);

  const auto out =
      graph.matmul(a, b);

  graph.set_outputs({out});

  Lowering lowering;
  const auto program =
      lowering.lower(graph);

  const auto &i_loop =
      require_for(program.instructions()[0]);

  const auto &j_loop =
      require_for(i_loop.body[2]);

  const auto &k_loop =
      require_for(j_loop.body[1]);

  REQUIRE(k_loop.body.size() == 7);

  require_instruction<lir::IndexAdd>(
      k_loop.body[0]);

  require_instruction<lir::IndexMul>(
      k_loop.body[1]);

  require_instruction<lir::IndexAdd>(
      k_loop.body[2]);

  require_instruction<lir::Load>(
      k_loop.body[3]);

  require_instruction<lir::Load>(
      k_loop.body[4]);

  require_instruction<lir::Mul>(
      k_loop.body[5]);

  require_instruction<lir::Add>(
      k_loop.body[6]);
}

TEST_CASE("Lowering rectangular MatMul uses M K and N as loop bounds") {
  Graph graph;

  const auto a =
      graph.input(
          Shape{5, 7},
          Dtype::F32);

  const auto b =
      graph.input(
          Shape{7, 11},
          Dtype::F32);

  const auto out =
      graph.matmul(a, b);

  graph.set_outputs({out});

  Lowering lowering;
  const auto program =
      lowering.lower(graph);

  const auto &i_loop =
      require_for(program.instructions()[0]);

  const auto &j_loop =
      require_for(i_loop.body[2]);

  const auto &k_loop =
      require_for(j_loop.body[1]);

  REQUIRE(
      require_constant_index(i_loop.end) ==
      5);

  REQUIRE(
      require_constant_index(j_loop.end) ==
      11);

  REQUIRE(
      require_constant_index(k_loop.end) ==
      7);

  REQUIRE(program.buffers()[2].shape ==
          Shape{5, 11});
}

// -----------------------------------------------------------------------------
// Dependency lowering
// -----------------------------------------------------------------------------

TEST_CASE("Lowering recursively lowers producer before consumer") {
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

  const auto product =
      graph.mul(x, y);

  const auto out =
      graph.add(product, z);

  graph.set_outputs({out});

  Lowering lowering;
  const auto program =
      lowering.lower(graph);

  // x, y, product, z, out
  REQUIRE(program.buffer_count() == 5);

  REQUIRE(program.inputs().size() == 3);
  REQUIRE(program.outputs().size() == 1);

  REQUIRE(program.instructions().size() == 2);

  const auto &mul_loop =
      require_for(program.instructions()[0]);

  const auto &add_loop =
      require_for(program.instructions()[1]);

  REQUIRE(
      std::holds_alternative<lir::Mul>(
          mul_loop.body[2].payload));

  REQUIRE(
      std::holds_alternative<lir::Add>(
          add_loop.body[2].payload));
}

// -----------------------------------------------------------------------------
// Memoization / shared subexpressions
// -----------------------------------------------------------------------------

TEST_CASE("Lowering lowers shared producer only once") {
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
      graph.add(shared, shared);

  graph.set_outputs({out});

  Lowering lowering;
  const auto program =
      lowering.lower(graph);

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

  Lowering lowering;
  const auto program =
      lowering.lower(graph);

  REQUIRE(program.buffer_count() == 3);

  REQUIRE(program.instructions().size() == 1);

  REQUIRE(program.outputs().size() == 2);

  REQUIRE(
      program.outputs()[0].value ==
      program.outputs()[1].value);

  REQUIRE(program.outputs()[0].value == 2);
}

// -----------------------------------------------------------------------------
// Multiple outputs
// -----------------------------------------------------------------------------

TEST_CASE("Lowering supports multiple computed graph outputs") {
  Graph graph;

  const auto x =
      graph.input(
          Shape{3},
          Dtype::F32);

  const auto y =
      graph.input(
          Shape{3},
          Dtype::F32);

  const auto add_out =
      graph.add(x, y);

  const auto mul_out =
      graph.mul(x, y);

  graph.set_outputs({
      add_out,
      mul_out,
  });

  Lowering lowering;
  const auto program =
      lowering.lower(graph);

  // x, y, add_out, mul_out
  REQUIRE(program.buffer_count() == 4);

  REQUIRE(program.inputs().size() == 2);
  REQUIRE(program.outputs().size() == 2);

  REQUIRE(program.instructions().size() == 2);

  REQUIRE(program.outputs()[0].value == 2);
  REQUIRE(program.outputs()[1].value == 3);

  const auto &add_loop =
      require_for(program.instructions()[0]);

  const auto &mul_loop =
      require_for(program.instructions()[1]);

  REQUIRE(
      std::holds_alternative<lir::Add>(
          add_loop.body[2].payload));

  REQUIRE(
      std::holds_alternative<lir::Mul>(
          mul_loop.body[2].payload));
}

TEST_CASE("Lowering supports input and computed value as outputs") {
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

  graph.set_outputs({
      x,
      sum,
  });

  Lowering lowering;
  const auto program =
      lowering.lower(graph);

  REQUIRE(program.outputs().size() == 2);

  // x is lowered first because it appears first in outputs.
  REQUIRE(program.outputs()[0].value == 0);

  // Lowering sum then discovers y and finally creates sum.
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

  const auto x =
      graph.input(
          Shape{3},
          Dtype::F32);

  const auto y =
      graph.input(
          Shape{3},
          Dtype::F32);

  const auto live =
      graph.add(x, y);

  const auto dead =
      graph.mul(x, y);

  (void)dead;

  graph.set_outputs({live});

  Lowering lowering;
  const auto program =
      lowering.lower(graph);

  REQUIRE(program.buffer_count() == 3);
  REQUIRE(program.inputs().size() == 2);

  // Only the live Add loop is lowered.
  REQUIRE(program.instructions().size() == 1);

  const auto &loop =
      require_for(program.instructions()[0]);

  REQUIRE(
      std::holds_alternative<lir::Add>(
          loop.body[2].payload));
}

TEST_CASE("Lowering excludes unreachable dependency subtree") {
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

  const auto live =
      graph.add(x, y);

  const auto dead_mul =
      graph.mul(y, z);

  const auto dead_add =
      graph.add(dead_mul, z);

  (void)dead_add;

  graph.set_outputs({live});

  Lowering lowering;
  const auto program =
      lowering.lower(graph);

  // Only x, y and live are reachable.
  REQUIRE(program.buffer_count() == 3);

  // Important Phase-3 lowering behavior:
  // unreachable z is not lowered as an LIR input either.
  REQUIRE(program.inputs().size() == 2);

  REQUIRE(program.instructions().size() == 1);

  REQUIRE(program.outputs().size() == 1);
}

// -----------------------------------------------------------------------------
// Lowering object reuse
// -----------------------------------------------------------------------------

TEST_CASE("Lowering clears memoization state between lower calls") {
  Lowering lowering;

  Graph first_graph;

  const auto first_x =
      first_graph.input(
          Shape{2},
          Dtype::F32);

  const auto first_y =
      first_graph.input(
          Shape{2},
          Dtype::F32);

  const auto first_out =
      first_graph.add(
          first_x,
          first_y);

  first_graph.set_outputs({
      first_out,
  });

  const auto first_program =
      lowering.lower(first_graph);

  REQUIRE(first_program.buffer_count() == 3);
  REQUIRE(first_program.inputs().size() == 2);

  Graph second_graph;

  const auto second_x =
      second_graph.input(
          Shape{5},
          Dtype::F32);

  second_graph.set_outputs({
      second_x,
  });

  const auto second_program =
      lowering.lower(second_graph);

  REQUIRE(second_program.buffer_count() == 1);
  REQUIRE(second_program.inputs().size() == 1);
  REQUIRE(second_program.outputs().size() == 1);

  REQUIRE(
      second_program.buffers()[0].shape ==
      Shape{5});

  REQUIRE(
      second_program.outputs()[0].value ==
      0);
}