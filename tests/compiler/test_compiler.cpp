#include <catch2/catch_test_macros.hpp>

#include "tcr/compiler/compiler.hpp"
#include "tcr/ir/dtype.hpp"
#include "tcr/ir/graph.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/runtime/instruction.hpp"
#include "tcr/runtime/opcode.hpp"
#include "tcr/runtime/program.hpp"
#include "tcr/runtime/slot_id.hpp"

#include <cstddef>
#include <vector>

using tcr::compiler::Compiler;

using tcr::ir::Dtype;
using tcr::ir::Graph;
using tcr::ir::Shape;

using tcr::runtime::IndexOperand;
using tcr::runtime::Opcode;
using tcr::runtime::SlotId;

namespace {

void require_constant_operand(const IndexOperand &operand,
                              std::size_t expected) {

  REQUIRE(operand.kind == IndexOperand::Kind::Constant);

  REQUIRE(operand.constant == expected);
}

void require_register_operand(const IndexOperand &operand,
                              std::size_t expected_register) {

  REQUIRE(operand.kind == IndexOperand::Kind::Register);

  REQUIRE(operand.reg.value == expected_register);
}

} // namespace

// =============================================================================
// Empty / unreachable graph
// =============================================================================

TEST_CASE("Compiler compiles empty graph to empty runtime program") {
  Graph graph;

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.instructions().empty());
  REQUIRE(program.inputs().empty());
  REQUIRE(program.outputs().empty());

  REQUIRE(program.slots_count() == 0);
  REQUIRE(program.scalar_count() == 0);
  REQUIRE(program.index_count() == 0);
}

TEST_CASE("Compiler does not compile graph inputs when graph has no outputs") {
  Graph graph;

  graph.input(Shape{2}, Dtype::F32);

  graph.input(Shape{3}, Dtype::F32);

  Compiler compiler;
  const auto program = compiler.compile(graph);

  // Scheduling starts from graph outputs.
  // Nothing is reachable when there are no outputs.
  REQUIRE(program.inputs().empty());
  REQUIRE(program.outputs().empty());
  REQUIRE(program.instructions().empty());

  REQUIRE(program.slots_count() == 0);
  REQUIRE(program.scalar_count() == 0);
  REQUIRE(program.index_count() == 0);
}

// =============================================================================
// Input directly as output
// =============================================================================

TEST_CASE("Compiler compiles graph input directly as runtime output") {
  Graph graph;

  const auto x = graph.input(Shape{3}, Dtype::F32);

  graph.set_outputs({x});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.instructions().empty());

  REQUIRE(program.inputs().size() == 1);
  REQUIRE(program.outputs().size() == 1);

  REQUIRE(program.inputs()[0].value_id == x);
  REQUIRE(program.inputs()[0].slot_id == SlotId{0});
  REQUIRE(program.get_slotid_from_valueid(x) == SlotId{0});
  REQUIRE(program.outputs()[0] == SlotId{0});

  REQUIRE(program.slots_count() == 1);
  REQUIRE(program.scalar_count() == 0);
  REQUIRE(program.index_count() == 0);

  const auto &info = program.tensor_meta_by_slot(SlotId{0});

  REQUIRE(info.id == SlotId{0});
  REQUIRE(info.shape == Shape{3});
  REQUIRE(info.dtype == Dtype::F32);
}

TEST_CASE("Compiler preserves scalar input metadata") {
  Graph graph;

  const auto x = graph.input(Shape{}, Dtype::F32);

  graph.set_outputs({x});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.slots_count() == 1);

  const auto &info = program.tensor_meta_by_slot(SlotId{0});

  REQUIRE(info.shape == Shape{});
  REQUIRE(info.dtype == Dtype::F32);
}

TEST_CASE("Compiler preserves matrix input metadata") {
  Graph graph;

  const auto x = graph.input(Shape{2, 5}, Dtype::F32);

  graph.set_outputs({x});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.tensor_meta_by_slot(SlotId{0}).shape == Shape{2, 5});
}

// =============================================================================
// Multiple direct input outputs
// =============================================================================

TEST_CASE("Compiler preserves requested output ordering for direct inputs") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);

  const auto y = graph.input(Shape{4}, Dtype::F32);

  graph.set_outputs({
      y,
      x,
  });

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.inputs().size() == 2);
  REQUIRE(program.outputs().size() == 2);
  REQUIRE(program.slots_count() == 2);

  // y is encountered first during reachable traversal.
  REQUIRE(program.inputs()[0].value_id == y);
  REQUIRE(program.inputs()[0].slot_id == SlotId{0});
  REQUIRE(program.inputs()[1].value_id == x);
  REQUIRE(program.inputs()[1].slot_id == SlotId{1});

  REQUIRE(program.get_slotid_from_valueid(y) == SlotId{0});
  REQUIRE(program.get_slotid_from_valueid(x) == SlotId{1});

  REQUIRE(program.outputs()[0] == SlotId{0});
  REQUIRE(program.outputs()[1] == SlotId{1});

  REQUIRE(program.tensor_meta_by_slot(SlotId{0}).shape == Shape{4});

  REQUIRE(program.tensor_meta_by_slot(SlotId{1}).shape == Shape{2});
}

// =============================================================================
// ADD
// =============================================================================

TEST_CASE("Compiler compiles Add into flat elementwise runtime loop") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);

  const auto y = graph.input(Shape{4}, Dtype::F32);

  const auto out = graph.add(x, y);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.inputs().size() == 2);
  REQUIRE(program.outputs().size() == 1);

  REQUIRE(program.slots_count() == 3);
  REQUIRE(program.scalar_count() == 3);
  REQUIRE(program.index_count() == 1);

  REQUIRE(program.inputs()[0].value_id == x);
  REQUIRE(program.inputs()[0].slot_id == SlotId{0});
  REQUIRE(program.inputs()[1].value_id == y);
  REQUIRE(program.inputs()[1].slot_id == SlotId{1});

  REQUIRE(program.get_slotid_from_valueid(x) == SlotId{0});
  REQUIRE(program.get_slotid_from_valueid(y) == SlotId{1});

  REQUIRE(program.outputs()[0] == SlotId{2});

  /*
   * 0 LoopBegin
   * 1 Load A
   * 2 Load B
   * 3 Add
   * 4 Store C
   * 5 LoopEnd
   */
  REQUIRE(program.instructions().size() == 6);

  REQUIRE(program.instructions()[0].opcode == Opcode::LoopBegin);

  REQUIRE(program.instructions()[1].opcode == Opcode::Load);

  REQUIRE(program.instructions()[2].opcode == Opcode::Load);

  REQUIRE(program.instructions()[3].opcode == Opcode::Add);

  REQUIRE(program.instructions()[4].opcode == Opcode::Store);

  REQUIRE(program.instructions()[5].opcode == Opcode::LoopEnd);
}

TEST_CASE("Compiler Add loop uses correct bounds and jumps") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);

  const auto y = graph.input(Shape{4}, Dtype::F32);

  const auto out = graph.add(x, y);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  const auto &begin = program.instructions()[0];

  const auto &end = program.instructions()[5];

  REQUIRE(begin.loop_index.value == 0);

  require_constant_operand(begin.loop_begin, 0);

  require_constant_operand(begin.loop_end, 4);

  // Skip entire loop.
  REQUIRE(begin.jump == 6);

  REQUIRE(end.loop_index.value == 0);

  require_constant_operand(end.loop_end, 4);

  // Repeat from first instruction in body.
  REQUIRE(end.jump == 1);
}

TEST_CASE("Compiler Add connects runtime scalar registers correctly") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);

  const auto y = graph.input(Shape{2}, Dtype::F32);

  const auto out = graph.add(x, y);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  const auto &load_a = program.instructions()[1];

  const auto &load_b = program.instructions()[2];

  const auto &add = program.instructions()[3];

  const auto &store = program.instructions()[4];

  REQUIRE(load_a.scalar_dst.value == 0);
  REQUIRE(load_a.buffer == SlotId{0});
  require_register_operand(load_a.offset, 0);

  REQUIRE(load_b.scalar_dst.value == 1);
  REQUIRE(load_b.buffer == SlotId{1});
  require_register_operand(load_b.offset, 0);

  REQUIRE(add.scalar_lhs.value == 0);
  REQUIRE(add.scalar_rhs.value == 1);
  REQUIRE(add.scalar_dst.value == 2);

  REQUIRE(store.scalar_lhs.value == 2);
  REQUIRE(store.buffer == SlotId{2});
  require_register_operand(store.offset, 0);
}

TEST_CASE("Compiler Add preserves output tensor metadata") {
  Graph graph;

  const auto x = graph.input(Shape{2, 3}, Dtype::F32);

  const auto y = graph.input(Shape{2, 3}, Dtype::F32);

  const auto out = graph.add(x, y);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.tensor_meta_by_slot(SlotId{2}).shape == Shape{2, 3});

  REQUIRE(program.tensor_meta_by_slot(SlotId{2}).dtype == Dtype::F32);

  const auto &loop = program.instructions()[0];

  require_constant_operand(loop.loop_end, 6);
}

// =============================================================================
// MUL
// =============================================================================

TEST_CASE("Compiler compiles Mul into flat elementwise runtime loop") {
  Graph graph;

  const auto x = graph.input(Shape{3}, Dtype::F32);

  const auto y = graph.input(Shape{3}, Dtype::F32);

  const auto out = graph.mul(x, y);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.slots_count() == 3);
  REQUIRE(program.scalar_count() == 3);
  REQUIRE(program.index_count() == 1);

  REQUIRE(program.instructions().size() == 6);

  REQUIRE(program.instructions()[0].opcode == Opcode::LoopBegin);

  REQUIRE(program.instructions()[1].opcode == Opcode::Load);

  REQUIRE(program.instructions()[2].opcode == Opcode::Load);

  REQUIRE(program.instructions()[3].opcode == Opcode::Mul);

  REQUIRE(program.instructions()[4].opcode == Opcode::Store);

  REQUIRE(program.instructions()[5].opcode == Opcode::LoopEnd);

  REQUIRE(program.outputs()[0] == SlotId{2});
}

TEST_CASE("Compiler distinguishes Add and Mul runtime arithmetic") {
  SECTION("Add") {
    Graph graph;

    const auto x = graph.input(Shape{2}, Dtype::F32);

    const auto y = graph.input(Shape{2}, Dtype::F32);

    const auto out = graph.add(x, y);

    graph.set_outputs({out});

    Compiler compiler;
    const auto program = compiler.compile(graph);

    REQUIRE(program.instructions()[3].opcode == Opcode::Add);
  }

  SECTION("Mul") {
    Graph graph;

    const auto x = graph.input(Shape{2}, Dtype::F32);

    const auto y = graph.input(Shape{2}, Dtype::F32);

    const auto out = graph.mul(x, y);

    graph.set_outputs({out});

    Compiler compiler;
    const auto program = compiler.compile(graph);

    REQUIRE(program.instructions()[3].opcode == Opcode::Mul);
  }
}

// =============================================================================
// SUM
// =============================================================================

TEST_CASE("Compiler compiles Sum into initialization loop and store") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);

  const auto out = graph.sum(x);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.inputs().size() == 1);
  REQUIRE(program.outputs().size() == 1);

  REQUIRE(program.slots_count() == 2);
  REQUIRE(program.scalar_count() == 2);
  REQUIRE(program.index_count() == 1);

  /*
   * 0 Constant acc = 0
   * 1 LoopBegin
   * 2 Load
   * 3 Add
   * 4 LoopEnd
   * 5 Store
   */
  REQUIRE(program.instructions().size() == 6);

  REQUIRE(program.instructions()[0].opcode == Opcode::Constant);

  REQUIRE(program.instructions()[1].opcode == Opcode::LoopBegin);

  REQUIRE(program.instructions()[2].opcode == Opcode::Load);

  REQUIRE(program.instructions()[3].opcode == Opcode::Add);

  REQUIRE(program.instructions()[4].opcode == Opcode::LoopEnd);

  REQUIRE(program.instructions()[5].opcode == Opcode::Store);
}

TEST_CASE("Compiler Sum initializes accumulator to zero") {
  Graph graph;

  const auto x = graph.input(Shape{3}, Dtype::F32);

  const auto out = graph.sum(x);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  const auto &init = program.instructions()[0];

  REQUIRE(init.opcode == Opcode::Constant);
  REQUIRE(init.scalar_dst.value == 1);
  REQUIRE(init.constant == 0.0F);
}

TEST_CASE("Compiler Sum uses complete input element count as loop bound") {
  Graph graph;

  const auto x = graph.input(Shape{2, 3}, Dtype::F32);

  const auto out = graph.sum(x);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  const auto &begin = program.instructions()[1];

  require_constant_operand(begin.loop_begin, 0);

  require_constant_operand(begin.loop_end, 6);

  REQUIRE(begin.jump == 5);

  const auto &end = program.instructions()[4];

  REQUIRE(end.jump == 2);
}

TEST_CASE("Compiler Sum produces scalar output metadata") {
  Graph graph;

  const auto x = graph.input(Shape{2, 4}, Dtype::F32);

  const auto out = graph.sum(x);

  graph.set_outputs({out});

  Compiler compiler;

  const auto program = compiler.compile(graph);

  REQUIRE(program.outputs()[0] == SlotId{1});

  const auto &output_info = program.tensor_meta_by_slot(SlotId{1});

  REQUIRE(output_info.shape == Shape{});
  REQUIRE(output_info.dtype == Dtype::F32);

  const auto &store = program.instructions()[5];

  REQUIRE(store.buffer == SlotId{1});
  require_constant_operand(store.offset, 0);
}

// =============================================================================
// PHASE 5: GROUPED ELEMENTWISE -> SUM
// =============================================================================

TEST_CASE("Compiler groups Add before Sum without materializing Add output") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);
  const auto y = graph.input(Shape{4}, Dtype::F32);

  const auto added = graph.add(x, y);
  const auto out = graph.sum(added);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  // x, y and the scalar reduction output are materialized.
  // There is no full Shape{4} buffer for `added`.
  REQUIRE(program.slots_count() == 3);
  REQUIRE(program.inputs().size() == 2);
  REQUIRE(program.outputs().size() == 1);

  const auto x_slot = program.get_slotid_from_valueid(x);
  const auto y_slot = program.get_slotid_from_valueid(y);
  const auto out_slot = program.outputs()[0];

  REQUIRE(program.tensor_meta_by_slot(x_slot).shape == Shape{4});
  REQUIRE(program.tensor_meta_by_slot(y_slot).shape == Shape{4});
  REQUIRE(program.tensor_meta_by_slot(out_slot).shape == Shape{});

  REQUIRE(program.scalar_count() == 4);
  REQUIRE(program.index_count() == 1);

  REQUIRE(program.instructions().size() == 8);

  REQUIRE(program.instructions()[0].opcode == Opcode::Constant);
  REQUIRE(program.instructions()[1].opcode == Opcode::LoopBegin);
  REQUIRE(program.instructions()[2].opcode == Opcode::Load);
  REQUIRE(program.instructions()[3].opcode == Opcode::Load);
  REQUIRE(program.instructions()[4].opcode == Opcode::Add);
  REQUIRE(program.instructions()[5].opcode == Opcode::Add);
  REQUIRE(program.instructions()[6].opcode == Opcode::LoopEnd);
  REQUIRE(program.instructions()[7].opcode == Opcode::Store);

  REQUIRE(program.instructions()[2].buffer == x_slot);
  REQUIRE(program.instructions()[3].buffer == y_slot);
  REQUIRE(program.instructions()[7].buffer == out_slot);
}

TEST_CASE("Compiler groups Mul before Sum without materializing Mul output") {
  Graph graph;

  const auto x = graph.input(Shape{5}, Dtype::F32);
  const auto y = graph.input(Shape{5}, Dtype::F32);

  const auto product = graph.mul(x, y);
  const auto out = graph.sum(product);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.slots_count() == 3);
  REQUIRE(program.inputs().size() == 2);
  REQUIRE(program.outputs().size() == 1);

  const auto x_slot = program.get_slotid_from_valueid(x);
  const auto y_slot = program.get_slotid_from_valueid(y);
  const auto out_slot = program.outputs()[0];

  REQUIRE(program.tensor_meta_by_slot(x_slot).shape == Shape{5});
  REQUIRE(program.tensor_meta_by_slot(y_slot).shape == Shape{5});
  REQUIRE(program.tensor_meta_by_slot(out_slot).shape == Shape{});

  REQUIRE(program.instructions().size() == 8);

  REQUIRE(program.instructions()[0].opcode == Opcode::Constant);
  REQUIRE(program.instructions()[1].opcode == Opcode::LoopBegin);
  REQUIRE(program.instructions()[2].opcode == Opcode::Load);
  REQUIRE(program.instructions()[3].opcode == Opcode::Load);
  REQUIRE(program.instructions()[4].opcode == Opcode::Mul);
  REQUIRE(program.instructions()[5].opcode == Opcode::Add);
  REQUIRE(program.instructions()[6].opcode == Opcode::LoopEnd);
  REQUIRE(program.instructions()[7].opcode == Opcode::Store);

  REQUIRE(program.instructions()[2].buffer == x_slot);
  REQUIRE(program.instructions()[3].buffer == y_slot);
  REQUIRE(program.instructions()[7].buffer == out_slot);
}

TEST_CASE("Compiler groups fused elementwise expression before Sum without "
          "intermediate tensor") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);
  const auto y = graph.input(Shape{4}, Dtype::F32);
  const auto z = graph.input(Shape{4}, Dtype::F32);

  const auto product = graph.mul(x, y);
  const auto mapped = graph.add(product, z);
  const auto out = graph.sum(mapped);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  // Phase 4 fuses Mul -> Add. Phase 5 groups the fused value with Sum.
  // Only x, y, z and the scalar reduction output are materialized.
  REQUIRE(program.slots_count() == 4);
  REQUIRE(program.inputs().size() == 3);
  REQUIRE(program.outputs().size() == 1);

  const auto x_slot = program.get_slotid_from_valueid(x);
  const auto y_slot = program.get_slotid_from_valueid(y);
  const auto z_slot = program.get_slotid_from_valueid(z);
  const auto out_slot = program.outputs()[0];

  REQUIRE(program.tensor_meta_by_slot(x_slot).shape == Shape{4});
  REQUIRE(program.tensor_meta_by_slot(y_slot).shape == Shape{4});
  REQUIRE(program.tensor_meta_by_slot(z_slot).shape == Shape{4});
  REQUIRE(program.tensor_meta_by_slot(out_slot).shape == Shape{});

  REQUIRE(program.instructions().size() == 10);

  REQUIRE(program.instructions()[0].opcode == Opcode::Constant);
  REQUIRE(program.instructions()[1].opcode == Opcode::LoopBegin);
  REQUIRE(program.instructions()[2].opcode == Opcode::Load);
  REQUIRE(program.instructions()[3].opcode == Opcode::Load);
  REQUIRE(program.instructions()[4].opcode == Opcode::Load);
  REQUIRE(program.instructions()[5].opcode == Opcode::Mul);
  REQUIRE(program.instructions()[6].opcode == Opcode::Add);
  REQUIRE(program.instructions()[7].opcode == Opcode::Add);
  REQUIRE(program.instructions()[8].opcode == Opcode::LoopEnd);
  REQUIRE(program.instructions()[9].opcode == Opcode::Store);

  REQUIRE(program.instructions()[2].buffer == x_slot);
  REQUIRE(program.instructions()[3].buffer == y_slot);
  REQUIRE(program.instructions()[4].buffer == z_slot);
  REQUIRE(program.instructions()[9].buffer == out_slot);
}

// =============================================================================
// MATMUL
// =============================================================================

TEST_CASE("Compiler compiles MatMul into nested flat runtime loops") {
  Graph graph;

  const auto a = graph.input(Shape{2, 3}, Dtype::F32);

  const auto b = graph.input(Shape{3, 4}, Dtype::F32);

  const auto out = graph.matmul(a, b);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.inputs().size() == 2);
  REQUIRE(program.outputs().size() == 1);

  REQUIRE(program.slots_count() == 3);

  // i,j,k + six arithmetic index registers.
  REQUIRE(program.index_count() == 9);

  // a, b, product, accumulator.
  REQUIRE(program.scalar_count() == 4);

  REQUIRE(program.instructions().size() == 18);

  REQUIRE(program.outputs()[0] == SlotId{2});

  REQUIRE(program.tensor_meta_by_slot(SlotId{2}).shape == Shape{2, 4});
}

TEST_CASE("Compiler MatMul emits expected flattened opcode sequence") {
  Graph graph;

  const auto a = graph.input(Shape{2, 3}, Dtype::F32);

  const auto b = graph.input(Shape{3, 4}, Dtype::F32);

  const auto out = graph.matmul(a, b);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  const std::vector<Opcode> expected{
      // i loop
      Opcode::LoopBegin,

      Opcode::IndexMul,
      Opcode::IndexMul,

      // j loop
      Opcode::LoopBegin,

      Opcode::Constant,

      // k loop
      Opcode::LoopBegin,

      Opcode::IndexAdd,
      Opcode::IndexMul,
      Opcode::IndexAdd,

      Opcode::Load,
      Opcode::Load,

      Opcode::Mul,
      Opcode::Add,

      // end k
      Opcode::LoopEnd,

      Opcode::IndexAdd,
      Opcode::Store,

      // end j
      Opcode::LoopEnd,

      // end i
      Opcode::LoopEnd,
  };

  REQUIRE(program.instructions().size() == expected.size());

  for (std::size_t i = 0; i < expected.size(); ++i) {

    REQUIRE(program.instructions()[i].opcode == expected[i]);
  }
}

TEST_CASE("Compiler MatMul creates correct M N K loop bounds") {
  Graph graph;

  const auto a = graph.input(Shape{5, 7}, Dtype::F32);

  const auto b = graph.input(Shape{7, 11}, Dtype::F32);

  const auto out = graph.matmul(a, b);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  // Outer i loop.
  const auto &i_begin = program.instructions()[0];

  REQUIRE(i_begin.loop_index.value == 0);
  require_constant_operand(i_begin.loop_begin, 0);
  require_constant_operand(i_begin.loop_end, 5);

  // j loop.
  const auto &j_begin = program.instructions()[3];

  REQUIRE(j_begin.loop_index.value == 1);
  require_constant_operand(j_begin.loop_begin, 0);
  require_constant_operand(j_begin.loop_end, 11);

  // k loop.
  const auto &k_begin = program.instructions()[5];

  REQUIRE(k_begin.loop_index.value == 2);
  require_constant_operand(k_begin.loop_begin, 0);
  require_constant_operand(k_begin.loop_end, 7);

  REQUIRE(program.tensor_meta_by_slot(SlotId{2}).shape == Shape{5, 11});
}

TEST_CASE("Compiler MatMul creates correct nested loop jumps") {
  Graph graph;

  const auto a = graph.input(Shape{2, 3}, Dtype::F32);

  const auto b = graph.input(Shape{3, 4}, Dtype::F32);

  const auto out = graph.matmul(a, b);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  /*
   *  0  i LoopBegin          jump = 18
   *  1  IndexMul
   *  2  IndexMul
   *  3  j LoopBegin          jump = 17
   *  4  Constant
   *  5  k LoopBegin          jump = 14
   *  6  IndexAdd
   *  7  IndexMul
   *  8  IndexAdd
   *  9  Load
   * 10  Load
   * 11  Mul
   * 12  Add
   * 13  k LoopEnd            jump = 6
   * 14  IndexAdd
   * 15  Store
   * 16  j LoopEnd            jump = 4
   * 17  i LoopEnd            jump = 1
   * 18  exit
   */

  REQUIRE(program.instructions()[0].jump == 18);

  REQUIRE(program.instructions()[3].jump == 17);

  REQUIRE(program.instructions()[5].jump == 14);

  REQUIRE(program.instructions()[13].jump == 6);

  REQUIRE(program.instructions()[16].jump == 4);

  REQUIRE(program.instructions()[17].jump == 1);
}

TEST_CASE("Compiler MatMul maps loads and output store to correct slots") {
  Graph graph;

  const auto a = graph.input(Shape{2, 3}, Dtype::F32);

  const auto b = graph.input(Shape{3, 4}, Dtype::F32);

  const auto out = graph.matmul(a, b);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  const auto &a_load = program.instructions()[9];

  const auto &b_load = program.instructions()[10];

  const auto &store = program.instructions()[15];

  REQUIRE(a_load.opcode == Opcode::Load);
  REQUIRE(a_load.buffer == SlotId{0});

  REQUIRE(b_load.opcode == Opcode::Load);
  REQUIRE(b_load.buffer == SlotId{1});

  REQUIRE(store.opcode == Opcode::Store);
  REQUIRE(store.buffer == SlotId{2});
}

// =============================================================================
// Dependency chain
// =============================================================================

TEST_CASE("Compiler fuses single-use elementwise dependency chain") {
  Graph graph;

  const auto x = graph.input(Shape{3}, Dtype::F32);
  const auto y = graph.input(Shape{3}, Dtype::F32);
  const auto z = graph.input(Shape{3}, Dtype::F32);

  const auto product = graph.mul(x, y);
  const auto out = graph.add(product, z);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  // Fusion removes the materialized product buffer.
  //
  // Current fused lowering allocates the fused output buffer before lowering
  // its boundary inputs:
  //
  // slot 0 = fused output
  // slot 1 = x
  // slot 2 = y
  // slot 3 = z
  REQUIRE(program.slots_count() == 4);

  REQUIRE(program.inputs().size() == 3);
  REQUIRE(program.get_slotid_from_valueid(x) == SlotId{1});
  REQUIRE(program.get_slotid_from_valueid(y) == SlotId{2});
  REQUIRE(program.get_slotid_from_valueid(z) == SlotId{3});

  REQUIRE(program.outputs().size() == 1);
  REQUIRE(program.outputs()[0] == SlotId{0});

  REQUIRE(program.scalar_count() == 5);
  REQUIRE(program.index_count() == 1);

  /*
   * One fused loop:
   *
   * 0 LoopBegin
   * 1 Load x
   * 2 Load y
   * 3 Load z
   * 4 Mul
   * 5 Add
   * 6 Store
   * 7 LoopEnd
   */
  REQUIRE(program.instructions().size() == 8);

  REQUIRE(program.instructions()[0].opcode == Opcode::LoopBegin);
  REQUIRE(program.instructions()[1].opcode == Opcode::Load);
  REQUIRE(program.instructions()[2].opcode == Opcode::Load);
  REQUIRE(program.instructions()[3].opcode == Opcode::Load);
  REQUIRE(program.instructions()[4].opcode == Opcode::Mul);
  REQUIRE(program.instructions()[5].opcode == Opcode::Add);
  REQUIRE(program.instructions()[6].opcode == Opcode::Store);
  REQUIRE(program.instructions()[7].opcode == Opcode::LoopEnd);

  REQUIRE(program.instructions()[1].buffer == SlotId{1});
  REQUIRE(program.instructions()[2].buffer == SlotId{2});
  REQUIRE(program.instructions()[3].buffer == SlotId{3});
  REQUIRE(program.instructions()[6].buffer == SlotId{0});

  // The fused expression computes Add(Mul(x, y), z).
  REQUIRE(program.instructions()[4].scalar_lhs.value == 0);
  REQUIRE(program.instructions()[4].scalar_rhs.value == 1);
  REQUIRE(program.instructions()[4].scalar_dst.value == 3);

  REQUIRE(program.instructions()[5].scalar_lhs.value == 3);
  REQUIRE(program.instructions()[5].scalar_rhs.value == 2);
  REQUIRE(program.instructions()[5].scalar_dst.value == 4);

  REQUIRE(program.instructions()[6].scalar_lhs.value == 4);
}

// =============================================================================
// Shared producer / memoization
// =============================================================================

TEST_CASE("Compiler compiles shared producer exactly once") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);

  const auto y = graph.input(Shape{2}, Dtype::F32);

  const auto shared = graph.mul(x, y);

  const auto out = graph.add(shared, shared);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  // x, y, shared, out.
  REQUIRE(program.slots_count() == 4);

  // Exactly two elementwise loops.
  REQUIRE(program.instructions().size() == 12);

  REQUIRE(program.instructions()[3].opcode == Opcode::Mul);

  REQUIRE(program.instructions()[9].opcode == Opcode::Add);

  // Both loads in Add must read the same shared slot.
  REQUIRE(program.instructions()[7].buffer == SlotId{2});

  REQUIRE(program.instructions()[8].buffer == SlotId{2});

  REQUIRE(program.outputs()[0] == SlotId{3});
}

TEST_CASE(
    "Compiler materializes shared producer once and fuses diamond consumers") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);
  const auto y = graph.input(Shape{2}, Dtype::F32);
  const auto z = graph.input(Shape{2}, Dtype::F32);

  const auto shared = graph.mul(x, y);
  const auto left = graph.add(shared, z);
  const auto right = graph.mul(shared, z);
  const auto out = graph.add(left, right);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.slots_count() == 5);
  REQUIRE(program.inputs().size() == 3);
  REQUIRE(program.outputs().size() == 1);

  const auto x_slot = program.get_slotid_from_valueid(x);
  const auto y_slot = program.get_slotid_from_valueid(y);
  const auto z_slot = program.get_slotid_from_valueid(z);
  const auto out_slot = program.outputs()[0];

  // One normal Mul loop for `shared`, followed by one fused loop.
  REQUIRE(program.instructions().size() == 16);

  REQUIRE(program.instructions()[0].opcode == Opcode::LoopBegin);
  REQUIRE(program.instructions()[1].opcode == Opcode::Load);
  REQUIRE(program.instructions()[2].opcode == Opcode::Load);
  REQUIRE(program.instructions()[3].opcode == Opcode::Mul);
  REQUIRE(program.instructions()[4].opcode == Opcode::Store);
  REQUIRE(program.instructions()[5].opcode == Opcode::LoopEnd);

  REQUIRE(program.instructions()[1].buffer == x_slot);
  REQUIRE(program.instructions()[2].buffer == y_slot);

  // Derive the materialized shared slot from the producer Store instead of
  // assuming the old recursive-lowering SlotId allocation order.
  const auto shared_slot = program.instructions()[4].buffer;

  REQUIRE(program.tensor_meta_by_slot(shared_slot).shape == Shape{2});

  REQUIRE(program.instructions()[6].opcode == Opcode::LoopBegin);

  REQUIRE(program.instructions()[7].opcode == Opcode::Load);
  REQUIRE(program.instructions()[7].buffer == shared_slot);

  REQUIRE(program.instructions()[8].opcode == Opcode::Load);
  REQUIRE(program.instructions()[8].buffer == z_slot);

  REQUIRE(program.instructions()[9].opcode == Opcode::Load);
  REQUIRE(program.instructions()[9].buffer == shared_slot);

  REQUIRE(program.instructions()[10].opcode == Opcode::Load);
  REQUIRE(program.instructions()[10].buffer == z_slot);

  REQUIRE(program.instructions()[11].opcode == Opcode::Add);
  REQUIRE(program.instructions()[12].opcode == Opcode::Mul);
  REQUIRE(program.instructions()[13].opcode == Opcode::Add);

  REQUIRE(program.instructions()[14].opcode == Opcode::Store);
  REQUIRE(program.instructions()[14].buffer == out_slot);

  REQUIRE(program.instructions()[15].opcode == Opcode::LoopEnd);
}

// =============================================================================
// Multiple graph outputs
// =============================================================================

TEST_CASE("Compiler preserves multiple computed output ordering") {
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
  REQUIRE(program.instructions().size() == 12);

  // Both independent kernels must be emitted exactly once. Their execution
  // order does not have to be identical to graph output ordering.
  std::size_t add_count = 0;
  std::size_t mul_count = 0;

  for (const auto &instruction : program.instructions()) {
    if (instruction.opcode == Opcode::Add) {
      ++add_count;
    } else if (instruction.opcode == Opcode::Mul) {
      ++mul_count;
    }
  }

  REQUIRE(add_count == 1);
  REQUIRE(mul_count == 1);

  // Requested graph output order is still preserved by the runtime program.
  REQUIRE(program.outputs()[0] != program.outputs()[1]);
  REQUIRE(program.tensor_meta_by_slot(program.outputs()[0]).shape == Shape{2});
  REQUIRE(program.tensor_meta_by_slot(program.outputs()[1]).shape == Shape{2});
}

TEST_CASE("Compiler reuses same runtime slot for duplicate graph output") {
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

  REQUIRE(program.slots_count() == 3);

  // Node is lowered only once.
  REQUIRE(program.instructions().size() == 6);

  REQUIRE(program.outputs().size() == 2);

  REQUIRE(program.outputs()[0] == SlotId{2});
  REQUIRE(program.outputs()[1] == SlotId{2});
}

TEST_CASE("Compiler supports input and computed value together as outputs") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);

  const auto y = graph.input(Shape{2}, Dtype::F32);

  const auto out = graph.add(x, y);

  graph.set_outputs({
      x,
      out,
  });

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.outputs().size() == 2);
  REQUIRE(program.inputs().size() == 2);

  REQUIRE(program.outputs()[0] == SlotId{0});
  REQUIRE(program.outputs()[1] == SlotId{2});

  REQUIRE(program.instructions().size() == 6);
  REQUIRE(program.slots_count() == 3);
}

// =============================================================================
// Reachability
// =============================================================================

TEST_CASE("Compiler excludes unreachable computed operation") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);

  const auto y = graph.input(Shape{2}, Dtype::F32);

  const auto live = graph.add(x, y);

  const auto dead = graph.mul(x, y);

  (void)dead;

  graph.set_outputs({live});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.slots_count() == 3);
  REQUIRE(program.inputs().size() == 2);

  // Only Add loop.
  REQUIRE(program.instructions().size() == 6);

  REQUIRE(program.instructions()[3].opcode == Opcode::Add);

  REQUIRE(program.outputs()[0] == SlotId{2});
}

TEST_CASE("Compiler excludes unreachable input and dependency subtree") {
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

  // z belongs only to the unreachable subtree and must not reach runtime.
  REQUIRE(program.inputs().size() == 2);

  REQUIRE(program.inputs()[0].value_id == x);
  REQUIRE(program.inputs()[0].slot_id == SlotId{0});
  REQUIRE(program.inputs()[1].value_id == y);
  REQUIRE(program.inputs()[1].slot_id == SlotId{1});

  REQUIRE(program.get_slotid_from_valueid(x) == SlotId{0});
  REQUIRE(program.get_slotid_from_valueid(y) == SlotId{1});

  REQUIRE(program.slots_count() == 3);

  REQUIRE(program.instructions().size() == 6);

  REQUIRE(program.instructions()[3].opcode == Opcode::Add);
}

// =============================================================================
// MatMul result consumed by another operation
// =============================================================================

TEST_CASE("Compiler propagates MatMul result metadata into dependent Add") {
  Graph graph;

  const auto lhs = graph.input(Shape{2, 3}, Dtype::F32);

  const auto rhs = graph.input(Shape{3, 4}, Dtype::F32);

  const auto bias = graph.input(Shape{2, 4}, Dtype::F32);

  const auto matmul = graph.matmul(lhs, rhs);

  const auto out = graph.add(matmul, bias);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  /*
   * Buffers:
   *
   * slot 0 lhs
   * slot 1 rhs
   * slot 2 matmul
   * slot 3 bias
   * slot 4 add output
   */
  REQUIRE(program.slots_count() == 5);

  REQUIRE(program.inputs().size() == 3);
  REQUIRE(program.get_slotid_from_valueid(lhs) == SlotId{0});
  REQUIRE(program.get_slotid_from_valueid(rhs) == SlotId{1});
  REQUIRE(program.get_slotid_from_valueid(bias) == SlotId{3});

  REQUIRE(program.tensor_meta_by_slot(SlotId{2}).shape == Shape{2, 4});

  REQUIRE(program.tensor_meta_by_slot(SlotId{4}).shape == Shape{2, 4});

  // MatMul = 18 runtime instructions.
  // Add = 6.
  REQUIRE(program.instructions().size() == 24);

  REQUIRE(program.outputs()[0] == SlotId{4});

  // First load of following Add consumes MatMul slot.
  REQUIRE(program.instructions()[19].opcode == Opcode::Load);

  REQUIRE(program.instructions()[19].buffer == SlotId{2});

  REQUIRE(program.instructions()[20].buffer == SlotId{3});

  REQUIRE(program.instructions()[21].opcode == Opcode::Add);

  REQUIRE(program.instructions()[22].buffer == SlotId{4});
}

// =============================================================================
// Longer mixed dependency chain
// =============================================================================

TEST_CASE(
    "Compiler fuses and groups longer elementwise chain before reduction") {
  Graph graph;

  const auto x = graph.input(Shape{4}, Dtype::F32);
  const auto y = graph.input(Shape{4}, Dtype::F32);

  const auto a = graph.mul(x, y);
  const auto b = graph.add(a, x);
  const auto c = graph.mul(b, y);
  const auto out = graph.sum(c);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  // Phase 4 forms one FusedElementwise value.
  // Phase 5 groups that value with Sum.
  // Only x, y and the scalar output are materialized.
  REQUIRE(program.slots_count() == 3);
  REQUIRE(program.inputs().size() == 2);
  REQUIRE(program.outputs().size() == 1);

  const auto x_slot = program.get_slotid_from_valueid(x);
  const auto y_slot = program.get_slotid_from_valueid(y);
  const auto out_slot = program.outputs()[0];

  REQUIRE(program.tensor_meta_by_slot(x_slot).shape == Shape{4});
  REQUIRE(program.tensor_meta_by_slot(y_slot).shape == Shape{4});
  REQUIRE(program.tensor_meta_by_slot(out_slot).shape == Shape{});

  REQUIRE(program.scalar_count() == 8);
  REQUIRE(program.index_count() == 1);

  REQUIRE(program.instructions().size() == 12);

  REQUIRE(program.instructions()[0].opcode == Opcode::Constant);
  REQUIRE(program.instructions()[1].opcode == Opcode::LoopBegin);

  REQUIRE(program.instructions()[2].opcode == Opcode::Load);
  REQUIRE(program.instructions()[3].opcode == Opcode::Load);
  REQUIRE(program.instructions()[4].opcode == Opcode::Load);
  REQUIRE(program.instructions()[5].opcode == Opcode::Load);

  REQUIRE(program.instructions()[6].opcode == Opcode::Mul);
  REQUIRE(program.instructions()[7].opcode == Opcode::Add);
  REQUIRE(program.instructions()[8].opcode == Opcode::Mul);
  REQUIRE(program.instructions()[9].opcode == Opcode::Add);

  REQUIRE(program.instructions()[10].opcode == Opcode::LoopEnd);
  REQUIRE(program.instructions()[11].opcode == Opcode::Store);
  REQUIRE(program.instructions()[11].buffer == out_slot);

  REQUIRE(program.instructions()[2].buffer == x_slot);
  REQUIRE(program.instructions()[3].buffer == y_slot);
  REQUIRE(program.instructions()[4].buffer == x_slot);
  REQUIRE(program.instructions()[5].buffer == y_slot);
}

// =============================================================================
// Buffer -> slot metadata invariant through complete compiler
// =============================================================================

TEST_CASE("Compiler runtime slot metadata remains aligned with slot ids") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);
  const auto y = graph.input(Shape{2}, Dtype::F32);

  const auto a = graph.add(x, y);
  const auto b = graph.mul(a, x);

  graph.set_outputs({b});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  // Add is absorbed into Mul, so there is no separate buffer for a.
  REQUIRE(program.slots_count() == 3);

  for (std::size_t i = 0; i < program.slots_count(); ++i) {
    REQUIRE(program.tensor_meta_by_slot(SlotId{i}).id == SlotId{i});
  }
}

TEST_CASE("Compiler preserves metadata for every reachable tensor slot") {
  Graph graph;

  const auto lhs = graph.input(Shape{2, 3}, Dtype::F32);

  const auto rhs = graph.input(Shape{3, 5}, Dtype::F32);

  const auto out = graph.matmul(lhs, rhs);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.slots_count() == 3);

  REQUIRE(program.tensor_meta_by_slot(SlotId{0}).shape == Shape{2, 3});

  REQUIRE(program.tensor_meta_by_slot(SlotId{1}).shape == Shape{3, 5});

  REQUIRE(program.tensor_meta_by_slot(SlotId{2}).shape == Shape{2, 5});

  REQUIRE(program.tensor_meta_by_slot(SlotId{0}).dtype == Dtype::F32);

  REQUIRE(program.tensor_meta_by_slot(SlotId{1}).dtype == Dtype::F32);

  REQUIRE(program.tensor_meta_by_slot(SlotId{2}).dtype == Dtype::F32);
}
