#include <catch2/catch_test_macros.hpp>

#include "tcr/compiler/compiler.hpp"
#include "tcr/ir/dtype.hpp"
#include "tcr/ir/graph.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/runtime/opcode.hpp"
#include "tcr/runtime/program.hpp"
#include "tcr/runtime/slot_id.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

using tcr::compiler::Compiler;
using tcr::ir::Dtype;
using tcr::ir::Graph;
using tcr::ir::Shape;
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

struct LogicalTensorInfo {
  SlotId slot_id;
  Shape shape;
  Dtype dtype;
};

LogicalTensorInfo
logical_tensor_info_by_slot(const tcr::runtime::Program &program,
                            SlotId slot_id) {
  for (const auto &input : program.inputs()) {
    if (input.slot_id == slot_id) {
      return LogicalTensorInfo{input.slot_id, input.shape, input.dtype};
    }
  }
  for (const auto &output : program.outputs()) {
    if (output.slot_id == slot_id) {
      return LogicalTensorInfo{output.slot_id, output.shape, output.dtype};
    }
  }
  throw std::logic_error(
      "runtime physical slot has no public logical tensor metadata");
}

} // namespace

TEST_CASE("Compiler compiles empty graph to empty runtime program") {
  Graph graph;

  const auto program = Compiler{}.compile(graph);

  REQUIRE(program.instructions().empty());
  REQUIRE(program.inputs().empty());
  REQUIRE(program.outputs().empty());
  REQUIRE(program.slots_count() == 0);
  REQUIRE(program.scalar_count() == 0);
  REQUIRE(program.index_count() == 0);
}

TEST_CASE("Compiler excludes graph input when graph has no reachable output") {
  Graph graph;
  graph.input(Shape{4}, Dtype::F32);

  const auto program = Compiler{}.compile(graph);

  REQUIRE(program.inputs().empty());
  REQUIRE(program.outputs().empty());
  REQUIRE(program.slots_count() == 0);
}

TEST_CASE(
    "Compiler can return graph input directly without planner-owned slot") {
  Graph graph;
  const auto x = graph.input(Shape{3}, Dtype::F32);
  graph.set_outputs({x});

  const auto program = Compiler{}.compile(graph);

  REQUIRE(program.inputs().size() == 1);
  REQUIRE(program.outputs().size() == 1);

  REQUIRE(program.inputs()[0].value_id == x);
  REQUIRE(program.inputs()[0].slot_id == SlotId{0});
  REQUIRE(program.get_slotid_from_valueid(x) == SlotId{0});
  REQUIRE(program.outputs()[0].slot_id == SlotId{0});

  REQUIRE(program.slots_count() == 1);
  REQUIRE(program.scalar_count() == 0);
  REQUIRE(program.index_count() == 0);

  const auto &info = logical_tensor_info_by_slot(program, SlotId{0});

  REQUIRE(info.slot_id == SlotId{0});
  REQUIRE(info.shape == Shape{3});
  REQUIRE(info.dtype == Dtype::F32);
}

TEST_CASE("Compiler preserves direct input output ordering") {
  Graph graph;

  const auto x = graph.input(Shape{}, Dtype::F32);

  graph.set_outputs({x});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(program.slots_count() == 1);

  const auto &info = logical_tensor_info_by_slot(program, SlotId{0});

  REQUIRE(info.shape == Shape{});
  REQUIRE(info.dtype == Dtype::F32);
}

TEST_CASE("Compiler preserves matrix input metadata") {
  Graph graph;

  const auto x = graph.input(Shape{2, 5}, Dtype::F32);

  graph.set_outputs({x});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(logical_tensor_info_by_slot(program, SlotId{0}).shape == Shape{2, 5});
}

// =============================================================================
// Multiple direct input outputs
// =============================================================================

TEST_CASE("Compiler preserves requested output ordering for direct inputs") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);

  const auto program = Compiler{}.compile(graph);

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

  REQUIRE(program.outputs()[0].slot_id == SlotId{0});
  REQUIRE(program.outputs()[1].slot_id == SlotId{1});

  REQUIRE(logical_tensor_info_by_slot(program, SlotId{0}).shape == Shape{4});

  REQUIRE(logical_tensor_info_by_slot(program, SlotId{1}).shape == Shape{2});
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

  const auto program = Compiler{}.compile(graph);

  REQUIRE(program.inputs().size() == 2);
  REQUIRE(program.outputs().size() == 1);
  REQUIRE(program.slots_count() == 3);
  REQUIRE(program.inputs()[0].slot_id == SlotId{0});
  REQUIRE(program.inputs()[1].slot_id == SlotId{1});

  REQUIRE(program.get_slotid_from_valueid(x) == SlotId{0});
  REQUIRE(program.get_slotid_from_valueid(y) == SlotId{1});

  REQUIRE(program.outputs()[0].slot_id == SlotId{2});

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

TEST_CASE("Compiler emits flat elementwise Add loop") {
  Graph graph;
  const auto x = graph.input(Shape{4}, Dtype::F32);
  const auto y = graph.input(Shape{4}, Dtype::F32);
  const auto out = graph.add(x, y);
  graph.set_outputs({out});

  const auto program = Compiler{}.compile(graph);
  const auto ops = opcodes(program);

  REQUIRE(ops.size() == 6);
  REQUIRE(ops[0] == Opcode::LoopBegin);
  REQUIRE(ops[1] == Opcode::Load);
  REQUIRE(ops[2] == Opcode::Load);
  REQUIRE(ops[3] == Opcode::Add);
  REQUIRE(ops[4] == Opcode::Store);
  REQUIRE(ops[5] == Opcode::LoopEnd);
  REQUIRE(program.instructions()[0].jump == 6);
  REQUIRE(program.instructions()[5].jump == 1);
  REQUIRE(program.instructions()[1].buffer == SlotId{0});
  REQUIRE(program.instructions()[2].buffer == SlotId{1});
  REQUIRE(program.instructions()[4].buffer == SlotId{2});
}

TEST_CASE("Compiler Mul emits Mul arithmetic and preserves output metadata") {
  Graph graph;
  const auto x = graph.input(Shape{2, 3}, Dtype::F32);
  const auto y = graph.input(Shape{2, 3}, Dtype::F32);

  const auto out = graph.add(x, y);

  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  REQUIRE(logical_tensor_info_by_slot(program, SlotId{2}).shape == Shape{2, 3});

  REQUIRE(logical_tensor_info_by_slot(program, SlotId{2}).dtype == Dtype::F32);

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

  const auto program = Compiler{}.compile(graph);

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

  REQUIRE(program.outputs()[0].slot_id == SlotId{2});
}

TEST_CASE("Compiler Sum produces scalar logical output") {
  Graph graph;
  const auto x = graph.input(Shape{5}, Dtype::F32);
  const auto out = graph.sum(x);
  graph.set_outputs({out});

  const auto program = Compiler{}.compile(graph);

  REQUIRE(program.inputs().size() == 1);
  REQUIRE(program.outputs().size() == 1);
  REQUIRE(program.slots_count() == 2);
  REQUIRE(program.outputs()[0].shape == Shape{});
  REQUIRE(program.outputs()[0].dtype == Dtype::F32);
  REQUIRE(program.get_slot_metadata(program.outputs()[0].slot_id).capacity ==
          sizeof(float));
  REQUIRE(count_opcode(program, Opcode::Constant) == 1);
  REQUIRE(count_opcode(program, Opcode::Add) == 1);
  REQUIRE(count_opcode(program, Opcode::Store) == 1);
}

TEST_CASE("Compiler groups elementwise Add before Sum without materializing "
          "Add tensor") {
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

  REQUIRE(program.outputs()[0].slot_id == SlotId{1});

  const auto &output_info = logical_tensor_info_by_slot(program, SlotId{1});

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

  const auto program = Compiler{}.compile(graph);

  REQUIRE(program.inputs().size() == 2);
  REQUIRE(program.outputs().size() == 1);

  const auto x_slot = program.get_slotid_from_valueid(x);
  const auto y_slot = program.get_slotid_from_valueid(y);
  const auto out_slot = program.outputs()[0].slot_id;

  REQUIRE(logical_tensor_info_by_slot(program, x_slot).shape == Shape{4});
  REQUIRE(logical_tensor_info_by_slot(program, y_slot).shape == Shape{4});
  REQUIRE(logical_tensor_info_by_slot(program, out_slot).shape == Shape{});

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
  const auto out_slot = program.outputs()[0].slot_id;

  REQUIRE(logical_tensor_info_by_slot(program, x_slot).shape == Shape{5});
  REQUIRE(logical_tensor_info_by_slot(program, y_slot).shape == Shape{5});
  REQUIRE(logical_tensor_info_by_slot(program, out_slot).shape == Shape{});

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

TEST_CASE("Compiler MatMul keeps logical matrix output metadata outside "
          "physical slot") {
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
  const auto out_slot = program.outputs()[0].slot_id;

  REQUIRE(logical_tensor_info_by_slot(program, x_slot).shape == Shape{4});
  REQUIRE(logical_tensor_info_by_slot(program, y_slot).shape == Shape{4});
  REQUIRE(logical_tensor_info_by_slot(program, z_slot).shape == Shape{4});
  REQUIRE(logical_tensor_info_by_slot(program, out_slot).shape == Shape{});

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

  const auto program = Compiler{}.compile(graph);

  REQUIRE(program.inputs().size() == 2);
  REQUIRE(program.outputs().size() == 1);
  REQUIRE(program.slots_count() == 3);

  // i,j,k + six arithmetic index registers.
  REQUIRE(program.index_count() == 9);

  // a, b, product, accumulator.
  REQUIRE(program.scalar_count() == 4);

  REQUIRE(program.instructions().size() == 18);

  REQUIRE(program.outputs()[0].slot_id == SlotId{2});

  REQUIRE(logical_tensor_info_by_slot(program, SlotId{2}).shape == Shape{2, 4});
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

  REQUIRE(logical_tensor_info_by_slot(program, SlotId{2}).shape ==
          Shape{5, 11});
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

  // Fusion removes the materialized product buffer. Phase 6 then assigns
  // runtime slots independently of the old lowering BufferId allocation order.
  REQUIRE(program.slots_count() == 4);

  REQUIRE(program.inputs().size() == 3);
  REQUIRE(program.outputs().size() == 1);

  const auto x_slot = program.get_slotid_from_valueid(x);
  const auto y_slot = program.get_slotid_from_valueid(y);
  const auto z_slot = program.get_slotid_from_valueid(z);
  const auto out_slot = program.outputs()[0].slot_id;

  REQUIRE(x_slot != y_slot);
  REQUIRE(x_slot != z_slot);
  REQUIRE(y_slot != z_slot);
  REQUIRE(out_slot != x_slot);
  REQUIRE(out_slot != y_slot);
  REQUIRE(out_slot != z_slot);

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

  REQUIRE(program.instructions()[1].buffer == x_slot);
  REQUIRE(program.instructions()[2].buffer == y_slot);
  REQUIRE(program.instructions()[3].buffer == z_slot);
  REQUIRE(program.instructions()[6].buffer == out_slot);

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

  REQUIRE(program.outputs()[0].slot_id == SlotId{3});
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
  const auto out_slot = program.outputs()[0].slot_id;

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

  // Internal materialized buffers no longer carry logical shape/dtype in the
  // runtime Program. The physical slot must instead have enough capacity.
  REQUIRE(program.get_slot_metadata(shared_slot).capacity >=
          Shape{2}.num_elements() * sizeof(float));

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

TEST_CASE("Compiler preserves multiple computed output ordering") {
  Graph graph;
  const auto x = graph.input(Shape{2}, Dtype::F32);
  const auto y = graph.input(Shape{2}, Dtype::F32);
  const auto added = graph.add(x, y);
  const auto multiplied = graph.mul(x, y);
  graph.set_outputs({multiplied, added});

  const auto program = Compiler{}.compile(graph);

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
  REQUIRE(program.outputs()[0].slot_id != program.outputs()[1].slot_id);
  REQUIRE(logical_tensor_info_by_slot(program, program.outputs()[0].slot_id)
              .shape == Shape{2});
  REQUIRE(logical_tensor_info_by_slot(program, program.outputs()[1].slot_id)
              .shape == Shape{2});
}

TEST_CASE("Compiler duplicate graph output reuses same runtime slot") {
  Graph graph;
  const auto x = graph.input(Shape{2}, Dtype::F32);
  const auto y = graph.input(Shape{2}, Dtype::F32);
  const auto out = graph.add(x, y);
  graph.set_outputs({out, out});

  const auto program = Compiler{}.compile(graph);

  REQUIRE(program.outputs().size() == 2);

  REQUIRE(program.outputs()[0].slot_id == SlotId{2});
  REQUIRE(program.outputs()[1].slot_id == SlotId{2});
}

TEST_CASE(
    "Compiler supports direct input and computed value together as outputs") {
  Graph graph;
  const auto x = graph.input(Shape{3}, Dtype::F32);
  const auto y = graph.input(Shape{3}, Dtype::F32);
  const auto out = graph.add(x, y);
  graph.set_outputs({x, out});

  const auto program = Compiler{}.compile(graph);

  REQUIRE(program.outputs().size() == 2);
  REQUIRE(program.inputs().size() == 2);

  REQUIRE(program.outputs()[0].slot_id == SlotId{0});
  REQUIRE(program.outputs()[1].slot_id == SlotId{2});

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

  REQUIRE(program.outputs()[0].slot_id == SlotId{2});
}

TEST_CASE("Compiler excludes unreachable input and dependency subtree") {
  Graph graph;

  const auto x = graph.input(Shape{2}, Dtype::F32);

  const auto y = graph.input(Shape{2}, Dtype::F32);

  const auto z = graph.input(Shape{2}, Dtype::F32);
  graph.set_outputs({z});

  const auto program = Compiler{}.compile(graph);

  REQUIRE(program.inputs().size() == 1);
  REQUIRE(program.outputs().size() == 1);
  REQUIRE(program.slots_count() == 1);
  REQUIRE(program.instructions().empty());
}

TEST_CASE("Compiler physical slot metadata is capacity and alignment only") {
  Graph graph;
  const auto x = graph.input(Shape{2, 3}, Dtype::F32);
  const auto y = graph.input(Shape{2, 3}, Dtype::F32);
  const auto out = graph.add(x, y);
  graph.set_outputs({out});

  Compiler compiler;
  const auto program = compiler.compile(graph);

  // Three external input slots plus two planner-owned physical slots:
  // one for the MatMul result and one for the Add output.
  REQUIRE(program.slots_count() == 5);

  REQUIRE(program.inputs().size() == 3);
  const auto lhs_slot = program.get_slotid_from_valueid(lhs);
  const auto rhs_slot = program.get_slotid_from_valueid(rhs);
  const auto bias_slot = program.get_slotid_from_valueid(bias);
  const auto out_slot = program.outputs()[0].slot_id;

  // MatMul's internal logical result is represented only by its physical slot
  // at runtime. Derive that slot from its Store instruction.
  const auto matmul_slot = program.instructions()[15].buffer;

  REQUIRE(program.get_slot_metadata(matmul_slot).capacity >=
          Shape{2, 4}.num_elements() * sizeof(float));
  REQUIRE(program.outputs()[0].shape == Shape{2, 4});
  REQUIRE(program.outputs()[0].dtype == Dtype::F32);

  // MatMul = 18 runtime instructions.
  // Add = 6.
  REQUIRE(program.instructions().size() == 24);

  REQUIRE(program.outputs()[0].slot_id == out_slot);

  // First load of following Add consumes MatMul storage; second consumes bias.
  REQUIRE(program.instructions()[19].opcode == Opcode::Load);
  REQUIRE(program.instructions()[19].buffer == matmul_slot);
  REQUIRE(program.instructions()[20].buffer == bias_slot);
  REQUIRE(program.instructions()[21].opcode == Opcode::Add);
  REQUIRE(program.instructions()[22].buffer == out_slot);

  REQUIRE(lhs_slot != rhs_slot);
  REQUIRE(lhs_slot != bias_slot);
  REQUIRE(rhs_slot != bias_slot);
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
  const auto out_slot = program.outputs()[0].slot_id;

  REQUIRE(logical_tensor_info_by_slot(program, x_slot).shape == Shape{4});
  REQUIRE(logical_tensor_info_by_slot(program, y_slot).shape == Shape{4});
  REQUIRE(logical_tensor_info_by_slot(program, out_slot).shape == Shape{});

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
    REQUIRE(logical_tensor_info_by_slot(program, SlotId{i}).slot_id ==
            SlotId{i});
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

  REQUIRE(logical_tensor_info_by_slot(program, SlotId{0}).shape == Shape{2, 3});

  REQUIRE(logical_tensor_info_by_slot(program, SlotId{1}).shape == Shape{3, 5});

  REQUIRE(logical_tensor_info_by_slot(program, SlotId{2}).shape == Shape{2, 5});

  REQUIRE(logical_tensor_info_by_slot(program, SlotId{0}).dtype == Dtype::F32);

  REQUIRE(logical_tensor_info_by_slot(program, SlotId{1}).dtype == Dtype::F32);

  REQUIRE(logical_tensor_info_by_slot(program, SlotId{2}).dtype == Dtype::F32);
}
