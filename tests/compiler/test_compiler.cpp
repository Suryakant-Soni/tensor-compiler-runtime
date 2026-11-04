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
#include <vector>

using tcr::compiler::Compiler;
using tcr::ir::Dtype;
using tcr::ir::Graph;
using tcr::ir::Shape;
using tcr::runtime::Opcode;
using tcr::runtime::SlotId;

namespace {

std::vector<Opcode> opcodes(const tcr::runtime::Program &program) {
  std::vector<Opcode> result;
  result.reserve(program.instructions().size());
  for (const auto &instruction : program.instructions()) {
    result.push_back(instruction.opcode);
  }
  return result;
}

std::size_t count_opcode(const tcr::runtime::Program &program, Opcode opcode) {
  return static_cast<std::size_t>(std::count_if(
      program.instructions().begin(), program.instructions().end(),
      [&](const auto &instruction) { return instruction.opcode == opcode; }));
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
  REQUIRE(program.slots_count() == 1);
  REQUIRE(program.inputs()[0].slot_id == SlotId{0});
  REQUIRE(program.outputs()[0].slot_id == SlotId{0});
  REQUIRE(program.inputs()[0].shape == Shape{3});
  REQUIRE(program.outputs()[0].shape == Shape{3});
  REQUIRE(program.inputs()[0].dtype == Dtype::F32);
  REQUIRE(program.outputs()[0].dtype == Dtype::F32);
}

TEST_CASE("Compiler preserves direct input output ordering") {
  Graph graph;
  const auto a = graph.input(Shape{4}, Dtype::F32);
  const auto b = graph.input(Shape{2}, Dtype::F32);
  graph.set_outputs({b, a});

  const auto program = Compiler{}.compile(graph);

  REQUIRE(program.inputs().size() == 2);
  REQUIRE(program.outputs().size() == 2);
  REQUIRE(program.outputs()[0].shape == Shape{2});
  REQUIRE(program.outputs()[1].shape == Shape{4});
  REQUIRE(program.outputs()[0].slot_id == program.get_slotid_from_valueid(b));
  REQUIRE(program.outputs()[1].slot_id == program.get_slotid_from_valueid(a));
}

TEST_CASE("Compiler Add uses two external input slots plus one planner slot") {
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
  REQUIRE(program.outputs()[0].slot_id == SlotId{2});
  REQUIRE(program.outputs()[0].shape == Shape{4});
  REQUIRE(program.get_slot_metadata(SlotId{2}).capacity == 4 * sizeof(float));
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
  const auto out = graph.mul(x, y);
  graph.set_outputs({out});

  const auto program = Compiler{}.compile(graph);

  REQUIRE(count_opcode(program, Opcode::Mul) == 1);
  REQUIRE(count_opcode(program, Opcode::Add) == 0);
  REQUIRE(program.outputs()[0].shape == Shape{2, 3});
  REQUIRE(program.outputs()[0].dtype == Dtype::F32);
  REQUIRE(program.get_slot_metadata(program.outputs()[0].slot_id).capacity ==
          6 * sizeof(float));
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
  const auto x = graph.input(Shape{4}, Dtype::F32);
  const auto y = graph.input(Shape{4}, Dtype::F32);
  const auto added = graph.add(x, y);
  const auto out = graph.sum(added);
  graph.set_outputs({out});

  const auto program = Compiler{}.compile(graph);

  REQUIRE(program.inputs().size() == 2);
  REQUIRE(program.outputs().size() == 1);
  REQUIRE(program.slots_count() == 3);
  REQUIRE(program.outputs()[0].shape == Shape{});
  REQUIRE(program.get_slot_metadata(program.outputs()[0].slot_id).capacity ==
          sizeof(float));
  REQUIRE(count_opcode(program, Opcode::Load) == 2);
  REQUIRE(count_opcode(program, Opcode::Add) == 2);
  REQUIRE(count_opcode(program, Opcode::Store) == 1);
}

TEST_CASE("Compiler MatMul keeps logical matrix output metadata outside "
          "physical slot") {
  Graph graph;
  const auto a = graph.input(Shape{2, 3}, Dtype::F32);
  const auto b = graph.input(Shape{3, 4}, Dtype::F32);
  const auto out = graph.matmul(a, b);
  graph.set_outputs({out});

  const auto program = Compiler{}.compile(graph);

  REQUIRE(program.inputs().size() == 2);
  REQUIRE(program.outputs().size() == 1);
  REQUIRE(program.slots_count() == 3);
  REQUIRE(program.outputs()[0].shape == Shape{2, 4});
  REQUIRE(program.outputs()[0].dtype == Dtype::F32);
  REQUIRE(program.get_slot_metadata(program.outputs()[0].slot_id).capacity ==
          8 * sizeof(float));
  REQUIRE(count_opcode(program, Opcode::LoopBegin) == 3);
  REQUIRE(count_opcode(program, Opcode::LoopEnd) == 3);
  REQUIRE(count_opcode(program, Opcode::Mul) >= 1);
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
  REQUIRE(program.outputs()[0].shape == Shape{2});
  REQUIRE(program.outputs()[1].shape == Shape{2});
  REQUIRE(program.outputs()[0].slot_id != program.outputs()[1].slot_id);
}

TEST_CASE("Compiler duplicate graph output reuses same runtime slot") {
  Graph graph;
  const auto x = graph.input(Shape{2}, Dtype::F32);
  const auto y = graph.input(Shape{2}, Dtype::F32);
  const auto out = graph.add(x, y);
  graph.set_outputs({out, out});

  const auto program = Compiler{}.compile(graph);

  REQUIRE(program.outputs().size() == 2);
  REQUIRE(program.outputs()[0].slot_id == program.outputs()[1].slot_id);
  REQUIRE(program.outputs()[0].shape == Shape{2});
  REQUIRE(program.outputs()[1].shape == Shape{2});
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
  REQUIRE(program.outputs()[0].slot_id == program.get_slotid_from_valueid(x));
  REQUIRE(program.outputs()[1].slot_id != program.outputs()[0].slot_id);
  REQUIRE(program.outputs()[0].shape == Shape{3});
  REQUIRE(program.outputs()[1].shape == Shape{3});
}

TEST_CASE("Compiler excludes unreachable dependency subtree") {
  Graph graph;
  const auto x = graph.input(Shape{2}, Dtype::F32);
  const auto y = graph.input(Shape{2}, Dtype::F32);
  const auto unused = graph.add(x, y);
  (void)unused;
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

  const auto program = Compiler{}.compile(graph);

  for (std::size_t i = 0; i < program.slots_count(); ++i) {
    const auto &meta = program.get_slot_metadata(SlotId{i});
    REQUIRE(meta.capacity >= sizeof(float));
    REQUIRE(meta.alignment >= alignof(std::max_align_t));
  }

  REQUIRE(program.inputs()[0].shape == Shape{2, 3});
  REQUIRE(program.inputs()[1].shape == Shape{2, 3});
  REQUIRE(program.outputs()[0].shape == Shape{2, 3});
}
