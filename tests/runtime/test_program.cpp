#include <catch2/catch_test_macros.hpp>

#include "tcr/ir/dtype.hpp"
#include "tcr/ir/ids.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/runtime/instruction.hpp"
#include "tcr/runtime/opcode.hpp"
#include "tcr/runtime/program.hpp"
#include "tcr/runtime/slot_id.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

using tcr::ir::Dtype;
using tcr::ir::Shape;
using tcr::ir::ValueId;
using tcr::runtime::InputInfo;
using tcr::runtime::Instruction;
using tcr::runtime::Opcode;
using tcr::runtime::OutputInfo;
using tcr::runtime::Program;
using tcr::runtime::SlotId;
using tcr::runtime::SlotMetadata;

TEST_CASE("runtime Program default constructor creates empty program") {
  const Program program;

  REQUIRE(program.instructions().empty());
  REQUIRE(program.inputs().empty());
  REQUIRE(program.outputs().empty());
  REQUIRE(program.slots_count() == 0);
  REQUIRE(program.scalar_count() == 0);
  REQUIRE(program.index_count() == 0);
}

TEST_CASE("runtime Program stores physical slot metadata by SlotId index") {
  const Program program({}, {}, {},
                        {
                            SlotMetadata{16, 16},
                            SlotMetadata{64, 32},
                        },
                        0, 0);

  REQUIRE(program.slots_count() == 2);
  REQUIRE(program.get_slot_metadata(SlotId{0}).capacity == 16);
  REQUIRE(program.get_slot_metadata(SlotId{0}).alignment == 16);
  REQUIRE(program.get_slot_metadata(SlotId{1}).capacity == 64);
  REQUIRE(program.get_slot_metadata(SlotId{1}).alignment == 32);
}

TEST_CASE("runtime Program rejects slot metadata lookup outside slot table") {
  const Program program({}, {}, {}, {SlotMetadata{16, 16}}, 0, 0);

  REQUIRE_THROWS_AS(program.get_slot_metadata(SlotId{1}), std::out_of_range);
  REQUIRE_THROWS_AS(program.get_slot_metadata(SlotId{100}), std::out_of_range);
}

TEST_CASE("runtime Program stores logical input metadata separately from slot "
          "metadata") {
  const Program program(
      {},
      {
          InputInfo{ValueId{10}, SlotId{0}, Shape{2, 3}, Dtype::F32},
          InputInfo{ValueId{20}, SlotId{1}, Shape{}, Dtype::F32},
      },
      {},
      {
          SlotMetadata{24, 16},
          SlotMetadata{4, 16},
      },
      0, 0);

  REQUIRE(program.inputs().size() == 2);
  REQUIRE(program.inputs()[0].value_id == ValueId{10});
  REQUIRE(program.inputs()[0].slot_id == SlotId{0});
  REQUIRE(program.inputs()[0].shape == Shape{2, 3});
  REQUIRE(program.inputs()[0].dtype == Dtype::F32);
  REQUIRE(program.inputs()[1].value_id == ValueId{20});
  REQUIRE(program.inputs()[1].slot_id == SlotId{1});
  REQUIRE(program.inputs()[1].shape == Shape{});
}

TEST_CASE("runtime Program maps input ValueId to runtime SlotId") {
  const Program program(
      {},
      {
          InputInfo{ValueId{10}, SlotId{2}, Shape{1}, Dtype::F32},
          InputInfo{ValueId{20}, SlotId{0}, Shape{1}, Dtype::F32},
      },
      {},
      {
          SlotMetadata{4, 16},
          SlotMetadata{4, 16},
          SlotMetadata{4, 16},
      },
      0, 0);

  REQUIRE(program.get_slotid_from_valueid(ValueId{10}) == SlotId{2});
  REQUIRE(program.get_slotid_from_valueid(ValueId{20}) == SlotId{0});
  REQUIRE_THROWS_AS(program.get_slotid_from_valueid(ValueId{30}),
                    std::logic_error);
}

TEST_CASE("runtime Program identifies external input slots") {
  const Program program(
      {},
      {
          InputInfo{ValueId{10}, SlotId{0}, Shape{1}, Dtype::F32},
          InputInfo{ValueId{20}, SlotId{2}, Shape{1}, Dtype::F32},
      },
      {},
      {
          SlotMetadata{4, 16},
          SlotMetadata{4, 16},
          SlotMetadata{4, 16},
      },
      0, 0);

  REQUIRE(program.is_input_slot(SlotId{0}));
  REQUIRE_FALSE(program.is_input_slot(SlotId{1}));
  REQUIRE(program.is_input_slot(SlotId{2}));
}

TEST_CASE("runtime Program stores logical output metadata separately from "
          "physical slot") {
  const Program program({}, {},
                        {
                            OutputInfo{SlotId{1}, Shape{2, 3}, Dtype::F32},
                            OutputInfo{SlotId{0}, Shape{}, Dtype::F32},
                        },
                        {
                            SlotMetadata{32, 16},
                            SlotMetadata{128, 16},
                        },
                        0, 0);

  REQUIRE(program.outputs().size() == 2);
  REQUIRE(program.outputs()[0].slot_id == SlotId{1});
  REQUIRE(program.outputs()[0].shape == Shape{2, 3});
  REQUIRE(program.outputs()[0].dtype == Dtype::F32);
  REQUIRE(program.outputs()[1].slot_id == SlotId{0});
  REQUIRE(program.outputs()[1].shape == Shape{});
}

TEST_CASE("runtime Program can expose same physical slot with different "
          "logical output metadata") {
  const Program program({}, {},
                        {
                            OutputInfo{SlotId{0}, Shape{4}, Dtype::F32},
                            OutputInfo{SlotId{0}, Shape{}, Dtype::F32},
                        },
                        {SlotMetadata{16, 16}}, 0, 0);

  REQUIRE(program.outputs()[0].slot_id == SlotId{0});
  REQUIRE(program.outputs()[1].slot_id == SlotId{0});
  REQUIRE(program.outputs()[0].shape == Shape{4});
  REQUIRE(program.outputs()[1].shape == Shape{});
}

TEST_CASE(
    "runtime Program stores scalar and index register counts independently") {
  const Program program({}, {}, {}, {}, 7, 11);

  REQUIRE(program.scalar_count() == 7);
  REQUIRE(program.index_count() == 11);
}

TEST_CASE("runtime Program stores instruction list in execution order") {
  const Program program(
      {
          Instruction{.opcode = Opcode::Constant, .constant = 3.0F},
          Instruction{.opcode = Opcode::Add},
          Instruction{.opcode = Opcode::Mul},
      },
      {}, {}, {}, 0, 0);

  REQUIRE(program.instructions().size() == 3);
  REQUIRE(program.instructions()[0].opcode == Opcode::Constant);
  REQUIRE(program.instructions()[1].opcode == Opcode::Add);
  REQUIRE(program.instructions()[2].opcode == Opcode::Mul);
}

TEST_CASE("runtime Program add_instruction appends instruction") {
  Program program;

  program.add_instruction(Instruction{.opcode = Opcode::Add});
  program.add_instruction(Instruction{.opcode = Opcode::Mul});

  REQUIRE(program.instructions().size() == 2);
  REQUIRE(program.instructions()[0].opcode == Opcode::Add);
  REQUIRE(program.instructions()[1].opcode == Opcode::Mul);
}

TEST_CASE(
    "runtime Program mutable instructions accessor can edit instructions") {
  Program program({Instruction{.opcode = Opcode::Add}}, {}, {}, {}, 0, 0);

  program.instructions()[0].opcode = Opcode::Mul;

  REQUIRE(program.instructions()[0].opcode == Opcode::Mul);
}

TEST_CASE("runtime Program accessors work through const reference") {
  Program program({Instruction{.opcode = Opcode::Constant}},
                  {InputInfo{ValueId{1}, SlotId{0}, Shape{2}, Dtype::F32}},
                  {OutputInfo{SlotId{1}, Shape{2}, Dtype::F32}},
                  {
                      SlotMetadata{8, 16},
                      SlotMetadata{32, 16},
                  },
                  3, 4);

  const Program &const_program = program;

  REQUIRE(const_program.instructions().size() == 1);
  REQUIRE(const_program.inputs().size() == 1);
  REQUIRE(const_program.outputs().size() == 1);
  REQUIRE(const_program.slots_count() == 2);
  REQUIRE(const_program.scalar_count() == 3);
  REQUIRE(const_program.index_count() == 4);
}
