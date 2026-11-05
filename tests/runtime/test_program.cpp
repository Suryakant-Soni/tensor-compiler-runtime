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

using tcr::runtime::IndexOperand;
using tcr::runtime::InputInfo;
using tcr::runtime::Instruction;
using tcr::runtime::Opcode;
using tcr::runtime::OutputInfo;
using tcr::runtime::OutputInfo;
using tcr::runtime::Program;
using tcr::runtime::SlotId;
using tcr::runtime::SlotMetadata;

namespace {

constexpr std::size_t kAlignment = alignof(std::max_align_t);

SlotMetadata slot(std::size_t capacity) {
  return SlotMetadata{capacity, kAlignment};
}

InputInfo input(ValueId value_id, SlotId slot_id, Shape shape = Shape{1},
                Dtype dtype = Dtype::F32) {
  return InputInfo{value_id, slot_id, std::move(shape), dtype};
}

OutputInfo output(SlotId slot_id, Shape shape = Shape{1},
                  Dtype dtype = Dtype::F32) {
  return OutputInfo{slot_id, std::move(shape), dtype};
}

} // namespace

// -----------------------------------------------------------------------------
// Default construction
// -----------------------------------------------------------------------------

TEST_CASE("runtime Program default constructor creates empty program") {
  const Program program;

  REQUIRE(program.instructions().empty());
  REQUIRE(program.inputs().empty());
  REQUIRE(program.outputs().empty());
  REQUIRE(program.slots_count() == 0);
  REQUIRE(program.scalar_count() == 0);
  REQUIRE(program.index_count() == 0);
}

// -----------------------------------------------------------------------------
// Slot metadata
// -----------------------------------------------------------------------------

TEST_CASE("runtime Program stores slot metadata") {
  const Program program{{}, {}, {}, {SlotMetadata{12, 32}}, 0, 0};

  REQUIRE(program.slots_count() == 1);

  const auto &info = program.get_slot_metadata(SlotId{0});
  REQUIRE(info.capacity == 12);
  REQUIRE(info.alignment == 32);
}

TEST_CASE("runtime Program stores metadata for multiple slots") {
  const Program program{{},
                        {},
                        {},
                        {
                            SlotMetadata{8, 8},
                            SlotMetadata{24, 16},
                            SlotMetadata{4, 32},
                        },
                        0,
                        0};

  REQUIRE(program.slots_count() == 3);

  const auto &first = program.get_slot_metadata(SlotId{0});
  const auto &second = program.get_slot_metadata(SlotId{1});
  const auto &third = program.get_slot_metadata(SlotId{2});

  REQUIRE(first.capacity == 8);
  REQUIRE(first.alignment == 8);
  REQUIRE(second.capacity == 24);
  REQUIRE(second.alignment == 16);
  REQUIRE(third.capacity == 4);
  REQUIRE(third.alignment == 32);
}

TEST_CASE("runtime Program supports zero slots") {
  const Program program{{}, {}, {}, {}, 0, 0};
  REQUIRE(program.slots_count() == 0);
}

TEST_CASE("runtime Program slot count comes from metadata vector size") {
  const Program program{{}, {}, {}, {slot(4), slot(4), slot(4), slot(4)}, 0, 0};
  REQUIRE(program.slots_count() == 4);
}

TEST_CASE("runtime Program get_slot_metadata returns metadata by slot index") {
  const Program program{{}, {}, {}, {SlotMetadata{10, 8}, SlotMetadata{20, 16}},
                        0,  0};

  REQUIRE(program.get_slot_metadata(SlotId{0}).capacity == 10);
  REQUIRE(program.get_slot_metadata(SlotId{1}).capacity == 20);
}

TEST_CASE(
    "runtime Program get_slot_metadata rejects slot equal to slot count") {
  const Program program{{}, {}, {}, {slot(4)}, 0, 0};
  REQUIRE_THROWS_AS(program.get_slot_metadata(SlotId{1}), std::out_of_range);
}

TEST_CASE("runtime Program get_slot_metadata rejects slot greater than slot "
          "count") {
  const Program program{{}, {}, {}, {slot(4)}, 0, 0};
  REQUIRE_THROWS_AS(program.get_slot_metadata(SlotId{100}), std::out_of_range);
}

TEST_CASE("runtime Program get_slot_metadata rejects any slot when program "
          "has no slots") {
  const Program program;
  REQUIRE_THROWS_AS(program.get_slot_metadata(SlotId{0}), std::out_of_range);
}

// -----------------------------------------------------------------------------
// Inputs
// -----------------------------------------------------------------------------

TEST_CASE("runtime Program stores input slots") {
  const Program program{{},
                        {
                            input(ValueId{10}, SlotId{0}, Shape{1}),
                            input(ValueId{20}, SlotId{1}, Shape{2}),
                            input(ValueId{30}, SlotId{3}, Shape{3}),
                        },
                        {},
                        {slot(4), slot(8), slot(4), slot(12)},
                        0,
                        0};

  REQUIRE(program.inputs().size() == 2);
  REQUIRE(program.inputs()[0].value_id == ValueId{10});
  REQUIRE(program.inputs()[0].slot_id == SlotId{0});
  REQUIRE(program.inputs()[0].shape == Shape{1});
  REQUIRE(program.inputs()[0].dtype == Dtype::F32);

  REQUIRE(program.inputs()[1].value_id == ValueId{20});
  REQUIRE(program.inputs()[1].slot_id == SlotId{1});
  REQUIRE(program.inputs()[1].shape == Shape{2});

  REQUIRE(program.inputs()[2].value_id == ValueId{30});
  REQUIRE(program.inputs()[2].slot_id == SlotId{3});
  REQUIRE(program.inputs()[2].shape == Shape{3});

  REQUIRE(program.get_slotid_from_valueid(ValueId{10}) == SlotId{0});
  REQUIRE(program.get_slotid_from_valueid(ValueId{20}) == SlotId{1});
  REQUIRE(program.get_slotid_from_valueid(ValueId{30}) == SlotId{3});
}

TEST_CASE("runtime Program preserves input slot ordering") {
  const Program program{{},
                        {
                            input(ValueId{10}, SlotId{4}),
                            input(ValueId{20}, SlotId{1}),
                            input(ValueId{30}, SlotId{3}),
                        },
                        {},
                        {slot(4), slot(4), slot(4), slot(4), slot(4)},
                        0,
                        0};

  REQUIRE(program.inputs().size() == 3);
  REQUIRE(program.inputs()[0].value_id == ValueId{10});
  REQUIRE(program.inputs()[0].slot_id == SlotId{4});
  REQUIRE(program.inputs()[1].value_id == ValueId{20});
  REQUIRE(program.inputs()[1].slot_id == SlotId{1});
  REQUIRE(program.inputs()[2].value_id == ValueId{30});
  REQUIRE(program.inputs()[2].slot_id == SlotId{3});

  REQUIRE(program.get_slotid_from_valueid(ValueId{10}) == SlotId{4});
  REQUIRE(program.get_slotid_from_valueid(ValueId{20}) == SlotId{1});
  REQUIRE(program.get_slotid_from_valueid(ValueId{30}) == SlotId{3});
}

TEST_CASE("runtime Program rejects ValueId that is not a runtime input") {
  const Program program{{}, {input(ValueId{10}, SlotId{0})}, {}, {slot(4)}, 0,
                        0};

  REQUIRE_THROWS_AS(program.get_slotid_from_valueid(ValueId{99}),
                    std::logic_error);
}

TEST_CASE("runtime Program supports no inputs") {
  const Program program{{}, {}, {output(SlotId{0})}, {slot(4)}, 0, 0};

  REQUIRE(program.inputs().empty());
  REQUIRE(program.outputs().size() == 1);
  REQUIRE(program.outputs()[0].slot_id == SlotId{0});
}

TEST_CASE("runtime Program stores output slots") {
  const Program program{{},
                        {},
                        {
                            output(SlotId{2}, Shape{2}),
                            output(SlotId{5}, Shape{}),
                        },
                        {slot(4), slot(4), slot(8), slot(4), slot(4), slot(16)},
                        0,
                        0};

  REQUIRE(program.outputs().size() == 2);
  REQUIRE(program.outputs()[0].slot_id == SlotId{2});
  REQUIRE(program.outputs()[1].slot_id == SlotId{5});
  REQUIRE(program.outputs()[0].shape == Shape{2});
  REQUIRE(program.outputs()[1].shape == Shape{});
}

TEST_CASE("runtime Program preserves output slot ordering") {
  const Program program{{},
                        {},
                        {
                            output(SlotId{4}),
                            output(SlotId{2}),
                            output(SlotId{3}),
                        },
                        {slot(4), slot(4), slot(4), slot(4), slot(4)},
                        0,
                        0};

  REQUIRE(program.outputs().size() == 3);
  REQUIRE(program.outputs()[0].slot_id == SlotId{4});
  REQUIRE(program.outputs()[1].slot_id == SlotId{2});
  REQUIRE(program.outputs()[2].slot_id == SlotId{3});
}

TEST_CASE("runtime Program supports no outputs") {
  const Program program{{}, {input(ValueId{10}, SlotId{0})}, {}, {slot(4)}, 0,
                        0};

  REQUIRE(program.inputs().size() == 1);
  REQUIRE(program.outputs().empty());
}

TEST_CASE("runtime Program supports multiple inputs and outputs") {
  const Program program{{},
                        {
                            input(ValueId{10}, SlotId{0}, Shape{2}),
                            input(ValueId{20}, SlotId{1}, Shape{2}),
                        },
                        {
                            output(SlotId{3}, Shape{2}),
                            output(SlotId{4}, Shape{2}),
                        },
                        {slot(8), slot(8), slot(8), slot(8), slot(8)},
                        0,
                        0};

  REQUIRE(program.inputs().size() == 2);
  REQUIRE(program.outputs().size() == 2);
  REQUIRE(program.slots_count() == 5);

  REQUIRE(program.inputs()[0].value_id == ValueId{10});
  REQUIRE(program.inputs()[0].slot_id == SlotId{0});
  REQUIRE(program.inputs()[1].value_id == ValueId{20});
  REQUIRE(program.inputs()[1].slot_id == SlotId{1});

  REQUIRE(program.outputs()[0].slot_id == SlotId{3});
  REQUIRE(program.outputs()[1].slot_id == SlotId{4});
}

TEST_CASE("runtime Program stores scalar count") {
  const Program program{{}, {}, {}, {}, 7, 0};
  REQUIRE(program.scalar_count() == 7);
}

TEST_CASE("runtime Program supports zero scalar registers") {
  const Program program{{}, {}, {}, {}, 0, 0};
  REQUIRE(program.scalar_count() == 0);
}

TEST_CASE("runtime Program stores index count") {
  const Program program{{}, {}, {}, {}, 0, 9};
  REQUIRE(program.index_count() == 9);
}

TEST_CASE("runtime Program supports zero index registers") {
  const Program program{{}, {}, {}, {}, 0, 0};
  REQUIRE(program.index_count() == 0);
}

TEST_CASE("runtime Program stores scalar and index counts independently") {
  const Program program{{}, {}, {}, {}, 4, 6};
  REQUIRE(program.scalar_count() == 4);
  REQUIRE(program.index_count() == 6);
}

// -----------------------------------------------------------------------------
// Instructions
// -----------------------------------------------------------------------------

TEST_CASE("runtime Program stores instruction list") {
  const Instruction first{
      .opcode = Opcode::Constant,
      .scalar_dst = ScalarId{0},
      .constant = 10.0F,
  };

  const Instruction second{
      .opcode = Opcode::Constant,
      .scalar_dst = ScalarId{1},
      .constant = 20.0F,
  };

  const Program program{{first, second}, {}, {}, {}, 2, 0};

  REQUIRE(program.instructions().size() == 2);
  REQUIRE(program.instructions()[0].opcode == Opcode::Constant);
  REQUIRE(program.instructions()[1].opcode == Opcode::Constant);
  REQUIRE(program.instructions()[0].scalar_dst.value == 0);
  REQUIRE(program.instructions()[1].scalar_dst.value == 1);
  REQUIRE(program.instructions()[0].constant == 10.0F);
  REQUIRE(program.instructions()[1].constant == 20.0F);
}

TEST_CASE("runtime Program preserves instruction execution order") {
  const Instruction constant{
      .opcode = Opcode::Constant,
      .scalar_dst = ScalarId{0},
      .constant = 1.0F,
  };

  const Instruction add{
      .opcode = Opcode::Add,
      .scalar_dst = ScalarId{2},
      .scalar_lhs = ScalarId{0},
      .scalar_rhs = ScalarId{1},
  };

  const Instruction store{
      .opcode = Opcode::Store,
      .scalar_lhs = ScalarId{2},
      .buffer = SlotId{0},
      .offset =
          IndexOperand{.kind = IndexOperand::Kind::Constant, .constant = 0},
  };

  const Program program{{constant, add, store}, {}, {}, {slot(4)}, 3, 0};

  REQUIRE(program.instructions().size() == 3);
  REQUIRE(program.instructions()[0].opcode == Opcode::Constant);
  REQUIRE(program.instructions()[1].opcode == Opcode::Add);
  REQUIRE(program.instructions()[2].opcode == Opcode::Store);
}

TEST_CASE("runtime Program add_instruction appends instruction") {
  Program program;

  program.add_instruction(Instruction{
      .opcode = Opcode::Constant,
      .scalar_dst = ScalarId{0},
      .constant = 42.0F,
  });

  REQUIRE(program.instructions().size() == 1);
  REQUIRE(program.instructions()[0].opcode == Opcode::Constant);
  REQUIRE(program.instructions()[0].scalar_dst.value == 0);
  REQUIRE(program.instructions()[0].constant == 42.0F);
}

TEST_CASE("runtime Program add_instruction preserves append order") {
  Program program;

  program.add_instruction(Instruction{
      .opcode = Opcode::Constant,
      .scalar_dst = ScalarId{0},
      .constant = 5.0F,
  });

  program.add_instruction(Instruction{
      .opcode = Opcode::Mul,
      .scalar_dst = ScalarId{2},
      .scalar_lhs = ScalarId{0},
      .scalar_rhs = ScalarId{1},
  });

  REQUIRE(program.instructions().size() == 2);
  REQUIRE(program.instructions()[0].opcode == Opcode::Constant);
  REQUIRE(program.instructions()[1].opcode == Opcode::Mul);
}

// -----------------------------------------------------------------------------
// Mutable instructions()
// -----------------------------------------------------------------------------

TEST_CASE("runtime Program mutable instructions accessor allows instruction "
          "modification") {
  Program program{{Instruction{.opcode = Opcode::Constant,
                               .scalar_dst = ScalarId{0},
                               .constant = 1.0F}},
                  {},
                  {},
                  {},
                  1,
                  0};

  program.instructions()[0].constant = 99.0F;
  REQUIRE(program.instructions()[0].constant == 99.0F);
}

TEST_CASE("runtime Program mutable instructions accessor allows appending "
          "instruction") {
  Program program;

  auto &instructions = program.instructions();
  instructions.push_back(Instruction{
      .opcode = Opcode::Constant,
      .scalar_dst = ScalarId{0},
      .constant = 7.0F,
  });

  REQUIRE(program.instructions().size() == 1);
  REQUIRE(program.instructions()[0].constant == 7.0F);
}

TEST_CASE("runtime Program accessors work through const Program reference") {
  Program program{{Instruction{.opcode = Opcode::Constant,
                               .scalar_dst = ScalarId{0},
                               .constant = 1.0F}},
                  {input(ValueId{10}, SlotId{0}, Shape{2})},
                  {output(SlotId{1}, Shape{2})},
                  {slot(8), slot(8)},
                  1,
                  2};

  const Program &const_program = program;

  REQUIRE(const_program.instructions().size() == 1);
  REQUIRE(const_program.inputs().size() == 1);
  REQUIRE(const_program.outputs().size() == 1);
  REQUIRE(const_program.slots_count() == 2);
  REQUIRE(const_program.scalar_count() == 1);
  REQUIRE(const_program.index_count() == 2);
  REQUIRE(const_program.inputs()[0].shape == Shape{2});
  REQUIRE(const_program.outputs()[0].shape == Shape{2});
}

// -----------------------------------------------------------------------------
// Complete runtime program state
// -----------------------------------------------------------------------------

TEST_CASE("runtime Program stores complete execution metadata independently") {
  const Instruction instruction{
      .opcode = Opcode::Load,
      .scalar_dst = ScalarId{0},
      .buffer = SlotId{0},
      .offset =
          IndexOperand{.kind = IndexOperand::Kind::Constant, .constant = 0},
  };

  const Program program{{instruction},
                        {input(ValueId{10}, SlotId{0}, Shape{2, 3})},
                        {output(SlotId{1}, Shape{2, 3})},
                        {slot(24), SlotMetadata{64, 32}},
                        4,
                        3};

  REQUIRE(program.instructions().size() == 1);
  REQUIRE(program.inputs().size() == 1);
  REQUIRE(program.outputs().size() == 1);

  REQUIRE(program.slots_count() == 2);
  REQUIRE(program.scalar_count() == 4);
  REQUIRE(program.index_count() == 3);

  REQUIRE(program.instructions()[0].opcode == Opcode::Load);
  REQUIRE(program.inputs()[0].slot_id == SlotId{0});
  REQUIRE(program.inputs()[0].value_id == ValueId{10});
  REQUIRE(program.get_slotid_from_valueid(ValueId{10}) == SlotId{0});

  REQUIRE(program.outputs()[0].slot_id == SlotId{1});
  REQUIRE(program.inputs()[0].shape == Shape{2, 3});
  REQUIRE(program.outputs()[0].shape == Shape{2, 3});

  // Physical storage metadata is independent of the logical tensor metadata.
  REQUIRE(program.get_slot_metadata(SlotId{0}).capacity == 24);
  REQUIRE(program.get_slot_metadata(SlotId{1}).capacity == 64);
  REQUIRE(program.get_slot_metadata(SlotId{1}).alignment == 32);
}

// -----------------------------------------------------------------------------
// Phase 6: logical metadata and physical storage are independent
// -----------------------------------------------------------------------------

TEST_CASE(
    "runtime Program can expose same physical slot with different logical "
    "output metadata") {
  const Program program{{},
                        {},
                        {
                            output(SlotId{0}, Shape{4}),
                            output(SlotId{0}, Shape{}),
                        },
                        {slot(16)},
                        0,
                        0};

  REQUIRE(program.outputs()[0].slot_id == SlotId{0});
  REQUIRE(program.outputs()[1].slot_id == SlotId{0});
  REQUIRE(program.outputs()[0].shape == Shape{4});
  REQUIRE(program.outputs()[1].shape == Shape{});
  REQUIRE(program.get_slot_metadata(SlotId{0}).capacity == 16);
}

TEST_CASE("runtime Program identifies external input slots") {
  const Program program{{},
                        {
                            input(ValueId{10}, SlotId{0}),
                            input(ValueId{20}, SlotId{2}),
                        },
                        {},
                        {slot(4), slot(4), slot(4)},
                        0,
                        0};

  REQUIRE(program.is_input_slot(SlotId{0}));
  REQUIRE_FALSE(program.is_input_slot(SlotId{1}));
  REQUIRE(program.is_input_slot(SlotId{2}));
}
