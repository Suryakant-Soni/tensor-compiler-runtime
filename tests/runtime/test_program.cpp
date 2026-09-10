#include <catch2/catch_test_macros.hpp>

#include "tcr/ir/dtype.hpp"
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

using tcr::runtime::IndexOperand;
using tcr::runtime::Instruction;
using tcr::runtime::Opcode;
using tcr::runtime::Program;
using tcr::runtime::ScalarId;
using tcr::runtime::SlotId;
using tcr::runtime::SlotTensorInfo;

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
  const Program program{
      {},
      {},
      {},
      {
          SlotTensorInfo{
              SlotId{0},
              Shape{3},
              Dtype::F32,
          },
      },
      0,
      0,
  };

  REQUIRE(program.slots_count() == 1);

  const auto &info =
      program.tensor_meta_by_slot(SlotId{0});

  REQUIRE(info.id == SlotId{0});
  REQUIRE(info.shape == Shape{3});
  REQUIRE(info.dtype == Dtype::F32);
}

TEST_CASE("runtime Program stores metadata for multiple slots") {
  const Program program{
      {},
      {},
      {},
      {
          SlotTensorInfo{
              SlotId{0},
              Shape{2},
              Dtype::F32,
          },
          SlotTensorInfo{
              SlotId{1},
              Shape{2, 3},
              Dtype::F32,
          },
          SlotTensorInfo{
              SlotId{2},
              Shape{},
              Dtype::F32,
          },
      },
      0,
      0,
  };

  REQUIRE(program.slots_count() == 3);

  const auto &first =
      program.tensor_meta_by_slot(SlotId{0});

  const auto &second =
      program.tensor_meta_by_slot(SlotId{1});

  const auto &third =
      program.tensor_meta_by_slot(SlotId{2});

  REQUIRE(first.id == SlotId{0});
  REQUIRE(first.shape == Shape{2});
  REQUIRE(first.dtype == Dtype::F32);

  REQUIRE(second.id == SlotId{1});
  REQUIRE(second.shape == Shape{2, 3});
  REQUIRE(second.dtype == Dtype::F32);

  REQUIRE(third.id == SlotId{2});
  REQUIRE(third.shape == Shape{});
  REQUIRE(third.dtype == Dtype::F32);
}

TEST_CASE("runtime Program supports zero slots") {
  const Program program{
      {},
      {},
      {},
      {},
      0,
      0,
  };

  REQUIRE(program.slots_count() == 0);
}

TEST_CASE("runtime Program slot count comes from metadata vector size") {
  const Program program{
      {},
      {},
      {},
      {
          SlotTensorInfo{
              SlotId{0},
              Shape{1},
              Dtype::F32,
          },
          SlotTensorInfo{
              SlotId{1},
              Shape{1},
              Dtype::F32,
          },
          SlotTensorInfo{
              SlotId{2},
              Shape{1},
              Dtype::F32,
          },
          SlotTensorInfo{
              SlotId{3},
              Shape{1},
              Dtype::F32,
          },
      },
      0,
      0,
  };

  REQUIRE(program.slots_count() == 4);
}

TEST_CASE("runtime Program tensor_meta_by_slot returns metadata by slot index") {
  const Program program{
      {},
      {},
      {},
      {
          SlotTensorInfo{
              SlotId{0},
              Shape{10},
              Dtype::F32,
          },
          SlotTensorInfo{
              SlotId{1},
              Shape{20},
              Dtype::F32,
          },
      },
      0,
      0,
  };

  REQUIRE(
      program.tensor_meta_by_slot(SlotId{0}).shape ==
      Shape{10});

  REQUIRE(
      program.tensor_meta_by_slot(SlotId{1}).shape ==
      Shape{20});
}

TEST_CASE("runtime Program tensor_meta_by_slot rejects slot equal to slot count") {
  const Program program{
      {},
      {},
      {},
      {
          SlotTensorInfo{
              SlotId{0},
              Shape{1},
              Dtype::F32,
          },
      },
      0,
      0,
  };

  REQUIRE_THROWS_AS(
      program.tensor_meta_by_slot(SlotId{1}),
      std::out_of_range);
}

TEST_CASE("runtime Program tensor_meta_by_slot rejects slot greater than slot count") {
  const Program program{
      {},
      {},
      {},
      {
          SlotTensorInfo{
              SlotId{0},
              Shape{1},
              Dtype::F32,
          },
      },
      0,
      0,
  };

  REQUIRE_THROWS_AS(
      program.tensor_meta_by_slot(SlotId{100}),
      std::out_of_range);
}

TEST_CASE("runtime Program tensor_meta_by_slot rejects any slot when program has no slots") {
  const Program program;

  REQUIRE_THROWS_AS(
      program.tensor_meta_by_slot(SlotId{0}),
      std::out_of_range);
}

// -----------------------------------------------------------------------------
// Inputs
// -----------------------------------------------------------------------------

TEST_CASE("runtime Program stores input slots") {
  const Program program{
      {},
      {
          SlotId{0},
          SlotId{1},
          SlotId{3},
      },
      {},
      {
          SlotTensorInfo{SlotId{0}, Shape{1}, Dtype::F32},
          SlotTensorInfo{SlotId{1}, Shape{1}, Dtype::F32},
          SlotTensorInfo{SlotId{2}, Shape{1}, Dtype::F32},
          SlotTensorInfo{SlotId{3}, Shape{1}, Dtype::F32},
      },
      0,
      0,
  };

  REQUIRE(program.inputs().size() == 3);

  REQUIRE(program.inputs()[0] == SlotId{0});
  REQUIRE(program.inputs()[1] == SlotId{1});
  REQUIRE(program.inputs()[2] == SlotId{3});
}

TEST_CASE("runtime Program preserves input slot ordering") {
  const Program program{
      {},
      {
          SlotId{4},
          SlotId{1},
          SlotId{3},
      },
      {},
      {
          SlotTensorInfo{SlotId{0}, Shape{1}, Dtype::F32},
          SlotTensorInfo{SlotId{1}, Shape{1}, Dtype::F32},
          SlotTensorInfo{SlotId{2}, Shape{1}, Dtype::F32},
          SlotTensorInfo{SlotId{3}, Shape{1}, Dtype::F32},
          SlotTensorInfo{SlotId{4}, Shape{1}, Dtype::F32},
      },
      0,
      0,
  };

  REQUIRE(program.inputs().size() == 3);

  REQUIRE(program.inputs()[0] == SlotId{4});
  REQUIRE(program.inputs()[1] == SlotId{1});
  REQUIRE(program.inputs()[2] == SlotId{3});
}

TEST_CASE("runtime Program supports no inputs") {
  const Program program{
      {},
      {},
      {SlotId{0}},
      {
          SlotTensorInfo{
              SlotId{0},
              Shape{1},
              Dtype::F32,
          },
      },
      0,
      0,
  };

  REQUIRE(program.inputs().empty());

  REQUIRE(program.outputs().size() == 1);
  REQUIRE(program.outputs()[0] == SlotId{0});
}

// -----------------------------------------------------------------------------
// Outputs
// -----------------------------------------------------------------------------

TEST_CASE("runtime Program stores output slots") {
  const Program program{
      {},
      {},
      {
          SlotId{2},
          SlotId{5},
      },
      {
          SlotTensorInfo{SlotId{0}, Shape{1}, Dtype::F32},
          SlotTensorInfo{SlotId{1}, Shape{1}, Dtype::F32},
          SlotTensorInfo{SlotId{2}, Shape{1}, Dtype::F32},
          SlotTensorInfo{SlotId{3}, Shape{1}, Dtype::F32},
          SlotTensorInfo{SlotId{4}, Shape{1}, Dtype::F32},
          SlotTensorInfo{SlotId{5}, Shape{1}, Dtype::F32},
      },
      0,
      0,
  };

  REQUIRE(program.outputs().size() == 2);

  REQUIRE(program.outputs()[0] == SlotId{2});
  REQUIRE(program.outputs()[1] == SlotId{5});
}

TEST_CASE("runtime Program preserves output slot ordering") {
  const Program program{
      {},
      {},
      {
          SlotId{4},
          SlotId{2},
          SlotId{3},
      },
      {
          SlotTensorInfo{SlotId{0}, Shape{1}, Dtype::F32},
          SlotTensorInfo{SlotId{1}, Shape{1}, Dtype::F32},
          SlotTensorInfo{SlotId{2}, Shape{1}, Dtype::F32},
          SlotTensorInfo{SlotId{3}, Shape{1}, Dtype::F32},
          SlotTensorInfo{SlotId{4}, Shape{1}, Dtype::F32},
      },
      0,
      0,
  };

  REQUIRE(program.outputs().size() == 3);

  REQUIRE(program.outputs()[0] == SlotId{4});
  REQUIRE(program.outputs()[1] == SlotId{2});
  REQUIRE(program.outputs()[2] == SlotId{3});
}

TEST_CASE("runtime Program supports no outputs") {
  const Program program{
      {},
      {SlotId{0}},
      {},
      {
          SlotTensorInfo{
              SlotId{0},
              Shape{1},
              Dtype::F32,
          },
      },
      0,
      0,
  };

  REQUIRE(program.inputs().size() == 1);
  REQUIRE(program.outputs().empty());
}

TEST_CASE("runtime Program supports multiple inputs and outputs") {
  const Program program{
      {},
      {
          SlotId{0},
          SlotId{1},
      },
      {
          SlotId{3},
          SlotId{4},
      },
      {
          SlotTensorInfo{SlotId{0}, Shape{2}, Dtype::F32},
          SlotTensorInfo{SlotId{1}, Shape{2}, Dtype::F32},
          SlotTensorInfo{SlotId{2}, Shape{2}, Dtype::F32},
          SlotTensorInfo{SlotId{3}, Shape{2}, Dtype::F32},
          SlotTensorInfo{SlotId{4}, Shape{2}, Dtype::F32},
      },
      0,
      0,
  };

  REQUIRE(program.inputs().size() == 2);
  REQUIRE(program.outputs().size() == 2);
  REQUIRE(program.slots_count() == 5);

  REQUIRE(program.inputs()[0] == SlotId{0});
  REQUIRE(program.inputs()[1] == SlotId{1});

  REQUIRE(program.outputs()[0] == SlotId{3});
  REQUIRE(program.outputs()[1] == SlotId{4});
}

// -----------------------------------------------------------------------------
// Scalar and index register counts
// -----------------------------------------------------------------------------

TEST_CASE("runtime Program stores scalar count") {
  const Program program{
      {},
      {},
      {},
      {},
      7,
      0,
  };

  REQUIRE(program.scalar_count() == 7);
}

TEST_CASE("runtime Program supports zero scalar registers") {
  const Program program{
      {},
      {},
      {},
      {},
      0,
      0,
  };

  REQUIRE(program.scalar_count() == 0);
}

TEST_CASE("runtime Program stores index count") {
  const Program program{
      {},
      {},
      {},
      {},
      0,
      9,
  };

  REQUIRE(program.index_count() == 9);
}

TEST_CASE("runtime Program supports zero index registers") {
  const Program program{
      {},
      {},
      {},
      {},
      0,
      0,
  };

  REQUIRE(program.index_count() == 0);
}

TEST_CASE("runtime Program stores scalar and index counts independently") {
  const Program program{
      {},
      {},
      {},
      {},
      4,
      6,
  };

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

  const Program program{
      {
          first,
          second,
      },
      {},
      {},
      {},
      2,
      0,
  };

  REQUIRE(program.instructions().size() == 2);

  REQUIRE(
      program.instructions()[0].opcode ==
      Opcode::Constant);

  REQUIRE(
      program.instructions()[1].opcode ==
      Opcode::Constant);

  REQUIRE(
      program.instructions()[0].scalar_dst.value ==
      0);

  REQUIRE(
      program.instructions()[1].scalar_dst.value ==
      1);

  REQUIRE(
      program.instructions()[0].constant ==
      10.0F);

  REQUIRE(
      program.instructions()[1].constant ==
      20.0F);
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
          IndexOperand{
              .kind =
                  IndexOperand::Kind::Constant,
              .constant = 0,
          },
  };

  const Program program{
      {
          constant,
          add,
          store,
      },
      {},
      {},
      {
          SlotTensorInfo{
              SlotId{0},
              Shape{1},
              Dtype::F32,
          },
      },
      3,
      0,
  };

  REQUIRE(program.instructions().size() == 3);

  REQUIRE(
      program.instructions()[0].opcode ==
      Opcode::Constant);

  REQUIRE(
      program.instructions()[1].opcode ==
      Opcode::Add);

  REQUIRE(
      program.instructions()[2].opcode ==
      Opcode::Store);
}

// -----------------------------------------------------------------------------
// add_instruction()
// -----------------------------------------------------------------------------

TEST_CASE("runtime Program add_instruction appends instruction") {
  Program program;

  program.add_instruction(
      Instruction{
          .opcode = Opcode::Constant,
          .scalar_dst = ScalarId{0},
          .constant = 42.0F,
      });

  REQUIRE(program.instructions().size() == 1);

  REQUIRE(
      program.instructions()[0].opcode ==
      Opcode::Constant);

  REQUIRE(
      program.instructions()[0].scalar_dst.value ==
      0);

  REQUIRE(
      program.instructions()[0].constant ==
      42.0F);
}

TEST_CASE("runtime Program add_instruction preserves append order") {
  Program program;

  program.add_instruction(
      Instruction{
          .opcode = Opcode::Constant,
          .scalar_dst = ScalarId{0},
          .constant = 5.0F,
      });

  program.add_instruction(
      Instruction{
          .opcode = Opcode::Mul,
          .scalar_dst = ScalarId{2},
          .scalar_lhs = ScalarId{0},
          .scalar_rhs = ScalarId{1},
      });

  REQUIRE(program.instructions().size() == 2);

  REQUIRE(
      program.instructions()[0].opcode ==
      Opcode::Constant);

  REQUIRE(
      program.instructions()[1].opcode ==
      Opcode::Mul);
}

// -----------------------------------------------------------------------------
// Mutable instructions()
// -----------------------------------------------------------------------------

TEST_CASE("runtime Program mutable instructions accessor allows instruction modification") {
  Program program{
      {
          Instruction{
              .opcode = Opcode::Constant,
              .scalar_dst = ScalarId{0},
              .constant = 1.0F,
          },
      },
      {},
      {},
      {},
      1,
      0,
  };

  program.instructions()[0].constant =
      99.0F;

  REQUIRE(
      program.instructions()[0].constant ==
      99.0F);
}

TEST_CASE("runtime Program mutable instructions accessor allows appending instruction") {
  Program program;

  auto &instructions =
      program.instructions();

  instructions.push_back(
      Instruction{
          .opcode = Opcode::Constant,
          .scalar_dst = ScalarId{0},
          .constant = 7.0F,
      });

  REQUIRE(program.instructions().size() == 1);

  REQUIRE(
      program.instructions()[0].constant ==
      7.0F);
}

// -----------------------------------------------------------------------------
// Const accessors
// -----------------------------------------------------------------------------

TEST_CASE("runtime Program accessors work through const Program reference") {
  Program program{
      {
          Instruction{
              .opcode = Opcode::Constant,
              .scalar_dst = ScalarId{0},
              .constant = 1.0F,
          },
      },
      {
          SlotId{0},
      },
      {
          SlotId{1},
      },
      {
          SlotTensorInfo{
              SlotId{0},
              Shape{2},
              Dtype::F32,
          },
          SlotTensorInfo{
              SlotId{1},
              Shape{2},
              Dtype::F32,
          },
      },
      1,
      2,
  };

  const Program &const_program =
      program;

  REQUIRE(
      const_program.instructions().size() ==
      1);

  REQUIRE(
      const_program.inputs().size() ==
      1);

  REQUIRE(
      const_program.outputs().size() ==
      1);

  REQUIRE(
      const_program.slots_count() ==
      2);

  REQUIRE(
      const_program.scalar_count() ==
      1);

  REQUIRE(
      const_program.index_count() ==
      2);

  REQUIRE(
      const_program
              .tensor_meta_by_slot(
                  SlotId{0})
              .shape ==
      Shape{2});
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
          IndexOperand{
              .kind =
                  IndexOperand::Kind::Constant,
              .constant = 0,
          },
  };

  const Program program{
      {
          instruction,
      },
      {
          SlotId{0},
      },
      {
          SlotId{1},
      },
      {
          SlotTensorInfo{
              SlotId{0},
              Shape{2, 3},
              Dtype::F32,
          },
          SlotTensorInfo{
              SlotId{1},
              Shape{2, 3},
              Dtype::F32,
          },
      },
      4,
      3,
  };

  REQUIRE(program.instructions().size() == 1);
  REQUIRE(program.inputs().size() == 1);
  REQUIRE(program.outputs().size() == 1);

  REQUIRE(program.slots_count() == 2);
  REQUIRE(program.scalar_count() == 4);
  REQUIRE(program.index_count() == 3);

  REQUIRE(
      program.instructions()[0].opcode ==
      Opcode::Load);

  REQUIRE(
      program.inputs()[0] ==
      SlotId{0});

  REQUIRE(
      program.outputs()[0] ==
      SlotId{1});

  REQUIRE(
      program
              .tensor_meta_by_slot(
                  SlotId{0})
              .shape ==
      Shape{2, 3});

  REQUIRE(
      program
              .tensor_meta_by_slot(
                  SlotId{1})
              .shape ==
      Shape{2, 3});
}