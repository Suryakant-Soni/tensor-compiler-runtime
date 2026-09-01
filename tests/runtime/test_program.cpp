#include <catch2/catch_test_macros.hpp>

#include "tcr/runtime/opcode.hpp"
#include "tcr/runtime/program.hpp"
#include "tcr/runtime/slot_id.hpp"
#include "tcr/runtime/instruction.hpp"

#include <cstddef>
#include <vector>

using tcr::runtime::Program;
using tcr::runtime::SlotId;
using tcr::runtime::Opcode;
using tcr::runtime::Instruction;

TEST_CASE("Program stores slot count") {
  const Program program{
      {},
      {},
      {},
      5,
  };

  REQUIRE(program.slots_count() == 5);
}

TEST_CASE("Program supports zero slots") {
  const Program program{
      {},
      {},
      {},
      0,
  };

  REQUIRE(program.slots_count() == 0);
}

TEST_CASE("Program stores input slots") {
  const Program program{
      {},
      std::vector<SlotId>{
          SlotId{0},
          SlotId{1},
          SlotId{3},
      },
      {},
      4,
  };

  REQUIRE(program.inputs().size() == 3);

  REQUIRE(program.inputs()[0] == SlotId{0});
  REQUIRE(program.inputs()[1] == SlotId{1});
  REQUIRE(program.inputs()[2] == SlotId{3});
}

TEST_CASE("Program preserves input slot ordering") {
  const Program program{
      {},
      std::vector<SlotId>{
          SlotId{4},
          SlotId{1},
          SlotId{7},
      },
      {},
      8,
  };

  REQUIRE(program.inputs().size() == 3);

  REQUIRE(program.inputs()[0] == SlotId{4});
  REQUIRE(program.inputs()[1] == SlotId{1});
  REQUIRE(program.inputs()[2] == SlotId{7});
}

TEST_CASE("Program stores output slots") {
  const Program program{
      {},
      {},
      std::vector<SlotId>{
          SlotId{2},
          SlotId{5},
      },
      6,
  };

  REQUIRE(program.outputs().size() == 2);

  REQUIRE(program.outputs()[0] == SlotId{2});
  REQUIRE(program.outputs()[1] == SlotId{5});
}

TEST_CASE("Program preserves output slot ordering") {
  const Program program{
      {},
      {},
      std::vector<SlotId>{
          SlotId{6},
          SlotId{2},
          SlotId{4},
      },
      7,
  };

  REQUIRE(program.outputs().size() == 3);

  REQUIRE(program.outputs()[0] == SlotId{6});
  REQUIRE(program.outputs()[1] == SlotId{2});
  REQUIRE(program.outputs()[2] == SlotId{4});
}

TEST_CASE("Program supports multiple inputs and outputs") {
  const Program program{
      {},
      std::vector<SlotId>{
          SlotId{0},
          SlotId{1},
      },
      std::vector<SlotId>{
          SlotId{3},
          SlotId{4},
      },
      5,
  };

  REQUIRE(program.inputs().size() == 2);
  REQUIRE(program.outputs().size() == 2);
  REQUIRE(program.slots_count() == 5);

  REQUIRE(program.inputs()[0] == SlotId{0});
  REQUIRE(program.inputs()[1] == SlotId{1});

  REQUIRE(program.outputs()[0] == SlotId{3});
  REQUIRE(program.outputs()[1] == SlotId{4});
}

TEST_CASE("Program supports no inputs") {
  const Program program{
      {},
      {},
      std::vector<SlotId>{SlotId{0}},
      1,
  };

  REQUIRE(program.inputs().empty());
  REQUIRE(program.outputs().size() == 1);
  REQUIRE(program.outputs()[0] == SlotId{0});
}

TEST_CASE("Program supports no outputs") {
  const Program program{
      {},
      std::vector<SlotId>{SlotId{0}},
      {},
      1,
  };

  REQUIRE(program.inputs().size() == 1);
  REQUIRE(program.outputs().empty());
}

TEST_CASE("Program supports empty execution plan") {
  const Program program{
      {},
      {},
      {},
      0,
  };

  REQUIRE(program.instructions().empty());
  REQUIRE(program.inputs().empty());
  REQUIRE(program.outputs().empty());
  REQUIRE(program.slots_count() == 0);
}

TEST_CASE("Program supports non-consecutive slot ids") {
  const Program program{
      {},
      std::vector<SlotId>{
          SlotId{1},
          SlotId{8},
      },
      std::vector<SlotId>{
          SlotId{12},
      },
      13,
  };

  REQUIRE(program.inputs()[0] == SlotId{1});
  REQUIRE(program.inputs()[1] == SlotId{8});
  REQUIRE(program.outputs()[0] == SlotId{12});
  REQUIRE(program.slots_count() == 13);
}

TEST_CASE("Program stores instructions in execution order") {
  const Instruction mul_instruction{
      Opcode::MUL,
      {SlotId{0}, SlotId{1}},
      SlotId{2},
  };

  const Instruction sum_instruction{
      Opcode::SUM,
      {SlotId{2}},
      SlotId{3},
  };

  const Program program{
      {
          mul_instruction,
          sum_instruction,
      },
      {
          SlotId{0},
          SlotId{1},
      },
      {
          SlotId{3},
      },
      4,
  };

  REQUIRE(program.instructions().size() == 2);

  REQUIRE(program.instructions()[0].opcode == Opcode::MUL);
  REQUIRE(program.instructions()[1].opcode == Opcode::SUM);
}