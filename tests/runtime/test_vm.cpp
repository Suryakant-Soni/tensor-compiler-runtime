#include <catch2/catch_test_macros.hpp>

#include "tcr/ir/dtype.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/runtime/bindings.hpp"
#include "tcr/runtime/instruction.hpp"
#include "tcr/runtime/program.hpp"
#include "tcr/runtime/slot_id.hpp"
#include "tcr/runtime/tensor.hpp"
#include "tcr/runtime/vm.hpp"

#include <stdexcept>
#include <vector>

using tcr::ir::Dtype;
using tcr::ir::Shape;

using tcr::runtime::InputBindings;
using tcr::runtime::Instruction;
using tcr::runtime::Opcode;
using tcr::runtime::Program;
using tcr::runtime::SlotId;
using tcr::runtime::Tensor;
using tcr::runtime::VM;

// -----------------------------------------------------------------------------
// ADD
// -----------------------------------------------------------------------------

TEST_CASE("VM executes ADD instruction") {
  const Program program{
      {
          Instruction{
              Opcode::ADD,
              {SlotId{0}, SlotId{1}},
              SlotId{2},
          },
      },
      {SlotId{0}, SlotId{1}},
      {SlotId{2}},
      3,
  };

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      Tensor{
          Shape{2},
          Dtype::F32,
          std::vector<float>{1.0F, 2.0F},
      });

  bindings.bind(
      SlotId{1},
      Tensor{
          Shape{2},
          Dtype::F32,
          std::vector<float>{3.0F, 4.0F},
      });

  VM vm;

  const auto outputs = vm.run(program, bindings);

  REQUIRE(outputs.size() == 1);
  REQUIRE(outputs[0].shape() == Shape{2});
  REQUIRE(outputs[0].dtype() == Dtype::F32);
  REQUIRE(outputs[0].data() == std::vector<float>{4.0F, 6.0F});
}

// -----------------------------------------------------------------------------
// MUL
// -----------------------------------------------------------------------------

TEST_CASE("VM executes MUL instruction") {
  const Program program{
      {
          Instruction{
              Opcode::MUL,
              {SlotId{0}, SlotId{1}},
              SlotId{2},
          },
      },
      {SlotId{0}, SlotId{1}},
      {SlotId{2}},
      3,
  };

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      Tensor{
          Shape{2},
          Dtype::F32,
          std::vector<float>{2.0F, 3.0F},
      });

  bindings.bind(
      SlotId{1},
      Tensor{
          Shape{2},
          Dtype::F32,
          std::vector<float>{4.0F, 5.0F},
      });

  VM vm;

  const auto outputs = vm.run(program, bindings);

  REQUIRE(outputs.size() == 1);
  REQUIRE(outputs[0].shape() == Shape{2});
  REQUIRE(outputs[0].dtype() == Dtype::F32);
  REQUIRE(outputs[0].data() == std::vector<float>{8.0F, 15.0F});
}

// -----------------------------------------------------------------------------
// SUM
// -----------------------------------------------------------------------------

TEST_CASE("VM executes SUM instruction") {
  const Program program{
      {
          Instruction{
              Opcode::SUM,
              {SlotId{0}},
              SlotId{1},
          },
      },
      {SlotId{0}},
      {SlotId{1}},
      2,
  };

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      Tensor{
          Shape{3},
          Dtype::F32,
          std::vector<float>{1.0F, 2.0F, 3.0F},
      });

  VM vm;

  const auto outputs = vm.run(program, bindings);

  REQUIRE(outputs.size() == 1);
  REQUIRE(outputs[0].shape() == Shape{});
  REQUIRE(outputs[0].dtype() == Dtype::F32);
  REQUIRE(outputs[0].data() == std::vector<float>{6.0F});
}

// -----------------------------------------------------------------------------
// MATMUL
// -----------------------------------------------------------------------------

TEST_CASE("VM executes MATMUL instruction") {
  const Program program{
      {
          Instruction{
              Opcode::MATMUL,
              {SlotId{0}, SlotId{1}},
              SlotId{2},
          },
      },
      {SlotId{0}, SlotId{1}},
      {SlotId{2}},
      3,
  };

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      Tensor{
          Shape{2, 2},
          Dtype::F32,
          std::vector<float>{
              1.0F, 2.0F,
              3.0F, 4.0F,
          },
      });

  bindings.bind(
      SlotId{1},
      Tensor{
          Shape{2, 2},
          Dtype::F32,
          std::vector<float>{
              5.0F, 6.0F,
              7.0F, 8.0F,
          },
      });

  VM vm;

  const auto outputs = vm.run(program, bindings);

  REQUIRE(outputs.size() == 1);
  REQUIRE(outputs[0].shape() == Shape{2, 2});
  REQUIRE(outputs[0].dtype() == Dtype::F32);

  REQUIRE(outputs[0].data() ==
          std::vector<float>{
              19.0F, 22.0F,
              43.0F, 50.0F,
          });
}

// -----------------------------------------------------------------------------
// Multi-instruction execution
// -----------------------------------------------------------------------------

TEST_CASE("VM executes multiple dependent instructions in order") {
  // t = x * y
  // out = t + z

  const Program program{
      {
          Instruction{
              Opcode::MUL,
              {SlotId{0}, SlotId{1}},
              SlotId{3},
          },
          Instruction{
              Opcode::ADD,
              {SlotId{3}, SlotId{2}},
              SlotId{4},
          },
      },
      {
          SlotId{0},
          SlotId{1},
          SlotId{2},
      },
      {
          SlotId{4},
      },
      5,
  };

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      Tensor{
          Shape{2},
          Dtype::F32,
          std::vector<float>{1.0F, 2.0F},
      });

  bindings.bind(
      SlotId{1},
      Tensor{
          Shape{2},
          Dtype::F32,
          std::vector<float>{3.0F, 4.0F},
      });

  bindings.bind(
      SlotId{2},
      Tensor{
          Shape{2},
          Dtype::F32,
          std::vector<float>{5.0F, 6.0F},
      });

  VM vm;

  const auto outputs = vm.run(program, bindings);

  REQUIRE(outputs.size() == 1);

  // x * y = [3, 8]
  // t + z = [8, 14]
  REQUIRE(outputs[0].data() == std::vector<float>{8.0F, 14.0F});
}

// -----------------------------------------------------------------------------
// Multiple outputs
// -----------------------------------------------------------------------------

TEST_CASE("VM gathers multiple outputs in declared program output order") {
  const Program program{
      {
          Instruction{
              Opcode::ADD,
              {SlotId{0}, SlotId{1}},
              SlotId{2},
          },
          Instruction{
              Opcode::MUL,
              {SlotId{0}, SlotId{1}},
              SlotId{3},
          },
      },
      {
          SlotId{0},
          SlotId{1},
      },
      {
          SlotId{3},
          SlotId{2},
      },
      4,
  };

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      Tensor{
          Shape{2},
          Dtype::F32,
          std::vector<float>{2.0F, 3.0F},
      });

  bindings.bind(
      SlotId{1},
      Tensor{
          Shape{2},
          Dtype::F32,
          std::vector<float>{4.0F, 5.0F},
      });

  VM vm;

  const auto outputs = vm.run(program, bindings);

  REQUIRE(outputs.size() == 2);

  // Slot 3 is MUL.
  REQUIRE(outputs[0].data() == std::vector<float>{8.0F, 15.0F});

  // Slot 2 is ADD.
  REQUIRE(outputs[1].data() == std::vector<float>{6.0F, 8.0F});
}

// -----------------------------------------------------------------------------
// Direct input -> output
// -----------------------------------------------------------------------------

TEST_CASE("VM can return program input directly as output") {
  const Program program{
      {},
      {SlotId{0}},
      {SlotId{0}},
      1,
  };

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      Tensor{
          Shape{2},
          Dtype::F32,
          std::vector<float>{7.0F, 8.0F},
      });

  VM vm;

  const auto outputs = vm.run(program, bindings);

  REQUIRE(outputs.size() == 1);
  REQUIRE(outputs[0].shape() == Shape{2});
  REQUIRE(outputs[0].data() == std::vector<float>{7.0F, 8.0F});
}

// -----------------------------------------------------------------------------
// Empty program
// -----------------------------------------------------------------------------

TEST_CASE("VM supports program with no outputs") {
  const Program program{
      {},
      {},
      {},
      0,
  };

  const InputBindings bindings;

  VM vm;

  const auto outputs = vm.run(program, bindings);

  REQUIRE(outputs.empty());
}

// -----------------------------------------------------------------------------
// Missing binding
// -----------------------------------------------------------------------------

TEST_CASE("VM rejects missing program input binding") {
  const Program program{
      {},
      {SlotId{0}},
      {SlotId{0}},
      1,
  };

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::out_of_range);
}

// -----------------------------------------------------------------------------
// Invalid program input slot
// -----------------------------------------------------------------------------

TEST_CASE("VM rejects program input slot outside slot table") {
  const Program program{
      {},
      {SlotId{3}},
      {},
      2,
  };

  InputBindings bindings;

  bindings.bind(
      SlotId{3},
      Tensor{
          Shape{1},
          Dtype::F32,
          std::vector<float>{1.0F},
      });

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

// -----------------------------------------------------------------------------
// Invalid instruction output slot
// -----------------------------------------------------------------------------

TEST_CASE("VM rejects instruction output slot outside slot table") {
  const Program program{
      {
          Instruction{
              Opcode::ADD,
              {SlotId{0}, SlotId{1}},
              SlotId{5},
          },
      },
      {SlotId{0}, SlotId{1}},
      {},
      2,
  };

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      Tensor{
          Shape{1},
          Dtype::F32,
          std::vector<float>{1.0F},
      });

  bindings.bind(
      SlotId{1},
      Tensor{
          Shape{1},
          Dtype::F32,
          std::vector<float>{2.0F},
      });

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

// -----------------------------------------------------------------------------
// Empty runtime input slot
// -----------------------------------------------------------------------------

TEST_CASE("VM rejects instruction reading empty runtime slot") {
  const Program program{
      {
          Instruction{
              Opcode::ADD,
              {SlotId{0}, SlotId{1}},
              SlotId{2},
          },
      },
      {SlotId{0}},
      {SlotId{2}},
      3,
  };

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      Tensor{
          Shape{1},
          Dtype::F32,
          std::vector<float>{1.0F},
      });

  // Slot 1 exists in the slot table,
  // but nothing initialized or produced it.

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

// -----------------------------------------------------------------------------
// Out-of-range instruction input
// -----------------------------------------------------------------------------

TEST_CASE("VM rejects instruction input slot outside slot table") {
  const Program program{
      {
          Instruction{
              Opcode::ADD,
              {SlotId{0}, SlotId{99}},
              SlotId{1},
          },
      },
      {SlotId{0}},
      {SlotId{1}},
      2,
  };

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      Tensor{
          Shape{1},
          Dtype::F32,
          std::vector<float>{1.0F},
      });

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

// -----------------------------------------------------------------------------
// ADD arity
// -----------------------------------------------------------------------------

TEST_CASE("VM rejects ADD with too few inputs") {
  const Program program{
      {
          Instruction{
              Opcode::ADD,
              {SlotId{0}},
              SlotId{1},
          },
      },
      {SlotId{0}},
      {},
      2,
  };

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      Tensor{
          Shape{1},
          Dtype::F32,
          std::vector<float>{1.0F},
      });

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

TEST_CASE("VM rejects ADD with too many inputs") {
  const Program program{
      {
          Instruction{
              Opcode::ADD,
              {SlotId{0}, SlotId{1}, SlotId{2}},
              SlotId{3},
          },
      },
      {
          SlotId{0},
          SlotId{1},
          SlotId{2},
      },
      {},
      4,
  };

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      Tensor{
          Shape{1},
          Dtype::F32,
          std::vector<float>{1.0F},
      });

  bindings.bind(
      SlotId{1},
      Tensor{
          Shape{1},
          Dtype::F32,
          std::vector<float>{2.0F},
      });

  bindings.bind(
      SlotId{2},
      Tensor{
          Shape{1},
          Dtype::F32,
          std::vector<float>{3.0F},
      });

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

// -----------------------------------------------------------------------------
// MUL arity
// -----------------------------------------------------------------------------

TEST_CASE("VM rejects MUL with too few inputs") {
  const Program program{
      {
          Instruction{
              Opcode::MUL,
              {SlotId{0}},
              SlotId{1},
          },
      },
      {SlotId{0}},
      {},
      2,
  };

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      Tensor{
          Shape{1},
          Dtype::F32,
          std::vector<float>{1.0F},
      });

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

TEST_CASE("VM rejects MUL with too many inputs") {
  const Program program{
      {
          Instruction{
              Opcode::MUL,
              {SlotId{0}, SlotId{1}, SlotId{2}},
              SlotId{3},
          },
      },
      {
          SlotId{0},
          SlotId{1},
          SlotId{2},
      },
      {},
      4,
  };

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      Tensor{Shape{1}, Dtype::F32, std::vector<float>{1.0F}});

  bindings.bind(
      SlotId{1},
      Tensor{Shape{1}, Dtype::F32, std::vector<float>{2.0F}});

  bindings.bind(
      SlotId{2},
      Tensor{Shape{1}, Dtype::F32, std::vector<float>{3.0F}});

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

// -----------------------------------------------------------------------------
// MATMUL arity
// -----------------------------------------------------------------------------

TEST_CASE("VM rejects MATMUL with too few inputs") {
  const Program program{
      {
          Instruction{
              Opcode::MATMUL,
              {SlotId{0}},
              SlotId{1},
          },
      },
      {SlotId{0}},
      {},
      2,
  };

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      Tensor{
          Shape{1, 1},
          Dtype::F32,
          std::vector<float>{1.0F},
      });

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

TEST_CASE("VM rejects MATMUL with too many inputs") {
  const Program program{
      {
          Instruction{
              Opcode::MATMUL,
              {SlotId{0}, SlotId{1}, SlotId{2}},
              SlotId{3},
          },
      },
      {
          SlotId{0},
          SlotId{1},
          SlotId{2},
      },
      {},
      4,
  };

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      Tensor{
          Shape{1, 1},
          Dtype::F32,
          std::vector<float>{1.0F},
      });

  bindings.bind(
      SlotId{1},
      Tensor{
          Shape{1, 1},
          Dtype::F32,
          std::vector<float>{2.0F},
      });

  bindings.bind(
      SlotId{2},
      Tensor{
          Shape{1, 1},
          Dtype::F32,
          std::vector<float>{3.0F},
      });

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

// -----------------------------------------------------------------------------
// SUM arity
// -----------------------------------------------------------------------------

TEST_CASE("VM rejects SUM with zero inputs") {
  const Program program{
      {
          Instruction{
              Opcode::SUM,
              {},
              SlotId{0},
          },
      },
      {},
      {},
      1,
  };

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

TEST_CASE("VM rejects SUM with too many inputs") {
  const Program program{
      {
          Instruction{
              Opcode::SUM,
              {SlotId{0}, SlotId{1}},
              SlotId{2},
          },
      },
      {
          SlotId{0},
          SlotId{1},
      },
      {},
      3,
  };

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      Tensor{
          Shape{1},
          Dtype::F32,
          std::vector<float>{1.0F},
      });

  bindings.bind(
      SlotId{1},
      Tensor{
          Shape{1},
          Dtype::F32,
          std::vector<float>{2.0F},
      });

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

// -----------------------------------------------------------------------------
// Invalid outputs
// -----------------------------------------------------------------------------

TEST_CASE("VM rejects program output slot outside slot table") {
  const Program program{
      {},
      {},
      {SlotId{5}},
      1,
  };

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

TEST_CASE("VM rejects program output referring to empty slot") {
  const Program program{
      {},
      {},
      {SlotId{0}},
      1,
  };

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}