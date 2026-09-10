#include <catch2/catch_test_macros.hpp>

#include "tcr/ir/dtype.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/runtime/bindings.hpp"
#include "tcr/runtime/instruction.hpp"
#include "tcr/runtime/program.hpp"
#include "tcr/runtime/slot_id.hpp"
#include "tcr/runtime/tensor.hpp"
#include "tcr/runtime/vm.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

using tcr::ir::Dtype;
using tcr::ir::Shape;

using tcr::runtime::IndexId;
using tcr::runtime::IndexOperand;
using tcr::runtime::InputBindings;
using tcr::runtime::Instruction;
using tcr::runtime::Opcode;
using tcr::runtime::Program;
using tcr::runtime::ScalarId;
using tcr::runtime::SlotId;
using tcr::runtime::SlotTensorInfo;
using tcr::runtime::Tensor;
using tcr::runtime::VM;

namespace {

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

IndexOperand reg(IndexId id) {
  return IndexOperand{
      .kind = IndexOperand::Kind::Register,
      .reg = id,
  };
}

IndexOperand constant_index(std::size_t value) {
  return IndexOperand{
      .kind = IndexOperand::Kind::Constant,
      .constant = value,
  };
}

SlotTensorInfo slot(std::size_t id, Shape shape) {
  return SlotTensorInfo{
      SlotId{id},
      std::move(shape),
      Dtype::F32,
  };
}

Program make_program(
    std::vector<Instruction> instructions,
    std::vector<SlotId> inputs,
    std::vector<SlotId> outputs,
    std::vector<SlotTensorInfo> slots,
    std::size_t scalar_count,
    std::size_t index_count) {

  return Program(
      std::move(instructions),
      std::move(inputs),
      std::move(outputs),
      std::move(slots),
      scalar_count,
      index_count);
}

Tensor vector_tensor(std::vector<float> data) {
  return Tensor{
      Shape{data.size()},
      Dtype::F32,
      std::move(data),
  };
}

} // namespace

// =============================================================================
// Empty program / allocation / output gathering
// =============================================================================

TEST_CASE("VM executes completely empty program") {
  const Program program =
      make_program(
          {},
          {},
          {},
          {},
          0,
          0);

  const InputBindings bindings;

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs.empty());
}

TEST_CASE("VM allocates non-input output tensor with zero values") {
  const Program program =
      make_program(
          {},
          {},
          {SlotId{0}},
          {
              slot(0, Shape{3}),
          },
          0,
          0);

  const InputBindings bindings;

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs.size() == 1);
  REQUIRE(outputs[0].shape() == Shape{3});
  REQUIRE(outputs[0].dtype() == Dtype::F32);

  REQUIRE(
      outputs[0].data() ==
      std::vector<float>{
          0.0F,
          0.0F,
          0.0F,
      });
}

TEST_CASE("VM allocates scalar output tensor") {
  const Program program =
      make_program(
          {},
          {},
          {SlotId{0}},
          {
              slot(0, Shape{}),
          },
          0,
          0);

  const InputBindings bindings;

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs.size() == 1);
  REQUIRE(outputs[0].shape() == Shape{});
  REQUIRE(outputs[0].data() == std::vector<float>{0.0F});
}

TEST_CASE("VM allocates matrix output tensor with correct shape") {
  const Program program =
      make_program(
          {},
          {},
          {SlotId{0}},
          {
              slot(0, Shape{2, 3}),
          },
          0,
          0);

  const InputBindings bindings;

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs.size() == 1);
  REQUIRE(outputs[0].shape() == Shape{2, 3});

  REQUIRE(
      outputs[0].data() ==
      std::vector<float>{
          0.0F, 0.0F, 0.0F,
          0.0F, 0.0F, 0.0F,
      });
}

TEST_CASE("VM can return input tensor directly as output") {
  const Program program =
      make_program(
          {},
          {SlotId{0}},
          {SlotId{0}},
          {
              slot(0, Shape{3}),
          },
          0,
          0);

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      Tensor{
          Shape{3},
          Dtype::F32,
          std::vector<float>{
              4.0F,
              5.0F,
              6.0F,
          },
      });

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs.size() == 1);
  REQUIRE(outputs[0].shape() == Shape{3});

  REQUIRE(
      outputs[0].data() ==
      std::vector<float>{
          4.0F,
          5.0F,
          6.0F,
      });
}

TEST_CASE("VM gathers multiple outputs in declared order") {
  const Program program =
      make_program(
          {},
          {SlotId{0}, SlotId{1}},
          {
              SlotId{1},
              SlotId{0},
          },
          {
              slot(0, Shape{1}),
              slot(1, Shape{1}),
          },
          0,
          0);

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      vector_tensor({10.0F}));

  bindings.bind(
      SlotId{1},
      vector_tensor({20.0F}));

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs.size() == 2);

  REQUIRE(outputs[0].data() ==
          std::vector<float>{20.0F});

  REQUIRE(outputs[1].data() ==
          std::vector<float>{10.0F});
}

TEST_CASE("VM supports same slot appearing multiple times as output") {
  const Program program =
      make_program(
          {},
          {SlotId{0}},
          {
              SlotId{0},
              SlotId{0},
          },
          {
              slot(0, Shape{1}),
          },
          0,
          0);

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      vector_tensor({9.0F}));

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs.size() == 2);
  REQUIRE(outputs[0].data() == std::vector<float>{9.0F});
  REQUIRE(outputs[1].data() == std::vector<float>{9.0F});
}

// =============================================================================
// Constant
// =============================================================================

TEST_CASE("VM executes Constant and Store") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = 7.5F,
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = constant_index(0),
              },
          },
          {},
          {SlotId{0}},
          {
              slot(0, Shape{1}),
          },
          1,
          0);

  const InputBindings bindings;

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs.size() == 1);
  REQUIRE(outputs[0].load(0) == 7.5F);
}

TEST_CASE("VM Constant supports zero") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = 0.0F,
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = constant_index(0),
              },
          },
          {},
          {SlotId{0}},
          {
              slot(0, Shape{1}),
          },
          1,
          0);

  const InputBindings bindings;

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs[0].load(0) == 0.0F);
}

TEST_CASE("VM Constant supports negative values") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = -13.25F,
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = constant_index(0),
              },
          },
          {},
          {SlotId{0}},
          {
              slot(0, Shape{1}),
          },
          1,
          0);

  const InputBindings bindings;

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs[0].load(0) == -13.25F);
}

// =============================================================================
// Load / Store
// =============================================================================

TEST_CASE("VM Load reads first tensor element") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Load,
                  .scalar_dst = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = constant_index(0),
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{0},
                  .buffer = SlotId{1},
                  .offset = constant_index(0),
              },
          },
          {SlotId{0}},
          {SlotId{1}},
          {
              slot(0, Shape{3}),
              slot(1, Shape{1}),
          },
          1,
          0);

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      vector_tensor({
          10.0F,
          20.0F,
          30.0F,
      }));

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs[0].load(0) == 10.0F);
}

TEST_CASE("VM Load reads middle tensor element") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Load,
                  .scalar_dst = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = constant_index(1),
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{0},
                  .buffer = SlotId{1},
                  .offset = constant_index(0),
              },
          },
          {SlotId{0}},
          {SlotId{1}},
          {
              slot(0, Shape{3}),
              slot(1, Shape{1}),
          },
          1,
          0);

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      vector_tensor({
          10.0F,
          20.0F,
          30.0F,
      }));

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs[0].load(0) == 20.0F);
}

TEST_CASE("VM Load reads last tensor element") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Load,
                  .scalar_dst = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = constant_index(2),
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{0},
                  .buffer = SlotId{1},
                  .offset = constant_index(0),
              },
          },
          {SlotId{0}},
          {SlotId{1}},
          {
              slot(0, Shape{3}),
              slot(1, Shape{1}),
          },
          1,
          0);

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      vector_tensor({
          10.0F,
          20.0F,
          30.0F,
      }));

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs[0].load(0) == 30.0F);
}

TEST_CASE("VM Store writes into selected tensor offset") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = 55.0F,
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = constant_index(2),
              },
          },
          {},
          {SlotId{0}},
          {
              slot(0, Shape{4}),
          },
          1,
          0);

  const InputBindings bindings;

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(
      outputs[0].data() ==
      std::vector<float>{
          0.0F,
          0.0F,
          55.0F,
          0.0F,
      });
}

TEST_CASE("VM Load and Store support register offsets") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::IndexAdd,
                  .index_dst = IndexId{0},
                  .index_lhs = constant_index(1),
                  .index_rhs = constant_index(1),
              },
              Instruction{
                  .opcode = Opcode::Load,
                  .scalar_dst = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = reg(IndexId{0}),
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{0},
                  .buffer = SlotId{1},
                  .offset = reg(IndexId{0}),
              },
          },
          {SlotId{0}},
          {SlotId{1}},
          {
              slot(0, Shape{4}),
              slot(1, Shape{4}),
          },
          1,
          1);

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      vector_tensor({
          10.0F,
          20.0F,
          30.0F,
          40.0F,
      }));

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(
      outputs[0].data() ==
      std::vector<float>{
          0.0F,
          0.0F,
          30.0F,
          0.0F,
      });
}

// =============================================================================
// Add
// =============================================================================

TEST_CASE("VM Add adds two scalar registers") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = 4.0F,
              },
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{1},
                  .constant = 7.0F,
              },
              Instruction{
                  .opcode = Opcode::Add,
                  .scalar_dst = ScalarId{2},
                  .scalar_lhs = ScalarId{0},
                  .scalar_rhs = ScalarId{1},
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{2},
                  .buffer = SlotId{0},
                  .offset = constant_index(0),
              },
          },
          {},
          {SlotId{0}},
          {
              slot(0, Shape{1}),
          },
          3,
          0);

  const InputBindings bindings;

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs[0].load(0) == 11.0F);
}

TEST_CASE("VM Add supports destination equal to lhs register") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = 4.0F,
              },
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{1},
                  .constant = 3.0F,
              },
              Instruction{
                  .opcode = Opcode::Add,
                  .scalar_dst = ScalarId{0},
                  .scalar_lhs = ScalarId{0},
                  .scalar_rhs = ScalarId{1},
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = constant_index(0),
              },
          },
          {},
          {SlotId{0}},
          {
              slot(0, Shape{1}),
          },
          2,
          0);

  const InputBindings bindings;

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs[0].load(0) == 7.0F);
}

TEST_CASE("VM Add supports destination equal to rhs register") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = 6.0F,
              },
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{1},
                  .constant = 5.0F,
              },
              Instruction{
                  .opcode = Opcode::Add,
                  .scalar_dst = ScalarId{1},
                  .scalar_lhs = ScalarId{0},
                  .scalar_rhs = ScalarId{1},
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{1},
                  .buffer = SlotId{0},
                  .offset = constant_index(0),
              },
          },
          {},
          {SlotId{0}},
          {
              slot(0, Shape{1}),
          },
          2,
          0);

  const InputBindings bindings;

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs[0].load(0) == 11.0F);
}

// =============================================================================
// Mul
// =============================================================================

TEST_CASE("VM Mul multiplies two scalar registers") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = 6.0F,
              },
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{1},
                  .constant = 7.0F,
              },
              Instruction{
                  .opcode = Opcode::Mul,
                  .scalar_dst = ScalarId{2},
                  .scalar_lhs = ScalarId{0},
                  .scalar_rhs = ScalarId{1},
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{2},
                  .buffer = SlotId{0},
                  .offset = constant_index(0),
              },
          },
          {},
          {SlotId{0}},
          {
              slot(0, Shape{1}),
          },
          3,
          0);

  const InputBindings bindings;

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs[0].load(0) == 42.0F);
}

TEST_CASE("VM Mul supports destination equal to input register") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = 5.0F,
              },
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{1},
                  .constant = 8.0F,
              },
              Instruction{
                  .opcode = Opcode::Mul,
                  .scalar_dst = ScalarId{0},
                  .scalar_lhs = ScalarId{0},
                  .scalar_rhs = ScalarId{1},
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = constant_index(0),
              },
          },
          {},
          {SlotId{0}},
          {
              slot(0, Shape{1}),
          },
          2,
          0);

  const InputBindings bindings;

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs[0].load(0) == 40.0F);
}

// =============================================================================
// Combined scalar arithmetic
// =============================================================================

TEST_CASE("VM executes dependent scalar arithmetic instructions in order") {
  // (3 * 4) + 5 = 17

  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = 3.0F,
              },
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{1},
                  .constant = 4.0F,
              },
              Instruction{
                  .opcode = Opcode::Mul,
                  .scalar_dst = ScalarId{2},
                  .scalar_lhs = ScalarId{0},
                  .scalar_rhs = ScalarId{1},
              },
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{3},
                  .constant = 5.0F,
              },
              Instruction{
                  .opcode = Opcode::Add,
                  .scalar_dst = ScalarId{4},
                  .scalar_lhs = ScalarId{2},
                  .scalar_rhs = ScalarId{3},
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{4},
                  .buffer = SlotId{0},
                  .offset = constant_index(0),
              },
          },
          {},
          {SlotId{0}},
          {
              slot(0, Shape{1}),
          },
          5,
          0);

  const InputBindings bindings;

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs[0].load(0) == 17.0F);
}

// =============================================================================
// IndexAdd
// =============================================================================

TEST_CASE("VM IndexAdd supports constant plus constant") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::IndexAdd,
                  .index_dst = IndexId{0},
                  .index_lhs = constant_index(2),
                  .index_rhs = constant_index(3),
              },
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = 9.0F,
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = reg(IndexId{0}),
              },
          },
          {},
          {SlotId{0}},
          {
              slot(0, Shape{6}),
          },
          1,
          1);

  const InputBindings bindings;

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs[0].load(5) == 9.0F);
}

TEST_CASE("VM IndexAdd supports register plus constant") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::IndexAdd,
                  .index_dst = IndexId{0},
                  .index_lhs = constant_index(2),
                  .index_rhs = constant_index(0),
              },
              Instruction{
                  .opcode = Opcode::IndexAdd,
                  .index_dst = IndexId{1},
                  .index_lhs = reg(IndexId{0}),
                  .index_rhs = constant_index(3),
              },
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = 11.0F,
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = reg(IndexId{1}),
              },
          },
          {},
          {SlotId{0}},
          {
              slot(0, Shape{6}),
          },
          1,
          2);

  const InputBindings bindings;

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs[0].load(5) == 11.0F);
}

TEST_CASE("VM IndexAdd supports constant plus register") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::IndexAdd,
                  .index_dst = IndexId{0},
                  .index_lhs = constant_index(4),
                  .index_rhs = constant_index(0),
              },
              Instruction{
                  .opcode = Opcode::IndexAdd,
                  .index_dst = IndexId{1},
                  .index_lhs = constant_index(2),
                  .index_rhs = reg(IndexId{0}),
              },
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = 15.0F,
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = reg(IndexId{1}),
              },
          },
          {},
          {SlotId{0}},
          {
              slot(0, Shape{7}),
          },
          1,
          2);

  const InputBindings bindings;

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs[0].load(6) == 15.0F);
}

TEST_CASE("VM IndexAdd supports register plus register") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::IndexAdd,
                  .index_dst = IndexId{0},
                  .index_lhs = constant_index(2),
                  .index_rhs = constant_index(0),
              },
              Instruction{
                  .opcode = Opcode::IndexAdd,
                  .index_dst = IndexId{1},
                  .index_lhs = constant_index(3),
                  .index_rhs = constant_index(0),
              },
              Instruction{
                  .opcode = Opcode::IndexAdd,
                  .index_dst = IndexId{2},
                  .index_lhs = reg(IndexId{0}),
                  .index_rhs = reg(IndexId{1}),
              },
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = 21.0F,
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = reg(IndexId{2}),
              },
          },
          {},
          {SlotId{0}},
          {
              slot(0, Shape{6}),
          },
          1,
          3);

  const InputBindings bindings;

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs[0].load(5) == 21.0F);
}

// =============================================================================
// IndexMul
// =============================================================================

TEST_CASE("VM IndexMul supports constant times constant") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::IndexMul,
                  .index_dst = IndexId{0},
                  .index_lhs = constant_index(2),
                  .index_rhs = constant_index(3),
              },
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = 12.0F,
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = reg(IndexId{0}),
              },
          },
          {},
          {SlotId{0}},
          {
              slot(0, Shape{7}),
          },
          1,
          1);

  const InputBindings bindings;

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs[0].load(6) == 12.0F);
}

TEST_CASE("VM IndexMul supports register times constant") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::IndexAdd,
                  .index_dst = IndexId{0},
                  .index_lhs = constant_index(3),
                  .index_rhs = constant_index(0),
              },
              Instruction{
                  .opcode = Opcode::IndexMul,
                  .index_dst = IndexId{1},
                  .index_lhs = reg(IndexId{0}),
                  .index_rhs = constant_index(2),
              },
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = 19.0F,
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = reg(IndexId{1}),
              },
          },
          {},
          {SlotId{0}},
          {
              slot(0, Shape{7}),
          },
          1,
          2);

  const InputBindings bindings;

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs[0].load(6) == 19.0F);
}

TEST_CASE("VM IndexMul and IndexAdd can build flat matrix offset") {
  // row = 2
  // width = 4
  // col = 3
  //
  // offset = row * width + col
  //        = 2 * 4 + 3
  //        = 11

  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::IndexAdd,
                  .index_dst = IndexId{0},
                  .index_lhs = constant_index(2),
                  .index_rhs = constant_index(0),
              },
              Instruction{
                  .opcode = Opcode::IndexMul,
                  .index_dst = IndexId{1},
                  .index_lhs = reg(IndexId{0}),
                  .index_rhs = constant_index(4),
              },
              Instruction{
                  .opcode = Opcode::IndexAdd,
                  .index_dst = IndexId{2},
                  .index_lhs = reg(IndexId{1}),
                  .index_rhs = constant_index(3),
              },
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = 88.0F,
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = reg(IndexId{2}),
              },
          },
          {},
          {SlotId{0}},
          {
              slot(0, Shape{3, 4}),
          },
          1,
          3);

  const InputBindings bindings;

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs[0].load(11) == 88.0F);
}

// =============================================================================
// LoopBegin / LoopEnd
// =============================================================================

TEST_CASE("VM skips zero-iteration loop") {
  // 0 LoopBegin -> exit 3
  // 1 Constant
  // 2 LoopEnd
  // 3 Constant
  // 4 Store

  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::LoopBegin,
                  .loop_index = IndexId{0},
                  .loop_begin = constant_index(0),
                  .loop_end = constant_index(0),
                  .jump = 3,
              },
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = 100.0F,
              },
              Instruction{
                  .opcode = Opcode::LoopEnd,
                  .loop_index = IndexId{0},
                  .loop_end = constant_index(0),
                  .jump = 1,
              },
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = 7.0F,
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = constant_index(0),
              },
          },
          {},
          {SlotId{0}},
          {
              slot(0, Shape{1}),
          },
          1,
          1);

  const InputBindings bindings;

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs[0].load(0) == 7.0F);
}

TEST_CASE("VM skips loop when begin is greater than end") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::LoopBegin,
                  .loop_index = IndexId{0},
                  .loop_begin = constant_index(5),
                  .loop_end = constant_index(2),
                  .jump = 3,
              },
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = 100.0F,
              },
              Instruction{
                  .opcode = Opcode::LoopEnd,
                  .loop_index = IndexId{0},
                  .loop_end = constant_index(2),
                  .jump = 1,
              },
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = 9.0F,
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = constant_index(0),
              },
          },
          {},
          {SlotId{0}},
          {
              slot(0, Shape{1}),
          },
          1,
          1);

  const InputBindings bindings;

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs[0].load(0) == 9.0F);
}

TEST_CASE("VM executes one loop iteration") {
  // for i = 0; i < 1; ++i:
  //     output[i] = 3

  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::LoopBegin,
                  .loop_index = IndexId{0},
                  .loop_begin = constant_index(0),
                  .loop_end = constant_index(1),
                  .jump = 4,
              },
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = 3.0F,
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = reg(IndexId{0}),
              },
              Instruction{
                  .opcode = Opcode::LoopEnd,
                  .loop_index = IndexId{0},
                  .loop_end = constant_index(1),
                  .jump = 1,
              },
          },
          {},
          {SlotId{0}},
          {
              slot(0, Shape{1}),
          },
          1,
          1);

  const InputBindings bindings;

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs[0].data() ==
          std::vector<float>{3.0F});
}

TEST_CASE("VM executes multiple loop iterations") {
  // for i = 0; i < 4; ++i:
  //     output[i] = 5

  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::LoopBegin,
                  .loop_index = IndexId{0},
                  .loop_begin = constant_index(0),
                  .loop_end = constant_index(4),
                  .jump = 4,
              },
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = 5.0F,
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = reg(IndexId{0}),
              },
              Instruction{
                  .opcode = Opcode::LoopEnd,
                  .loop_index = IndexId{0},
                  .loop_end = constant_index(4),
                  .jump = 1,
              },
          },
          {},
          {SlotId{0}},
          {
              slot(0, Shape{4}),
          },
          1,
          1);

  const InputBindings bindings;

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(
      outputs[0].data() ==
      std::vector<float>{
          5.0F,
          5.0F,
          5.0F,
          5.0F,
      });
}

TEST_CASE("VM loop supports non-zero begin") {
  // Only offsets 2, 3 and 4 should be written.

  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::LoopBegin,
                  .loop_index = IndexId{0},
                  .loop_begin = constant_index(2),
                  .loop_end = constant_index(5),
                  .jump = 4,
              },
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = 6.0F,
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = reg(IndexId{0}),
              },
              Instruction{
                  .opcode = Opcode::LoopEnd,
                  .loop_index = IndexId{0},
                  .loop_end = constant_index(5),
                  .jump = 1,
              },
          },
          {},
          {SlotId{0}},
          {
              slot(0, Shape{6}),
          },
          1,
          1);

  const InputBindings bindings;

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(
      outputs[0].data() ==
      std::vector<float>{
          0.0F,
          0.0F,
          6.0F,
          6.0F,
          6.0F,
          0.0F,
      });
}

// =============================================================================
// Real elementwise loop
// =============================================================================

TEST_CASE("VM executes elementwise Add loop") {
  // for i in [0, 4):
  //     a = A[i]
  //     b = B[i]
  //     c = a + b
  //     C[i] = c

  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::LoopBegin,
                  .loop_index = IndexId{0},
                  .loop_begin = constant_index(0),
                  .loop_end = constant_index(4),
                  .jump = 7,
              },
              Instruction{
                  .opcode = Opcode::Load,
                  .scalar_dst = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = reg(IndexId{0}),
              },
              Instruction{
                  .opcode = Opcode::Load,
                  .scalar_dst = ScalarId{1},
                  .buffer = SlotId{1},
                  .offset = reg(IndexId{0}),
              },
              Instruction{
                  .opcode = Opcode::Add,
                  .scalar_dst = ScalarId{2},
                  .scalar_lhs = ScalarId{0},
                  .scalar_rhs = ScalarId{1},
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{2},
                  .buffer = SlotId{2},
                  .offset = reg(IndexId{0}),
              },
              Instruction{
                  .opcode = Opcode::LoopEnd,
                  .loop_index = IndexId{0},
                  .loop_end = constant_index(4),
                  .jump = 1,
              },
          },
          {SlotId{0}, SlotId{1}},
          {SlotId{2}},
          {
              slot(0, Shape{4}),
              slot(1, Shape{4}),
              slot(2, Shape{4}),
          },
          3,
          1);

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      vector_tensor({
          1.0F,
          2.0F,
          3.0F,
          4.0F,
      }));

  bindings.bind(
      SlotId{1},
      vector_tensor({
          10.0F,
          20.0F,
          30.0F,
          40.0F,
      }));

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(
      outputs[0].data() ==
      std::vector<float>{
          11.0F,
          22.0F,
          33.0F,
          44.0F,
      });
}

TEST_CASE("VM executes elementwise Mul loop") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::LoopBegin,
                  .loop_index = IndexId{0},
                  .loop_begin = constant_index(0),
                  .loop_end = constant_index(3),
                  .jump = 6,
              },
              Instruction{
                  .opcode = Opcode::Load,
                  .scalar_dst = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = reg(IndexId{0}),
              },
              Instruction{
                  .opcode = Opcode::Load,
                  .scalar_dst = ScalarId{1},
                  .buffer = SlotId{1},
                  .offset = reg(IndexId{0}),
              },
              Instruction{
                  .opcode = Opcode::Mul,
                  .scalar_dst = ScalarId{2},
                  .scalar_lhs = ScalarId{0},
                  .scalar_rhs = ScalarId{1},
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{2},
                  .buffer = SlotId{2},
                  .offset = reg(IndexId{0}),
              },
              Instruction{
                  .opcode = Opcode::LoopEnd,
                  .loop_index = IndexId{0},
                  .loop_end = constant_index(3),
                  .jump = 1,
              },
          },
          {SlotId{0}, SlotId{1}},
          {SlotId{2}},
          {
              slot(0, Shape{3}),
              slot(1, Shape{3}),
              slot(2, Shape{3}),
          },
          3,
          1);

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      vector_tensor({
          2.0F,
          3.0F,
          4.0F,
      }));

  bindings.bind(
      SlotId{1},
      vector_tensor({
          5.0F,
          6.0F,
          7.0F,
      }));

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(
      outputs[0].data() ==
      std::vector<float>{
          10.0F,
          18.0F,
          28.0F,
      });
}

// =============================================================================
// Reduction-style loop
// =============================================================================

TEST_CASE("VM can execute reduction accumulator loop") {
  // acc = 0
  // for i in [0,4):
  //     x = input[i]
  //     acc = acc + x
  // output[0] = acc

  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{1},
                  .constant = 0.0F,
              },
              Instruction{
                  .opcode = Opcode::LoopBegin,
                  .loop_index = IndexId{0},
                  .loop_begin = constant_index(0),
                  .loop_end = constant_index(4),
                  .jump = 5,
              },
              Instruction{
                  .opcode = Opcode::Load,
                  .scalar_dst = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = reg(IndexId{0}),
              },
              Instruction{
                  .opcode = Opcode::Add,
                  .scalar_dst = ScalarId{1},
                  .scalar_lhs = ScalarId{1},
                  .scalar_rhs = ScalarId{0},
              },
              Instruction{
                  .opcode = Opcode::LoopEnd,
                  .loop_index = IndexId{0},
                  .loop_end = constant_index(4),
                  .jump = 2,
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{1},
                  .buffer = SlotId{1},
                  .offset = constant_index(0),
              },
          },
          {SlotId{0}},
          {SlotId{1}},
          {
              slot(0, Shape{4}),
              slot(1, Shape{}),
          },
          2,
          1);

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      vector_tensor({
          1.0F,
          2.0F,
          3.0F,
          4.0F,
      }));

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs[0].shape() == Shape{});
  REQUIRE(outputs[0].load(0) == 10.0F);
}

// =============================================================================
// Nested loops
// =============================================================================

TEST_CASE("VM executes nested loops") {
  // for i in [0,2):
  //   row = i * 3
  //   for j in [0,3):
  //     offset = row + j
  //     output[offset] = 8

  const Program program =
      make_program(
          {
              // 0
              Instruction{
                  .opcode = Opcode::LoopBegin,
                  .loop_index = IndexId{0},
                  .loop_begin = constant_index(0),
                  .loop_end = constant_index(2),
                  .jump = 8,
              },

              // 1
              Instruction{
                  .opcode = Opcode::IndexMul,
                  .index_dst = IndexId{2},
                  .index_lhs = reg(IndexId{0}),
                  .index_rhs = constant_index(3),
              },

              // 2
              Instruction{
                  .opcode = Opcode::LoopBegin,
                  .loop_index = IndexId{1},
                  .loop_begin = constant_index(0),
                  .loop_end = constant_index(3),
                  .jump = 7,
              },

              // 3
              Instruction{
                  .opcode = Opcode::IndexAdd,
                  .index_dst = IndexId{3},
                  .index_lhs = reg(IndexId{2}),
                  .index_rhs = reg(IndexId{1}),
              },

              // 4
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = 8.0F,
              },

              // 5
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = reg(IndexId{3}),
              },

              // 6
              Instruction{
                  .opcode = Opcode::LoopEnd,
                  .loop_index = IndexId{1},
                  .loop_end = constant_index(3),
                  .jump = 3,
              },

              // 7
              Instruction{
                  .opcode = Opcode::LoopEnd,
                  .loop_index = IndexId{0},
                  .loop_end = constant_index(2),
                  .jump = 1,
              },
          },
          {},
          {SlotId{0}},
          {
              slot(0, Shape{2, 3}),
          },
          1,
          4);

  const InputBindings bindings;

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(
      outputs[0].data() ==
      std::vector<float>{
          8.0F, 8.0F, 8.0F,
          8.0F, 8.0F, 8.0F,
      });
}

// =============================================================================
// Register-valued loop bounds
// =============================================================================

TEST_CASE("VM LoopBegin supports register begin and end operands") {
  const Program program =
      make_program(
          {
              // begin = 1
              Instruction{
                  .opcode = Opcode::IndexAdd,
                  .index_dst = IndexId{1},
                  .index_lhs = constant_index(1),
                  .index_rhs = constant_index(0),
              },

              // end = 4
              Instruction{
                  .opcode = Opcode::IndexAdd,
                  .index_dst = IndexId{2},
                  .index_lhs = constant_index(4),
                  .index_rhs = constant_index(0),
              },

              Instruction{
                  .opcode = Opcode::LoopBegin,
                  .loop_index = IndexId{0},
                  .loop_begin = reg(IndexId{1}),
                  .loop_end = reg(IndexId{2}),
                  .jump = 6,
              },

              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = 4.0F,
              },

              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = reg(IndexId{0}),
              },

              Instruction{
                  .opcode = Opcode::LoopEnd,
                  .loop_index = IndexId{0},
                  .loop_end = reg(IndexId{2}),
                  .jump = 3,
              },
          },
          {},
          {SlotId{0}},
          {
              slot(0, Shape{5}),
          },
          1,
          3);

  const InputBindings bindings;

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(
      outputs[0].data() ==
      std::vector<float>{
          0.0F,
          4.0F,
          4.0F,
          4.0F,
          0.0F,
      });
}

// =============================================================================
// Input and output allocation together
// =============================================================================

TEST_CASE("VM keeps bound inputs and allocates intermediate and output slots") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Load,
                  .scalar_dst = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = constant_index(0),
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{0},
                  .buffer = SlotId{1},
                  .offset = constant_index(0),
              },
              Instruction{
                  .opcode = Opcode::Load,
                  .scalar_dst = ScalarId{1},
                  .buffer = SlotId{1},
                  .offset = constant_index(0),
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{1},
                  .buffer = SlotId{2},
                  .offset = constant_index(0),
              },
          },
          {SlotId{0}},
          {SlotId{2}},
          {
              slot(0, Shape{1}),
              slot(1, Shape{1}),
              slot(2, Shape{1}),
          },
          2,
          0);

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      vector_tensor({37.0F}));

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs.size() == 1);
  REQUIRE(outputs[0].load(0) == 37.0F);
}

// =============================================================================
// Missing / invalid input slots
// =============================================================================

TEST_CASE("VM rejects missing program input binding") {
  const Program program =
      make_program(
          {},
          {SlotId{0}},
          {},
          {
              slot(0, Shape{1}),
          },
          0,
          0);

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::out_of_range);
}

TEST_CASE("VM rejects program input slot outside slot table") {
  const Program program =
      make_program(
          {},
          {SlotId{3}},
          {},
          {
              slot(0, Shape{1}),
              slot(1, Shape{1}),
          },
          0,
          0);

  InputBindings bindings;

  bindings.bind(
      SlotId{3},
      vector_tensor({1.0F}));

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

// =============================================================================
// Invalid scalar registers
// =============================================================================

TEST_CASE("VM rejects Constant destination scalar outside register table") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{1},
                  .constant = 1.0F,
              },
          },
          {},
          {},
          {},
          1,
          0);

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

TEST_CASE("VM rejects Load destination scalar outside register table") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Load,
                  .scalar_dst = ScalarId{2},
                  .buffer = SlotId{0},
                  .offset = constant_index(0),
              },
          },
          {SlotId{0}},
          {},
          {
              slot(0, Shape{1}),
          },
          1,
          0);

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      vector_tensor({1.0F}));

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

TEST_CASE("VM rejects Store source scalar outside register table") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{1},
                  .buffer = SlotId{0},
                  .offset = constant_index(0),
              },
          },
          {},
          {},
          {
              slot(0, Shape{1}),
          },
          1,
          0);

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

TEST_CASE("VM rejects Add destination scalar outside register table") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Add,
                  .scalar_dst = ScalarId{3},
                  .scalar_lhs = ScalarId{0},
                  .scalar_rhs = ScalarId{1},
              },
          },
          {},
          {},
          {},
          2,
          0);

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

TEST_CASE("VM rejects Add lhs scalar outside register table") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Add,
                  .scalar_dst = ScalarId{0},
                  .scalar_lhs = ScalarId{5},
                  .scalar_rhs = ScalarId{0},
              },
          },
          {},
          {},
          {},
          1,
          0);

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

TEST_CASE("VM rejects Add rhs scalar outside register table") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Add,
                  .scalar_dst = ScalarId{0},
                  .scalar_lhs = ScalarId{0},
                  .scalar_rhs = ScalarId{5},
              },
          },
          {},
          {},
          {},
          1,
          0);

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

TEST_CASE("VM rejects Mul destination scalar outside register table") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Mul,
                  .scalar_dst = ScalarId{4},
                  .scalar_lhs = ScalarId{0},
                  .scalar_rhs = ScalarId{1},
              },
          },
          {},
          {},
          {},
          2,
          0);

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

TEST_CASE("VM rejects Mul lhs scalar outside register table") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Mul,
                  .scalar_dst = ScalarId{0},
                  .scalar_lhs = ScalarId{2},
                  .scalar_rhs = ScalarId{0},
              },
          },
          {},
          {},
          {},
          1,
          0);

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

TEST_CASE("VM rejects Mul rhs scalar outside register table") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Mul,
                  .scalar_dst = ScalarId{0},
                  .scalar_lhs = ScalarId{0},
                  .scalar_rhs = ScalarId{2},
              },
          },
          {},
          {},
          {},
          1,
          0);

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

// =============================================================================
// Invalid tensor slots
// =============================================================================

TEST_CASE("VM rejects Load buffer outside slot table") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Load,
                  .scalar_dst = ScalarId{0},
                  .buffer = SlotId{4},
                  .offset = constant_index(0),
              },
          },
          {},
          {},
          {
              slot(0, Shape{1}),
          },
          1,
          0);

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

TEST_CASE("VM rejects Store buffer outside slot table") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = 1.0F,
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{0},
                  .buffer = SlotId{3},
                  .offset = constant_index(0),
              },
          },
          {},
          {},
          {
              slot(0, Shape{1}),
          },
          1,
          0);

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

// =============================================================================
// Invalid tensor offsets
// =============================================================================

TEST_CASE("VM rejects Load offset equal to tensor size") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Load,
                  .scalar_dst = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = constant_index(3),
              },
          },
          {SlotId{0}},
          {},
          {
              slot(0, Shape{3}),
          },
          1,
          0);

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      vector_tensor({
          1.0F,
          2.0F,
          3.0F,
      }));

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

TEST_CASE("VM rejects Load offset greater than tensor size") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Load,
                  .scalar_dst = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = constant_index(100),
              },
          },
          {SlotId{0}},
          {},
          {
              slot(0, Shape{3}),
          },
          1,
          0);

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      vector_tensor({
          1.0F,
          2.0F,
          3.0F,
      }));

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

TEST_CASE("VM rejects Store offset equal to tensor size") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = 5.0F,
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = constant_index(2),
              },
          },
          {},
          {},
          {
              slot(0, Shape{2}),
          },
          1,
          0);

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

// =============================================================================
// Invalid index operands
// =============================================================================

TEST_CASE("VM rejects Load register offset outside index table") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Load,
                  .scalar_dst = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = reg(IndexId{3}),
              },
          },
          {SlotId{0}},
          {},
          {
              slot(0, Shape{1}),
          },
          1,
          1);

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      vector_tensor({1.0F}));

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::out_of_range);
}

TEST_CASE("VM rejects Store register offset outside index table") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Constant,
                  .scalar_dst = ScalarId{0},
                  .constant = 1.0F,
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = reg(IndexId{4}),
              },
          },
          {},
          {},
          {
              slot(0, Shape{1}),
          },
          1,
          1);

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::out_of_range);
}

TEST_CASE("VM rejects IndexAdd destination outside index table") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::IndexAdd,
                  .index_dst = IndexId{1},
                  .index_lhs = constant_index(1),
                  .index_rhs = constant_index(2),
              },
          },
          {},
          {},
          {},
          0,
          1);

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

TEST_CASE("VM rejects IndexAdd lhs register outside index table") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::IndexAdd,
                  .index_dst = IndexId{0},
                  .index_lhs = reg(IndexId{3}),
                  .index_rhs = constant_index(1),
              },
          },
          {},
          {},
          {},
          0,
          1);

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::out_of_range);
}

TEST_CASE("VM rejects IndexAdd rhs register outside index table") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::IndexAdd,
                  .index_dst = IndexId{0},
                  .index_lhs = constant_index(1),
                  .index_rhs = reg(IndexId{3}),
              },
          },
          {},
          {},
          {},
          0,
          1);

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::out_of_range);
}

TEST_CASE("VM rejects IndexMul destination outside index table") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::IndexMul,
                  .index_dst = IndexId{2},
                  .index_lhs = constant_index(2),
                  .index_rhs = constant_index(3),
              },
          },
          {},
          {},
          {},
          0,
          1);

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

TEST_CASE("VM rejects IndexMul lhs register outside index table") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::IndexMul,
                  .index_dst = IndexId{0},
                  .index_lhs = reg(IndexId{4}),
                  .index_rhs = constant_index(2),
              },
          },
          {},
          {},
          {},
          0,
          1);

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::out_of_range);
}

TEST_CASE("VM rejects IndexMul rhs register outside index table") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::IndexMul,
                  .index_dst = IndexId{0},
                  .index_lhs = constant_index(2),
                  .index_rhs = reg(IndexId{4}),
              },
          },
          {},
          {},
          {},
          0,
          1);

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::out_of_range);
}

// =============================================================================
// Invalid loop registers / operands
// =============================================================================

TEST_CASE("VM rejects LoopBegin loop index outside index table") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::LoopBegin,
                  .loop_index = IndexId{2},
                  .loop_begin = constant_index(0),
                  .loop_end = constant_index(1),
                  .jump = 1,
              },
          },
          {},
          {},
          {},
          0,
          1);

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

TEST_CASE("VM rejects LoopBegin begin register outside index table") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::LoopBegin,
                  .loop_index = IndexId{0},
                  .loop_begin = reg(IndexId{3}),
                  .loop_end = constant_index(1),
                  .jump = 1,
              },
          },
          {},
          {},
          {},
          0,
          1);

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::out_of_range);
}

TEST_CASE("VM rejects LoopBegin end register outside index table") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::LoopBegin,
                  .loop_index = IndexId{0},
                  .loop_begin = constant_index(0),
                  .loop_end = reg(IndexId{4}),
                  .jump = 1,
              },
          },
          {},
          {},
          {},
          0,
          1);

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::out_of_range);
}

TEST_CASE("VM rejects LoopEnd loop index outside index table") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::LoopEnd,
                  .loop_index = IndexId{5},
                  .loop_end = constant_index(1),
                  .jump = 0,
              },
          },
          {},
          {},
          {},
          0,
          1);

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

TEST_CASE("VM rejects LoopEnd end register outside index table") {
  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::LoopEnd,
                  .loop_index = IndexId{0},
                  .loop_end = reg(IndexId{4}),
                  .jump = 0,
              },
          },
          {},
          {},
          {},
          0,
          1);

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::out_of_range);
}

// =============================================================================
// Invalid outputs
// =============================================================================

TEST_CASE("VM rejects output slot outside slot table") {
  const Program program =
      make_program(
          {},
          {},
          {SlotId{4}},
          {
              slot(0, Shape{1}),
          },
          0,
          0);

  const InputBindings bindings;

  VM vm;

  REQUIRE_THROWS_AS(
      vm.run(program, bindings),
      std::logic_error);
}

// =============================================================================
// Larger mixed execution
// =============================================================================

TEST_CASE("VM executes mixed Load Mul Add Store sequence") {
  // output = (a * b) + c

  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Load,
                  .scalar_dst = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = constant_index(0),
              },
              Instruction{
                  .opcode = Opcode::Load,
                  .scalar_dst = ScalarId{1},
                  .buffer = SlotId{1},
                  .offset = constant_index(0),
              },
              Instruction{
                  .opcode = Opcode::Mul,
                  .scalar_dst = ScalarId{2},
                  .scalar_lhs = ScalarId{0},
                  .scalar_rhs = ScalarId{1},
              },
              Instruction{
                  .opcode = Opcode::Load,
                  .scalar_dst = ScalarId{3},
                  .buffer = SlotId{2},
                  .offset = constant_index(0),
              },
              Instruction{
                  .opcode = Opcode::Add,
                  .scalar_dst = ScalarId{4},
                  .scalar_lhs = ScalarId{2},
                  .scalar_rhs = ScalarId{3},
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{4},
                  .buffer = SlotId{3},
                  .offset = constant_index(0),
              },
          },
          {
              SlotId{0},
              SlotId{1},
              SlotId{2},
          },
          {SlotId{3}},
          {
              slot(0, Shape{1}),
              slot(1, Shape{1}),
              slot(2, Shape{1}),
              slot(3, Shape{1}),
          },
          5,
          0);

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      vector_tensor({3.0F}));

  bindings.bind(
      SlotId{1},
      vector_tensor({4.0F}));

  bindings.bind(
      SlotId{2},
      vector_tensor({5.0F}));

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs[0].load(0) == 17.0F);
}

TEST_CASE("VM executes two independent outputs") {
  // output0 = a + b
  // output1 = a * b

  const Program program =
      make_program(
          {
              Instruction{
                  .opcode = Opcode::Load,
                  .scalar_dst = ScalarId{0},
                  .buffer = SlotId{0},
                  .offset = constant_index(0),
              },
              Instruction{
                  .opcode = Opcode::Load,
                  .scalar_dst = ScalarId{1},
                  .buffer = SlotId{1},
                  .offset = constant_index(0),
              },
              Instruction{
                  .opcode = Opcode::Add,
                  .scalar_dst = ScalarId{2},
                  .scalar_lhs = ScalarId{0},
                  .scalar_rhs = ScalarId{1},
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{2},
                  .buffer = SlotId{2},
                  .offset = constant_index(0),
              },
              Instruction{
                  .opcode = Opcode::Mul,
                  .scalar_dst = ScalarId{3},
                  .scalar_lhs = ScalarId{0},
                  .scalar_rhs = ScalarId{1},
              },
              Instruction{
                  .opcode = Opcode::Store,
                  .scalar_lhs = ScalarId{3},
                  .buffer = SlotId{3},
                  .offset = constant_index(0),
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
          {
              slot(0, Shape{1}),
              slot(1, Shape{1}),
              slot(2, Shape{1}),
              slot(3, Shape{1}),
          },
          4,
          0);

  InputBindings bindings;

  bindings.bind(
      SlotId{0},
      vector_tensor({3.0F}));

  bindings.bind(
      SlotId{1},
      vector_tensor({5.0F}));

  VM vm;

  const auto outputs =
      vm.run(program, bindings);

  REQUIRE(outputs.size() == 2);

  // Declared output order is product first.
  REQUIRE(outputs[0].load(0) == 15.0F);

  // Sum second.
  REQUIRE(outputs[1].load(0) == 8.0F);
}
