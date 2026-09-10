#include <catch2/catch_test_macros.hpp>

#include "tcr/compiler/program_emitter.hpp"
#include "tcr/ir/dtype.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/lir/index.hpp"
#include "tcr/lir/instruction.hpp"
#include "tcr/lir/program.hpp"
#include "tcr/runtime/instruction.hpp"
#include "tcr/runtime/opcode.hpp"
#include "tcr/runtime/program.hpp"
#include "tcr/runtime/slot_id.hpp"

#include <cstddef>
#include <memory>
#include <vector>

using tcr::compiler::ProgramEmitter;

using tcr::ir::Dtype;
using tcr::ir::Shape;
using tcr::ir::ValueId;

namespace lir = tcr::lir;
namespace runtime = tcr::runtime;

// -----------------------------------------------------------------------------
// Empty program
// -----------------------------------------------------------------------------

TEST_CASE("ProgramEmitter emits empty runtime Program from empty LIR Program") {
  lir::Program lir_program;

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  REQUIRE(rt_program.instructions().empty());
  REQUIRE(rt_program.inputs().empty());
  REQUIRE(rt_program.outputs().empty());

  REQUIRE(rt_program.slots_count() == 0);
  REQUIRE(rt_program.scalar_count() == 0);
  REQUIRE(rt_program.index_count() == 0);
}

// -----------------------------------------------------------------------------
// Buffer -> Slot mapping
// -----------------------------------------------------------------------------

TEST_CASE("ProgramEmitter creates one runtime slot per LIR buffer") {
  lir::Program lir_program;

  lir_program.create_buffer(
      Shape{3},
      Dtype::F32);

  lir_program.create_buffer(
      Shape{2, 4},
      Dtype::F32);

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  REQUIRE(rt_program.slots_count() == 2);
}

TEST_CASE("ProgramEmitter maps BufferId to same-numbered SlotId") {
  lir::Program lir_program;

  lir_program.create_buffer(
      Shape{2},
      Dtype::F32);

  lir_program.create_buffer(
      Shape{3},
      Dtype::F32);

  lir_program.create_buffer(
      Shape{4},
      Dtype::F32);

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  REQUIRE(
      rt_program.tensor_meta_by_slot(
          runtime::SlotId{0}).id ==
      runtime::SlotId{0});

  REQUIRE(
      rt_program.tensor_meta_by_slot(
          runtime::SlotId{1}).id ==
      runtime::SlotId{1});

  REQUIRE(
      rt_program.tensor_meta_by_slot(
          runtime::SlotId{2}).id ==
      runtime::SlotId{2});
}

TEST_CASE("ProgramEmitter preserves buffer shape metadata in runtime slots") {
  lir::Program lir_program;

  lir_program.create_buffer(
      Shape{2},
      Dtype::F32);

  lir_program.create_buffer(
      Shape{2, 3},
      Dtype::F32);

  lir_program.create_buffer(
      Shape{},
      Dtype::F32);

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  REQUIRE(
      rt_program.tensor_meta_by_slot(
          runtime::SlotId{0}).shape ==
      Shape{2});

  REQUIRE(
      rt_program.tensor_meta_by_slot(
          runtime::SlotId{1}).shape ==
      Shape{2, 3});

  REQUIRE(
      rt_program.tensor_meta_by_slot(
          runtime::SlotId{2}).shape ==
      Shape{});
}

TEST_CASE("ProgramEmitter preserves buffer dtype metadata in runtime slots") {
  lir::Program lir_program;

  lir_program.create_buffer(
      Shape{3},
      Dtype::F32);

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  REQUIRE(
      rt_program.tensor_meta_by_slot(
          runtime::SlotId{0}).dtype ==
      Dtype::F32);
}

// -----------------------------------------------------------------------------
// Inputs
// -----------------------------------------------------------------------------

TEST_CASE("ProgramEmitter converts LIR inputs to runtime slots") {
  lir::Program lir_program;

  const auto first =
      lir_program.create_buffer(
          Shape{2},
          Dtype::F32);

  const auto second =
      lir_program.create_buffer(
          Shape{2},
          Dtype::F32);

  lir_program.add_input(
      ValueId{10},
      first);

  lir_program.add_input(
      ValueId{20},
      second);

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  REQUIRE(rt_program.inputs().size() == 2);

  REQUIRE(
      rt_program.inputs()[0] ==
      runtime::SlotId{0});

  REQUIRE(
      rt_program.inputs()[1] ==
      runtime::SlotId{1});
}

TEST_CASE("ProgramEmitter preserves LIR input ordering") {
  lir::Program lir_program;

  const auto first =
      lir_program.create_buffer(
          Shape{1},
          Dtype::F32);

  const auto second =
      lir_program.create_buffer(
          Shape{1},
          Dtype::F32);

  const auto third =
      lir_program.create_buffer(
          Shape{1},
          Dtype::F32);

  lir_program.add_input(
      ValueId{0},
      third);

  lir_program.add_input(
      ValueId{1},
      first);

  lir_program.add_input(
      ValueId{2},
      second);

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  REQUIRE(rt_program.inputs().size() == 3);

  REQUIRE(
      rt_program.inputs()[0] ==
      runtime::SlotId{2});

  REQUIRE(
      rt_program.inputs()[1] ==
      runtime::SlotId{0});

  REQUIRE(
      rt_program.inputs()[2] ==
      runtime::SlotId{1});
}

// -----------------------------------------------------------------------------
// Outputs
// -----------------------------------------------------------------------------

TEST_CASE("ProgramEmitter converts LIR outputs to runtime slots") {
  lir::Program lir_program;

  const auto first =
      lir_program.create_buffer(
          Shape{2},
          Dtype::F32);

  const auto second =
      lir_program.create_buffer(
          Shape{2},
          Dtype::F32);

  lir_program.add_output(second);
  lir_program.add_output(first);

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  REQUIRE(rt_program.outputs().size() == 2);

  REQUIRE(
      rt_program.outputs()[0] ==
      runtime::SlotId{1});

  REQUIRE(
      rt_program.outputs()[1] ==
      runtime::SlotId{0});
}

// -----------------------------------------------------------------------------
// Scalar / index count propagation
// -----------------------------------------------------------------------------

TEST_CASE("ProgramEmitter preserves scalar register count") {
  lir::Program lir_program;

  lir_program.create_scalar();
  lir_program.create_scalar();
  lir_program.create_scalar();

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  REQUIRE(rt_program.scalar_count() == 3);
}

TEST_CASE("ProgramEmitter preserves index register count") {
  lir::Program lir_program;

  lir_program.create_index();
  lir_program.create_index();

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  REQUIRE(rt_program.index_count() == 2);
}

TEST_CASE("ProgramEmitter preserves scalar and index counts independently") {
  lir::Program lir_program;

  for (std::size_t i = 0; i < 4; ++i) {
    lir_program.create_scalar();
  }

  for (std::size_t i = 0; i < 7; ++i) {
    lir_program.create_index();
  }

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  REQUIRE(rt_program.scalar_count() == 4);
  REQUIRE(rt_program.index_count() == 7);
}

// -----------------------------------------------------------------------------
// Constant
// -----------------------------------------------------------------------------

TEST_CASE("ProgramEmitter translates LIR Constant") {
  lir::Program lir_program;

  const auto scalar =
      lir_program.create_scalar();

  lir_program.add_instruction(
      lir::Constant{
          scalar,
          42.5F,
      });

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  REQUIRE(rt_program.instructions().size() == 1);

  const auto &instruction =
      rt_program.instructions()[0];

  REQUIRE(
      instruction.opcode ==
      runtime::Opcode::Constant);

  REQUIRE(
      instruction.scalar_dst.value ==
      scalar.value);

  REQUIRE(instruction.constant == 42.5F);
}

// -----------------------------------------------------------------------------
// Load
// -----------------------------------------------------------------------------

TEST_CASE("ProgramEmitter translates LIR Load with register offset") {
  lir::Program lir_program;

  const auto buffer =
      lir_program.create_buffer(
          Shape{4},
          Dtype::F32);

  const auto scalar =
      lir_program.create_scalar();

  const auto index =
      lir_program.create_index();

  lir_program.add_instruction(
      lir::Load(
          scalar,
          buffer,
          index));

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  REQUIRE(rt_program.instructions().size() == 1);

  const auto &instruction =
      rt_program.instructions()[0];

  REQUIRE(
      instruction.opcode ==
      runtime::Opcode::Load);

  REQUIRE(
      instruction.scalar_dst.value ==
      scalar.value);

  REQUIRE(
      instruction.buffer ==
      runtime::SlotId{buffer.value});

  REQUIRE(
      instruction.offset.kind ==
      runtime::IndexOperand::Kind::Register);

  REQUIRE(
      instruction.offset.reg.value ==
      index.value);
}

TEST_CASE("ProgramEmitter translates LIR Load with constant offset") {
  lir::Program lir_program;

  const auto buffer =
      lir_program.create_buffer(
          Shape{8},
          Dtype::F32);

  const auto scalar =
      lir_program.create_scalar();

  lir_program.add_instruction(
      lir::Load(
          scalar,
          buffer,
          lir::ConstantIndex{5}));

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  const auto &instruction =
      rt_program.instructions()[0];

  REQUIRE(
      instruction.offset.kind ==
      runtime::IndexOperand::Kind::Constant);

  REQUIRE(
      instruction.offset.constant ==
      5);
}

// -----------------------------------------------------------------------------
// Store
// -----------------------------------------------------------------------------

TEST_CASE("ProgramEmitter translates LIR Store with register offset") {
  lir::Program lir_program;

  const auto buffer =
      lir_program.create_buffer(
          Shape{4},
          Dtype::F32);

  const auto scalar =
      lir_program.create_scalar();

  const auto index =
      lir_program.create_index();

  lir_program.add_instruction(
      lir::Store(
          scalar,
          buffer,
          index));

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  REQUIRE(rt_program.instructions().size() == 1);

  const auto &instruction =
      rt_program.instructions()[0];

  REQUIRE(
      instruction.opcode ==
      runtime::Opcode::Store);

  REQUIRE(
      instruction.scalar_lhs.value ==
      scalar.value);

  REQUIRE(
      instruction.buffer ==
      runtime::SlotId{buffer.value});

  REQUIRE(
      instruction.offset.kind ==
      runtime::IndexOperand::Kind::Register);

  REQUIRE(
      instruction.offset.reg.value ==
      index.value);
}

TEST_CASE("ProgramEmitter translates LIR Store with constant offset") {
  lir::Program lir_program;

  const auto buffer =
      lir_program.create_buffer(
          Shape{4},
          Dtype::F32);

  const auto scalar =
      lir_program.create_scalar();

  lir_program.add_instruction(
      lir::Store(
          scalar,
          buffer,
          lir::ConstantIndex{3}));

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  const auto &instruction =
      rt_program.instructions()[0];

  REQUIRE(
      instruction.offset.kind ==
      runtime::IndexOperand::Kind::Constant);

  REQUIRE(
      instruction.offset.constant ==
      3);
}

// -----------------------------------------------------------------------------
// Add
// -----------------------------------------------------------------------------

TEST_CASE("ProgramEmitter translates LIR Add") {
  lir::Program lir_program;

  const auto lhs =
      lir_program.create_scalar();

  const auto rhs =
      lir_program.create_scalar();

  const auto dst =
      lir_program.create_scalar();

  lir_program.add_instruction(
      lir::Add(
          lhs,
          rhs,
          dst));

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  REQUIRE(rt_program.instructions().size() == 1);

  const auto &instruction =
      rt_program.instructions()[0];

  REQUIRE(
      instruction.opcode ==
      runtime::Opcode::Add);

  REQUIRE(
      instruction.scalar_lhs.value ==
      lhs.value);

  REQUIRE(
      instruction.scalar_rhs.value ==
      rhs.value);

  REQUIRE(
      instruction.scalar_dst.value ==
      dst.value);
}

// -----------------------------------------------------------------------------
// Mul
// -----------------------------------------------------------------------------

TEST_CASE("ProgramEmitter translates LIR Mul") {
  lir::Program lir_program;

  const auto lhs =
      lir_program.create_scalar();

  const auto rhs =
      lir_program.create_scalar();

  const auto dst =
      lir_program.create_scalar();

  lir_program.add_instruction(
      lir::Mul(
          lhs,
          rhs,
          dst));

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  REQUIRE(rt_program.instructions().size() == 1);

  const auto &instruction =
      rt_program.instructions()[0];

  REQUIRE(
      instruction.opcode ==
      runtime::Opcode::Mul);

  REQUIRE(
      instruction.scalar_lhs.value ==
      lhs.value);

  REQUIRE(
      instruction.scalar_rhs.value ==
      rhs.value);

  REQUIRE(
      instruction.scalar_dst.value ==
      dst.value);
}

// -----------------------------------------------------------------------------
// IndexAdd
// -----------------------------------------------------------------------------

TEST_CASE("ProgramEmitter translates IndexAdd with register operands") {
  lir::Program lir_program;

  const auto lhs =
      lir_program.create_index();

  const auto rhs =
      lir_program.create_index();

  const auto dst =
      lir_program.create_index();

  lir_program.add_instruction(
      lir::IndexAdd(
          lhs,
          rhs,
          dst));

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  REQUIRE(rt_program.instructions().size() == 1);

  const auto &instruction =
      rt_program.instructions()[0];

  REQUIRE(
      instruction.opcode ==
      runtime::Opcode::IndexAdd);

  REQUIRE(
      instruction.index_dst.value ==
      dst.value);

  REQUIRE(
      instruction.index_lhs.kind ==
      runtime::IndexOperand::Kind::Register);

  REQUIRE(
      instruction.index_lhs.reg.value ==
      lhs.value);

  REQUIRE(
      instruction.index_rhs.kind ==
      runtime::IndexOperand::Kind::Register);

  REQUIRE(
      instruction.index_rhs.reg.value ==
      rhs.value);
}

TEST_CASE("ProgramEmitter translates IndexAdd with constant operand") {
  lir::Program lir_program;

  const auto lhs =
      lir_program.create_index();

  const auto dst =
      lir_program.create_index();

  lir_program.add_instruction(
      lir::IndexAdd(
          lhs,
          lir::ConstantIndex{7},
          dst));

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  const auto &instruction =
      rt_program.instructions()[0];

  REQUIRE(
      instruction.index_lhs.kind ==
      runtime::IndexOperand::Kind::Register);

  REQUIRE(
      instruction.index_lhs.reg.value ==
      lhs.value);

  REQUIRE(
      instruction.index_rhs.kind ==
      runtime::IndexOperand::Kind::Constant);

  REQUIRE(
      instruction.index_rhs.constant ==
      7);

  REQUIRE(
      instruction.index_dst.value ==
      dst.value);
}

// -----------------------------------------------------------------------------
// IndexMul
// -----------------------------------------------------------------------------

TEST_CASE("ProgramEmitter translates IndexMul with register operands") {
  lir::Program lir_program;

  const auto lhs =
      lir_program.create_index();

  const auto rhs =
      lir_program.create_index();

  const auto dst =
      lir_program.create_index();

  lir_program.add_instruction(
      lir::IndexMul(
          lhs,
          rhs,
          dst));

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  REQUIRE(rt_program.instructions().size() == 1);

  const auto &instruction =
      rt_program.instructions()[0];

  REQUIRE(
      instruction.opcode ==
      runtime::Opcode::IndexMul);

  REQUIRE(
      instruction.index_lhs.reg.value ==
      lhs.value);

  REQUIRE(
      instruction.index_rhs.reg.value ==
      rhs.value);

  REQUIRE(
      instruction.index_dst.value ==
      dst.value);
}

TEST_CASE("ProgramEmitter translates IndexMul with constant operand") {
  lir::Program lir_program;

  const auto lhs =
      lir_program.create_index();

  const auto dst =
      lir_program.create_index();

  lir_program.add_instruction(
      lir::IndexMul(
          lhs,
          lir::ConstantIndex{16},
          dst));

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  const auto &instruction =
      rt_program.instructions()[0];

  REQUIRE(
      instruction.index_lhs.kind ==
      runtime::IndexOperand::Kind::Register);

  REQUIRE(
      instruction.index_rhs.kind ==
      runtime::IndexOperand::Kind::Constant);

  REQUIRE(
      instruction.index_rhs.constant ==
      16);

  REQUIRE(
      instruction.index_dst.value ==
      dst.value);
}

// -----------------------------------------------------------------------------
// Instruction ordering
// -----------------------------------------------------------------------------

TEST_CASE("ProgramEmitter preserves flat non-loop instruction ordering") {
  lir::Program lir_program;

  const auto a =
      lir_program.create_scalar();

  const auto b =
      lir_program.create_scalar();

  const auto c =
      lir_program.create_scalar();

  lir_program.add_instruction(
      lir::Constant{
          a,
          1.0F,
      });

  lir_program.add_instruction(
      lir::Constant{
          b,
          2.0F,
      });

  lir_program.add_instruction(
      lir::Add(
          a,
          b,
          c));

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  REQUIRE(rt_program.instructions().size() == 3);

  REQUIRE(
      rt_program.instructions()[0].opcode ==
      runtime::Opcode::Constant);

  REQUIRE(
      rt_program.instructions()[1].opcode ==
      runtime::Opcode::Constant);

  REQUIRE(
      rt_program.instructions()[2].opcode ==
      runtime::Opcode::Add);
}

// -----------------------------------------------------------------------------
// Single loop flattening
// -----------------------------------------------------------------------------

TEST_CASE("ProgramEmitter flattens one LIR loop") {
  lir::Program lir_program;

  const auto index =
      lir_program.create_index();

  const auto scalar =
      lir_program.create_scalar();

  auto loop =
      std::make_unique<lir::For>();

  loop->index = index;
  loop->begin = lir::ConstantIndex{0};
  loop->end = lir::ConstantIndex{4};

  loop->body.emplace_back(
      lir::Constant{
          scalar,
          1.0F,
      });

  lir_program.add_instruction(
      std::move(loop));

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  REQUIRE(rt_program.instructions().size() == 3);

  REQUIRE(
      rt_program.instructions()[0].opcode ==
      runtime::Opcode::LoopBegin);

  REQUIRE(
      rt_program.instructions()[1].opcode ==
      runtime::Opcode::Constant);

  REQUIRE(
      rt_program.instructions()[2].opcode ==
      runtime::Opcode::LoopEnd);
}

TEST_CASE("ProgramEmitter preserves loop bounds") {
  lir::Program lir_program;

  const auto index =
      lir_program.create_index();

  auto loop =
      std::make_unique<lir::For>();

  loop->index = index;
  loop->begin = lir::ConstantIndex{2};
  loop->end = lir::ConstantIndex{9};

  lir_program.add_instruction(
      std::move(loop));

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  REQUIRE(rt_program.instructions().size() == 2);

  const auto &begin =
      rt_program.instructions()[0];

  const auto &end =
      rt_program.instructions()[1];

  REQUIRE(
      begin.opcode ==
      runtime::Opcode::LoopBegin);

  REQUIRE(
      begin.loop_index.value ==
      index.value);

  REQUIRE(
      begin.loop_begin.kind ==
      runtime::IndexOperand::Kind::Constant);

  REQUIRE(begin.loop_begin.constant == 2);

  REQUIRE(
      begin.loop_end.kind ==
      runtime::IndexOperand::Kind::Constant);

  REQUIRE(begin.loop_end.constant == 9);

  REQUIRE(
      end.opcode ==
      runtime::Opcode::LoopEnd);

  REQUIRE(
      end.loop_index.value ==
      index.value);

  REQUIRE(
      end.loop_end.kind ==
      runtime::IndexOperand::Kind::Constant);

  REQUIRE(end.loop_end.constant == 9);
}

// -----------------------------------------------------------------------------
// Loop jump targets
// -----------------------------------------------------------------------------

TEST_CASE("ProgramEmitter single loop uses correct jump targets") {
  lir::Program lir_program;

  const auto index =
      lir_program.create_index();

  const auto scalar =
      lir_program.create_scalar();

  auto loop =
      std::make_unique<lir::For>();

  loop->index = index;
  loop->begin = lir::ConstantIndex{0};
  loop->end = lir::ConstantIndex{3};

  loop->body.emplace_back(
      lir::Constant{
          scalar,
          5.0F,
      });

  lir_program.add_instruction(
      std::move(loop));

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  /*
   * Flat form:
   *
   * 0 LoopBegin jump=3
   * 1 Constant
   * 2 LoopEnd   jump=1
   * 3 <exit>
   */

  REQUIRE(rt_program.instructions().size() == 3);

  REQUIRE(
      rt_program.instructions()[0].jump ==
      3);

  REQUIRE(
      rt_program.instructions()[2].jump ==
      1);
}

TEST_CASE("ProgramEmitter empty loop uses correct jump targets") {
  lir::Program lir_program;

  const auto index =
      lir_program.create_index();

  auto loop =
      std::make_unique<lir::For>();

  loop->index = index;
  loop->begin = lir::ConstantIndex{0};
  loop->end = lir::ConstantIndex{0};

  lir_program.add_instruction(
      std::move(loop));

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  /*
   * 0 LoopBegin jump=2
   * 1 LoopEnd   jump=1
   * 2 exit
   */

  REQUIRE(rt_program.instructions().size() == 2);

  REQUIRE(
      rt_program.instructions()[0].opcode ==
      runtime::Opcode::LoopBegin);

  REQUIRE(
      rt_program.instructions()[1].opcode ==
      runtime::Opcode::LoopEnd);

  REQUIRE(
      rt_program.instructions()[0].jump ==
      2);

  REQUIRE(
      rt_program.instructions()[1].jump ==
      1);
}

// -----------------------------------------------------------------------------
// Loop with multiple body instructions
// -----------------------------------------------------------------------------

TEST_CASE("ProgramEmitter loop body keeps instruction ordering") {
  lir::Program lir_program;

  const auto index =
      lir_program.create_index();

  const auto first =
      lir_program.create_scalar();

  const auto second =
      lir_program.create_scalar();

  const auto sum =
      lir_program.create_scalar();

  auto loop =
      std::make_unique<lir::For>();

  loop->index = index;
  loop->begin = lir::ConstantIndex{0};
  loop->end = lir::ConstantIndex{2};

  loop->body.emplace_back(
      lir::Constant{
          first,
          10.0F,
      });

  loop->body.emplace_back(
      lir::Constant{
          second,
          20.0F,
      });

  loop->body.emplace_back(
      lir::Add(
          first,
          second,
          sum));

  lir_program.add_instruction(
      std::move(loop));

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  REQUIRE(rt_program.instructions().size() == 5);

  REQUIRE(
      rt_program.instructions()[0].opcode ==
      runtime::Opcode::LoopBegin);

  REQUIRE(
      rt_program.instructions()[1].opcode ==
      runtime::Opcode::Constant);

  REQUIRE(
      rt_program.instructions()[2].opcode ==
      runtime::Opcode::Constant);

  REQUIRE(
      rt_program.instructions()[3].opcode ==
      runtime::Opcode::Add);

  REQUIRE(
      rt_program.instructions()[4].opcode ==
      runtime::Opcode::LoopEnd);

  REQUIRE(
      rt_program.instructions()[0].jump ==
      5);

  REQUIRE(
      rt_program.instructions()[4].jump ==
      1);
}

// -----------------------------------------------------------------------------
// Nested loops
// -----------------------------------------------------------------------------

TEST_CASE("ProgramEmitter flattens nested LIR loops") {
  lir::Program lir_program;

  const auto outer_index =
      lir_program.create_index();

  const auto inner_index =
      lir_program.create_index();

  const auto scalar =
      lir_program.create_scalar();

  auto inner =
      std::make_unique<lir::For>();

  inner->index = inner_index;
  inner->begin = lir::ConstantIndex{0};
  inner->end = lir::ConstantIndex{3};

  inner->body.emplace_back(
      lir::Constant{
          scalar,
          1.0F,
      });

  auto outer =
      std::make_unique<lir::For>();

  outer->index = outer_index;
  outer->begin = lir::ConstantIndex{0};
  outer->end = lir::ConstantIndex{2};

  outer->body.emplace_back(
      std::move(inner));

  lir_program.add_instruction(
      std::move(outer));

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  /*
   * 0 outer LoopBegin
   * 1 inner LoopBegin
   * 2 Constant
   * 3 inner LoopEnd
   * 4 outer LoopEnd
   */

  REQUIRE(rt_program.instructions().size() == 5);

  REQUIRE(
      rt_program.instructions()[0].opcode ==
      runtime::Opcode::LoopBegin);

  REQUIRE(
      rt_program.instructions()[1].opcode ==
      runtime::Opcode::LoopBegin);

  REQUIRE(
      rt_program.instructions()[2].opcode ==
      runtime::Opcode::Constant);

  REQUIRE(
      rt_program.instructions()[3].opcode ==
      runtime::Opcode::LoopEnd);

  REQUIRE(
      rt_program.instructions()[4].opcode ==
      runtime::Opcode::LoopEnd);
}

TEST_CASE("ProgramEmitter nested loops have correct independent jump targets") {
  lir::Program lir_program;

  const auto outer_index =
      lir_program.create_index();

  const auto inner_index =
      lir_program.create_index();

  const auto scalar =
      lir_program.create_scalar();

  auto inner =
      std::make_unique<lir::For>();

  inner->index = inner_index;
  inner->begin = lir::ConstantIndex{0};
  inner->end = lir::ConstantIndex{3};

  inner->body.emplace_back(
      lir::Constant{
          scalar,
          1.0F,
      });

  auto outer =
      std::make_unique<lir::For>();

  outer->index = outer_index;
  outer->begin = lir::ConstantIndex{0};
  outer->end = lir::ConstantIndex{2};

  outer->body.emplace_back(
      std::move(inner));

  lir_program.add_instruction(
      std::move(outer));

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  /*
   * 0 Outer LoopBegin   jump = 5
   * 1 Inner LoopBegin   jump = 4
   * 2 Constant
   * 3 Inner LoopEnd     jump = 2
   * 4 Outer LoopEnd     jump = 1
   * 5 exit
   */

  REQUIRE(rt_program.instructions().size() == 5);

  REQUIRE(
      rt_program.instructions()[0].jump ==
      5);

  REQUIRE(
      rt_program.instructions()[1].jump ==
      4);

  REQUIRE(
      rt_program.instructions()[3].jump ==
      2);

  REQUIRE(
      rt_program.instructions()[4].jump ==
      1);
}

TEST_CASE("ProgramEmitter nested loops preserve separate loop index ids") {
  lir::Program lir_program;

  const auto outer_index =
      lir_program.create_index();

  const auto inner_index =
      lir_program.create_index();

  auto inner =
      std::make_unique<lir::For>();

  inner->index = inner_index;
  inner->begin = lir::ConstantIndex{0};
  inner->end = lir::ConstantIndex{4};

  auto outer =
      std::make_unique<lir::For>();

  outer->index = outer_index;
  outer->begin = lir::ConstantIndex{0};
  outer->end = lir::ConstantIndex{2};

  outer->body.emplace_back(
      std::move(inner));

  lir_program.add_instruction(
      std::move(outer));

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  REQUIRE(
      rt_program.instructions()[0].loop_index.value ==
      outer_index.value);

  REQUIRE(
      rt_program.instructions()[1].loop_index.value ==
      inner_index.value);

  REQUIRE(
      rt_program.instructions()[2].loop_index.value ==
      inner_index.value);

  REQUIRE(
      rt_program.instructions()[3].loop_index.value ==
      outer_index.value);
}

// -----------------------------------------------------------------------------
// Register loop bounds
// -----------------------------------------------------------------------------

TEST_CASE("ProgramEmitter converts register-valued loop bound") {
  lir::Program lir_program;

  const auto loop_index =
      lir_program.create_index();

  const auto end_index =
      lir_program.create_index();

  auto loop =
      std::make_unique<lir::For>();

  loop->index = loop_index;
  loop->begin = lir::ConstantIndex{0};
  loop->end = end_index;

  lir_program.add_instruction(
      std::move(loop));

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  const auto &begin =
      rt_program.instructions()[0];

  REQUIRE(
      begin.loop_end.kind ==
      runtime::IndexOperand::Kind::Register);

  REQUIRE(
      begin.loop_end.reg.value ==
      end_index.value);

  const auto &end =
      rt_program.instructions()[1];

  REQUIRE(
      end.loop_end.kind ==
      runtime::IndexOperand::Kind::Register);

  REQUIRE(
      end.loop_end.reg.value ==
      end_index.value);
}

// -----------------------------------------------------------------------------
// Combined program
// -----------------------------------------------------------------------------

TEST_CASE("ProgramEmitter emits complete runtime program metadata and instructions") {
  lir::Program lir_program;

  const auto input =
      lir_program.create_buffer(
          Shape{3},
          Dtype::F32);

  const auto output =
      lir_program.create_buffer(
          Shape{3},
          Dtype::F32);

  lir_program.add_input(
      ValueId{7},
      input);

  lir_program.add_output(
      output);

  const auto scalar =
      lir_program.create_scalar();

  const auto index =
      lir_program.create_index();

  auto loop =
      std::make_unique<lir::For>();

  loop->index = index;
  loop->begin = lir::ConstantIndex{0};
  loop->end = lir::ConstantIndex{3};

  loop->body.emplace_back(
      lir::Load(
          scalar,
          input,
          index));

  loop->body.emplace_back(
      lir::Store(
          scalar,
          output,
          index));

  lir_program.add_instruction(
      std::move(loop));

  ProgramEmitter emitter;
  const auto rt_program =
      emitter.emit(lir_program);

  REQUIRE(rt_program.slots_count() == 2);
  REQUIRE(rt_program.scalar_count() == 1);
  REQUIRE(rt_program.index_count() == 1);

  REQUIRE(rt_program.inputs().size() == 1);
  REQUIRE(rt_program.outputs().size() == 1);

  REQUIRE(
      rt_program.inputs()[0] ==
      runtime::SlotId{0});

  REQUIRE(
      rt_program.outputs()[0] ==
      runtime::SlotId{1});

  REQUIRE(
      rt_program.tensor_meta_by_slot(
          runtime::SlotId{0}).shape ==
      Shape{3});

  REQUIRE(
      rt_program.tensor_meta_by_slot(
          runtime::SlotId{1}).shape ==
      Shape{3});

  REQUIRE(rt_program.instructions().size() == 4);

  REQUIRE(
      rt_program.instructions()[0].opcode ==
      runtime::Opcode::LoopBegin);

  REQUIRE(
      rt_program.instructions()[1].opcode ==
      runtime::Opcode::Load);

  REQUIRE(
      rt_program.instructions()[2].opcode ==
      runtime::Opcode::Store);

  REQUIRE(
      rt_program.instructions()[3].opcode ==
      runtime::Opcode::LoopEnd);

  REQUIRE(
      rt_program.instructions()[1].buffer ==
      runtime::SlotId{0});

  REQUIRE(
      rt_program.instructions()[2].buffer ==
      runtime::SlotId{1});

  REQUIRE(
      rt_program.instructions()[0].jump ==
      4);

  REQUIRE(
      rt_program.instructions()[3].jump ==
      1);
}