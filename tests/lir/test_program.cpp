#include <catch2/catch_test_macros.hpp>

#include "tcr/ir/dtype.hpp"
#include "tcr/ir/ids.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/lir/ids.hpp"
#include "tcr/lir/instruction.hpp"
#include "tcr/lir/program.hpp"

#include <cstddef>
#include <vector>

using tcr::ir::Dtype;
using tcr::ir::Shape;
using tcr::ir::ValueId;

using tcr::lir::BufferId;
using tcr::lir::IndexId;
using tcr::lir::Program;
using tcr::lir::ScalarId;

// -----------------------------------------------------------------------------
// Initial state
// -----------------------------------------------------------------------------

TEST_CASE("lir Program starts empty") {
  const Program program;

  REQUIRE(program.buffers().empty());
  REQUIRE(program.inputs().empty());
  REQUIRE(program.outputs().empty());
  REQUIRE(program.instructions().empty());

  REQUIRE(program.buffer_count() == 0);
  REQUIRE(program.scalar_count() == 0);
  REQUIRE(program.index_count() == 0);
}

// -----------------------------------------------------------------------------
// Buffer creation
// -----------------------------------------------------------------------------

TEST_CASE("lir Program create_buffer creates first buffer with id zero") {
  Program program;

  const BufferId buffer =
      program.create_buffer(Shape{3}, Dtype::F32);

  REQUIRE(buffer.value == 0);

  REQUIRE(program.buffer_count() == 1);
  REQUIRE(program.buffers().size() == 1);

  REQUIRE(program.buffers()[0].id.value == 0);
  REQUIRE(program.buffers()[0].shape == Shape{3});
  REQUIRE(program.buffers()[0].dtype == Dtype::F32);
}

TEST_CASE("lir Program create_buffer creates sequential buffer ids") {
  Program program;

  const BufferId first =
      program.create_buffer(Shape{2}, Dtype::F32);

  const BufferId second =
      program.create_buffer(Shape{3}, Dtype::F32);

  const BufferId third =
      program.create_buffer(Shape{4}, Dtype::F32);

  REQUIRE(first.value == 0);
  REQUIRE(second.value == 1);
  REQUIRE(third.value == 2);

  REQUIRE(program.buffer_count() == 3);
}

TEST_CASE("lir Program buffers preserve creation order") {
  Program program;

  program.create_buffer(
      Shape{2},
      Dtype::F32);

  program.create_buffer(
      Shape{2, 3},
      Dtype::F32);

  program.create_buffer(
      Shape{},
      Dtype::F32);

  REQUIRE(program.buffers().size() == 3);

  REQUIRE(program.buffers()[0].id.value == 0);
  REQUIRE(program.buffers()[0].shape == Shape{2});

  REQUIRE(program.buffers()[1].id.value == 1);
  REQUIRE(program.buffers()[1].shape == Shape{2, 3});

  REQUIRE(program.buffers()[2].id.value == 2);
  REQUIRE(program.buffers()[2].shape == Shape{});
}

TEST_CASE("lir Program create_buffer stores matrix metadata") {
  Program program;

  const auto buffer =
      program.create_buffer(
          Shape{2, 3},
          Dtype::F32);

  REQUIRE(buffer.value == 0);

  const auto &stored =
      program.buffers()[buffer.value];

  REQUIRE(stored.id.value == 0);
  REQUIRE(stored.shape == Shape{2, 3});
  REQUIRE(stored.shape.rank() == 2);
  REQUIRE(stored.shape.num_elements() == 6);
  REQUIRE(stored.dtype == Dtype::F32);
}

TEST_CASE("lir Program create_buffer supports scalar shape") {
  Program program;

  const auto buffer =
      program.create_buffer(
          Shape{},
          Dtype::F32);

  REQUIRE(buffer.value == 0);

  const auto &stored =
      program.buffers()[0];

  REQUIRE(stored.shape == Shape{});
  REQUIRE(stored.shape.rank() == 0);
  REQUIRE(stored.shape.num_elements() == 1);
  REQUIRE(stored.dtype == Dtype::F32);
}

TEST_CASE("lir Program buffer id equals its buffers vector index") {
  Program program;

  constexpr std::size_t buffer_count = 5;

  for (std::size_t i = 0;
       i < buffer_count;
       ++i) {
    program.create_buffer(
        Shape{i + 1},
        Dtype::F32);
  }

  REQUIRE(program.buffers().size() ==
          buffer_count);

  for (std::size_t i = 0;
       i < program.buffers().size();
       ++i) {
    REQUIRE(
        program.buffers()[i].id.value == i);
  }
}

// -----------------------------------------------------------------------------
// Scalar creation
// -----------------------------------------------------------------------------

TEST_CASE("lir Program create_scalar starts with id zero") {
  Program program;

  const ScalarId scalar =
      program.create_scalar();

  REQUIRE(scalar.value == 0);
  REQUIRE(program.scalar_count() == 1);
}

TEST_CASE("lir Program create_scalar creates sequential ids") {
  Program program;

  const ScalarId first =
      program.create_scalar();

  const ScalarId second =
      program.create_scalar();

  const ScalarId third =
      program.create_scalar();

  REQUIRE(first.value == 0);
  REQUIRE(second.value == 1);
  REQUIRE(third.value == 2);

  REQUIRE(program.scalar_count() == 3);
}

TEST_CASE("lir Program scalar count changes only when scalar is created") {
  Program program;

  program.create_buffer(
      Shape{2},
      Dtype::F32);

  REQUIRE(program.scalar_count() == 0);

  program.create_scalar();

  REQUIRE(program.scalar_count() == 1);

  program.create_buffer(
      Shape{3},
      Dtype::F32);

  REQUIRE(program.scalar_count() == 1);

  program.create_scalar();

  REQUIRE(program.scalar_count() == 2);
}

// -----------------------------------------------------------------------------
// Index creation
// -----------------------------------------------------------------------------

TEST_CASE("lir Program create_index starts with id zero") {
  Program program;

  const IndexId index =
      program.create_index();

  REQUIRE(index.value == 0);
  REQUIRE(program.index_count() == 1);
}

TEST_CASE("lir Program create_index creates sequential ids") {
  Program program;

  const IndexId first =
      program.create_index();

  const IndexId second =
      program.create_index();

  const IndexId third =
      program.create_index();

  REQUIRE(first.value == 0);
  REQUIRE(second.value == 1);
  REQUIRE(third.value == 2);

  REQUIRE(program.index_count() == 3);
}

TEST_CASE("lir Program index count changes only when index is created") {
  Program program;

  program.create_buffer(
      Shape{2},
      Dtype::F32);

  REQUIRE(program.index_count() == 0);

  program.create_index();

  REQUIRE(program.index_count() == 1);

  program.create_scalar();

  REQUIRE(program.index_count() == 1);

  program.create_index();

  REQUIRE(program.index_count() == 2);
}

// -----------------------------------------------------------------------------
// Scalar and index namespaces are independent
// -----------------------------------------------------------------------------

TEST_CASE("lir Program scalar and index ids have independent numbering") {
  Program program;

  const auto scalar0 =
      program.create_scalar();

  const auto index0 =
      program.create_index();

  const auto scalar1 =
      program.create_scalar();

  const auto index1 =
      program.create_index();

  REQUIRE(scalar0.value == 0);
  REQUIRE(scalar1.value == 1);

  REQUIRE(index0.value == 0);
  REQUIRE(index1.value == 1);

  REQUIRE(program.scalar_count() == 2);
  REQUIRE(program.index_count() == 2);
}

TEST_CASE("lir Program buffer scalar and index ids each start at zero") {
  Program program;

  const auto buffer =
      program.create_buffer(
          Shape{1},
          Dtype::F32);

  const auto scalar =
      program.create_scalar();

  const auto index =
      program.create_index();

  REQUIRE(buffer.value == 0);
  REQUIRE(scalar.value == 0);
  REQUIRE(index.value == 0);

  REQUIRE(program.buffer_count() == 1);
  REQUIRE(program.scalar_count() == 1);
  REQUIRE(program.index_count() == 1);
}

// -----------------------------------------------------------------------------
// Inputs
// -----------------------------------------------------------------------------

TEST_CASE("lir Program add_input stores value id and buffer id") {
  Program program;

  const ValueId value{7};

  const auto buffer =
      program.create_buffer(
          Shape{3},
          Dtype::F32);

  program.add_input(
      value,
      buffer);

  REQUIRE(program.inputs().size() == 1);

  REQUIRE(
      program.inputs()[0].value.value == 7);

  REQUIRE(
      program.inputs()[0].buffer.value ==
      buffer.value);
}

TEST_CASE("lir Program add_input preserves insertion order") {
  Program program;

  const auto first_buffer =
      program.create_buffer(
          Shape{2},
          Dtype::F32);

  const auto second_buffer =
      program.create_buffer(
          Shape{2},
          Dtype::F32);

  program.add_input(
      ValueId{10},
      first_buffer);

  program.add_input(
      ValueId{20},
      second_buffer);

  REQUIRE(program.inputs().size() == 2);

  REQUIRE(
      program.inputs()[0].value.value == 10);

  REQUIRE(
      program.inputs()[0].buffer.value == 0);

  REQUIRE(
      program.inputs()[1].value.value == 20);

  REQUIRE(
      program.inputs()[1].buffer.value == 1);
}

TEST_CASE("lir Program adding input does not create another buffer") {
  Program program;

  const auto buffer =
      program.create_buffer(
          Shape{3},
          Dtype::F32);

  REQUIRE(program.buffer_count() == 1);

  program.add_input(
      ValueId{0},
      buffer);

  REQUIRE(program.buffer_count() == 1);
  REQUIRE(program.inputs().size() == 1);
}

TEST_CASE("lir Program can register multiple values as inputs") {
  Program program;

  const auto a =
      program.create_buffer(
          Shape{2, 2},
          Dtype::F32);

  const auto b =
      program.create_buffer(
          Shape{2, 2},
          Dtype::F32);

  const auto c =
      program.create_buffer(
          Shape{2, 2},
          Dtype::F32);

  program.add_input(ValueId{0}, a);
  program.add_input(ValueId{1}, b);
  program.add_input(ValueId{2}, c);

  REQUIRE(program.inputs().size() == 3);

  REQUIRE(
      program.inputs()[0].buffer.value == 0);
  REQUIRE(
      program.inputs()[1].buffer.value == 1);
  REQUIRE(
      program.inputs()[2].buffer.value == 2);
}

// -----------------------------------------------------------------------------
// Outputs
// -----------------------------------------------------------------------------

TEST_CASE("lir Program add_output stores buffer id") {
  Program program;

  const auto buffer =
      program.create_buffer(
          Shape{3},
          Dtype::F32);

  program.add_output(buffer);

  REQUIRE(program.outputs().size() == 1);
  REQUIRE(program.outputs()[0].value ==
          buffer.value);
}

TEST_CASE("lir Program add_output preserves insertion order") {
  Program program;

  const auto first =
      program.create_buffer(
          Shape{1},
          Dtype::F32);

  const auto second =
      program.create_buffer(
          Shape{1},
          Dtype::F32);

  const auto third =
      program.create_buffer(
          Shape{1},
          Dtype::F32);

  program.add_output(second);
  program.add_output(first);
  program.add_output(third);

  REQUIRE(program.outputs().size() == 3);

  REQUIRE(
      program.outputs()[0].value ==
      second.value);

  REQUIRE(
      program.outputs()[1].value ==
      first.value);

  REQUIRE(
      program.outputs()[2].value ==
      third.value);
}

TEST_CASE("lir Program adding output does not modify buffer count") {
  Program program;

  const auto buffer =
      program.create_buffer(
          Shape{4},
          Dtype::F32);

  REQUIRE(program.buffer_count() == 1);

  program.add_output(buffer);

  REQUIRE(program.buffer_count() == 1);
  REQUIRE(program.outputs().size() == 1);
}

// -----------------------------------------------------------------------------
// Instructions
// -----------------------------------------------------------------------------

TEST_CASE("lir Program add_instruction stores constant instruction") {
  Program program;

  const auto scalar =
      program.create_scalar();

  program.add_instruction(
      tcr::lir::Constant{
          scalar,
          42.0F,
      });

  REQUIRE(program.instructions().size() == 1);
}

TEST_CASE("lir Program add_instruction preserves instruction insertion order") {
  Program program;

  const auto first =
      program.create_scalar();

  const auto second =
      program.create_scalar();

  program.add_instruction(
      tcr::lir::Constant{
          first,
          10.0F,
      });

  program.add_instruction(
      tcr::lir::Constant{
          second,
          20.0F,
      });

  REQUIRE(program.instructions().size() == 2);
}

TEST_CASE("lir Program adding instructions does not affect resource counts") {
  Program program;

  const auto scalar =
      program.create_scalar();

  const auto buffer =
      program.create_buffer(
          Shape{1},
          Dtype::F32);

  const auto index =
      program.create_index();

  REQUIRE(program.scalar_count() == 1);
  REQUIRE(program.buffer_count() == 1);
  REQUIRE(program.index_count() == 1);

  program.add_instruction(
      tcr::lir::Constant{
          scalar,
          1.0F,
      });

  REQUIRE(program.instructions().size() == 1);

  REQUIRE(program.scalar_count() == 1);
  REQUIRE(program.buffer_count() == 1);
  REQUIRE(program.index_count() == 1);

  // Prevent unused-variable warnings in strict builds.
  REQUIRE(buffer.value == 0);
  REQUIRE(index.value == 0);
}

// -----------------------------------------------------------------------------
// Combined Program state
// -----------------------------------------------------------------------------

TEST_CASE("lir Program maintains buffers inputs outputs scalars indices and instructions independently") {
  Program program;

  const auto a =
      program.create_buffer(
          Shape{3},
          Dtype::F32);

  const auto b =
      program.create_buffer(
          Shape{3},
          Dtype::F32);

  const auto c =
      program.create_buffer(
          Shape{3},
          Dtype::F32);

  program.add_input(
      ValueId{0},
      a);

  program.add_input(
      ValueId{1},
      b);

  program.add_output(c);

  const auto scalar0 =
      program.create_scalar();

  const auto scalar1 =
      program.create_scalar();

  const auto index0 =
      program.create_index();

  program.add_instruction(
      tcr::lir::Constant{
          scalar0,
          5.0F,
      });

  REQUIRE(program.buffer_count() == 3);
  REQUIRE(program.buffers().size() == 3);

  REQUIRE(program.inputs().size() == 2);
  REQUIRE(program.outputs().size() == 1);

  REQUIRE(program.scalar_count() == 2);
  REQUIRE(program.index_count() == 1);

  REQUIRE(program.instructions().size() == 1);

  REQUIRE(a.value == 0);
  REQUIRE(b.value == 1);
  REQUIRE(c.value == 2);

  REQUIRE(scalar0.value == 0);
  REQUIRE(scalar1.value == 1);

  REQUIRE(index0.value == 0);
}

// -----------------------------------------------------------------------------
// Const accessors
// -----------------------------------------------------------------------------

TEST_CASE("lir Program accessors can be used through const Program reference") {
  Program program;

  const auto buffer =
      program.create_buffer(
          Shape{2},
          Dtype::F32);

  program.add_input(
      ValueId{0},
      buffer);

  program.add_output(buffer);

  const auto scalar =
      program.create_scalar();

  program.create_index();

  program.add_instruction(
      tcr::lir::Constant{
          scalar,
          1.0F,
      });

  const Program &const_program =
      program;

  REQUIRE(
      const_program.buffers().size() == 1);

  REQUIRE(
      const_program.inputs().size() == 1);

  REQUIRE(
      const_program.outputs().size() == 1);

  REQUIRE(
      const_program.instructions().size() == 1);

  REQUIRE(
      const_program.buffer_count() == 1);

  REQUIRE(
      const_program.scalar_count() == 1);

  REQUIRE(
      const_program.index_count() == 1);
}