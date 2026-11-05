#include <catch2/catch_test_macros.hpp>

#include "tcr/compiler/lowering.hpp"
#include "tcr/compiler/memory_planner.hpp"
#include "tcr/compiler/schedule.hpp"
#include "tcr/ir/dtype.hpp"
#include "tcr/ir/ids.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/lir/program.hpp"
#include "tcr/runtime/slot_id.hpp"

#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using tcr::compiler::KernelUnit;
using tcr::compiler::LoweringResult;
using tcr::compiler::MemoryPlanner;
using tcr::compiler::Schedule;
using tcr::ir::Dtype;
using tcr::ir::Shape;
using tcr::ir::ValueId;

namespace lir = tcr::lir;
namespace runtime = tcr::runtime;

namespace {

struct LinearFixture {
  LoweringResult lowered;
  Schedule schedule;
  ValueId input{0};
  ValueId first{1};
  ValueId second{2};
  ValueId third{3};
  lir::BufferId input_buffer{};
  lir::BufferId first_buffer{};
  lir::BufferId second_buffer{};
  lir::BufferId third_buffer{};
};

LinearFixture make_three_kernel_chain(Shape first_shape = Shape{4},
                                      Shape second_shape = Shape{4},
                                      Shape third_shape = Shape{4}) {
  LinearFixture f;

  f.input_buffer = f.lowered.program.create_buffer(Shape{4}, Dtype::F32);
  f.first_buffer = f.lowered.program.create_buffer(first_shape, Dtype::F32);
  f.second_buffer = f.lowered.program.create_buffer(second_shape, Dtype::F32);
  f.third_buffer = f.lowered.program.create_buffer(third_shape, Dtype::F32);

  f.lowered.program.add_input(f.input, f.input_buffer);
  f.lowered.program.add_output(f.third_buffer);

  f.lowered.value_to_buffer = {
      {f.input, f.input_buffer},
      {f.first, f.first_buffer},
      {f.second, f.second_buffer},
      {f.third, f.third_buffer},
  };

  f.schedule.kernels = {
      KernelUnit{{f.first}, {f.input}, {f.first}},
      KernelUnit{{f.second}, {f.first}, {f.second}},
      KernelUnit{{f.third}, {f.second}, {f.third}},
  };

  return f;
}

} // namespace

TEST_CASE("MemoryPlanner leaves external graph inputs without planner-owned slots") {
  const auto f = make_three_kernel_chain();

  const auto plan = MemoryPlanner{}.plan(f.schedule, f.lowered);

  REQUIRE(plan.buffer_to_slot.size() == f.lowered.program.buffer_count());
  REQUIRE_FALSE(plan.buffer_to_slot[f.input_buffer.value].has_value());
  REQUIRE(plan.buffer_to_slot[f.first_buffer.value].has_value());
  REQUIRE(plan.buffer_to_slot[f.second_buffer.value].has_value());
  REQUIRE(plan.buffer_to_slot[f.third_buffer.value].has_value());
}

TEST_CASE("MemoryPlanner reuses a slot after a non-overlapping lifetime") {
  const auto f = make_three_kernel_chain();

  const auto plan = MemoryPlanner{}.plan(f.schedule, f.lowered);

  REQUIRE(plan.slots.size() == 2);
  REQUIRE(plan.buffer_to_slot[f.first_buffer.value] == runtime::SlotId{0});
  REQUIRE(plan.buffer_to_slot[f.second_buffer.value] == runtime::SlotId{1});
  REQUIRE(plan.buffer_to_slot[f.third_buffer.value] == runtime::SlotId{0});
}

TEST_CASE("MemoryPlanner does not reuse storage when last use equals new birth kernel") {
  LoweringResult lowered;
  const ValueId input{0};
  const ValueId first{1};
  const ValueId output{2};

  const auto input_buffer = lowered.program.create_buffer(Shape{4}, Dtype::F32);
  const auto first_buffer = lowered.program.create_buffer(Shape{4}, Dtype::F32);
  const auto output_buffer = lowered.program.create_buffer(Shape{4}, Dtype::F32);

  lowered.program.add_input(input, input_buffer);
  lowered.program.add_output(output_buffer);
  lowered.value_to_buffer = {
      {input, input_buffer},
      {first, first_buffer},
      {output, output_buffer},
  };

  const Schedule schedule{{
      KernelUnit{{first}, {input}, {first}},
      KernelUnit{{output}, {first}, {output}},
  }};

  const auto plan = MemoryPlanner{}.plan(schedule, lowered);

  REQUIRE(plan.slots.size() == 2);
  REQUIRE(plan.buffer_to_slot[first_buffer.value] !=
          plan.buffer_to_slot[output_buffer.value]);
}

TEST_CASE("MemoryPlanner keeps graph outputs live through program completion") {
  auto f = make_three_kernel_chain();
  f.lowered.program.add_output(f.first_buffer);

  const auto plan = MemoryPlanner{}.plan(f.schedule, f.lowered);

  REQUIRE(plan.slots.size() == 3);
  REQUIRE(plan.buffer_to_slot[f.first_buffer.value] !=
          plan.buffer_to_slot[f.third_buffer.value]);
}

TEST_CASE("MemoryPlanner grows a reused physical slot to the largest assigned buffer") {
  const auto f = make_three_kernel_chain(Shape{1}, Shape{1}, Shape{8});

  const auto plan = MemoryPlanner{}.plan(f.schedule, f.lowered);

  REQUIRE(plan.buffer_to_slot[f.first_buffer.value] == runtime::SlotId{0});
  REQUIRE(plan.buffer_to_slot[f.third_buffer.value] == runtime::SlotId{0});
  REQUIRE(plan.slots[0].capacity_bytes == 8 * sizeof(float));
  REQUIRE(plan.slots[0].alignment >= alignof(std::max_align_t));
}

TEST_CASE("MemoryPlanner assigns deterministic distinct slots to outputs born in same kernel") {
  LoweringResult lowered;
  const ValueId input{0};
  const ValueId a{1};
  const ValueId b{2};

  const auto input_buffer = lowered.program.create_buffer(Shape{2}, Dtype::F32);
  const auto a_buffer = lowered.program.create_buffer(Shape{2}, Dtype::F32);
  const auto b_buffer = lowered.program.create_buffer(Shape{2}, Dtype::F32);

  lowered.program.add_input(input, input_buffer);
  lowered.program.add_output(a_buffer);
  lowered.program.add_output(b_buffer);
  lowered.value_to_buffer = {
      {input, input_buffer},
      {a, a_buffer},
      {b, b_buffer},
  };

  const Schedule schedule{{KernelUnit{{a, b}, {input}, {b, a}}}};

  const auto plan = MemoryPlanner{}.plan(schedule, lowered);

  REQUIRE(plan.slots.size() == 2);
  REQUIRE(plan.buffer_to_slot[a_buffer.value] == runtime::SlotId{0});
  REQUIRE(plan.buffer_to_slot[b_buffer.value] == runtime::SlotId{1});
}

TEST_CASE("MemoryPlanner rejects planner-owned buffer without complete lifetime") {
  LoweringResult lowered;
  const ValueId input{0};
  const ValueId dead{1};

  const auto input_buffer = lowered.program.create_buffer(Shape{2}, Dtype::F32);
  const auto dead_buffer = lowered.program.create_buffer(Shape{2}, Dtype::F32);

  lowered.program.add_input(input, input_buffer);
  lowered.value_to_buffer = {
      {input, input_buffer},
      {dead, dead_buffer},
  };

  const Schedule schedule{{KernelUnit{{dead}, {input}, {dead}}}};

  REQUIRE_THROWS_AS(MemoryPlanner{}.plan(schedule, lowered), std::logic_error);
}
