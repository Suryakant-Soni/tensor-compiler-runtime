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
#include <iostream>
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

std::size_t naive_planner_bytes(const LoweringResult &lowered,
                                const tcr::compiler::MemoryPlan &plan) {
  std::size_t total = 0;

  for (std::size_t b = 0; b < plan.buffer_to_slot.size(); ++b) {
    // nullopt means external input storage. The planner does not allocate it.
    if (!plan.buffer_to_slot[b].has_value()) {
      continue;
    }

    const auto &buffer = lowered.program.buffers()[b];
    total += tcr::ir::size_in_bytes(buffer.dtype) * buffer.shape.num_elements();
  }

  return total;
}

std::size_t planned_slot_bytes(const tcr::compiler::MemoryPlan &plan) {
  std::size_t total = 0;

  for (const auto &slot : plan.slots) {
    total += slot.capacity_bytes;
  }

  return total;
}

} // namespace

TEST_CASE(
    "MemoryPlanner leaves external graph inputs without planner-owned slots") {
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

TEST_CASE("MemoryPlanner does not reuse storage when last use equals new birth "
          "kernel") {
  LoweringResult lowered;
  const ValueId input{0};
  const ValueId first{1};
  const ValueId output{2};

  const auto input_buffer = lowered.program.create_buffer(Shape{4}, Dtype::F32);
  const auto first_buffer = lowered.program.create_buffer(Shape{4}, Dtype::F32);
  const auto output_buffer =
      lowered.program.create_buffer(Shape{4}, Dtype::F32);

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

TEST_CASE("MemoryPlanner grows a reused physical slot to the largest assigned "
          "buffer") {
  const auto f = make_three_kernel_chain(Shape{1}, Shape{1}, Shape{8});

  const auto plan = MemoryPlanner{}.plan(f.schedule, f.lowered);

  REQUIRE(plan.buffer_to_slot[f.first_buffer.value] == runtime::SlotId{0});
  REQUIRE(plan.buffer_to_slot[f.third_buffer.value] == runtime::SlotId{0});
  REQUIRE(plan.slots[0].capacity_bytes == 8 * sizeof(float));
  REQUIRE(plan.slots[0].alignment >= alignof(std::max_align_t));
}

TEST_CASE("MemoryPlanner assigns deterministic distinct slots to outputs born "
          "in same kernel") {
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

TEST_CASE(
    "MemoryPlanner rejects planner-owned buffer without complete lifetime") {
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

TEST_CASE("MemoryPlanner rejects same logical buffer produced by two kernels") {
  LoweringResult lowered;

  const ValueId input{0};
  const ValueId shared{1};

  const auto input_buffer = lowered.program.create_buffer(Shape{4}, Dtype::F32);

  const auto shared_buffer =
      lowered.program.create_buffer(Shape{4}, Dtype::F32);

  lowered.program.add_input(input, input_buffer);
  lowered.program.add_output(shared_buffer);

  lowered.value_to_buffer = {
      {input, input_buffer},
      {shared, shared_buffer},
  };

  // Invalid schedule:
  //
  // K0 claims to produce `shared`
  // K1 also claims to produce the same logical value `shared`.
  //
  // One materialized logical value must have exactly one producer kernel.
  const Schedule schedule{{
      KernelUnit{{shared}, {input}, {shared}},
      KernelUnit{{shared}, {input}, {shared}},
  }};

  REQUIRE_THROWS_AS(MemoryPlanner{}.plan(schedule, lowered), std::logic_error);
}

TEST_CASE("MemoryPlanner rejects buffer consumed before producer") {
  LoweringResult lowered;

  const ValueId input{0};
  const ValueId late{1};
  const ValueId consumer{2};

  const auto input_buffer = lowered.program.create_buffer(Shape{4}, Dtype::F32);

  const auto late_buffer = lowered.program.create_buffer(Shape{4}, Dtype::F32);

  const auto consumer_buffer =
      lowered.program.create_buffer(Shape{4}, Dtype::F32);

  lowered.program.add_input(input, input_buffer);
  lowered.program.add_output(consumer_buffer);

  lowered.value_to_buffer = {
      {input, input_buffer},
      {late, late_buffer},
      {consumer, consumer_buffer},
  };

  // Invalid schedule:
  //
  // K0 consumes `late`
  // K1 produces `late`
  //
  // Therefore:
  //
  // last_use(late) = K0 = 0
  // birth(late)    = K1 = 1
  //
  // last_use < birth is impossible in a legal schedule.
  const Schedule schedule{{
      KernelUnit{{consumer}, {late}, {consumer}},
      KernelUnit{{late}, {input}, {late}},
  }};

  REQUIRE_THROWS_AS(MemoryPlanner{}.plan(schedule, lowered), std::logic_error);
}

TEST_CASE("MemoryPlanner keeps shared producer alive through its last diamond "
          "consumer") {
  LoweringResult lowered;

  const ValueId input{0};
  const ValueId t0{1};
  const ValueId t1{2};
  const ValueId t2{3};
  const ValueId t3{4};

  const auto input_buffer = lowered.program.create_buffer(Shape{4}, Dtype::F32);

  const auto t0_buffer = lowered.program.create_buffer(Shape{4}, Dtype::F32);

  const auto t1_buffer = lowered.program.create_buffer(Shape{4}, Dtype::F32);

  const auto t2_buffer = lowered.program.create_buffer(Shape{4}, Dtype::F32);

  const auto t3_buffer = lowered.program.create_buffer(Shape{4}, Dtype::F32);

  lowered.program.add_input(input, input_buffer);
  lowered.program.add_output(t3_buffer);

  lowered.value_to_buffer = {
      {input, input_buffer}, {t0, t0_buffer}, {t1, t1_buffer},
      {t2, t2_buffer},       {t3, t3_buffer},
  };

  // T0 is consumed by BOTH K1 and K2.
  //
  // Therefore:
  //
  // birth(T0)    = K0
  // last_use(T0) = K2
  //
  // It cannot share storage with T1 or T2,
  // but its slot becomes reusable before T3 is born.

  const Schedule schedule{{
      KernelUnit{{t0}, {input}, {t0}},
      KernelUnit{{t1}, {t0}, {t1}},
      KernelUnit{{t2}, {t0}, {t2}},
      KernelUnit{{t3}, {t1, t2}, {t3}},
  }};

  const auto plan = MemoryPlanner{}.plan(schedule, lowered);

  // At maximum overlap:
  //
  // T0, T1, and T2 require three distinct slots.
  REQUIRE(plan.slots.size() == 3);

  const auto t0_slot = plan.buffer_to_slot[t0_buffer.value];

  const auto t1_slot = plan.buffer_to_slot[t1_buffer.value];

  const auto t2_slot = plan.buffer_to_slot[t2_buffer.value];

  const auto t3_slot = plan.buffer_to_slot[t3_buffer.value];

  REQUIRE(t0_slot.has_value());
  REQUIRE(t1_slot.has_value());
  REQUIRE(t2_slot.has_value());
  REQUIRE(t3_slot.has_value());

  // T0 is still alive while T1 exists.
  REQUIRE(t0_slot != t1_slot);

  // Most important assertion:
  // T0 is STILL alive when T2 is born because K2 consumes T0.
  REQUIRE(t0_slot != t2_slot);

  // T1 and T2 are simultaneously needed by K3.
  REQUIRE(t1_slot != t2_slot);

  // After K2 completes, T0 dies.
  // T3 is born at K3, so T0's physical storage may now be reused.
  REQUIRE(t3_slot == t0_slot);
}
TEST_CASE(
    "MemoryPlanner reduces allocation footprint versus naive allocation") {
  const auto f = make_three_kernel_chain();

  const auto plan = MemoryPlanner{}.plan(f.schedule, f.lowered);

  const auto naive_bytes = naive_planner_bytes(f.lowered, plan);
  const auto planned_bytes = planned_slot_bytes(plan);

  std::cout << "Naive bytes:   " << naive_bytes << '\n';
  std::cout << "Planned bytes: " << planned_bytes << '\n';
  std::cout << "Saved bytes:   " << (naive_bytes - planned_bytes) << '\n';
  std::cout << "Reduction:     "
            << (100.0 * (naive_bytes - planned_bytes) / naive_bytes) << "%\n";

  // Three planner-owned logical buffers, each Shape{4} / F32:
  //
  //   first  = 4 * sizeof(float) = 16 bytes
  //   second = 4 * sizeof(float) = 16 bytes
  //   third  = 4 * sizeof(float) = 16 bytes
  //
  // Naive allocation:
  //   16 + 16 + 16 = 48 bytes
  //
  // Lifetime reuse:
  //   first  -> S0
  //   second -> S1
  //   third  -> S0
  //
  // Planned allocation:
  //   S0 = 16 bytes
  //   S1 = 16 bytes
  //   total = 32 bytes

  REQUIRE(naive_bytes == 3 * 4 * sizeof(float));
  REQUIRE(planned_bytes == 2 * 4 * sizeof(float));
  REQUIRE(planned_bytes < naive_bytes);
}
