#pragma once

#include "tcr/compiler/lowering.hpp"
#include "tcr/compiler/memory_plan.hpp"
#include "tcr/compiler/schedule.hpp"
namespace tcr::compiler {
/*
Invariants of memory planning
1. Every planner managed logical buffer has exactly one slot assignment
2. External graph inputs are not planner-owned slot allocations
3. A graph output remains live throgh program completion
4. Two logical buffers may share a slot only when their lifetimes do not overlap
5. If A.last_use == B.birth, they may not share storage in the current version
6. Slot capacity must be >= every logical buffer assigned to it throughout
7. Slot alignment must satisfy alignment of every buffer
8. Buffers on incompatible devices/memory spaces never share a slot.
9.  Memory planning never changes graph semantics or scheduling decisions
10. Memory plannin never introduces recomputation
*/
class MemoryPlanner {

public:
  MemoryPlan plan(const Schedule &schedule,
                  const LoweringResult &lowered) const;
};
} // namespace tcr::compiler