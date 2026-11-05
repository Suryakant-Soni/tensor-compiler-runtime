# Tensor Compiler + Runtime Architecture

**Last updated:** November 5, 2026

```text
graph object
(inputs, outputs, intermediate values, operation nodes)
    │
    │ Compiler::compile(graph)
    ▼
┌─────────────────────────────┐
│ High IR: ir::Graph          │
│                             │
│ ValueId, Node, Op           │
│ Shape, DType                │
│                             │
│ Add, Mul, MatMul, Sum       │
│ FusedElementwise            │
└─────────────┬───────────────┘
              │
              │ ElementwiseFusion
              │
              ▼
┌─────────────────────────────┐
│ Optimized High IR           │
│ ir::Graph                   │
│                             │
│ Eligible Add/Mul chains     │
│ represented as              │
│ FusedElementwise nodes      │
└─────────────┬───────────────┘
              │
              │ Scheduler::schedule(graph)
              ▼
┌─────────────────────────────┐
│ compiler::Schedule          │
│                             │
│ ordered KernelUnits         │
│                             │
│ members                     │
│ boundary_inputs             │
│ materialized_outputs        │
│                             │
│ decides:                    │
│ - execution order           │
│ - kernel grouping           │
│ - materialization           │
└─────────────┬───────────────┘
              │
              │ Lowering::lower(graph, schedule)
              │
              │ lowering obeys the schedule;
              │ computed dependencies are not
              │ recursively lowered
              ▼
┌─────────────────────────────┐
│ Low IR: lir::Program        │
│                             │
│ BufferId  → tensor buffers  │
│ ScalarId  → scalar temps    │
│ IndexId   → index temps     │
│                             │
│ For, Load, Store,           │
│ Add, Mul, IndexAdd...       │
│                             │
│ Grouped kernels may lower   │
│ together so internal values │
│ need not be materialized    │
└─────────────┬───────────────┘
              │
              │ ProgramEmitter::emit(lir_program)
              │ BufferId → SlotId
              │ structured loops → flat jumps
              ▼
┌─────────────────────────────┐
│ runtime::Program            │
│                             │
│ vector<Instruction>         │
│ input/output SlotIds        │
│ slots_count                 │
│ scalar_count                │
│ index_count                 │
└─────────────┬───────────────┘
              │
              │ VM::run(program, bindings)
              ▼
┌─────────────────────────────┐
│ VM runtime state            │
│                             │
│ slots   : vector<Tensor>    │ ← SlotId
│ scalars : vector<float>     │ ← ScalarId
│ indices : vector<size_t>    │ ← IndexId
└─────────────┬───────────────┘
              │
              │ pc = 0
              │ while (pc < instructions.size())
              ▼
┌─────────────────────────────┐
│ switch (instruction.opcode) │
│                             │
│ Constant  → scalars[...]    │
│ Load      → slot → scalar   │
│ Store     → scalar → slot   │
│ Add/Mul   → scalar ops      │
│ Index*    → index ops       │
│ LoopBegin → initialize/jump │
│ LoopEnd   → increment/jump  │
└─────────────┬───────────────┘
              │
              ▼
        Output Tensor(s)
```

## Phase 6 Memory Planner — Measured Allocation Reduction

The memory planner now has a direct regression test comparing naive logical-buffer allocation with lifetime-based physical-slot reuse.

Test:

```text
MemoryPlanner reduces allocation footprint versus naive allocation
```

Measured result:

| Metric             |    Bytes |
| ------------------ | -------: |
| Naive allocation   |       48 |
| Planned allocation |       32 |
| Bytes saved        |       16 |
| Reduction          | 33.3333% |

For the tested three-kernel chain:

```text
first  = Shape{4}, F32 = 16 bytes
second = Shape{4}, F32 = 16 bytes
third  = Shape{4}, F32 = 16 bytes
```

Naive allocation gives every planner-owned logical buffer separate storage:

```text
16 + 16 + 16 = 48 bytes
```

Lifetime-aware planning reuses storage:

```text
first  → S0
second → S1
third  → S0
```

Therefore only two physical slots are required:

```text
S0 = 16 bytes
S1 = 16 bytes

Total planned allocation = 32 bytes
```

So the planner reduces the physical allocation footprint from **48 bytes to 32 bytes**, saving **16 bytes**, or **33.3333%**, for this test graph.

To reproduce the measurement from the build directory:

```bash
ctest -R "MemoryPlanner reduces allocation footprint versus naive allocation" -V
```

Expected diagnostic output:

```text
Naive bytes:   48
Planned bytes: 32
Saved bytes:   16
Reduction:     33.3333%
```
