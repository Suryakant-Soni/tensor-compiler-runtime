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