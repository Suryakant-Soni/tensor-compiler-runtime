```text
graph object with inputs,outputs and intermeditate values and operation nodes
    │
    │ Compiler::compile(graph, outputs)
    ▼
┌─────────────────────────────┐
│ High IR: ir::Graph          │
│                             │
│ ValueId, Node, Op           │
│ Shape, Dtype                │
│ e.g. MatMul(A, B)           │
└─────────────┬───────────────┘
              │
              │ Lowering::lower(graph, outputs)
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