# Tensor Compiler + Runtime — Phase Plan

This is not "Component 1: build an IR." It is one integrated C++ artifact — a
tensor compiler and its runtime — built in phases where every phase extends the
same pipeline rather than sitting beside it. The name matters: thinking of it as
*IR alone* is what produces disconnected mini-projects. Thinking of it as one
runtime that grows around its IR is what produces compounding depth.

The finished shape, present in outline from the very first phase:

```
Frontend / High-level IR
        │
        ▼
   Rewrite system
        │
        ▼
    Scheduler
        │
        ▼
   Low-level IR
        │
        ▼
  Execution plan
        │
        ▼
       VM
        │
        ▼
  Memory planner
        │
        ▼
 CPU / (later) CUDA backend
```

Early on, several of these boxes are deliberately trivial. But every box sits in
its final architectural position from the day it is introduced. That is the
whole discipline.

---

## The two rules that govern every phase

**Rule 1 — Trivial implementation of a permanent abstraction is allowed. A
temporary abstraction is not, even when it is easier.**

A trivial scheduler is fine. A trivial VM is fine. A trivial allocator is fine.
Each is allowed to be dumb. None is allowed to occupy a position it will later
vacate. A one-operation-per-kernel scheduler that emits a real execution plan is
a trivial *permanent* abstraction. A compiler that compiles the graph straight
into arithmetic opcodes with no plan in between is a *temporary* abstraction —
it will be bypassed later, so it is scaffolding, so it is forbidden.

**Rule 2 — The correctness oracle must be semantically independent from the
production pipeline.**

Every phase is checked by executing the same graph two ways and demanding the
same number. That check is worthless if both paths share code. If the oracle and
the VM both reach their answer through the same lowering pass and the same scalar
kernels, they are one witness written twice — a shared bug lies to both of them
identically and the check passes while both are wrong.

So the oracle computes each operation the **direct** way (a matmul is three
nested loops, right there, no lowering, no IR transformation), while the
production side **reaches** that operation by transforming IR through the full
pipeline. Two independent roads to the same number. Only then does

```
oracle(graph) == VM(compile(graph))
```

actually mean something.

---

## The test for "this phase is not a toy"

"Does the source code survive?" is too weak — a dispatch loop can survive as code
while everything around it is rearchitected, in which case the design was thrown
away even though the file wasn't. Use three questions instead:

| Test               | Question                                                                 |
|--------------------|--------------------------------------------------------------------------|
| Artifact survival  | Is this data structure / code still part of the final system?            |
| Interface survival | Does the next phase *extend* this same boundary rather than bypass it?    |
| Semantic survival  | Does this phase establish an invariant that later phases depend upon?     |

Interface survival is the one that is easy to fail silently. It is the reason the
VM must consume an execution plan from day one, not recursively consume the
graph — so that the boundary the trivial scheduler emits through is the exact
boundary the sophisticated scheduler will emit through five phases later.

---

## Phase 0 — High-level IR + independent reference oracle

**Build:** the real high-level IR — `ValueId`, `Node`, `Op`, `Graph`, `Shape`,
`DType`. This is not a temporary AST to be discarded; it is the frontend the
scheduler and lowering will consume unchanged.

```
x ──────┐
        MUL ─────┐
y ──────┘        │
                 ADD ─── output
z ───────────────┘
```

**Also build:** a tree-walking interpreter `Tensor interpret(NodeId)` that gives
the IR direct reference semantics. Keep it **aggressively boring**. A matmul is:

```
for i
  for j
    for k
      C[i][j] += A[i][k] * B[k][j];
```

No fusion. No lowering. No shared kernels. No cleverness. Its boringness is the
point — it must fail *differently* from the production pipeline (Rule 2).

**Survival:** artifact — the IR is the permanent frontend. Interface — scheduler
and lowering consume it forever. Semantic — value identity and graph
dependencies are foundational invariants everything else assumes.

**Head start:** minitensor's autograd graph is this shape already. You are
re-meeting the DAG, not learning it. Move fast; spend the saved time making the
oracle so simple you trust it unconditionally.

**Done when:** you build a small graph by hand, run the oracle, and match a
hand computation.

---

## Phase 1 — Permanent execution-plan boundary + VM dispatch core

The mental frame is *not* "learn to build a bytecode VM." It is "establish the
final execution boundary of the compiler, today, at its smallest."

**Build:** the permanent runtime interface —

```
struct Instruction { Opcode opcode; /* operands */ };
struct Program     { std::vector<Instruction> instructions; };
VM::run(const Program&);
```

— and the dumbest possible compiler that targets it: one high-level op becomes
one kernel unit. For `c = a * b + d`:

```
ALLOC r0
MUL   r0, a, b
ALLOC r1
ADD   r1, r0, d
RETURN r1
```

This is unoptimized, but it is **not toy architecture** — it is the *baseline
legal schedule*. When fusion arrives later it transforms this plan; it does not
replace the boundary. The VM consumes a `Program`, never the graph. That single
decision is what makes every later scheduler upgrade land without touching the
VM interface.

**The dispatch loop** is flat — `enum` + `switch` (or computed goto) — never
virtual dispatch, because it runs millions of times. But: correct and permanent,
**not fast**, at this stage. Optimizing the dispatch loop before the scheduler
exists is depth in the wrong place. Speed is a later, separate pass over a
working system.

**Survival:** artifact — this dispatch loop is the literal execution core of the
final system. Interface — the sophisticated scheduler emits the same `Program`
this trivial one does. Semantic — instruction ordering and buffer semantics
become the runtime contract.

**Note (cold start):** unlike the IR, this is genuinely new ground — no prior
bytecode-VM muscle to lean on. It arrives in Phase 1, not Phase 6, precisely so
you meet the hard unfamiliar object while everything around it is still trivial
and it is the only hard thing on screen.

**Done when:** the same graph from Phase 0, compiled to a `Program` and run
through the VM, matches the oracle.

---

## Phase 2 — Shape + dtype inference and verification

Not "print `[M,N]` so I understand shapes." This is a compiler frontend /
type system, and its output is consumed structurally downstream.

```
A : [M,K]
B : [K,N]
   MATMUL(A,B)
      ↓
    [M,N]
```

```
A : [3,7]
B : [9,4]
   MATMUL(A,B)
      ↓
  compile-time graph error   (rejected, not mis-run)
```

Nodes carry shape and dtype; the pass propagates them through the graph, and an
illegal graph is refused at compile time. This is where C++ compile-time
reasoning becomes natural, and it feeds everything after it: scheduler legality,
buffer sizing, and the index arithmetic that lowering will emit.

**Survival:** artifact — shape metadata is permanent. Interface — scheduler and
lowering consume it. Semantic — "every legal graph is shape-correct" is an
invariant later phases rely on rather than re-checking.

**Done when:** graphs built without explicit shapes get correct shapes inferred,
and an illegal graph produces a real compile-time error.

---

## Phase 3 — Low-level IR + real lowering

This is the most important refinement over a naive plan. Do **not** think "add
LOAD/STORE opcodes to the VM." Introduce an actual **second IR level**.

High IR:

```
%2 = MATMUL %0, %1
```

lowers to Low IR:

```
for i = 0..M
  for j = 0..N
    %acc = 0
    for k = 0..K
      %a = LOAD A[i,k]
      %b = LOAD B[k,j]
      %p = MUL %a, %b
      %acc = ADD %acc, %p
    STORE C[i,j], %acc
```

then Low IR → execution plan → VM.

This is substantially deeper than "MATMUL opcode → a C++ matmul function,"
because that only teaches runtime dispatch, while this teaches compiler
*lowering* — the high-level-vs-low-level distinction that is the spine of
tinygrad's UOp system. If eventually reading tinygrad's UOps is a goal, this
distinction is the prerequisite, not an optional flourish.

The VM gains the load/store/loop opcodes to execute the lowered form — but it is
the *same* VM from Phase 1, extended, not a new one.

**Survival:** artifact — the low IR and lowering pass are permanent stages.
Interface — every graph flows through lowering from here on. Semantic — "the plan
the VM runs is always lowered low IR" becomes a structural guarantee.

**Done when:** a lowered matmul, executed through the VM, matches the oracle's
direct triple-loop — the two independent roads meet (Rule 2 pays off here).

---

## Phase 4 — Fusion first, rewrite engine second

Here is the one place where **both** source arguments are overridden in favor of
the principle they share.

One argument says: build a general `Pattern / Matcher / RewriteRule /
RewriteEngine` up front, with elementwise fusion as merely its first customer,
because graph-rewriting-as-a-mechanism (not fusion-as-a-feature) is the thing to
learn. That is correct about tinygrad — its whole optimizer is a rewrite engine.

But it violates **Rule 1**. A general rewrite engine is not the trivial
implementation of a permanent abstraction — it is heavy, generic infrastructure
(pattern language, matching, rule ordering, fixpoint iteration) built to serve
constant-folding, canonicalization, and algebraic simplification that *do not
exist yet*. Building it now is speculative generality: guessing the interface
from zero concrete rewrites.

So follow the shared principle, not either instruction:

1. Write elementwise-into-elementwise fusion as a **direct pass** first. Merge
   an elementwise op feeding another into a single kernel unit so the
   intermediate never touches memory. Prove it fired by counting eliminated
   intermediates; prove it is correct against the independent oracle.
2. **Extract** the general `RewriteEngine` at the moment you write your *second*
   rewrite — because that is the first moment the abstraction is real instead of
   imagined. One concrete rewrite teaches you what the engine must be; zero means
   you are inventing requirements.

The mechanism still ends up general and reusable (constant folding,
canonicalization, movement simplification, algebraic rewriting all become
customers) — but it is extracted from evidence, not designed from speculation.

**Survival:** artifact — the fusion pass, then the engine, are permanent.
Interface — the engine becomes the substrate the scheduler's hard fusion reuses.
Semantic — "rewrites preserve the oracle's number" is the invariant that makes
every future optimization safe to add.

**Done when:** a graph run with and without fusion produces an identical number,
with provably fewer materialized intermediates.

---

## Phase 5 — The real scheduler

This is the hardest and highest-value phase, and it is allowed to hurt. It is
where a toy compiler becomes a compiler. Everything it needs now exists: shapes
(Phase 2), low IR (Phase 3), the rewrite substrate (Phase 4).

Three jobs, in order:

**1. Dependency analysis.** Topologically order the graph so every op runs after
what it depends on. Straightforward; do it first.

**2. Kernel grouping / fusion decisions.** The core job. Easy start:

```
A
 \
  MUL ──┐
 /      │
B       ADD ── RELU
        │
C ──────┘
```

initially schedules as three kernels with two materialized temporaries; then
fuses to one kernel materializing only the result. The **hard** case is
reductions:

```
x
│
MUL
│
SUM(axis)
│
ADD bias
│
RELU
```

Now you must reason about dependencies, reduction boundaries, fusion legality,
recomputation, lifetimes, kernel grouping, materialization, memory traffic, and
register/liveness pressure — and, critically, about when **not** to fuse, because
fusing everything is not always fastest. This is the matmul-epilogue problem
(reduce + surrounding elementwise) made concrete.

**3. Materialization decisions.** Which intermediates get written to memory vs
kept live between fused ops.

Rule-based is enough. You do **not** need search (tinygrad's BEAM tries many
configurations and benchmarks them) — just know it exists as the sophisticated
version. The scheduler emits the execution plan; the VM interface does not
change. Expect to loop back and enrich the IR when the scheduler reveals missing
information — that backpressure is the compounding working as intended.

**Survival:** artifact — the scheduler is the heart of the final system.
Interface — it emits the same plan the trivial Phase 1 scheduler did, now
intelligent. Semantic — "the plan is a legal, dependency-respecting, fusion-valid
schedule" is the contract the runtime depends on.

**Done when:** a multi-op graph produces an explicit plan (ordered kernel units,
each marked with what it fuses and materializes), and executing that plan matches
the oracle.

---

## Phase 6 — Liveness + memory planning

Now the compiler becomes a runtime. Given a schedule:

```
K0 → T0
K1(T0) → T1
K2(T0) → T2
K3(T1,T2) → T3
```

derive the dependency structure and compute last-use:

```
T0 dies after K2
T1 dies at K3
T2 dies at K3
```

then allocate and reuse:

```
buffer A → T0
buffer B → T1
buffer C → T2
after T0 dies: buffer A becomes available again
```

This is where the C++ ownership decision bites — arena vs refcounted — and where
the VM stops being "an interpreter that runs arithmetic" and becomes an execution
engine. Measure success by peak-memory reduction; check the number still matches
the oracle one final time. This same muscle is what later KV-cache paging reuses.

**Survival:** artifact — memory planning is the last architectural piece.
Interface — allocation sits inside the VM/runtime boundary. Semantic — buffer
lifetime correctness becomes a runtime guarantee.

**Done when:** naive vs reuse allocation shows a real peak-memory drop, and the
result still matches the oracle.

---

## Phase 7 — Backend evolution (CPU → CUDA)

The compute ops in the VM start as plain CPU implementations. Once the whole
spine works and is verified, the backend becomes the join point where prior
kernel work rejoins the runtime: dispatch the VM's compute ops to real CUDA
kernels instead of CPU loops. This slots in at the VM's backend boundary without
disturbing anything above it — the oracle, the plan, the scheduler, and the VM
interface all stay fixed. It is deliberately last so the entire spine is
CPU-only and differentially verifiable before a GPU is ever involved.

---

## How the pipeline densifies (and never changes direction)

```
Phase 0:  IR ───────────────────────────────► oracle
Phase 1:  IR ─► trivial plan ─► VM
Phase 2:  IR+types ─► trivial plan ─► VM
Phase 3:  IR+types ─► low IR ─► plan ─► VM
Phase 4:  IR ─► rewrites ─► low IR ─► plan ─► VM
Phase 5:  IR ─► rewrites ─► scheduler ─► low IR ─► plan ─► VM
Phase 6:  IR ─► rewrites ─► scheduler ─► low IR ─► memory plan ─► VM
Phase 7:  (same) ─► VM ─► CPU / CUDA backend
```

The pipeline gets denser. It never reverses. That density-without-redirection is
the concrete meaning of "compounded depth."

---

## Boundaries to hold (so no phase expands to infinity)

- Each phase's stop condition is: its differential check against the oracle is
  green, **and** you can now read the corresponding tinygrad source (UOp,
  scheduler, codegen) and recognize your own design in it. Not
  production-grade — *readable-source-grade*.
- The scheduler (Phase 5) can absorb unlimited time. It deserves the most, but
  rule-based-and-correct is the ceiling for this pass; search is out of scope.
- Speed is always a later, separate pass over a working, verified system. A fast
  dispatch loop before the scheduler exists is effort in the wrong place.
- Learn concepts from general/compiler/paper sources first; use tinygrad as the
  worked example you read *after* building your own simplified version. Build in
  the middle is what makes the reading land.
