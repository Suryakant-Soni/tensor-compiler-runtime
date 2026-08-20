## Component 1: IR (Intermediate Representation)

**What it is, plainly:** an IR is the data structure that represents a computation after you've parsed the user's code but before you execute it. Instead of running c = a @ b + d directly, you first build a graph: nodes for the matmul, the add, the inputs. That graph is the IR. Everything a compiler does — optimization, fusion, lowering to kernels — is transformation of this graph. This is the single most important thing to understand deeply, because it is the core of tinygrad (its UOp graph) and the thing that makes you able to read tinygrad's source.

**What to study first:**

The concept of a computation graph / DAG (directed acyclic graph) — nodes are operations, edges are data dependencies. You already met this in minitensor's autograd (the backward graph is exactly this), so you have a head start.
The distinction between levels of IR: a high-level IR (tensor ops: matmul, add, relu) vs a low-level IR (loops, loads, stores, scalar arithmetic). Real compilers lower from high to low. Tinygrad does exactly this — its UOp graph starts high and gets rewritten down toward something that maps to a kernel.
SSA (Static Single Assignment) form — a way of structuring IR where each value is assigned exactly once. It's the standard IR discipline in real compilers (LLVM, and tinygrad's UOp is SSA-like). Understand why it exists (it makes dependencies explicit and optimization easier).
Graph rewriting / pattern matching — how optimizations are expressed as "find this pattern in the graph, replace with that pattern." Tinygrad's whole optimization system is pattern-match rewrites on the UOp graph. This is the mechanism to understand.

**What to build (in your C++ project):**

A shape-aware IR: nodes carry tensor shape and dtype, and shape is propagated through the graph (a matmul of [M,K] and [K,N] produces [M,N] — the IR computes this). This is the "shape-aware" part your minitensor plan named, and it's where constexpr/compile-time reasoning in C++ becomes natural.
A few operation node types (elementwise add/mul, matmul, a reduction like sum, a reshape/movement op).
At least one graph transformation: the canonical one is operator fusion — detecting that an elementwise op feeding another elementwise op can be merged into one kernel. This is the optimization that makes tinygrad/Inductor fast, and building even a simple version teaches you the core of what these systems do.
A lowering pass: take the high-level IR (matmul node) and lower it to a low-level IR (nested loops with loads/stores). This is the bridge to the VM.

**The tinygrad connection to keep in view:** tinygrad's uop/ops.py (the UOp definition), its pattern-matcher/rewrite system, and its scheduler are exactly this component. Build your IR with the explicit goal of being able to read those files afterward and recognize your own design decisions in theirs.

## Component 2: VM (Virtual Machine / execution engine)

**What it is, plainly:** once you have a low-level IR (or bytecode), the VM is what actually executes it. It's the interpreter loop: read an instruction, do the operation, move to the next. For a tensor VM, the "instructions" are things like "load this buffer, run this kernel, store the result." The VM manages the buffers (memory), dispatches the operations, and handles the execution order.

**What to study first:**

Bytecode vs tree-walking interpretation — two ways to execute an IR. A tree-walker recursively evaluates the graph directly; a bytecode VM first compiles the graph to a flat sequence of instructions (bytecode) then executes that sequence. Bytecode is faster and is what "bytecode VM" in your plan means. Understand the tradeoff.
The dispatch loop — the hot inner loop that reads an opcode and jumps to its handler. This is the performance-critical part (recall our C++ discussion: this loop wants flat dispatch — enum+switch or computed goto — not virtual dispatch, because it runs millions of times).
Buffer/memory management — the VM has to allocate, track, and free the tensor buffers that flow between operations. This is where you decide ownership (the C++ ownership decision we discussed — arena? refcounted? this is where it bites).
Memory planning — a real optimization: the VM can reuse buffers (once a tensor is no longer needed, its memory can be reused by a later one). Tinygrad does this ("Memory Planning and Buffer Management" is a whole section of its architecture).

**What to build:**

A bytecode format: define the instruction set for your VM (LOAD, STORE, a set of compute ops, etc.).
A compiler from your low-level IR to that bytecode.
The execution loop that runs the bytecode, dispatching each instruction (flat dispatch, per our C++ discussion).
Buffer management with at least simple reuse (a buffer freed when its last consumer has run).
The execution backend: your VM's compute ops actually need to do the math — and here's where your CUDA work connects. Your VM can dispatch to CPU implementations first (simplest), then you can make it dispatch to GPU kernels — at which point the kernels you spent a month writing become the execution backend of your own runtime. That's the moment the two halves of your work (kernels + systems) join.

**The tinygrad/vLLM connection:** tinygrad's runner/execution system and JIT are this component. vLLM's execution engine (how it actually runs the model's operations) is a production version. Building this teaches you what both are doing under the hood.

## Component 3: Flash Attention

**What it is, plainly:** attention (the core operation in transformers) naively requires computing a huge N×N score matrix (for sequence length N), which is memory-expensive — you materialize the whole thing in memory. Flash attention is an algorithm that computes the same result without ever materializing the full score matrix, by processing it in blocks and keeping running statistics (the "online softmax"). It's faster and uses far less memory. It's arguably the single most important kernel in modern LLM inference.

**What to study first (this is where your CUDA warp knowledge pays off):**

The attention operation itself: Q, K, V matrices, scores = QKᵀ, softmax(scores), output = softmax·V. Understand this math cold first — it's the prerequisite.
The memory problem: why materializing the N×N score matrix is the bottleneck (it's O(N²) memory, which explodes for long sequences).
Online softmax — the mathematical trick at the heart of flash attention. You compute softmax incrementally, block by block, maintaining a running maximum and running sum, rescaling as you go, so you never hold the full row of scores at once. This is the conceptual core — study it until you can derive it.
The tiling/blocking structure — how flash attention blocks Q, K, V and processes them in a nested loop, keeping the working set in fast memory (shared memory/registers). This is where warptiling pays off: flash attention is a warp-cooperative kernel, and the "organize the warp to hold and share running state in registers" mental model you built (even though your shuffle kernel lost on throughput) is exactly what makes flash attention's structure readable.

**What to build:**

First, a naive attention implementation (materialize the score matrix) — as a correctness reference and to feel the memory problem.
Then a flash-attention-style implementation: blocked, with online softmax, never materializing the full matrix. Start on CPU or simple CUDA for correctness, then optimize. You don't need to match the real FlashAttention library's performance — you need to understand and implement the algorithm, because that understanding is what makes you able to read and contribute to vLLM's attention kernels.

**The connection:** vLLM's core is efficient attention + serving. Understanding flash attention is close to a prerequisite for meaningful vLLM kernel contribution.

## Component 4: KV Cache

**What it is, plainly:** during LLM inference (generating text token by token), each new token needs to attend to all previous tokens. Recomputing the K and V matrices for all previous tokens every step would be wasteful — so you cache them (the "KV cache"). But this cache grows with sequence length and there's one per request, so managing this memory efficiently — across many concurrent requests of different lengths — is a hard systems problem. This is a memory-management problem, which is why it comes after you've built VM-level buffer management.

**What to study first:**

Why the KV cache exists — the token-by-token generation loop and the redundant recomputation it avoids. Understand the inference loop first.
The memory challenge — the cache is large, variable-length, and there's one per concurrent request. Naive allocation (a fixed max-length buffer per request) wastes enormous memory.
PagedAttention — vLLM's signature innovation. It applies the operating-system paging idea to the KV cache: instead of one contiguous buffer per request, break the cache into fixed-size blocks (pages) and allocate them on demand, with a lookup table mapping logical positions to physical blocks. This lets many requests share memory efficiently. This is the thing to understand deeply — it's vLLM's core contribution, and it's pure systems (it's literally OS memory paging applied to ML).
Continuous batching / scheduling — how vLLM serves many requests at once, adding and removing them from batches dynamically. This is the serving-system layer around the KV cache.

**What to build:**

A simple token-by-token generation loop with a naive KV cache (contiguous per request) — to feel the problem.
A paged KV cache: fixed-size blocks, on-demand allocation, a block table mapping logical→physical. Even a simplified version teaches you the core idea and makes vLLM's implementation readable.

**The connection:** this is vLLM's heart. Understanding paged KV cache is close to essential for vLLM contribution, and it's the most directly "systems" of the four — it connects to your interest in memory management and runtime design.

## The sequence, and why
IR first — it's the foundation; everything represents computation as IR. Also the biggest tinygrad overlap.
VM second — it executes the IR, and it's where your CUDA kernels rejoin the story as the execution backend. IR+VM together are your C++ project and your tinygrad-readiness.
Flash attention third — a compute algorithm that runs on a runtime; needs the attention-math and warp foundations. Your CUDA warp work pays off here.
KV cache fourth — the memory-management system around attention inference; needs you to understand attention (from 3) and buffer management (from 2) first. Most directly vLLM.

Do IR and VM as one integrated C++ project (they're two halves of one runtime). Do flash attention and KV cache as focused implementations that can be smaller/rougher — their job is understanding for contribution readiness, not being production runtimes.

## What you're missing — three things worth adding

**1. The scheduler, explicitly.** You've mentioned IR/VM but the scheduler — the component that decides execution order, what gets fused, what runs when — is arguably the hardest and most valuable part, and it sits between IR and VM. Tinygrad's scheduler is a huge part of its architecture. Treat scheduling as a first-class sub-topic of the IR/VM work, not an afterthought. It's also, as we discussed, the same thing as the "scheduler in C++" project you were independently drawn to.

**2. Quantization — at least conceptually.** Both tinygrad and vLLM deal heavily with quantization (representing weights in fewer bits — int8, int4 — to save memory and bandwidth). You don't need to implement a full quantization scheme, but understand what it is and why it matters for inference, because it comes up constantly in vLLM kernel work and you'll hit it immediately.

**3. Actually reading the target codebases, as a deliberate step.** After IR/VM and before/during flash attention and KV cache, spend real time reading tinygrad's source (the UOp, scheduler, codegen files) and vLLM's core (paged attention, the engine). Building your own versions is to make their code readable — so the reading is not optional, it's the payoff. Budget it as its own phase: "read tinygrad's scheduler with my IR/VM understanding" is a concrete, valuable activity that directly precedes contributing.

One honest note given your runway: this is a lot, and each piece can expand indefinitely. The discipline from the CUDA arc applies — build enough of each to understand it and read the real thing, then move; don't turn any one into a production system. The IR/VM is your deep C++ artifact and deserves the most time; flash attention and KV cache are "understand deeply, implement enough to prove you get it, then read the real implementations." The goal across all four is contribution-readiness, not four more finished frameworks — the finished artifact is the IR/VM; the other two are understanding-vehicles that make tinygrad and vLLM's code yours to modify.


## Now the two components you asked to fit in.

### Scheduler (sits between IR and VM)

**What it is, plainly:** given the IR graph, the scheduler decides what runs when and what gets grouped together. It answers: which operations fuse into a single kernel? In what order do kernels execute given their dependencies? Which intermediate results need to be materialized in memory versus kept in registers? It's the component that turns a graph of individual operations into an efficient execution plan. It's the hardest and highest-value part of a tensor compiler, and it's the heart of tinygrad specifically.

**What to learn, in order:**

Dependency analysis — building the order constraints from the IR graph (op B needs op A's output, so A before B). This is topological ordering of the DAG. Straightforward, do it first.
Kernel fusion decisions — the core scheduler job. Which adjacent operations should merge into one kernel? Elementwise-into-elementwise is easy; the hard cases are fusing a reduction with the elementwise ops around it, and deciding when not to fuse (fusing everything isn't always best). This is where tinygrad spends enormous effort ("Schedule Creation and Kernel Grouping" in its architecture).
Materialization decisions — which intermediate tensors get written to memory vs kept live. Fewer materializations = less memory traffic = faster, but you can't keep everything live. This trades off against the memory planning you built in the VM.
Scheduling heuristics vs search — simple schedulers use rules ("always fuse elementwise chains"); advanced ones search over possible schedules and pick the fastest (tinygrad's BEAM search does this — it literally tries multiple kernel configurations and benchmarks them). Understand the rule-based version first, then know that search exists.

**What to build:** extend your IR/VM project with a scheduler pass that (1) topologically orders the graph, (2) groups fusable ops into kernel-units, (3) decides materialization. A rule-based scheduler is enough — you don't need BEAM search. The point is to understand the decisions, which makes tinygrad's far more sophisticated scheduler readable.

**Where it goes in the sequence:** it's not a separate phase — it's the middle of the IR/VM work. You build IR (represent the graph), then scheduler (plan its execution), then VM (execute the plan). So it's woven into components 1-2, not added after. This is the "scheduler in C++" project you were independently drawn to — it's the same thing as the IR/VM's planning layer.

### Quantization (a focused study topic, lighter than the others)

**What it is, plainly:** representing numbers with fewer bits. Instead of 32-bit floats for every weight, use 8-bit or 4-bit integers, with a scale factor to map the small integer range back to real values. This cuts memory (a model takes 4-8× less space) and memory bandwidth (the usual inference bottleneck), at some accuracy cost. It's everywhere in modern inference.

**What to learn, in order:**

The basic idea — how a float tensor maps to (int values + a scale + maybe a zero-point). Understand the arithmetic of quantize and dequantize. This is the foundation and it's quick.
Quantization granularity — per-tensor vs per-channel vs per-group scales. Finer granularity = better accuracy, more overhead. Know the tradeoff.
The main formats you'll see in vLLM/tinygrad — int8, int4, and the named schemes (GPTQ, AWQ for weights; FP8 on newer hardware). You don't need to implement these — you need to recognize them and know what problem each solves, because they appear constantly in vLLM kernel code.
Why it's a kernel problem — quantized matmul needs special kernels (the multiply happens in int, accumulation and rescaling need care). This connects back to your CUDA work: quantized GEMM is a variant of the SGEMM you already understand, and this is where your kernel depth meets the inference-serving world. This connection is worth internalizing — it's a natural bridge from what you've done to what vLLM does.

**What to build:** minimal — implement quantize/dequantize for int8, and optionally a simple quantized matmul (int8 inputs, int32 accumulate, rescale). This is much smaller than the other components. Its purpose is literacy, not mastery — enough to read quantized kernels, not to write production ones.

**Where it goes:** after flash attention and KV cache, or interleaved with them as a lighter side-study. It's the smallest of the six and doesn't gate anything else, so it's flexible.

## Will you learn these from tinygrad and vLLM resources?

Partly, and here's the honest breakdown, because relying only on their codebases as learning resources is a trap:

**For the concepts:** no, don't learn the concepts from tinygrad/vLLM source first — their code assumes you already understand the concepts, so reading it cold is inefficient. Learn the concepts from general sources (compiler and systems material for scheduling; the standard quantization explanations and the GPTQ/AWQ papers for quantization; the FlashAttention paper for attention), then use tinygrad/vLLM as the worked example you read once you understand the idea.

**For the specific implementations:** yes, absolutely — once you understand a concept, tinygrad and vLLM are the best resources for seeing it done in practice. Tinygrad's scheduler is a real, readable (small) scheduler. vLLM's paged attention is the reference implementation of that idea. Reading them after building your own simplified version is where they become teachers rather than walls.

So the pattern for every component: concept from general/paper sources → build a simplified version yourself → read the tinygrad/vLLM implementation as the "how the pros do it" capstone. The building-in-the-middle is what makes the reading land.

## Full sequence with rough time allocation

I'll give proportions rather than hard weeks, since your pace varies and I don't want to overspecify against your runway. Treat these as relative sizes.

IR (largest) — the foundation, most tinygrad overlap, your deep C++ artifact begins here.
Scheduler (large) — woven in after IR, before/with VM. The hardest, highest-value piece.
VM (large) — executes the plan; your CUDA kernels rejoin as the backend. IR+scheduler+VM together are one integrated C++ project and the bulk of this phase.
Flash attention (medium) — compute algorithm on top of the runtime; warp knowledge pays off.
KV cache (medium) — memory system around attention inference; most directly vLLM.
Quantization (smallest) — literacy study, interleaved or after; bridges your kernel work to inference serving.

The IR/scheduler/VM block (1-3) is the majority of the effort and is the C++ artifact that anchors everything. Components 4-6 are lighter, understanding-focused, and more directly contribution-preparation for the specific target codebases.