#include "tcr/compiler/compiler.hpp"
#include "tcr/ir/graph.hpp"
#include "tcr/ir/ids.hpp"
#include "tcr/ir/nodes.hpp"
#include "tcr/ir/op.hpp"
#include "tcr/ir/value.hpp"
#include "tcr/runtime/instruction.hpp"
#include "tcr/runtime/opcode.hpp"
#include "tcr/runtime/slot_id.hpp"
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
namespace tcr::compiler {
using tcr::ir::Graph;
using tcr::ir::NodeId;
using tcr::ir::Op;
using tcr::ir::Value;
using tcr::ir::ValueId;
using tcr::runtime::Opcode;
using tcr::runtime::SlotId;

namespace {
Opcode lower_opcode(Op op) {
  switch (op) {
  case Op::Add:
    return Opcode::ADD;
  case Op::Mul:
    return Opcode::MUL;
  case Op::MatMul:
    return Opcode::MATMUL;
  case Op::Sum:
    return Opcode::SUM;
  default:
    throw std::logic_error("unsupported IR operation");
  }
}
} // namespace

Program Compiler::compile(const Graph &graph) const {
  // visited nodes index = node_id and value = visited or not bool
  std::vector<bool> visited_nodes(graph.nodes_count(), false);

  // value_to_slot - a vector where index = valueid and value-at-index = slotid,
  // its size will be total values count in the graph values store
  std::vector<std::optional<SlotId>> value_to_slot(graph.values_count());

  // instructions vector which will keep all instructions emmited in the
  // recursive journey of lambda
  std::vector<runtime::Instruction> instructions;
  instructions.reserve(graph.nodes_count());

  // program inputs which will be a list of slotid which are pure/external
  // inputs
  std::vector<SlotId> program_inputs;
  program_inputs.reserve(graph.inputs_count());

  // slot incrementor to get next assignable slot id
  std::size_t next_slot = 0;

  // fill up external inputs in value_to_slot(which will later be fed with
  // intermediate inputs and outputs) table and also in program_inputs
  for (const auto input_value_id : graph.inputs()) {
    const auto slot = SlotId{next_slot++};
    value_to_slot[input_value_id.value] = slot;
    program_inputs.push_back(slot);
  }

  // recursive lambda - visiting a value_id output recursively till it reached
  // the external/pure inputs and then in return of the recursion emit the Node
  // operations as instructions
  auto visit = [&](auto &&self, ValueId output_value_id) -> void {
    // check the value details of output_value_id param
    const Value &output_value = graph.value(output_value_id);
    // if the producer is null it means this is a pure/external input so return
    if (!output_value.producer.has_value()) {
      return;
    }
    // so if it is a producer take the node and subject its input to same
    // visit() lambda
    const NodeId node_id = *output_value.producer;
    const ir::Node &node = graph.node(node_id);

    // if the node was already visited as part of earlier recursion
    if (visited_nodes[node_id.value]) {
      return;
    }

    for (const auto node_input : node.inputs) {
      self(self, node_input);
    }
    // after return from external input we come here in cases where the
    // output_value_id is a produced one not pure inputs now for the current
    // node prepare the instruction and emit it what are the inputs to this
    // instruction-with in current node, they are already in value_to_slot
    // table, put them in inputs
    std::vector<SlotId> instruction_inputs;
    instruction_inputs.reserve(node.inputs.size());
    for (const auto node_input : node.inputs) {
      const auto &slot = value_to_slot[node_input.value];
      if (!slot.has_value()) {
        throw std::logic_error(
            "runtime slot not available for operation-node's input");
      }
      instruction_inputs.push_back(*slot);
    }
    // save the output_slot to value_to_slot(input slots already there, both
    // intermediate and external output slots are collected as they are output
    // of nodes encountered in the recursion)
    const auto output_slot = SlotId(next_slot++);
    value_to_slot[output_value_id.value] = output_slot;

    // emit instruction finally
    instructions.push_back(runtime::Instruction{
        lower_opcode(node.op), std::move(instruction_inputs), output_slot});
    // mark the node as visited now
    visited_nodes[node_id.value] = true;
  }; // end of lambda

  // for eachoutput traverse the unvisited nodes
  for (const auto output_id : graph.outputs()) {
    visit(visit, output_id);
  }

  // program return object prep
  // program output is slotid list corresponding to valueid list of graph
  // output(external outputs)
  std::vector<SlotId> program_outputs;
  program_outputs.reserve(graph.outputs_count());
  for (const auto output_id : graph.outputs()) {
    // take out the slot for output value
    const auto slot = value_to_slot[output_id.value];
    if (!slot.has_value()) {
      throw std::logic_error(
          "slot id not available for the graph.output[n].valueid");
    }
    program_outputs.push_back(*slot);
  }
  return Program{std::move(instructions), std::move(program_inputs),
                 std::move(program_outputs), next_slot};
}
} // namespace tcr::compiler