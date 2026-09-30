#include "tcr/compiler/scheduler.hpp"
#include "tcr/ir/graph.hpp"
#include "tcr/ir/ids.hpp"
#include "tcr/ir/op.hpp"
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
struct Analysis {
  std::vector<bool> is_reachable;
  std::vector<bool> is_graph_output;
  std::vector<std::vector<tcr::ir::ValueId>> consumers;
  std::vector<tcr::ir::ValueId> topological_values;
};

// visit each node and then visit the inputs recursively to prepare initial
// topological sequnce and other analysis details
void visit(tcr::ir::ValueId value_id, Analysis &analysis,
           const tcr::ir::Graph &graph) {
  if (analysis.is_reachable[value_id.value]) {
    return;
  }
  analysis.is_reachable[value_id.value] = true;
  analysis.is_reachable[value_id.value] = true;
  const auto &value = graph.value(value_id);
  if (!value.producer.has_value()) {
    return;
  }
  const auto &node = graph.node(*value.producer);
  for (auto const node_input : node.inputs) {
    visit(node_input, analysis, graph);
    // consumer of input value id is the output value id
    auto &consumers = analysis.consumers[node_input.value];
    if (std::find(consumers.begin(), consumers.end(), value_id) ==
        consumers.end()) {
      analysis.consumers[node_input.value].push_back(value_id);
    }
  }
  // in last while returning add topological outcomes, final and intermediate,
  // input would not come here
  analysis.topological_values.push_back(value_id);
}

// preparing analysis records for a graph and its values
Analysis analyze(const tcr::ir::Graph &graph) {
  std::vector<bool> visited(graph.values_count(), false);
  Analysis analysis;
  const auto value_count = graph.values_count();
  analysis.is_graph_output.resize(value_count, false);
  analysis.is_reachable.resize(value_count, false);
  analysis.consumers.resize(value_count);

  // for all the output
  for (const auto output_value_id : graph.outputs()) {
    visit(output_value_id, analysis, graph);
    analysis.is_graph_output[output_value_id.value] = true;
  }
  return analysis;
}

// make each operation a separate kernel initially before grouping like applies
tcr::compiler::Schedule create_singleton_partition(const tcr::ir::Graph &graph,
                                                   const Analysis &analysis) {
  std::vector<tcr::compiler::KernelUnit> kernels;
  for (const auto topo_value : analysis.topological_values) {
    const auto &value = graph.value(topo_value);
    if (!value.producer.has_value()) {
      throw std::logic_error("topological values of analysis should have "
                             "inputs assigned to the node");
    }
    tcr::compiler::KernelUnit kernel_unit;
    kernel_unit.members.push_back(topo_value);
    kernels.push_back(std::move(kernel_unit));
  }
  return tcr::compiler::Schedule{std::move(kernels)};
}

bool can_group_elementwise_sum(tcr::ir::ValueId producer,
                               tcr::ir::ValueId consumer,
                               const tcr::ir::Graph &graph,
                               const Analysis &analysis) {
  // consumer is sum,
  const auto &value_con = graph.value(consumer);
  if (!value_con.producer.has_value()) {
    throw std::logic_error("consumer valueid should be a node op");
  }
  const auto &node_con = graph.node(*value_con.producer);
  if (node_con.op != tcr::ir::Op::Sum) {
    return false;
  }
  // only one consumer,
  if (analysis.consumers[producer.value].size() != 1) {
    return false;
  }
  // producer is elementwise,
  const auto &value_pro = graph.value(producer);
  if (!value_pro.producer.has_value()) {
    throw std::logic_error("producer valueid should be a node op");
  }
  const auto &node_pro = graph.node(*value_pro.producer);
  if (node_pro.op != tcr::ir::Op::Add && node_pro.op != tcr::ir::Op::Mul &&
      node_pro.op != tcr::ir::Op::FusedElementwise) {
    return false;
  }
  // producer is not the graph output
  if (analysis.is_graph_output[producer.value]) {
    return false;
  }
  return true;
}

// a vector map where index-> valueid and value -> kernel index
std::vector<size_t> build_owner(const tcr::compiler::Schedule &schedule,
                                size_t value_count) {
  constexpr size_t no_owner = std::numeric_limits<size_t>::max();
  std::vector<size_t> owner(value_count, no_owner);
  for (size_t k = 0; k < schedule.Kernels.size(); ++k) {
    for (const auto member : schedule.Kernels[k].members) {
      owner[member.value] = k;
    }
  }
  return owner;
}

void apply_grouping_policies(tcr::compiler::Schedule &schedule,
                             const Analysis &analysis,
                             const tcr::ir::Graph &graph,
                             std::vector<size_t> &owner) {
  // loop all topological values as producer and check its corresponding
  // consumers if the grouping can be done
  std::vector<std::pair<tcr::ir::ValueId, tcr::ir::ValueId>> merges;
  for (const auto producer : analysis.topological_values) {
    for (const auto consumer : analysis.consumers[producer.value]) {
      if (can_group_elementwise_sum(producer, consumer, graph, analysis)) {
        merges.push_back({producer, consumer});
      }
    }
  }
  for (const auto merge : merges) {
    schedule.Kernels[owner[merge.first.value]].members.push_back(merge.second);
    std::erase(schedule.Kernels[owner[merge.second.value]].members,
               merge.second);
    owner[merge.second.value] = owner[merge.first.value];
  }
  // after merges are done, remove all the empty kernels which do not have any
  // member
  std::erase_if(schedule.Kernels,
                [](const auto &kernel) { return kernel.members.empty(); });
}

// you only have kernel units and members into it, now calculate kernelunit's
// boundary inputs and outputs which need materialization
void recompute_boundaries(tcr::compiler::Schedule &schedule,
                          const Analysis &analysis,
                          const std::vector<size_t> &owner,
                          const tcr::ir::Graph &graph) {
  for (size_t k = 0; k < schedule.Kernels.size(); k++) {
    auto &kernel_unit = schedule.Kernels[k];
    kernel_unit.boundary_inputs.clear();
    kernel_unit.materialized_outputs.clear();
    const auto &members = kernel_unit.members;
    for (const auto member : members) {
      // for each member, find inputs and check if the input is co-member or not
      const auto &value = graph.value(member);
      if (!value.producer.has_value()) {
        throw std::logic_error(
            "kernel member should be an operation value not input");
      }
      const auto &node = graph.node(*value.producer);
      for (const auto input_value : node.inputs) {
        if (std::find(members.begin(), members.end(), input_value) ==
            members.end()) {
          auto &inputs = kernel_unit.boundary_inputs;
          if (std::find(inputs.begin(), inputs.end(), input_value) ==
              inputs.end()) {
            inputs.emplace_back(input_value);
          }
        }
      }
      // for each member check if it is the output to be materialized
      if (analysis.is_graph_output[member.value]) {
        kernel_unit.materialized_outputs.emplace_back(member);
        continue;
      }
      // find all the consumers of this member in the wholge graph
      const auto &consumers = analysis.consumers[member.value];
      // if any consumer lies in any other kernel than the value needs to be
      // materialized
      for (const auto consumer : consumers) {
        if (owner[consumer.value] != k) {
          kernel_unit.materialized_outputs.emplace_back(member);
          break;
        }
      }
    }
  }
}

} // namespace

namespace tcr::compiler {
Schedule Scheduler::schedule(const ir::Graph &graph) const {
  // first understand the incoming graph
  const Analysis analysis = analyze(graph);

  // create a plain single partion kernels for graph
  Schedule schedule = create_singleton_partition(graph, analysis);

  std::vector<size_t> owner = build_owner(schedule, graph.values_count());

  // apply grouping policies to the singleton kernels
  apply_grouping_policies(schedule, analysis, graph, owner);

  owner = build_owner(schedule, graph.values_count());
  // recompute the inputs and outputs to be materialized as per the new
  // rewritten/grouped schedule
  recompute_boundaries(schedule, analysis, owner, graph);

  return schedule;
}
} // namespace tcr::compiler