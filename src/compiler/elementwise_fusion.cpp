#include "tcr/compiler/elementwise_fusion.hpp"
#include "tcr/ir/graph.hpp"
#include "tcr/ir/ids.hpp"
#include "tcr/ir/nodes.hpp"
#include "tcr/ir/op.hpp"
#include "tcr/ir/value.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
namespace tcr::compiler {
using tcr::ir::Op;
namespace {

struct FusionCandidate {
  // boundary inputs after fusion expression is completed
  std::vector<ir::ValueId> boundary_inputs;
  // false - root node, nothing absorbed till now
  // true - atleast one node(apart from root node) is absorbed
  bool is_absorbed{false};
};

bool is_elementwise_op(ir::Op op) {
  switch (op) {
  case ir::Op::Add:
  case ir::Op::Mul:
    return true;
  default:
    return false;
  }
}

ir::ElementwiseExprKind get_elementwise_kind(Op op) {
  switch (op) {
  case Op::Add:
    return ir::ElementwiseExprKind::Add;
  case Op::Mul:
    return ir::ElementwiseExprKind::Mul;
  default:
    throw std::logic_error("not elementwise operation kind");
  }
}

ir::ValueId recreate_node(const ir::Node &old_node, ir::Graph &new_graph,
                          const std::vector<ir::ValueId> &new_inputs) {
  switch (old_node.op) {
  case Op::Add:
    return new_graph.add(new_inputs[0], new_inputs[1]);
  case Op::Mul:
    return new_graph.mul(new_inputs[0], new_inputs[1]);
  case Op::Sum:
    return new_graph.sum(new_inputs[0]);
  case Op::MatMul:
    return new_graph.matmul(new_inputs[0], new_inputs[1]);
  case tcr::ir::Op::Input:
    throw std::logic_error("Input value also has a producer by mistake");
  case tcr::ir::Op::FusedElementwise:
    throw std::logic_error(
        "FusedElementwise node already present wrongly; before fusion");
  }
}

// -----------------------------------------------------------------------------
// prepare input kind elementwise expression for the new fused node
// ------------------------------------------------------------------------------
const ir::ElementwiseExpr
make_fusion_boundary_input(const ir::ValueId &old_valueid,
                           FusionCandidate &candidate) {

  const auto index = candidate.boundary_inputs.size();
  candidate.boundary_inputs.push_back(old_valueid);
  return ir::ElementwiseExpr{
      .kind = ir::ElementwiseExprKind::Input,
      .input_index = index,
      .lhs = nullptr,
      .rhs = nullptr,
  };
}

const ir::ElementwiseExpr
build_fusion_expr(ir::ValueId v, const ir::Graph &old_graph,
                  const std::vector<std::size_t> &use_count,
                  const std::vector<bool> &is_output,
                  FusionCandidate &candidate) {
  const auto &value = old_graph.value(v);

  // -----------------------------------------------------------------------------
  // Valueid received is an external input
  // ------------------------------------------------------------------------------
  if (!value.producer.has_value()) {
    return make_fusion_boundary_input(v, candidate);
  }

  // -----------------------------------------------------------------------------
  // Valueid received has more than one consumer nodes
  // ------------------------------------------------------------------------------
  if (use_count[v.value] != 1) {
    return make_fusion_boundary_input(v, candidate);
  }

  // -----------------------------------------------------------------------------
  // Valueid received is one of the external outputs
  // ------------------------------------------------------------------------------
  if (is_output[v.value]) {
    return make_fusion_boundary_input(v, candidate);
  }

  const auto &node = old_graph.node(*value.producer);

  // -----------------------------------------------------------------------------
  // node of received value id is not elementwise, return it as boundary i/p
  // ------------------------------------------------------------------------------
  if (!is_elementwise_op(node.op)) {
    return make_fusion_boundary_input(v, candidate);
  }

  // cross check if the operation is add or mul for fusion
  if (node.inputs.size() != 2) {
    throw std::logic_error(
        "seleceted elementwise op for fusion should have 2 inputs");
  }

  // -----------------------------------------------------------------------------
  // Case: Node is eligible to be absorbed into the fused island;
  // ------------------------------------------------------------------------------
  // recurse to get deep expressions of lhs&rhs and then combine it to get
  // parent expresson
  candidate.is_absorbed = true;
  auto lhs = build_fusion_expr(node.inputs[0], old_graph, use_count, is_output,
                               candidate);
  auto rhs = build_fusion_expr(node.inputs[1], old_graph, use_count, is_output,
                               candidate);
  return ir::ElementwiseExpr{
      .kind = get_elementwise_kind(node.op),
      .input_index = 0,
      .lhs = std::make_unique<ir::ElementwiseExpr>(std::move(lhs)),
      .rhs = std::make_unique<ir::ElementwiseExpr>(std::move(rhs)),
  };
}

// takes in old output_value, recurses and rewrites internal components and come
// out and returns back the equivalent-semantic new output_value for new graph
ir::ValueId rewrite_value(const ir::ValueId old_v, const ir::Graph &old_graph,
                          ir::Graph &new_graph,
                          const std::vector<std::size_t> &use_count,
                          const std::vector<bool> &is_output,
                          std::vector<std::optional<ir::ValueId>> &remap) {

  // -----------------------------------------------------------------------------
  // Value is already written in new graph
  // ------------------------------------------------------------------------------
  if (remap[old_v.value]) {
    return *remap[old_v.value];
  }

  const auto &output_value = old_graph.value(old_v);

  // -----------------------------------------------------------------------------
  // producer not present,unexpected since before starting rewrite we populated
  // all external inputs to new graph
  // ------------------------------------------------------------------------------
  if (!output_value.producer.has_value()) {
    throw std::logic_error("graph input should already be present in the remap");
  }

  const auto &node = old_graph.node(*output_value.producer);

  // -----------------------------------------------------------------------------
  // node is non-elementwise, non-fusible; recursion/rewrite into inputs of node
  // ------------------------------------------------------------------------------
  if (!is_elementwise_op(node.op)) {
    std::vector<ir::ValueId> new_node_inputs;
    new_node_inputs.reserve(node.inputs.size());
    for (const auto input : node.inputs) {
      new_node_inputs.push_back(rewrite_value(input, old_graph, new_graph,
                                              use_count, is_output, remap));
    }
    const auto new_v = recreate_node(node, new_graph, new_node_inputs);
    remap[old_v.value] = new_v;
    return new_v;
  }

  // check if no. of inputs is not 2
  if (node.inputs.size() != 2) {
    throw std::logic_error(
        "no. of inputs should be 2, since the node is an elementwise op");
  }

  // -----------------------------------------------------------------------------
  // node is an elementwise(fusible); build deep expr for both inputs - lhs,rhs
  // ------------------------------------------------------------------------------
  FusionCandidate candidate;

  auto lhs = build_fusion_expr(node.inputs[0], old_graph, use_count, is_output,
                               candidate);
  auto rhs = build_fusion_expr(node.inputs[1], old_graph, use_count, is_output,
                               candidate);

  ir::ElementwiseExpr root_expr{
      .kind = get_elementwise_kind(node.op),
      .input_index = 0,
      .lhs = std::make_unique<ir::ElementwiseExpr>(std::move(lhs)),
      .rhs = std::make_unique<ir::ElementwiseExpr>(std::move(rhs))};

  // -----------------------------------------------------------------------------
  // no node is absorbed after recursing using build_fusion_expr; rewrite root
  // node as plain node to new graph
  // ------------------------------------------------------------------------------
  if (!candidate.is_absorbed) {
    std::vector<ir::ValueId> new_inputs;
    new_inputs.reserve(node.inputs.size());
    for (const auto input : node.inputs) {
      new_inputs.push_back(rewrite_value(input, old_graph, new_graph, use_count,
                                         is_output, remap));
    }
    const auto new_v = recreate_node(node, new_graph, new_inputs);
    remap[old_v.value] = new_v;
    return new_v;
  }

  // -----------------------------------------------------------------------------
  // after returning from recursion-build-expression, recurse on boundary inputs
  // of fusible island
  // ------------------------------------------------------------------------------
  // you need to rewrite boundary values which were non fusible and left out
  // values returned from boundary inputs will be the node.inputs for
  // new fused node, use them while creating the fused node operation on new
  // graph
  std::vector<ir::ValueId> fused_node_inputs;
  fused_node_inputs.reserve(candidate.boundary_inputs.size());
  for (const auto &boundary_input : candidate.boundary_inputs) {
    auto input = rewrite_value(boundary_input, old_graph, new_graph, use_count,
                               is_output, remap);
    fused_node_inputs.push_back(input);
  }
  // create the fused node
  const auto new_value_id =
      new_graph.fused_elementwise(fused_node_inputs, std::move(root_expr));
  remap[old_v.value] = new_value_id;
  return new_value_id;
}

} // namespace

/**
ElementwiseFusion::run
        ↓
rewrite_value
        ↓
   Add/Mul root?
     /      \
   no        yes
   ↓          ↓
rewrite     build_fusion_expr
inputs          ↓
             discover
          fusible island
                ↓
         boundary old ValueIds
                ↓
          rewrite_value
          on boundaries
                ↓
        create FusedElementwise
 */

const ir::Graph ElementwiseFusion::run(const ir::Graph &old_graph) const {

  // -----------------------------------------------------------------------------
  // to validate if there is only single consumer for a value for it to be
  // fusible; index-> valueid and value - count total
  // ------------------------------------------------------------------------------
  std::vector<std::size_t> use_count(old_graph.values_count(), 0);
  for (std::int32_t i = 0; i < old_graph.nodes_count(); ++i) {
    const auto &inputs = old_graph.node(ir::NodeId{i}).inputs;
    for (const auto &input : inputs) {
      ++use_count[input.value];
    }
  }
  // -----------------------------------------------------------------------------
  // check if the valueid is external output or not
  // ------------------------------------------------------------------------------
  std::vector<bool> is_output(old_graph.values_count(), false);
  for (const auto out_v : old_graph.outputs()) {
    is_output[out_v.value] = true;
  }

  // map of old valueid->index to new valueid->value
  std::vector<std::optional<ir::ValueId>> remap(old_graph.values_count());

  ir::Graph new_graph;

  // -----------------------------------------------------------------------------
  // populate external inputs from old graph to new graph, fill remap for them
  // ------------------------------------------------------------------------------
  for (const auto old_input : old_graph.inputs()) {
    const auto value = old_graph.value(old_input);
    const auto new_v = new_graph.input(value.shape, value.dtype);

    remap[old_input.value] = new_v;
  }

  // -----------------------------------------------------------------------------
  // starting from all outputs of old graph, rewrite into new graph replacing
  // with fused node wherever possible
  // ------------------------------------------------------------------------------

  for (const auto output : old_graph.outputs()) {
    const auto new_out = rewrite_value(output, old_graph, new_graph, use_count,
                                       is_output, remap);
    new_graph.add_output(new_out);
  }

  // -----------------------------------------------------------------------------
  // setup outputs and return new graph
  // ------------------------------------------------------------------------------
  return new_graph;
}

} // namespace tcr::compiler