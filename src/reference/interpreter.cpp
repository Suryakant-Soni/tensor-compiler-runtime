#include "tcr/reference/interpreter.hpp"
#include "tcr/ir/graph.hpp"
#include "tcr/ir/ids.hpp"
#include "tcr/ir/nodes.hpp"
#include "tcr/ir/op.hpp"
#include "tcr/reference/bindings.hpp"
#include "tcr/reference/ops.hpp"
#include "tcr/reference/tensor.hpp"
#include <optional>
#include <stdexcept>

namespace tcr::reference {
using tcr::ir::Graph;
using tcr::ir::Op;
using tcr::ir::ValueId;

Tensor Interpreter::run(const Graph &graph, ValueId output_id,
                        const InputBindings &inputs) {
  CacheType cache;
  return eval(graph, output_id, inputs, cache);
}

Tensor Interpreter::eval(const Graph &graph, ValueId output_id,
                         const InputBindings &inputs, CacheType &cache) {
  // you have got the output id of graph, first check out its value details
  const auto &value = graph.value(output_id);
  // if this value is itself registered as input, return its Tensor(actual data)
  if (value.producer == std::nullopt) {
    return inputs.get(output_id);
  }
  if (const auto it = cache.find(output_id); it != cache.end()) {
    return it->second;
  }

  // if not find the node/operation which produced the ouptut
  const auto &node = graph.node(value.producer.value());
  std::optional<Tensor> result;
  switch (node.op) {
  case Op::Add:
    result = add(eval(graph, node.inputs[0], inputs, cache),
                 eval(graph, node.inputs[1], inputs, cache));
    break;
  case ir::Op::Mul:
    result = mul(eval(graph, node.inputs[0], inputs, cache),
                 eval(graph, node.inputs[1], inputs, cache));
    break;
  case ir::Op::Sum:
    result = sum(eval(graph, node.inputs[0], inputs, cache));
    break;
  case ir::Op::MatMul:
    result = matmul(eval(graph, node.inputs[0], inputs, cache),
                    eval(graph, node.inputs[1], inputs, cache));
    break;
  case ir::Op::Input:
    throw std::logic_error("input should not have any producer");
  case ir::Op::FusedElementwise:
    throw std::logic_error("High Ir reference graph does not use fusion nodes");
  }
  if (!result.has_value()) {
    throw std::logic_error("unsupported reference interpreter operation");
  }
  cache.emplace(output_id, *result);
  return *result;
}
} // namespace tcr::reference