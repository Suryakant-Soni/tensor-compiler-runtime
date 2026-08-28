#include "tcr/reference/interpreter.hpp"
#include "tcr/ir/graph.hpp"
#include "tcr/ir/ids.hpp"
#include "tcr/ir/nodes.hpp"
#include "tcr/reference/bindings.hpp"
#include "tcr/reference/ops.hpp"
#include "tcr/reference/tensor.hpp"
#include <stdexcept>

namespace tcr::reference {
using tcr::ir::Graph;
using tcr::ir::Op;
using tcr::ir::ValueId;

Tensor Interpreter::run(const Graph &graph, ValueId output_id,
                        const InputBindings &inputs) {
  return eval(graph, output_id, inputs);
}

Tensor Interpreter::eval(const Graph &graph, ValueId output_id,
                         const InputBindings &inputs) {
  // you have got the output id of graph, first check out its value details
  const auto &value = graph.value(output_id);
  // if this value is itself registered as input, return its Tensor(actual data)
  if (value.producer == std::nullopt) {
    return inputs.get(output_id);
  }

  // if not find the node/operation which produced the ouptut
  const auto &node = graph.node(value.producer.value());

  switch (node.op) {
  case Op::Add:
    return add(eval(graph, node.inputs[0], inputs),
               eval(graph, node.inputs[1], inputs));
  case ir::Op::Mul:
    return mul(eval(graph, node.inputs[0], inputs),
               eval(graph, node.inputs[1], inputs));
  case ir::Op::Sum:
    return sum(eval(graph, node.inputs[0], inputs));
  case ir::Op::MatMul:
    return matmul(eval(graph, node.inputs[0], inputs),
                  eval(graph, node.inputs[1], inputs));
  case ir::Op::Input:
    throw std::logic_error("input should not have any producer");
  case ir::Op::Constant:
    throw std::logic_error("constant evaluation is not implemented yet");
  }
  throw std::logic_error("unknown operation");
}
} // namespace tcr::reference