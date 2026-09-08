#include "tcr/compiler/lowering.hpp"
#include "tcr/ir/graph.hpp"
#include "tcr/ir/ids.hpp"
#include "tcr/ir/op.hpp"
#include "tcr/ir/value.hpp"
#include "tcr/lir/ids.hpp"
#include "tcr/lir/index.hpp"
#include "tcr/lir/instruction.hpp"
#include "tcr/lir/program.hpp"
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
namespace tcr::compiler {

lir::Program Lowering::lower(const ir::Graph &graph,
                             const std::vector<ir::ValueId> &outputs) {

  lowered_value_to_buffer_.clear();
  lir::Program program;

  for (const auto output_value_id : outputs) {
    const auto buffer_id = lower_value(graph, output_value_id, program);
    // collect the buffer id as output
    program.add_output(buffer_id);
  }
  return program;
}

lir::BufferId Lowering::lower_value(const ir::Graph &graph,
                                    ir::ValueId value_id,
                                    lir::Program &program) {
  // if value already visited
  if (auto it = lowered_value_to_buffer_.find(value_id);
      it != lowered_value_to_buffer_.end()) {
    return it->second;
  }
  const auto &output_value = graph.value(value_id);

  if (!output_value.producer.has_value()) {
    return lower_input(graph, value_id, program);
  }
  const auto &node = graph.node(*output_value.producer);

  switch (node.op) {
  case ir::Op::Add:
    return lower_add(graph, value_id, program);
  case ir::Op::Mul:
    return lower_mul(graph, value_id, program);
  case ir::Op::MatMul:
    return lower_matmul(graph, value_id, program);
  case ir::Op::Sum:
    return lower_sum(graph, value_id, program);
  case ir::Op::Input:
    throw std::logic_error("Input value unexpectedly has a producer");
  }
  throw std::logic_error("Unsupported high level Op in lowering");
}

lir::BufferId Lowering::lower_input(const ir::Graph &graph,
                                    ir::ValueId input_value_id,
                                    lir::Program &program) {
  // create a new buffer for this input, add it as an input,  add to
  // value_to_buffer map
  const auto &value = graph.value(input_value_id);

  const auto buffer_id = program.create_buffer(value.shape, value.dtype);
  program.add_input(input_value_id, buffer_id);

  lowered_value_to_buffer_.emplace(input_value_id, buffer_id);
  return buffer_id;
}

lir::BufferId Lowering::lower_matmul(const ir::Graph &graph,
                                     ir::ValueId output_value_id,
                                     lir::Program &program) {
  const auto &output_value = graph.value(output_value_id);
  const auto &node = graph.node(*output_value.producer);

  // lower recursively the inputs to this node
  const auto a_buffer = lower_value(graph, node.inputs[0], program);
  const auto b_buffer = lower_value(graph, node.inputs[1], program);

  const auto c_buffer =
      program.create_buffer(output_value.shape, output_value.dtype);
  const auto &a = graph.value(node.inputs[0]);
  const auto &b = graph.value(node.inputs[1]);

  const auto M = lir::ConstantIndex(a.shape.dim(0));
  const auto K = lir::ConstantIndex(a.shape.dim(1));
  const auto N = lir::ConstantIndex(b.shape.dim(1));

  // indexids for running loops
  const auto i = program.create_index();
  const auto j = program.create_index();
  const auto k = program.create_index();

  // indexids for index arithmetic
  const auto a_row = program.create_index();
  const auto b_row = program.create_index();
  const auto a_off = program.create_index();
  const auto b_off = program.create_index();
  const auto c_row = program.create_index();
  const auto c_off = program.create_index();

  const auto a_scalar = program.create_scalar();
  const auto b_scalar = program.create_scalar();
  const auto product = program.create_scalar();
  const auto acc = program.create_scalar();

  auto k_loop = std::make_unique<lir::For>();

  k_loop->index = k;
  k_loop->begin = lir::ConstantIndex(0);
  k_loop->end = K;
  k_loop->body.emplace_back(lir::IndexAdd(a_row, k, a_off));
  k_loop->body.emplace_back(lir::IndexMul(k, N, b_row));
  k_loop->body.emplace_back(lir::IndexAdd(b_row, j, b_off));
  k_loop->body.emplace_back(lir::Load(a_scalar, a_buffer, a_off));
  k_loop->body.emplace_back(lir::Load(b_scalar, b_buffer, b_off));
  k_loop->body.emplace_back(lir::Mul(a_scalar, b_scalar, product));
  k_loop->body.emplace_back(lir::Add(acc, product, acc));

  auto j_loop = std::make_unique<lir::For>();
  j_loop->index = j;
  j_loop->begin = lir::ConstantIndex(0);
  j_loop->end = N;
  j_loop->body.emplace_back(lir::Constant{acc, 0.0f});
  j_loop->body.emplace_back(std::move(k_loop));
  j_loop->body.emplace_back(lir::IndexAdd(c_row, j, c_off));
  j_loop->body.emplace_back(lir::Store(acc, c_buffer, c_off));

  auto i_loop = std::make_unique<lir::For>();

  i_loop->index = i;
  i_loop->begin = lir::ConstantIndex(0);
  i_loop->end = M;
  i_loop->body.emplace_back(lir::IndexMul(i, K, a_row));
  i_loop->body.emplace_back(lir::IndexMul(i, N, c_row));
  i_loop->body.emplace_back(std::move(j_loop));

  program.add_instruction(std::move(i_loop));

  lowered_value_to_buffer_.emplace(output_value_id, c_buffer);
  return c_buffer;
}

lir::BufferId Lowering::lower_elementwise_binary(const ir::Graph &graph,
                                                 ir::ValueId output_value_id,
                                                 lir::Program &program,
                                                 ElementwiseBinaryKind kind) {
  const auto &value = graph.value(output_value_id);
  const auto &node = graph.node(*value.producer);

  // recursively lower the inputs
  const auto a_value = node.inputs[0];
  const auto b_value = node.inputs[1];
  const auto a_buffer = lower_value(graph, a_value, program);
  const auto b_buffer = lower_value(graph, b_value, program);

  const auto c_buffer = program.create_buffer(value.shape, value.dtype);
  const auto a_scalar = program.create_scalar();
  const auto b_scalar = program.create_scalar();
  const auto c_scalar = program.create_scalar();
  const auto i = program.create_index();

  auto i_loop = std::make_unique<lir::For>();
  i_loop->index = i;
  i_loop->begin = lir::ConstantIndex(0);
  i_loop->end = lir::ConstantIndex(value.shape.num_elements());
  i_loop->body.emplace_back(lir::Load(a_scalar, a_buffer, i));
  i_loop->body.emplace_back(lir::Load(b_scalar, b_buffer, i));

  switch (kind) {
  case ElementwiseBinaryKind::Add:
    i_loop->body.emplace_back(lir::Add(a_scalar, b_scalar, c_scalar));
    break;
  case ElementwiseBinaryKind::Mul:
    i_loop->body.emplace_back(lir::Mul(a_scalar, b_scalar, c_scalar));
    break;
  }
  i_loop->body.emplace_back(lir::Store(c_scalar, c_buffer, i));

  program.add_instruction(std::move(i_loop));

  lowered_value_to_buffer_.emplace(output_value_id, c_buffer);
  return c_buffer;
}

lir::BufferId Lowering::lower_add(const ir::Graph &graph,
                                  ir::ValueId output_value_id,
                                  lir::Program &program) {
  return lower_elementwise_binary(graph, output_value_id, program,
                                  ElementwiseBinaryKind::Add);
}

lir::BufferId Lowering::lower_mul(const ir::Graph &graph,
                                  ir::ValueId output_value_id,
                                  lir::Program &program) {
  return lower_elementwise_binary(graph, output_value_id, program,
                                  ElementwiseBinaryKind::Mul);
}

lir::BufferId Lowering::lower_sum(const ir::Graph &graph,
                                  ir::ValueId output_value_id,
                                  lir::Program &program) {
  const auto &output_value = graph.value(output_value_id);
  const auto &node = graph.node(*output_value.producer);
  const auto &input_value = graph.value(node.inputs[0]);
  const auto input_buffer = lower_value(graph, node.inputs[0], program);
  const auto output_buffer =
      program.create_buffer(output_value.shape, output_value.dtype);
  const auto input_scalar = program.create_scalar();
  const auto acc = program.create_scalar();
  const auto i = program.create_index();
  auto i_loop = std::make_unique<lir::For>();
  i_loop->index = i;
  i_loop->begin = lir::ConstantIndex(0);
  i_loop->end = lir::ConstantIndex(input_value.shape.num_elements());
  i_loop->body.emplace_back(lir::Load(input_scalar, input_buffer, i));
  i_loop->body.emplace_back(lir::Add(acc, input_scalar, acc));

  program.add_instruction(lir::Constant(acc, 0.0f));
  program.add_instruction(std::move(i_loop));
  program.add_instruction(
      lir::Store(acc, output_buffer, lir::ConstantIndex(0)));
  lowered_value_to_buffer_.emplace(output_value_id, output_buffer);
  return output_buffer;
}
} // namespace tcr::compiler