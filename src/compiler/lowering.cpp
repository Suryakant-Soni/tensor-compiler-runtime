#include "tcr/compiler/lowering.hpp"
#include "tcr/compiler/schedule.hpp"
#include "tcr/ir/elementwise_expr.hpp"
#include "tcr/ir/graph.hpp"
#include "tcr/ir/ids.hpp"
#include "tcr/ir/op.hpp"
#include "tcr/ir/value.hpp"
#include "tcr/lir/ids.hpp"
#include "tcr/lir/index.hpp"
#include "tcr/lir/instruction.hpp"
#include "tcr/lir/program.hpp"
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
namespace tcr::compiler {

namespace {
bool is_elementwise_sum_group(const KernelUnit &kernel,
                              const ir::Graph &graph) {
  const auto first_member = kernel.members[0];
  const auto &first_value = graph.value(first_member);

  const auto second_member = kernel.members[1];
  const auto &second_value = graph.value(second_member);
  if (!first_value.producer.has_value() || !second_value.producer.has_value()) {
    throw std::logic_error("kernel members should be computed values");
  }

  const auto &first_node = graph.node(*first_value.producer);
  const auto &second_node = graph.node(*second_value.producer);

  if (second_node.inputs.size() != 1) {
    throw std::logic_error(
        "sum should have only one input : is_elementwise_sum_group");
  }

  if (first_node.op != ir::Op::Add && first_node.op != ir::Op::Mul &&
      first_node.op != ir::Op::FusedElementwise) {
    return false;
  }
  if (second_node.op != ir::Op::Sum || second_node.inputs[0] != first_member) {
    return false;
  }
  return true;
}
} // namespace

// manages returning buffer for inputs and computed values from previous kernels
// and members
lir::BufferId Lowering::require_buffer(const ir::Graph &graph,
                                       ir::ValueId value_id,
                                       lir::Program &program) {
  // return if the value already exist in the map
  auto it = valueid_to_bufferid_.find(value_id);
  if (it != valueid_to_bufferid_.end()) {
    return it->second;
  }
  // check if this value is external input and not in the buffer yet
  const auto &value = graph.value(value_id);
  if (!value.producer.has_value()) {
    return lower_input(graph, value_id, program);
  }
  throw std::logic_error(
      "buffer should be available for the previously computed value");
}

lir::Program Lowering::lower(const ir::Graph &graph, const Schedule &schedule) {

  valueid_to_bufferid_.clear();
  lir::Program program;

  for (const auto &kernel : schedule.kernels) {
    lower_kernel(graph, kernel, program);
  }
  for (const auto output : graph.outputs()) {
    program.add_output(require_buffer(graph, output, program));
  }
  return program;
}

lir::BufferId Lowering::lower_kernel(const ir::Graph &graph,
                                     const KernelUnit &kernel_unit,
                                     lir::Program &program) {

  // for a single ungrouped kernel, run the singleton operation as is
  if (kernel_unit.members.size() == 1) {
    const auto member = kernel_unit.members[0];
    // check if member is i/p, it should not be
    const auto &value = graph.value(member);
    if (!value.producer.has_value()) {
      throw std::logic_error(
          "kernel memeber should not be an input : lowerkernel");
    }
    const auto &node = graph.node(*value.producer);
    switch (node.op) {
    case ir::Op::Add:
      return lower_add(graph, member, program);
    case ir::Op::Mul:
      return lower_mul(graph, member, program);
    case ir::Op::MatMul:
      return lower_matmul(graph, member, program);
    case ir::Op::Sum:
      return lower_sum(graph, member, program);
    case ir::Op::FusedElementwise:
      return lower_fused_elementwise(graph, member, program);
    case ir::Op::Input:
      throw std::logic_error("Input value unexpectedly has a producer");
    }
  }
  if (kernel_unit.members.size() == 2) {
    if (is_elementwise_sum_group(kernel_unit, graph)) {
      return lower_elementwise_with_sum(graph, kernel_unit, program);
    }
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

  valueid_to_bufferid_.emplace(input_value_id, buffer_id);
  return buffer_id;
}

lir::BufferId Lowering::lower_matmul(const ir::Graph &graph,
                                     ir::ValueId output_value_id,
                                     lir::Program &program) {
  const auto &output_value = graph.value(output_value_id);
  const auto &node = graph.node(*output_value.producer);

  // inputs are already lowered from previous members/kernels
  const auto a_buffer = require_buffer(graph, node.inputs[0], program);
  const auto b_buffer = require_buffer(graph, node.inputs[1], program);

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

  valueid_to_bufferid_.emplace(output_value_id, c_buffer);
  return c_buffer;
}

namespace {
lir::ScalarId
lower_elementwise_expression(const ir::ElementwiseExpr &expr,
                             const std::vector<lir::ScalarId> &in_scalars,
                             lir::For &i_loop, lir::Program &program) {
  if (in_scalars.empty()) {
    throw std::invalid_argument(
        "input scalars are not supplied to lower elementwise expr");
  }
  if (expr.kind == ir::ElementwiseExprKind::Input) {
    if (expr.input_index >= in_scalars.size()) {
      throw std::out_of_range("input index is out of range");
    }
    return in_scalars[expr.input_index];
  }
  if (!expr.lhs || !expr.rhs) {
    throw std::logic_error(
        "elementwise expression op missing operand expression");
  }
  const auto scalar_lhs =
      lower_elementwise_expression(*expr.lhs, in_scalars, i_loop, program);
  const auto scalar_rhs =
      lower_elementwise_expression(*expr.rhs, in_scalars, i_loop, program);

  const auto out_scalar = program.create_scalar();
  switch (expr.kind) {
  case tcr::ir::ElementwiseExprKind::Add:
    i_loop.body.emplace_back(lir::Add(scalar_lhs, scalar_rhs, out_scalar));
    break;
  case tcr::ir::ElementwiseExprKind::Mul:
    i_loop.body.emplace_back(lir::Mul(scalar_lhs, scalar_rhs, out_scalar));
    break;
  default:
    throw std::logic_error("unknow elementwise expression Kind");
  }
  return out_scalar;
}
} // namespace
lir::BufferId Lowering::lower_fused_elementwise(const ir::Graph &graph,
                                                lir::ValueId output_value_id,
                                                lir::Program &program) {

  const auto &value = graph.value(output_value_id);
  const auto &node = graph.node(*value.producer);

  if (!node.elementwise_expr.has_value()) {
    throw std::invalid_argument(
        "elementwise expression param does not have value inside");
  }

  // recursively lower all the inputs first
  std::vector<lir::ScalarId> input_scalars;
  input_scalars.reserve(node.inputs.size());

  auto output_buffer = program.create_buffer(value.shape, value.dtype);
  const auto i = program.create_index();

  auto i_loop = std::make_unique<lir::For>();
  i_loop->index = i;
  i_loop->begin = lir::ConstantIndex{0};
  i_loop->end = lir::ConstantIndex{value.shape.num_elements()};

  for (std::size_t j = 0; j < node.inputs.size(); ++j) {
    auto scalar = program.create_scalar();
    auto buffer = require_buffer(graph, node.inputs[j], program);

    input_scalars.push_back(scalar);

    // load statements;
    i_loop->body.emplace_back(lir::Load(scalar, buffer, i));
  }

  // lower elementwise expression recursively
  const auto out_scalar = lower_elementwise_expression(
      *node.elementwise_expr, input_scalars, *i_loop, program);

  // after getting the top most scalarid store the scalarid single slot in
  // output
  i_loop->body.emplace_back(lir::Store(out_scalar, output_buffer, i));

  program.add_instruction(std::move(i_loop));

  valueid_to_bufferid_.emplace(output_value_id, output_buffer);

  return output_buffer;
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
  const auto a_buffer = require_buffer(graph, a_value, program);
  const auto b_buffer = require_buffer(graph, b_value, program);

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

  valueid_to_bufferid_.emplace(output_value_id, c_buffer);
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
  const auto input_buffer = require_buffer(graph, node.inputs[0], program);
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
  valueid_to_bufferid_.emplace(output_value_id, output_buffer);
  return output_buffer;
}

// lower the group of elementwise member and then sum member
lir::BufferId
Lowering::lower_elementwise_with_sum(const ir::Graph &graph,
                                     const KernelUnit &kernel_unit,
                                     lir::Program &program) {
  const auto elementwise_member = kernel_unit.members[0];
  const auto &ele_value = graph.value(elementwise_member);
  const auto &ele_node = graph.node(*ele_value.producer);

  const auto sum_member = kernel_unit.members[1];
  const auto &sum_value = graph.value(sum_member);

  // recursively lower all the inputs first
  std::vector<lir::ScalarId> input_scalars;
  input_scalars.reserve(ele_node.inputs.size());

  // scalar for accumulation for SUM operation
  const auto sum_scalar = program.create_scalar();
  program.add_instruction(lir::Constant(sum_scalar, 0.0f));

  auto output_buffer = program.create_buffer(sum_value.shape, sum_value.dtype);
  const auto i = program.create_index();

  auto i_loop = std::make_unique<lir::For>();
  i_loop->index = i;
  i_loop->begin = lir::ConstantIndex{0};
  i_loop->end = lir::ConstantIndex{ele_value.shape.num_elements()};

  for (std::size_t j = 0; j < ele_node.inputs.size(); ++j) {
    auto scalar = program.create_scalar();
    auto buffer = require_buffer(graph, ele_node.inputs[j], program);

    input_scalars.push_back(scalar);

    // load statements;
    i_loop->body.emplace_back(lir::Load(scalar, buffer, i));
  }

  // now lower the elementwise op add/mul/fusednode and then add sum
  // instructions to it
  lir::ScalarId out_scalar;
  switch (ele_node.op) {
  case ir::Op::FusedElementwise:
    if (!ele_node.elementwise_expr.has_value()) {
      throw std::invalid_argument(
          "elementwise expression param does not have value inside");
    }
    // lower elementwise expression recursively
    out_scalar = lower_elementwise_expression(*ele_node.elementwise_expr,
                                              input_scalars, *i_loop, program);
    break;

  case ir::Op::Add:
    out_scalar = program.create_scalar();
    i_loop->body.emplace_back(
        lir::Add(input_scalars[0], input_scalars[1], out_scalar));
    break;
  case ir::Op::Mul:
    out_scalar = program.create_scalar();
    i_loop->body.emplace_back(
        lir::Mul(input_scalars[0], input_scalars[1], out_scalar));
    break;
  default:
    throw std::logic_error("no other elementwise op supported");
  }

  // accumulative addition in the last statement of loop
  i_loop->body.emplace_back(lir::Add(sum_scalar, out_scalar, sum_scalar));

  program.add_instruction(std::move(i_loop));

  program.add_instruction(
      lir::Store(sum_scalar, output_buffer, lir::ConstantIndex(0)));
  valueid_to_bufferid_.emplace(sum_member, output_buffer);
  return output_buffer;
}
} // namespace tcr::compiler