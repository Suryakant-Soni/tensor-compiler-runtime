#include "tcr/ir/type_inference.hpp"
#include <cstddef>
#include <stdexcept>
namespace tcr::ir {

InferredType infer_type(Op op, const std::vector<const Value *> &inputs) {
  if (inputs.empty()) {
    throw std::invalid_argument("no input supplied in param");
  }
  switch (op) {
  case Op::Add:
    if (inputs.size() != 2) {
      throw std::invalid_argument("ADD operation requires 2 operands/inputs");
    }
    if (inputs[0]->shape != inputs[1]->shape) {
      throw std::invalid_argument(
          "ADD operation needs matching shape of operands");
    }
    if (inputs[0]->dtype != inputs[1]->dtype) {
      throw std::invalid_argument(
          "ADD operation needs matching dtypes of operands");
    }
    return InferredType{
        inputs[0]->shape,
        inputs[0]->dtype,
    };

  case Op::Mul:
    if (inputs.size() != 2) {
      throw std::invalid_argument("MUL operation requires 2 operands/inputs");
    }
    if (inputs[0]->shape != inputs[1]->shape) {
      throw std::invalid_argument(
          "MUL operation needs matching shape of operands");
    }
    if (inputs[0]->dtype != inputs[1]->dtype) {
      throw std::invalid_argument(
          "MUL operation needs matching dtypes of operands");
    }
    return InferredType{
        inputs[0]->shape,
        inputs[0]->dtype,
    };

  case Op::MatMul:
    if (inputs.size() != 2) {
      throw std::invalid_argument(
          "MATMUL operation requires 2 operands/inputs");
    }
    if (inputs[0]->dtype != inputs[1]->dtype) {
      throw std::invalid_argument(
          "MATMUL operation needs matching dtypes of operands");
    }
    if (inputs[0]->shape.rank() != 2 || inputs[1]->shape.rank() != 2) {
      throw std::invalid_argument(
          "MATMUL operation needs operands to have rank=2");
    }
    if (inputs[0]->shape.dim(1) != inputs[1]->shape.dim(0)) {
      throw std::invalid_argument(
          "MATMUL operation needs compatible lhs and rhs shapes");
    }
    return InferredType{
        {inputs[0]->shape.dim(0), inputs[1]->shape.dim(1)},
        inputs[0]->dtype,
    };

  case Op::Sum:
    if (inputs.size() != 1) {
      throw std::invalid_argument(
          "SUM operation needs only single operand to be aggregated upon");
    }
    return InferredType{
        Shape{},
        inputs[0]->dtype,
    };
  case Op::FusedElementwise: {
    const auto &first_shape = inputs[0]->shape;
    const auto &first_dtype = inputs[0]->dtype;
    for (std::size_t i = 1; i < inputs.size(); ++i) {
      if (first_shape != inputs[i]->shape || first_dtype != inputs[i]->dtype) {
        throw std::invalid_argument(
            "shape and dtype metadata should be same for all "
            "inputs to fused node");
      }
    }
    return InferredType{
        .shape = first_shape,
        .dtype = first_dtype,
    };
  }
  default:
    throw std::logic_error("unsupported operation type");
  }
}

} // namespace tcr::ir