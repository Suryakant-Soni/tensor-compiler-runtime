
#include "tcr/ir/graph.hpp"
#include "tcr/ir/dtype.hpp"
#include "tcr/ir/ids.hpp"
#include "tcr/ir/nodes.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/ir/value.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace tcr::ir {
void Graph::validate_value_id(ValueId id) const {
  if (id.value < 0 || static_cast<std::size_t>(id.value) >= values_.size()) {
    throw std::out_of_range("invalid ValueId");
  }
}
ValueId Graph::input(Shape shape, Dtype dtype) {
  // get a new valueid
  ValueId id{static_cast<std::int32_t>(values_.size())};
  // create a new entry in values_ and producer will be nullopt
  values_.push_back(Value{
      // std::move uses the same shape local copy available in function and move
      // it to shape member avoiding deep copy of vector
      .shape = std::move(shape),
      .dtype = dtype,
      .producer = std::nullopt,
  });
  // add this entry along with valueid in inputs_ and then return the valueid
  inputs_.push_back(id);
  return id;
}

ValueId Graph::add(ValueId a, ValueId b) {
  validate_value_id(a);
  validate_value_id(b);
  // gather shape of input elements
  const Value &lhs = values_[a.value];
  const Value &rhs = values_[b.value];

  if (lhs.shape != rhs.shape) {
    throw std::invalid_argument("add requires matching shape");
  }

  if (lhs.dtype != rhs.dtype) {
    throw std::invalid_argument("add requires matching element types/dtypes");
  }

  // create a new value id for output of add
  ValueId value_id{static_cast<std::int32_t>(values_.size())};

  // create a new nodeid for the node operation add
  NodeId node_id{static_cast<int32_t>(nodes_.size())};

  // push a new entry in nodes_ for add node
  nodes_.push_back(Node{
      .op = Op::Add,
      .inputs = {a, b},
  });

  values_.push_back(Value{
      .dtype = lhs.dtype,
      .producer = node_id,
      .shape = lhs.shape,
  });
  return value_id;
}

ValueId Graph::mul(ValueId a, ValueId b) {
  validate_value_id(a);
  validate_value_id(b);
  const Value &lhs = values_[a.value];
  const Value &rhs = values_[b.value];
  if (lhs.shape != rhs.shape) {
    throw std::invalid_argument("mul requires matching shape");
  }
  if (lhs.dtype != rhs.dtype) {
    throw std::invalid_argument("mul requires matching dtypes");
  }
  // create a valueid for result of MUL
  ValueId value_id{static_cast<int32_t>(values_.size())};
  // create a nodeid for operation MUL
  NodeId node_id{static_cast<int32_t>(nodes_.size())};
  // push entry in nodes_
  nodes_.push_back(Node{
      .op = Op::Mul,
      .inputs = {a, b},
  });
  // push an entry into values since the output received from MUL node is also a
  // value
  values_.push_back(Value{
      .dtype = lhs.dtype,
      .shape = lhs.shape,
      .producer = node_id,
  });
  return value_id;
}

ValueId Graph::matmul(ValueId a, ValueId b) {
  // valudate the values of a and b operands
  validate_value_id(a);
  validate_value_id(b);
  const Value &lhs = values_[a.value];
  const Value &rhs = values_[b.value];
  if (lhs.dtype != rhs.dtype) {
    throw std::invalid_argument("matmul requires matching arguement");
  }
  // rank for matmul should be 2 for both operands
  if (lhs.shape.rank() != 2 || rhs.shape.rank() != 2) {
    throw std::invalid_argument(
        "for matmul operation rank of operands should be 2");
  }
  // shape should match for matrix multiplication
  if (lhs.shape.dim(1) != rhs.shape.dim(0)) {
    throw std::invalid_argument(
        "the shapes of matmul operands do not match for matrix multiplication");
  }
  // create a valueid for matmul output and node id for operation
  ValueId value_id{static_cast<int32_t>(values_.size())};

  NodeId node_id{static_cast<int32_t>(nodes_.size())};
  // push the matmul node in nodes_
  nodes_.push_back(Node{
      .op = Op::MatMul,
      .inputs = {a, b},
  });
  // push the value to values
  values_.push_back(Value{
      .dtype = lhs.dtype,
      .shape = {lhs.shape.dim(0), rhs.shape.dim(1)},
      .producer = node_id,
  });
  return value_id;
}

ValueId Graph::sum(ValueId input) {
  validate_value_id(input);
  const Value &input_value = values_[input.value];
  // create a valueid for representing sum output and node id for representing
  // SUM operation
  ValueId value_id{static_cast<int32_t>(values_.size())};

  NodeId node_id{static_cast<int32_t>(nodes_.size())};
  nodes_.push_back(Node{
      .op = Op::Sum,
      .inputs = {input},
  });
  values_.push_back(Value{
      .dtype = input_value.dtype,
      .shape = Shape{},
      .producer = node_id,
  });
  return value_id;
}

void Graph::set_outputs(std::vector<ValueId> outputs) {
  for (ValueId id : outputs) {
    validate_value_id(id);
  }
  outputs_ = std::move(outputs);
}

void Graph::add_output(ValueId id) {
  validate_value_id(id);
  outputs_.push_back(id);
}

const Value &Graph::value(ValueId id) const {
  validate_value_id(id);
  return values_[id.value];
}
const Node &Graph::node(NodeId id) const {
  if (id.value < 0 || static_cast<size_t>(id.value) >= nodes_.size()) {
    throw std::out_of_range("node id is out of bound");
  }
  return nodes_[id.value];
}
const std::vector<ValueId>& Graph::inputs() const { return inputs_; };
const std::vector<ValueId>& Graph::outputs() const { return outputs_; };
} // namespace tcr::ir