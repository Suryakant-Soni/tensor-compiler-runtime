
#include "tcr/ir/graph.hpp"
#include "tcr/ir/dtype.hpp"
#include "tcr/ir/ids.hpp"
#include "tcr/ir/nodes.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/ir/type_inference.hpp"
#include "tcr/ir/value.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
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
  // gather input elements
  std::vector<const Value *> inputs{
      &values_[a.value],
      &values_[b.value],
  };

  const auto inferred = infer_type(Op::Add, inputs);

  // create a new value id for output of add
  ValueId value_id{static_cast<std::int32_t>(values_.size())};

  // create a new nodeid for the node operation add
  NodeId node_id{static_cast<std::int32_t>(nodes_.size())};

  // push a new entry in nodes_ for add node
  nodes_.push_back(Node{
      .op = Op::Add,
      .inputs = {a, b},
  });

  values_.push_back(Value{
      .shape = inferred.shape,
      .dtype = inferred.dtype,
      .producer = node_id,
  });
  return value_id;
}

ValueId Graph::mul(ValueId a, ValueId b) {
  validate_value_id(a);
  validate_value_id(b);
  // gather input elements
  std::vector<const Value *> inputs{
      &values_[a.value],
      &values_[b.value],
  };

  const auto inferred = infer_type(Op::Mul, inputs);
  // create a valueid for result of MUL
  ValueId value_id{static_cast<std::int32_t>(values_.size())};
  // create a nodeid for operation MUL
  NodeId node_id{static_cast<std::int32_t>(nodes_.size())};
  // push entry in nodes_
  nodes_.push_back(Node{
      .op = Op::Mul,
      .inputs = {a, b},
  });
  // push an entry into values since the output received from MUL node is also a
  // value
  values_.push_back(Value{
      .shape = inferred.shape,
      .dtype = inferred.dtype,
      .producer = node_id,
  });
  return value_id;
}

ValueId Graph::matmul(ValueId a, ValueId b) {
  // valudate the values of a and b operands
  validate_value_id(a);
  validate_value_id(b);
  // gather input elements
  std::vector<const Value *> inputs{
      &values_[a.value],
      &values_[b.value],
  };

  const auto inferred = infer_type(Op::MatMul, inputs);
  // create a valueid for matmul output and node id for operation
  ValueId value_id{static_cast<std::int32_t>(values_.size())};

  NodeId node_id{static_cast<std::int32_t>(nodes_.size())};
  // push the matmul node in nodes_
  nodes_.push_back(Node{
      .op = Op::MatMul,
      .inputs = {a, b},
  });
  // push the value to values
  values_.push_back(Value{
      .shape = inferred.shape,
      .dtype = inferred.dtype,
      .producer = node_id,
  });
  return value_id;
}

ValueId Graph::sum(ValueId input) {
  validate_value_id(input);
  // gather input elements
  std::vector<const Value *> inputs{
      &values_[input.value],
  };

  const auto inferred = infer_type(Op::Sum, inputs);
  // create a valueid for representing sum output and node id for representing
  // SUM operation
  ValueId value_id{static_cast<std::int32_t>(values_.size())};

  NodeId node_id{static_cast<std::int32_t>(nodes_.size())};
  nodes_.push_back(Node{
      .op = Op::Sum,
      .inputs = {input},
  });
  values_.push_back(Value{
      .shape = inferred.shape,
      .dtype = inferred.dtype,
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
  if (id.value < 0 || static_cast<std::size_t>(id.value) >= nodes_.size()) {
    throw std::out_of_range("node id is out of bound");
  }
  return nodes_[id.value];
}
std::size_t Graph::nodes_count() const { return nodes_.size(); }
std::size_t Graph::values_count() const { return values_.size(); }
std::size_t Graph::inputs_count() const { return inputs_.size(); }
std::size_t Graph::outputs_count() const { return outputs_.size(); }

const std::vector<ValueId> &Graph::inputs() const { return inputs_; }
const std::vector<ValueId> &Graph::outputs() const { return outputs_; }
} // namespace tcr::ir