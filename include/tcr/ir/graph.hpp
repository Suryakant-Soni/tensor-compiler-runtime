#pragma once
#include "tcr/ir/dtype.hpp"
#include "tcr/ir/ids.hpp"
#include "tcr/ir/nodes.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/ir/value.hpp"
#include <vector>
namespace tcr::ir {
class Graph {
public:
  // public apis
  // create graph inputs
  ValueId input(Shape shape, Dtype dtype);
  // record operation nodes + the value of output in value record

  ValueId add(ValueId a, ValueId b);
  ValueId mul(ValueId a, ValueId b);
  ValueId matmul(ValueId a, ValueId b);
  ValueId sum(ValueId input);

  // mark final output for graph result
  void set_outputs(std::vector<ValueId> outputs);
  void add_output(ValueId id);

  // read graph contents
  const Value &value(ValueId id) const;
  const Node &node(NodeId id) const;

  const std::vector<ValueId>& inputs() const;
  const std::vector<ValueId>& outputs() const;

  // private attributes
private:
  // validate value id
  void validate_value_id(ValueId id) const;
  std::vector<Value> values_;
  std::vector<Node> nodes_;

  std::vector<ValueId> inputs_;
  std::vector<ValueId> outputs_;
};
} // namespace tcr::ir