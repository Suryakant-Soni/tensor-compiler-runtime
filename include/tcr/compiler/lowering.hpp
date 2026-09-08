// public api to lower, private api for lowering a value, input, and all
// operations, then member unordered map with ==operator() and hasher type with
// operator()
#pragma once
#include "tcr/ir/graph.hpp"
#include "tcr/ir/ids.hpp"
#include "tcr/lir/ids.hpp"
#include "tcr/lir/program.hpp"
#include <unordered_map>
#include <vector>
namespace tcr::compiler {
class Lowering {
public:
  lir::Program lower(const ir::Graph &graph,
                     const std::vector<ir::ValueId> &outputs);

private:
  enum class ElementwiseBinaryKind {
    Add,
    Mul,
  };
  lir::BufferId lower_value(const ir::Graph &graph, ir::ValueId value_id,
                            lir::Program &program);

  lir::BufferId lower_input(const ir::Graph &graph, ir::ValueId input_value_id,
                            lir::Program &program);

  lir::BufferId lower_add(const ir::Graph &graph, ir::ValueId output_value_id,
                          lir::Program &program);
  lir::BufferId lower_mul(const ir::Graph &graph, ir::ValueId output_value_id,
                          lir::Program &program);
  lir::BufferId lower_matmul(const ir::Graph &graph,
                             ir::ValueId output_value_id,
                             lir::Program &program);
  lir::BufferId lower_sum(const ir::Graph &graph, ir::ValueId output_value_id,
                          lir::Program &program);
  lir::BufferId lower_elementwise_binary(const ir::Graph &graph,
                                         ir::ValueId output_value_id,
                                         lir::Program &program,
                                         ElementwiseBinaryKind kind);
  std::unordered_map<ir::ValueId, lir::BufferId, ir::ValueIdHash>
      lowered_value_to_buffer_;
};
} // namespace tcr::compiler