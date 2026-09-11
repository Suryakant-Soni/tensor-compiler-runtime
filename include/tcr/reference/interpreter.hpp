#pragma once
#include "tcr/ir/graph.hpp"
#include "tcr/ir/ids.hpp"
#include "tcr/reference/bindings.hpp"
#include "tcr/reference/tensor.hpp"
#include <unordered_map>
namespace tcr::reference {
using tcr::ir::Graph;
using tcr::ir::ValueId;
class Interpreter {

  using CacheType = std::unordered_map<ir::ValueId, Tensor, ir::ValueIdHash>;

public:
  Tensor run(const Graph &graph, ValueId output_id,
             const InputBindings &inputs);

private:
  Tensor eval(const Graph &graph, ValueId output_id,
              const InputBindings &inputs, CacheType &cache);
};
} // namespace tcr::reference