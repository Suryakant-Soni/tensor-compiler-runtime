#include "tcr/ir/graph.hpp"
#include "tcr/ir/ids.hpp"
#include "tcr/reference/bindings.hpp"
#include "tcr/reference/tensor.hpp"
namespace tcr::reference {
using tcr::ir::Graph;
using tcr::ir::ValueId;
class Interpreter {

public:
  Tensor run(const Graph &graph, ValueId output_id,
             const InputBindings &inputs);

private:
  Tensor eval(const Graph &graph, ValueId output_id,
              const InputBindings &inputs);
};
} // namespace tcr::reference