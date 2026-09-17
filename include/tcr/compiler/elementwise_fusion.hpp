#include "tcr/ir/graph.hpp"
namespace tcr::compiler {
class ElementwiseFusion {
public:
  const ir::Graph run(const ir::Graph &graph) const;
};
} // namespace tcr::compiler