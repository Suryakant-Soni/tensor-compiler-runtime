#include "tcr/ir/dtype.hpp"
#include "tcr/ir/graph.hpp"
#include "tcr/ir/ids.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/reference/bindings.hpp"
#include "tcr/reference/interpreter.hpp"
#include "tcr/reference/tensor.hpp"
#include <catch2/catch_test_macros.hpp>

using tcr::ir::Dtype;
using tcr::ir::Graph;
using tcr::ir::Shape;
using tcr::ir::ValueId;
using tcr::reference::InputBindings;
using tcr::reference::Interpreter;
using tcr::reference::Tensor;

TEST_CASE("reference matmul matches interpreter results",
          "[reference][interpreter]") {
  Graph g;
  ValueId a = g.input(Shape{2, 2}, Dtype::F32);
  ValueId b = g.input(Shape{2, 2}, Dtype::F32);

  ValueId out_id = g.matmul(a, b);

  InputBindings bindings;

  bindings.bind(a, Tensor{
                       Shape{2, 2},
                       Dtype::F32,
                       {1.0f, 2.0f, 3.0f, 4.0f},
                   });
  bindings.bind(b, Tensor{
                       Shape{2, 2},
                       Dtype::F32,
                       {5.0f, 6.0f, 7.0f, 8.0f},
                   });
  Interpreter interpreter;
  Tensor result = interpreter.run(g, out_id, bindings);

  REQUIRE(result.shape() == Shape{2, 2});

  REQUIRE(result.data()[0] == 19.0f);
  REQUIRE(result.data()[1] == 22.0f);
  REQUIRE(result.data()[2] == 43.0f);
  REQUIRE(result.data()[3] == 50.0f);
}
