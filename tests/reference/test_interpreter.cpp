#include "tcr/reference/interpreter.hpp"

#include "tcr/ir/dtype.hpp"
#include "tcr/ir/graph.hpp"
#include "tcr/ir/ids.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/reference/bindings.hpp"
#include "tcr/reference/tensor.hpp"

#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <vector>

using namespace tcr::ir;
using namespace tcr::reference;

TEST_CASE("Interpreter returns bound graph input", "[reference][interpreter]") {
  Graph graph;

  ValueId x = graph.input(Shape{2}, Dtype::F32);

  InputBindings bindings;
  bindings.bind(x, Tensor{
                       Shape{2},
                       Dtype::F32,
                       std::vector<float>{1.0f, 2.0f},
                   });

  Interpreter interpreter;

  Tensor result = interpreter.run(graph, x, bindings);

  REQUIRE(result.shape() == Shape{2});
  REQUIRE(result.dtype() == Dtype::F32);
  REQUIRE(result.data() == std::vector<float>{1.0f, 2.0f});
}

TEST_CASE("Interpreter evaluates add", "[reference][interpreter][add]") {
  Graph graph;

  ValueId a = graph.input(Shape{3}, Dtype::F32);
  ValueId b = graph.input(Shape{3}, Dtype::F32);

  ValueId out = graph.add(a, b);

  InputBindings bindings;

  bindings.bind(a, Tensor{
                       Shape{3},
                       Dtype::F32,
                       std::vector<float>{1.0f, 2.0f, 3.0f},
                   });

  bindings.bind(b, Tensor{
                       Shape{3},
                       Dtype::F32,
                       std::vector<float>{4.0f, 5.0f, 6.0f},
                   });

  Interpreter interpreter;

  Tensor result = interpreter.run(graph, out, bindings);

  REQUIRE(result.shape() == Shape{3});
  REQUIRE(result.data() == std::vector<float>{5.0f, 7.0f, 9.0f});
}

TEST_CASE("Interpreter evaluates mul", "[reference][interpreter][mul]") {
  Graph graph;

  ValueId a = graph.input(Shape{3}, Dtype::F32);
  ValueId b = graph.input(Shape{3}, Dtype::F32);

  ValueId out = graph.mul(a, b);

  InputBindings bindings;

  bindings.bind(a, Tensor{
                       Shape{3},
                       Dtype::F32,
                       std::vector<float>{1.0f, 2.0f, 3.0f},
                   });

  bindings.bind(b, Tensor{
                       Shape{3},
                       Dtype::F32,
                       std::vector<float>{4.0f, 5.0f, 6.0f},
                   });

  Interpreter interpreter;

  Tensor result = interpreter.run(graph, out, bindings);

  REQUIRE(result.data() == std::vector<float>{4.0f, 10.0f, 18.0f});
}

TEST_CASE("Interpreter evaluates sum", "[reference][interpreter][sum]") {
  Graph graph;

  ValueId x = graph.input(Shape{4}, Dtype::F32);
  ValueId out = graph.sum(x);

  InputBindings bindings;

  bindings.bind(x, Tensor{
                       Shape{4},
                       Dtype::F32,
                       std::vector<float>{
                           1.0f,
                           2.0f,
                           3.0f,
                           4.0f,
                       },
                   });

  Interpreter interpreter;

  Tensor result = interpreter.run(graph, out, bindings);

  REQUIRE(result.shape() == Shape{});
  REQUIRE(result.shape().rank() == 0);
  REQUIRE(result.data() == std::vector<float>{10.0f});
}

TEST_CASE("Interpreter evaluates matmul", "[reference][interpreter][matmul]") {
  Graph graph;

  ValueId a = graph.input(Shape{2, 3}, Dtype::F32);
  ValueId b = graph.input(Shape{3, 2}, Dtype::F32);

  ValueId out = graph.matmul(a, b);

  InputBindings bindings;

  bindings.bind(a, Tensor{
                       Shape{2, 3},
                       Dtype::F32,
                       std::vector<float>{
                           1.0f,
                           2.0f,
                           3.0f,
                           4.0f,
                           5.0f,
                           6.0f,
                       },
                   });

  bindings.bind(b, Tensor{
                       Shape{3, 2},
                       Dtype::F32,
                       std::vector<float>{
                           7.0f,
                           8.0f,
                           9.0f,
                           10.0f,
                           11.0f,
                           12.0f,
                       },
                   });

  Interpreter interpreter;

  Tensor result = interpreter.run(graph, out, bindings);

  REQUIRE(result.shape() == Shape{2, 2});

  REQUIRE(result.data() == std::vector<float>{
                               58.0f,
                               64.0f,
                               139.0f,
                               154.0f,
                           });
}

TEST_CASE("Interpreter recursively evaluates computed operands",
          "[reference][interpreter][recursive]") {
  Graph graph;

  ValueId x = graph.input(Shape{2}, Dtype::F32);
  ValueId y = graph.input(Shape{2}, Dtype::F32);
  ValueId z = graph.input(Shape{2}, Dtype::F32);

  // intermediate = x * y
  ValueId intermediate = graph.mul(x, y);

  // output = intermediate + z
  ValueId output = graph.add(intermediate, z);

  InputBindings bindings;

  bindings.bind(x, Tensor{
                       Shape{2},
                       Dtype::F32,
                       std::vector<float>{1.0f, 2.0f},
                   });

  bindings.bind(y, Tensor{
                       Shape{2},
                       Dtype::F32,
                       std::vector<float>{3.0f, 4.0f},
                   });

  bindings.bind(z, Tensor{
                       Shape{2},
                       Dtype::F32,
                       std::vector<float>{5.0f, 6.0f},
                   });

  Interpreter interpreter;

  Tensor result = interpreter.run(graph, output, bindings);

  // x * y = [3, 8]
  // [3, 8] + [5, 6] = [8, 14]
  REQUIRE(result.data() == std::vector<float>{8.0f, 14.0f});
}

TEST_CASE("Interpreter evaluates deeper expression tree",
          "[reference][interpreter][recursive]") {
  Graph graph;

  ValueId a = graph.input(Shape{2}, Dtype::F32);
  ValueId b = graph.input(Shape{2}, Dtype::F32);
  ValueId c = graph.input(Shape{2}, Dtype::F32);

  // ab = a + b
  ValueId ab = graph.add(a, b);

  // product = ab * c
  ValueId product = graph.mul(ab, c);

  // result = sum(product)
  ValueId output = graph.sum(product);

  InputBindings bindings;

  bindings.bind(a, Tensor{
                       Shape{2},
                       Dtype::F32,
                       std::vector<float>{1.0f, 2.0f},
                   });

  bindings.bind(b, Tensor{
                       Shape{2},
                       Dtype::F32,
                       std::vector<float>{3.0f, 4.0f},
                   });

  bindings.bind(c, Tensor{
                       Shape{2},
                       Dtype::F32,
                       std::vector<float>{2.0f, 3.0f},
                   });

  Interpreter interpreter;

  Tensor result = interpreter.run(graph, output, bindings);

  // a + b = [4, 6]
  // [4, 6] * [2, 3] = [8, 18]
  // sum = 26
  REQUIRE(result.shape() == Shape{});
  REQUIRE(result.data() == std::vector<float>{26.0f});
}

TEST_CASE("Interpreter can consume matmul result in another op",
          "[reference][interpreter][recursive][matmul]") {
  Graph graph;

  ValueId a = graph.input(Shape{2, 2}, Dtype::F32);
  ValueId b = graph.input(Shape{2, 2}, Dtype::F32);
  ValueId bias = graph.input(Shape{2, 2}, Dtype::F32);

  ValueId product = graph.matmul(a, b);
  ValueId output = graph.add(product, bias);

  InputBindings bindings;

  bindings.bind(a, Tensor{
                       Shape{2, 2},
                       Dtype::F32,
                       std::vector<float>{
                           1.0f,
                           2.0f,
                           3.0f,
                           4.0f,
                       },
                   });

  bindings.bind(b, Tensor{
                       Shape{2, 2},
                       Dtype::F32,
                       std::vector<float>{
                           5.0f,
                           6.0f,
                           7.0f,
                           8.0f,
                       },
                   });

  bindings.bind(bias, Tensor{
                          Shape{2, 2},
                          Dtype::F32,
                          std::vector<float>{
                              1.0f,
                              1.0f,
                              1.0f,
                              1.0f,
                          },
                      });

  Interpreter interpreter;

  Tensor result = interpreter.run(graph, output, bindings);

  // A * B =
  // [19, 22]
  // [43, 50]
  //
  // + bias =
  // [20, 23]
  // [44, 51]

  REQUIRE(result.shape() == Shape{2, 2});

  REQUIRE(result.data() == std::vector<float>{
                               20.0f,
                               23.0f,
                               44.0f,
                               51.0f,
                           });
}

TEST_CASE("Interpreter throws when graph input has no binding",
          "[reference][interpreter][negative]") {
  Graph graph;

  ValueId x = graph.input(Shape{2}, Dtype::F32);

  InputBindings bindings;
  Interpreter interpreter;

  REQUIRE_THROWS_AS(interpreter.run(graph, x, bindings), std::out_of_range);
}

TEST_CASE("Interpreter throws when one operation operand has no binding",
          "[reference][interpreter][negative]") {
  Graph graph;

  ValueId a = graph.input(Shape{2}, Dtype::F32);
  ValueId b = graph.input(Shape{2}, Dtype::F32);

  ValueId output = graph.add(a, b);

  InputBindings bindings;

  bindings.bind(a, Tensor{
                       Shape{2},
                       Dtype::F32,
                       std::vector<float>{1.0f, 2.0f},
                   });

  // b deliberately has no binding.

  Interpreter interpreter;

  REQUIRE_THROWS_AS(interpreter.run(graph, output, bindings),
                    std::out_of_range);
}

TEST_CASE("Interpreter rejects invalid output ValueId",
          "[reference][interpreter][negative]") {
  Graph graph;

  graph.input(Shape{2}, Dtype::F32);

  InputBindings bindings;
  Interpreter interpreter;

  REQUIRE_THROWS_AS(interpreter.run(graph, ValueId{999}, bindings),
                    std::out_of_range);
}

TEST_CASE("Interpreter rejects negative output ValueId",
          "[reference][interpreter][negative]") {
  Graph graph;

  InputBindings bindings;
  Interpreter interpreter;

  REQUIRE_THROWS_AS(interpreter.run(graph, ValueId{-1}, bindings),
                    std::out_of_range);
}