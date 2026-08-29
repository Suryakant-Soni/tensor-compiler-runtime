#include "tcr/ir/dtype.hpp"
#include "tcr/ir/graph.hpp"
#include "tcr/ir/ids.hpp"
#include "tcr/ir/shape.hpp"
#include <catch2/catch_test_macros.hpp>

using namespace tcr::ir;
#include "tcr/ir/graph.hpp"

using namespace tcr::ir;

TEST_CASE("input creates a graph input value", "[graph]") {
  Graph g;
  ValueId id = g.input(Shape{2, 3}, Dtype::F32);

  const Value &v = g.value(id);
  REQUIRE(v.shape == Shape{2, 3});
  REQUIRE(v.dtype == Dtype::F32);
  REQUIRE(v.producer == std::nullopt);

  auto ins = g.inputs();
  REQUIRE(ins.size() == 1);
  REQUIRE(ins[0] == id);
}

TEST_CASE("add records correct node and output value", "[graph]") {
  Graph g;
  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{2, 3}, Dtype::F32);

  ValueId out = g.add(a, b);

  const Node &n = g.node(NodeId{0}); // add's node — first node created
  REQUIRE(n.op == Op::Add);
  REQUIRE(n.inputs == std::vector<ValueId>{a, b});

  const Value &out_val = g.value(out);
  REQUIRE(out_val.shape == Shape{2, 3});
  REQUIRE(out_val.producer == NodeId{0});
}

TEST_CASE("add throws on mismatched shapes or dtypes") {
  Graph g;
  ValueId a = g.input(Shape{2, 3}, Dtype::F32);

  SECTION("mismatched shape") {
    ValueId b = g.input(Shape{3, 2}, Dtype::F32);
    REQUIRE_THROWS_AS(g.add(a, b), std::invalid_argument);
  }

  SECTION("mismatched dtype — once you add a second Dtype") {
    // skip until Dtype has more than one enumerator to test against
  }
}

TEST_CASE("matmul produces correct output shape", "[graph]") {
  Graph g;
  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{3, 4}, Dtype::F32);

  ValueId out = g.matmul(a, b);

  REQUIRE(g.value(out).shape == Shape{2, 4});
}

TEST_CASE("matmul throws on rank mismatch and inner-dim mismatch") {
  Graph g;

  SECTION("rank != 2") {
    ValueId a = g.input(Shape{3}, Dtype::F32);    // rank 1
    ValueId b = g.input(Shape{3, 4}, Dtype::F32); // rank 2
    REQUIRE_THROWS_AS(g.matmul(a, b), std::invalid_argument);
  }

  SECTION("inner dims don't match") {
    ValueId a = g.input(Shape{2, 3}, Dtype::F32);
    ValueId b = g.input(Shape{5, 4}, Dtype::F32); // 3 != 5
    REQUIRE_THROWS_AS(g.matmul(a, b), std::invalid_argument);
  }
}

TEST_CASE("out-of-range ValueId throws in value() and add()") {
  Graph g;
  ValueId bad{99};

  SECTION("value() with invalid id") {
    REQUIRE_THROWS_AS(g.value(bad), std::out_of_range);
  }

  SECTION("negative id") {
    ValueId negative{-1};
    REQUIRE_THROWS_AS(g.value(negative), std::out_of_range);
  }

  SECTION("add() with one invalid operand") {
    ValueId a = g.input(Shape{2, 3}, Dtype::F32);
    REQUIRE_THROWS_AS(g.add(a, bad), std::out_of_range);
  }
}

TEST_CASE("empty graph has empty inputs and outputs") {
  Graph g;
  REQUIRE(g.inputs().empty());
  REQUIRE(g.outputs().empty());
}

TEST_CASE("add_output appends without clobbering existing outputs") {
  Graph g;
  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{2, 3}, Dtype::F32);

  g.add_output(a);
  g.add_output(b);

  REQUIRE(g.outputs() == std::vector<ValueId>{a, b});
}

TEST_CASE("mul records correct node and output value", "[graph]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{2, 3}, Dtype::F32);

  ValueId out = g.mul(a, b);

  const Node &n = g.node(NodeId{0});
  REQUIRE(n.op == Op::Mul);
  REQUIRE(n.inputs == std::vector<ValueId>{a, b});

  const Value &v = g.value(out);
  REQUIRE(v.shape == Shape{2, 3});
  REQUIRE(v.dtype == Dtype::F32);
  REQUIRE(v.producer == NodeId{0});
}

TEST_CASE("mul rejects mismatched shapes", "[graph]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{3, 2}, Dtype::F32);

  REQUIRE_THROWS_AS(g.mul(a, b), std::invalid_argument);
}

TEST_CASE("sum creates scalar output", "[graph]") {
  Graph g;

  ValueId input = g.input(Shape{2, 3}, Dtype::F32);
  ValueId out = g.sum(input);

  const Node &n = g.node(NodeId{0});
  REQUIRE(n.op == Op::Sum);
  REQUIRE(n.inputs == std::vector<ValueId>{input});

  const Value &v = g.value(out);

  REQUIRE(v.shape == Shape{});
  REQUIRE(v.shape.rank() == 0);
  REQUIRE(v.dtype == Dtype::F32);
  REQUIRE(v.producer == NodeId{0});
}

TEST_CASE("sum rejects invalid ValueId", "[graph]") {
  Graph g;

  REQUIRE_THROWS_AS(g.sum(ValueId{42}), std::out_of_range);
}

TEST_CASE("matmul records correct node and output metadata", "[graph]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{3, 4}, Dtype::F32);

  ValueId out = g.matmul(a, b);

  const Node &n = g.node(NodeId{0});

  REQUIRE(n.op == Op::MatMul);
  REQUIRE(n.inputs == std::vector<ValueId>{a, b});

  const Value &v = g.value(out);

  REQUIRE(v.shape == Shape{2, 4});
  REQUIRE(v.dtype == Dtype::F32);
  REQUIRE(v.producer == NodeId{0});
}

TEST_CASE("node rejects invalid NodeId", "[graph]") {
  Graph g;

  SECTION("positive out-of-range NodeId") {
    REQUIRE_THROWS_AS(g.node(NodeId{99}), std::out_of_range);
  }

  SECTION("negative NodeId") {
    REQUIRE_THROWS_AS(g.node(NodeId{-1}), std::out_of_range);
  }
}

TEST_CASE("set_outputs stores graph outputs", "[graph]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{2, 3}, Dtype::F32);
  ValueId c = g.add(a, b);

  g.set_outputs({a, c});

  REQUIRE(g.outputs() == std::vector<ValueId>{a, c});
}

TEST_CASE("set_outputs replaces previous outputs", "[graph]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{2, 3}, Dtype::F32);
  ValueId c = g.add(a, b);

  g.set_outputs({a, b});
  REQUIRE(g.outputs() == std::vector<ValueId>{a, b});

  g.set_outputs({c});

  REQUIRE(g.outputs() == std::vector<ValueId>{c});
}

TEST_CASE("set_outputs validates every ValueId", "[graph]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId invalid{999};

  REQUIRE_THROWS_AS(g.set_outputs({a, invalid}), std::out_of_range);
}

TEST_CASE("failed set_outputs preserves existing outputs", "[graph]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{2, 3}, Dtype::F32);

  g.set_outputs({a});

  REQUIRE_THROWS_AS(g.set_outputs({b, ValueId{999}}), std::out_of_range);

  REQUIRE(g.outputs() == std::vector<ValueId>{a});
}

TEST_CASE("add_output rejects invalid ValueId", "[graph]") {
  Graph g;

  REQUIRE_THROWS_AS(g.add_output(ValueId{99}), std::out_of_range);

  REQUIRE(g.outputs().empty());
}

TEST_CASE("operation results are not automatically graph inputs", "[graph]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{2, 3}, Dtype::F32);

  ValueId out = g.add(a, b);

  REQUIRE(g.inputs() == std::vector<ValueId>{a, b});

  for (ValueId id : g.inputs()) {
    REQUIRE(id != out);
  }
}

TEST_CASE("ValueIds increase as values are created", "[graph]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{2, 3}, Dtype::F32);
  ValueId c = g.add(a, b);
  ValueId d = g.mul(a, b);

  REQUIRE(a == ValueId{0});
  REQUIRE(b == ValueId{1});
  REQUIRE(c == ValueId{2});
  REQUIRE(d == ValueId{3});
}

TEST_CASE("NodeIds increase as operations are created", "[graph]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{2, 3}, Dtype::F32);

  ValueId add_out = g.add(a, b);
  ValueId mul_out = g.mul(a, b);

  REQUIRE(g.value(add_out).producer == NodeId{0});
  REQUIRE(g.value(mul_out).producer == NodeId{1});

  REQUIRE(g.node(NodeId{0}).op == Op::Add);
  REQUIRE(g.node(NodeId{1}).op == Op::Mul);
}

TEST_CASE("failed add does not create a node or value", "[graph]") {
  Graph g;

  ValueId a = g.input(Shape{2, 3}, Dtype::F32);
  ValueId b = g.input(Shape{3, 2}, Dtype::F32);

  REQUIRE_THROWS_AS(g.add(a, b), std::invalid_argument);

  // If add had incorrectly created a node before validation,
  // NodeId{0} would exist.
  REQUIRE_THROWS_AS(g.node(NodeId{0}), std::out_of_range);

  // The next created value should still be ValueId{2}.
  ValueId c = g.input(Shape{1}, Dtype::F32);
  REQUIRE(c == ValueId{2});
}