#include "tcr/ir/dtype.hpp"
#include "tcr/ir/op.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/ir/type_inference.hpp"
#include "tcr/ir/value.hpp"

#include <catch2/catch_test_macros.hpp>

#include <optional>
#include <stdexcept>
#include <vector>

using namespace tcr::ir;

namespace {

Value make_value(Shape shape, Dtype dtype = Dtype::F32) {
  return Value{
      .shape = std::move(shape),
      .dtype = dtype,
      .producer = std::nullopt,
  };
}

} // namespace

// ============================================================================
// ADD
// ============================================================================

TEST_CASE("infer_type infers Add result type for matching matrices",
          "[type_inference][add]") {
  const Value lhs = make_value(Shape{2, 3});
  const Value rhs = make_value(Shape{2, 3});

  const InferredType result =
      infer_type(Op::Add, std::vector<const Value *>{&lhs, &rhs});

  REQUIRE(result.shape == Shape{2, 3});
  REQUIRE(result.dtype == Dtype::F32);
}

TEST_CASE("infer_type infers Add result type for matching vectors",
          "[type_inference][add]") {
  const Value lhs = make_value(Shape{5});
  const Value rhs = make_value(Shape{5});

  const InferredType result =
      infer_type(Op::Add, std::vector<const Value *>{&lhs, &rhs});

  REQUIRE(result.shape == Shape{5});
  REQUIRE(result.dtype == Dtype::F32);
}

TEST_CASE("infer_type infers Add result type for matching scalars",
          "[type_inference][add]") {
  const Value lhs = make_value(Shape{});
  const Value rhs = make_value(Shape{});

  const InferredType result =
      infer_type(Op::Add, std::vector<const Value *>{&lhs, &rhs});

  REQUIRE(result.shape == Shape{});
  REQUIRE(result.dtype == Dtype::F32);
}

TEST_CASE("infer_type infers Add result type for higher-rank tensors",
          "[type_inference][add]") {
  const Value lhs = make_value(Shape{2, 3, 4});
  const Value rhs = make_value(Shape{2, 3, 4});

  const InferredType result =
      infer_type(Op::Add, std::vector<const Value *>{&lhs, &rhs});

  REQUIRE(result.shape == Shape{2, 3, 4});
  REQUIRE(result.dtype == Dtype::F32);
}

TEST_CASE("infer_type rejects Add operands with different shapes",
          "[type_inference][add]") {
  const Value lhs = make_value(Shape{2, 3});
  const Value rhs = make_value(Shape{3, 2});

  REQUIRE_THROWS_AS(
      infer_type(Op::Add, std::vector<const Value *>{&lhs, &rhs}),
      std::invalid_argument);
}

TEST_CASE("infer_type rejects Add operands with different ranks",
          "[type_inference][add]") {
  const Value lhs = make_value(Shape{2, 3});
  const Value rhs = make_value(Shape{2, 3, 1});

  REQUIRE_THROWS_AS(
      infer_type(Op::Add, std::vector<const Value *>{&lhs, &rhs}),
      std::invalid_argument);
}

// ============================================================================
// MUL
// ============================================================================

TEST_CASE("infer_type infers Mul result type for matching matrices",
          "[type_inference][mul]") {
  const Value lhs = make_value(Shape{2, 3});
  const Value rhs = make_value(Shape{2, 3});

  const InferredType result =
      infer_type(Op::Mul, std::vector<const Value *>{&lhs, &rhs});

  REQUIRE(result.shape == Shape{2, 3});
  REQUIRE(result.dtype == Dtype::F32);
}

TEST_CASE("infer_type infers Mul result type for matching vectors",
          "[type_inference][mul]") {
  const Value lhs = make_value(Shape{7});
  const Value rhs = make_value(Shape{7});

  const InferredType result =
      infer_type(Op::Mul, std::vector<const Value *>{&lhs, &rhs});

  REQUIRE(result.shape == Shape{7});
  REQUIRE(result.dtype == Dtype::F32);
}

TEST_CASE("infer_type infers Mul result type for matching scalars",
          "[type_inference][mul]") {
  const Value lhs = make_value(Shape{});
  const Value rhs = make_value(Shape{});

  const InferredType result =
      infer_type(Op::Mul, std::vector<const Value *>{&lhs, &rhs});

  REQUIRE(result.shape == Shape{});
  REQUIRE(result.dtype == Dtype::F32);
}

TEST_CASE("infer_type infers Mul result type for higher-rank tensors",
          "[type_inference][mul]") {
  const Value lhs = make_value(Shape{2, 3, 4});
  const Value rhs = make_value(Shape{2, 3, 4});

  const InferredType result =
      infer_type(Op::Mul, std::vector<const Value *>{&lhs, &rhs});

  REQUIRE(result.shape == Shape{2, 3, 4});
  REQUIRE(result.dtype == Dtype::F32);
}

TEST_CASE("infer_type rejects Mul operands with different shapes",
          "[type_inference][mul]") {
  const Value lhs = make_value(Shape{2, 3});
  const Value rhs = make_value(Shape{3, 2});

  REQUIRE_THROWS_AS(
      infer_type(Op::Mul, std::vector<const Value *>{&lhs, &rhs}),
      std::invalid_argument);
}

TEST_CASE("infer_type rejects Mul operands with different ranks",
          "[type_inference][mul]") {
  const Value lhs = make_value(Shape{2, 3});
  const Value rhs = make_value(Shape{2, 3, 1});

  REQUIRE_THROWS_AS(
      infer_type(Op::Mul, std::vector<const Value *>{&lhs, &rhs}),
      std::invalid_argument);
}

// ============================================================================
// MATMUL
// ============================================================================

TEST_CASE("infer_type infers MatMul result shape",
          "[type_inference][matmul]") {
  const Value lhs = make_value(Shape{2, 3});
  const Value rhs = make_value(Shape{3, 4});

  const InferredType result =
      infer_type(Op::MatMul, std::vector<const Value *>{&lhs, &rhs});

  REQUIRE(result.shape == Shape{2, 4});
  REQUIRE(result.dtype == Dtype::F32);
}

TEST_CASE("infer_type infers MatMul result for square matrices",
          "[type_inference][matmul]") {
  const Value lhs = make_value(Shape{4, 4});
  const Value rhs = make_value(Shape{4, 4});

  const InferredType result =
      infer_type(Op::MatMul, std::vector<const Value *>{&lhs, &rhs});

  REQUIRE(result.shape == Shape{4, 4});
  REQUIRE(result.dtype == Dtype::F32);
}

TEST_CASE("infer_type infers MatMul result for matrix times column matrix",
          "[type_inference][matmul]") {
  const Value lhs = make_value(Shape{5, 3});
  const Value rhs = make_value(Shape{3, 1});

  const InferredType result =
      infer_type(Op::MatMul, std::vector<const Value *>{&lhs, &rhs});

  REQUIRE(result.shape == Shape{5, 1});
  REQUIRE(result.dtype == Dtype::F32);
}

TEST_CASE("infer_type infers MatMul result for row matrix times matrix",
          "[type_inference][matmul]") {
  const Value lhs = make_value(Shape{1, 3});
  const Value rhs = make_value(Shape{3, 8});

  const InferredType result =
      infer_type(Op::MatMul, std::vector<const Value *>{&lhs, &rhs});

  REQUIRE(result.shape == Shape{1, 8});
  REQUIRE(result.dtype == Dtype::F32);
}

TEST_CASE("infer_type rejects MatMul with incompatible inner dimensions",
          "[type_inference][matmul]") {
  const Value lhs = make_value(Shape{2, 3});
  const Value rhs = make_value(Shape{5, 4});

  REQUIRE_THROWS_AS(
      infer_type(Op::MatMul, std::vector<const Value *>{&lhs, &rhs}),
      std::invalid_argument);
}

TEST_CASE("infer_type rejects MatMul when lhs is rank one",
          "[type_inference][matmul]") {
  const Value lhs = make_value(Shape{3});
  const Value rhs = make_value(Shape{3, 4});

  REQUIRE_THROWS_AS(
      infer_type(Op::MatMul, std::vector<const Value *>{&lhs, &rhs}),
      std::invalid_argument);
}

TEST_CASE("infer_type rejects MatMul when rhs is rank one",
          "[type_inference][matmul]") {
  const Value lhs = make_value(Shape{2, 3});
  const Value rhs = make_value(Shape{3});

  REQUIRE_THROWS_AS(
      infer_type(Op::MatMul, std::vector<const Value *>{&lhs, &rhs}),
      std::invalid_argument);
}

TEST_CASE("infer_type rejects MatMul when lhs has rank greater than two",
          "[type_inference][matmul]") {
  const Value lhs = make_value(Shape{2, 3, 4});
  const Value rhs = make_value(Shape{4, 5});

  REQUIRE_THROWS_AS(
      infer_type(Op::MatMul, std::vector<const Value *>{&lhs, &rhs}),
      std::invalid_argument);
}

TEST_CASE("infer_type rejects MatMul when rhs has rank greater than two",
          "[type_inference][matmul]") {
  const Value lhs = make_value(Shape{2, 3});
  const Value rhs = make_value(Shape{3, 4, 5});

  REQUIRE_THROWS_AS(
      infer_type(Op::MatMul, std::vector<const Value *>{&lhs, &rhs}),
      std::invalid_argument);
}

TEST_CASE("infer_type rejects MatMul on scalar operands",
          "[type_inference][matmul]") {
  const Value lhs = make_value(Shape{});
  const Value rhs = make_value(Shape{});

  REQUIRE_THROWS_AS(
      infer_type(Op::MatMul, std::vector<const Value *>{&lhs, &rhs}),
      std::invalid_argument);
}

// ============================================================================
// SUM
// ============================================================================

TEST_CASE("infer_type infers Sum of matrix as scalar",
          "[type_inference][sum]") {
  const Value input = make_value(Shape{2, 3});

  const InferredType result =
      infer_type(Op::Sum, std::vector<const Value *>{&input});

  REQUIRE(result.shape == Shape{});
  REQUIRE(result.dtype == Dtype::F32);
}

TEST_CASE("infer_type infers Sum of vector as scalar",
          "[type_inference][sum]") {
  const Value input = make_value(Shape{8});

  const InferredType result =
      infer_type(Op::Sum, std::vector<const Value *>{&input});

  REQUIRE(result.shape == Shape{});
  REQUIRE(result.dtype == Dtype::F32);
}

TEST_CASE("infer_type infers Sum of higher-rank tensor as scalar",
          "[type_inference][sum]") {
  const Value input = make_value(Shape{2, 3, 4, 5});

  const InferredType result =
      infer_type(Op::Sum, std::vector<const Value *>{&input});

  REQUIRE(result.shape == Shape{});
  REQUIRE(result.dtype == Dtype::F32);
}

TEST_CASE("infer_type permits Sum of scalar and produces scalar",
          "[type_inference][sum]") {
  const Value input = make_value(Shape{});

  const InferredType result =
      infer_type(Op::Sum, std::vector<const Value *>{&input});

  REQUIRE(result.shape == Shape{});
  REQUIRE(result.dtype == Dtype::F32);
}

// ============================================================================
// ZERO-SIZED DIMENSIONS
// ============================================================================

TEST_CASE("infer_type preserves zero-sized dimensions for Add",
          "[type_inference][edge]") {
  const Value lhs = make_value(Shape{2, 0, 4});
  const Value rhs = make_value(Shape{2, 0, 4});

  const InferredType result =
      infer_type(Op::Add, std::vector<const Value *>{&lhs, &rhs});

  REQUIRE(result.shape == Shape{2, 0, 4});
  REQUIRE(result.dtype == Dtype::F32);
}

TEST_CASE("infer_type preserves zero-sized dimensions for Mul",
          "[type_inference][edge]") {
  const Value lhs = make_value(Shape{0, 3});
  const Value rhs = make_value(Shape{0, 3});

  const InferredType result =
      infer_type(Op::Mul, std::vector<const Value *>{&lhs, &rhs});

  REQUIRE(result.shape == Shape{0, 3});
  REQUIRE(result.dtype == Dtype::F32);
}

TEST_CASE("infer_type handles MatMul with zero output rows",
          "[type_inference][matmul][edge]") {
  const Value lhs = make_value(Shape{0, 3});
  const Value rhs = make_value(Shape{3, 4});

  const InferredType result =
      infer_type(Op::MatMul, std::vector<const Value *>{&lhs, &rhs});

  REQUIRE(result.shape == Shape{0, 4});
  REQUIRE(result.dtype == Dtype::F32);
}

TEST_CASE("infer_type handles MatMul with zero output columns",
          "[type_inference][matmul][edge]") {
  const Value lhs = make_value(Shape{2, 3});
  const Value rhs = make_value(Shape{3, 0});

  const InferredType result =
      infer_type(Op::MatMul, std::vector<const Value *>{&lhs, &rhs});

  REQUIRE(result.shape == Shape{2, 0});
  REQUIRE(result.dtype == Dtype::F32);
}

TEST_CASE("infer_type handles MatMul with zero shared dimension",
          "[type_inference][matmul][edge]") {
  const Value lhs = make_value(Shape{2, 0});
  const Value rhs = make_value(Shape{0, 4});

  const InferredType result =
      infer_type(Op::MatMul, std::vector<const Value *>{&lhs, &rhs});

  REQUIRE(result.shape == Shape{2, 4});
  REQUIRE(result.dtype == Dtype::F32);
}

// ============================================================================
// ARITY VALIDATION
// ============================================================================

TEST_CASE("infer_type rejects Add with no operands",
          "[type_inference][arity]") {
  REQUIRE_THROWS_AS(
      infer_type(Op::Add, std::vector<const Value *>{}),
      std::invalid_argument);
}

TEST_CASE("infer_type rejects Add with one operand",
          "[type_inference][arity]") {
  const Value input = make_value(Shape{2, 3});

  REQUIRE_THROWS_AS(
      infer_type(Op::Add, std::vector<const Value *>{&input}),
      std::invalid_argument);
}

TEST_CASE("infer_type rejects Add with more than two operands",
          "[type_inference][arity]") {
  const Value a = make_value(Shape{2, 3});
  const Value b = make_value(Shape{2, 3});
  const Value c = make_value(Shape{2, 3});

  REQUIRE_THROWS_AS(
      infer_type(Op::Add, std::vector<const Value *>{&a, &b, &c}),
      std::invalid_argument);
}

TEST_CASE("infer_type rejects Mul with wrong arity",
          "[type_inference][arity]") {
  const Value input = make_value(Shape{2, 3});

  REQUIRE_THROWS_AS(
      infer_type(Op::Mul, std::vector<const Value *>{&input}),
      std::invalid_argument);
}

TEST_CASE("infer_type rejects MatMul with wrong arity",
          "[type_inference][arity]") {
  const Value input = make_value(Shape{2, 3});

  REQUIRE_THROWS_AS(
      infer_type(Op::MatMul, std::vector<const Value *>{&input}),
      std::invalid_argument);
}

TEST_CASE("infer_type rejects Sum with no operands",
          "[type_inference][arity]") {
  REQUIRE_THROWS_AS(
      infer_type(Op::Sum, std::vector<const Value *>{}),
      std::invalid_argument);
}

TEST_CASE("infer_type rejects Sum with more than one operand",
          "[type_inference][arity]") {
  const Value a = make_value(Shape{2, 3});
  const Value b = make_value(Shape{2, 3});

  REQUIRE_THROWS_AS(
      infer_type(Op::Sum, std::vector<const Value *>{&a, &b}),
      std::invalid_argument);
}

// ============================================================================
// UNSUPPORTED OPERATION
// ============================================================================

TEST_CASE("infer_type rejects Input because Input has no inferred result type",
          "[type_inference][unsupported]") {
  REQUIRE_THROWS_AS(infer_type(Op::Input, std::vector<const Value *>{}),
                    std::logic_error);
}