#include "tcr/ir/shape.hpp"
#include <catch2/catch_test_macros.hpp>

using namespace tcr::ir;

TEST_CASE("Shape stores dimensions and reports rank", "[shape]") {
    Shape shape{2, 3, 4};

    REQUIRE(shape.rank() == 3);
    REQUIRE(shape.dim(0) == 2);
    REQUIRE(shape.dim(1) == 3);
    REQUIRE(shape.dim(2) == 4);
}

TEST_CASE("Shape rank matches number of dimensions", "[shape]") {
    REQUIRE(Shape{5}.rank() == 1);
    REQUIRE(Shape{2, 3}.rank() == 2);
    REQUIRE(Shape{2, 3, 4}.rank() == 3);
    REQUIRE(Shape{}.rank() == 0);
}

TEST_CASE("Shape num_elements computes product of dimensions", "[shape]") {
    REQUIRE(Shape{2, 3}.num_elements() == 6);
    REQUIRE(Shape{2, 3, 4}.num_elements() == 24);
    REQUIRE(Shape{5}.num_elements() == 5);
}

TEST_CASE("Scalar Shape has rank zero and one element", "[shape]") {
    Shape scalar{};

    REQUIRE(scalar.rank() == 0);
    REQUIRE(scalar.num_elements() == 1);
}

TEST_CASE("Shape with zero-sized dimension has zero elements", "[shape]") {
    REQUIRE(Shape{0}.num_elements() == 0);
    REQUIRE(Shape{2, 0, 4}.num_elements() == 0);
}

TEST_CASE("Shape equality compares dimensions", "[shape]") {
    Shape a{2, 3};
    Shape b{2, 3};
    Shape c{3, 2};
    Shape d{2, 3, 1};

    REQUIRE(a == b);
    REQUIRE_FALSE(a == c);
    REQUIRE_FALSE(a == d);
}

TEST_CASE("Shape dimension values preserve construction order", "[shape]") {
    Shape shape{8, 16, 32, 64};

    REQUIRE(shape.dim(0) == 8);
    REQUIRE(shape.dim(1) == 16);
    REQUIRE(shape.dim(2) == 32);
    REQUIRE(shape.dim(3) == 64);
}

TEST_CASE("One-dimensional Shape behaves correctly", "[shape]") {
    Shape shape{7};

    REQUIRE(shape.rank() == 1);
    REQUIRE(shape.dim(0) == 7);
    REQUIRE(shape.num_elements() == 7);
}

TEST_CASE("Large dimensions produce expected element count", "[shape]") {
    Shape shape{64, 128, 32};

    REQUIRE(shape.num_elements() == 64 * 128 * 32);
}