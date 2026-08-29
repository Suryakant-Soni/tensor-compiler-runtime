#include "tcr/reference/ops.hpp"
#include "tcr/ir/dtype.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/reference/tensor.hpp"

#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <vector>

using namespace tcr::ir;
using namespace tcr::reference;

TEST_CASE("add performs elementwise addition", "[reference][ops][add]") {
    Tensor a{
        Shape{2, 2},
        Dtype::F32,
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
    };

    Tensor b{
        Shape{2, 2},
        Dtype::F32,
        std::vector<float>{10.0f, 20.0f, 30.0f, 40.0f},
    };

    Tensor out = add(a, b);

    REQUIRE(out.shape() == Shape{2, 2});
    REQUIRE(out.dtype() == Dtype::F32);
    REQUIRE(out.data() ==
            std::vector<float>{11.0f, 22.0f, 33.0f, 44.0f});
}

TEST_CASE("add works for rank-1 tensors", "[reference][ops][add]") {
    Tensor a{
        Shape{3},
        Dtype::F32,
        std::vector<float>{1.0f, 2.0f, 3.0f},
    };

    Tensor b{
        Shape{3},
        Dtype::F32,
        std::vector<float>{4.0f, 5.0f, 6.0f},
    };

    Tensor out = add(a, b);

    REQUIRE(out.shape() == Shape{3});
    REQUIRE(out.data() ==
            std::vector<float>{5.0f, 7.0f, 9.0f});
}

TEST_CASE("add works for higher-rank tensors", "[reference][ops][add]") {
    Tensor a{
        Shape{2, 1, 2},
        Dtype::F32,
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
    };

    Tensor b{
        Shape{2, 1, 2},
        Dtype::F32,
        std::vector<float>{5.0f, 6.0f, 7.0f, 8.0f},
    };

    Tensor out = add(a, b);

    REQUIRE(out.shape() == Shape{2, 1, 2});
    REQUIRE(out.data() ==
            std::vector<float>{6.0f, 8.0f, 10.0f, 12.0f});
}

TEST_CASE("add rejects mismatched shapes", "[reference][ops][add]") {
    Tensor a{
        Shape{2, 2},
        Dtype::F32,
        std::vector<float>{1, 2, 3, 4},
    };

    Tensor b{
        Shape{4},
        Dtype::F32,
        std::vector<float>{1, 2, 3, 4},
    };

    REQUIRE_THROWS_AS(add(a, b), std::invalid_argument);
}

TEST_CASE("mul performs elementwise multiplication",
          "[reference][ops][mul]") {
    Tensor a{
        Shape{2, 2},
        Dtype::F32,
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
    };

    Tensor b{
        Shape{2, 2},
        Dtype::F32,
        std::vector<float>{5.0f, 6.0f, 7.0f, 8.0f},
    };

    Tensor out = mul(a, b);

    REQUIRE(out.shape() == Shape{2, 2});
    REQUIRE(out.dtype() == Dtype::F32);
    REQUIRE(out.data() ==
            std::vector<float>{5.0f, 12.0f, 21.0f, 32.0f});
}

TEST_CASE("mul handles zero and negative values",
          "[reference][ops][mul]") {
    Tensor a{
        Shape{4},
        Dtype::F32,
        std::vector<float>{0.0f, -2.0f, 3.0f, -4.0f},
    };

    Tensor b{
        Shape{4},
        Dtype::F32,
        std::vector<float>{5.0f, 3.0f, -2.0f, -1.0f},
    };

    Tensor out = mul(a, b);

    REQUIRE(out.data() ==
            std::vector<float>{0.0f, -6.0f, -6.0f, 4.0f});
}

TEST_CASE("mul rejects mismatched shapes", "[reference][ops][mul]") {
    Tensor a{
        Shape{2, 2},
        Dtype::F32,
        std::vector<float>{1, 2, 3, 4},
    };

    Tensor b{
        Shape{2, 1},
        Dtype::F32,
        std::vector<float>{1, 2},
    };

    REQUIRE_THROWS_AS(mul(a, b), std::invalid_argument);
}

TEST_CASE("elementwise_binary applies supplied function",
          "[reference][ops][elementwise]") {
    Tensor a{
        Shape{3},
        Dtype::F32,
        std::vector<float>{5.0f, 7.0f, 9.0f},
    };

    Tensor b{
        Shape{3},
        Dtype::F32,
        std::vector<float>{1.0f, 2.0f, 3.0f},
    };

    Tensor out = elementwise_binary(
        a,
        b,
        [](float lhs, float rhs) {
            return lhs - rhs;
        });

    REQUIRE(out.shape() == Shape{3});
    REQUIRE(out.data() ==
            std::vector<float>{4.0f, 5.0f, 6.0f});
}

TEST_CASE("elementwise_binary rejects mismatched shapes",
          "[reference][ops][elementwise]") {
    Tensor a{
        Shape{2, 2},
        Dtype::F32,
        std::vector<float>{1, 2, 3, 4},
    };

    Tensor b{
        Shape{4},
        Dtype::F32,
        std::vector<float>{1, 2, 3, 4},
    };

    REQUIRE_THROWS_AS(
        elementwise_binary(
            a,
            b,
            [](float x, float y) { return x + y; }),
        std::invalid_argument);
}

TEST_CASE("sum reduces rank-1 tensor to scalar",
          "[reference][ops][sum]") {
    Tensor input{
        Shape{4},
        Dtype::F32,
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
    };

    Tensor out = sum(input);

    REQUIRE(out.shape() == Shape{});
    REQUIRE(out.shape().rank() == 0);
    REQUIRE(out.dtype() == Dtype::F32);
    REQUIRE(out.data() == std::vector<float>{10.0f});
}

TEST_CASE("sum reduces higher-rank tensor to scalar",
          "[reference][ops][sum]") {
    Tensor input{
        Shape{2, 2},
        Dtype::F32,
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
    };

    Tensor out = sum(input);

    REQUIRE(out.shape() == Shape{});
    REQUIRE(out.data() == std::vector<float>{10.0f});
}

TEST_CASE("sum handles negative values", "[reference][ops][sum]") {
    Tensor input{
        Shape{4},
        Dtype::F32,
        std::vector<float>{10.0f, -2.0f, -3.0f, 1.0f},
    };

    Tensor out = sum(input);

    REQUIRE(out.data() == std::vector<float>{6.0f});
}

TEST_CASE("sum handles a one-element tensor", "[reference][ops][sum]") {
    Tensor input{
        Shape{1},
        Dtype::F32,
        std::vector<float>{42.0f},
    };

    Tensor out = sum(input);

    REQUIRE(out.shape() == Shape{});
    REQUIRE(out.data() == std::vector<float>{42.0f});
}

TEST_CASE("matmul computes matrix product", "[reference][ops][matmul]") {
    Tensor lhs{
        Shape{2, 3},
        Dtype::F32,
        std::vector<float>{
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
        },
    };

    Tensor rhs{
        Shape{3, 2},
        Dtype::F32,
        std::vector<float>{
            7.0f, 8.0f,
            9.0f, 10.0f,
            11.0f, 12.0f,
        },
    };

    Tensor out = matmul(lhs, rhs);

    REQUIRE(out.shape() == Shape{2, 2});
    REQUIRE(out.dtype() == Dtype::F32);

    REQUIRE(out.data() ==
            std::vector<float>{
                58.0f, 64.0f,
                139.0f, 154.0f,
            });
}

TEST_CASE("matmul works for rectangular matrices",
          "[reference][ops][matmul]") {
    Tensor lhs{
        Shape{1, 3},
        Dtype::F32,
        std::vector<float>{1.0f, 2.0f, 3.0f},
    };

    Tensor rhs{
        Shape{3, 4},
        Dtype::F32,
        std::vector<float>{
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
        },
    };

    Tensor out = matmul(lhs, rhs);

    REQUIRE(out.shape() == Shape{1, 4});
    REQUIRE(out.data() ==
            std::vector<float>{
                38.0f, 44.0f, 50.0f, 56.0f,
            });
}

TEST_CASE("matmul works with identity matrix",
          "[reference][ops][matmul]") {
    Tensor lhs{
        Shape{2, 2},
        Dtype::F32,
        std::vector<float>{
            1.0f, 2.0f,
            3.0f, 4.0f,
        },
    };

    Tensor identity{
        Shape{2, 2},
        Dtype::F32,
        std::vector<float>{
            1.0f, 0.0f,
            0.0f, 1.0f,
        },
    };

    Tensor out = matmul(lhs, identity);

    REQUIRE(out.shape() == Shape{2, 2});
    REQUIRE(out.data() == lhs.data());
}

TEST_CASE("matmul rejects lhs rank other than two",
          "[reference][ops][matmul]") {
    Tensor lhs{
        Shape{3},
        Dtype::F32,
        std::vector<float>{1.0f, 2.0f, 3.0f},
    };

    Tensor rhs{
        Shape{3, 2},
        Dtype::F32,
        std::vector<float>{
            1, 2,
            3, 4,
            5, 6,
        },
    };

    REQUIRE_THROWS_AS(matmul(lhs, rhs), std::invalid_argument);
}

TEST_CASE("matmul rejects rhs rank other than two",
          "[reference][ops][matmul]") {
    Tensor lhs{
        Shape{2, 3},
        Dtype::F32,
        std::vector<float>{
            1, 2, 3,
            4, 5, 6,
        },
    };

    Tensor rhs{
        Shape{3},
        Dtype::F32,
        std::vector<float>{1, 2, 3},
    };

    REQUIRE_THROWS_AS(matmul(lhs, rhs), std::invalid_argument);
}

TEST_CASE("matmul rejects incompatible inner dimensions",
          "[reference][ops][matmul]") {
    Tensor lhs{
        Shape{2, 3},
        Dtype::F32,
        std::vector<float>{
            1, 2, 3,
            4, 5, 6,
        },
    };

    Tensor rhs{
        Shape{4, 2},
        Dtype::F32,
        std::vector<float>{
            1, 2,
            3, 4,
            5, 6,
            7, 8,
        },
    };

    REQUIRE_THROWS_AS(matmul(lhs, rhs), std::invalid_argument);
}