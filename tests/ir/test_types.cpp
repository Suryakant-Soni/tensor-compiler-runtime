#include "tcr/ir/ids.hpp"
#include <catch2/catch_test_macros.hpp>

using namespace tcr::ir;

TEST_CASE("value and node constructor") {
REQUIRE(ValueId{3} == ValueId{3});
REQUIRE(ValueId{3} != ValueId{4});
REQUIRE(NodeId{3} == NodeId{3});
REQUIRE(NodeId{3} != NodeId{4});
}