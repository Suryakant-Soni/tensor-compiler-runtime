#include "tcr/ir/ids.hpp"
#include <catch2/catch_test_macros.hpp>

using namespace tcr::ir;

int main() {
assert(ValueId{3} == ValueId{3});
assert(ValueId{3} != ValueId{4});
assert(NodeId{3} == NodeId{3});
assert(NodeId{3} != NodeId{4});
return 0;
}