// documents the data type of elements of the inputs and outputs of these tensors
#include <cstddef>
#include <string_view>
namespace tcr::ir {
enum class Dtype {
  F32,
};

std::string_view to_string(Dtype);
std::size_t size_in_bytes(Dtype);
} // namespace tcr::ir