#include "tcr/ir/dtype.hpp"
#include <string_view>

namespace tcr::ir {

std::string_view to_string(Dtype dtype) {
  switch (dtype) {
  case Dtype::F32:
    return "f32";
  }

  throw std::invalid_argument("unknown dtype");
}

std::size_t size_in_bytes(Dtype dt) {
  switch (dt) {
  case Dtype::F32:
    return 4;
  }
}
} // namespace tcr::ir