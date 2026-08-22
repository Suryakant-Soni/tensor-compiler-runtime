#include "tcr/ir/dtype.hpp"
#include <string_view>

namespace tcr::ir {

std::string_view to_string(Dtype dt) {
  switch (dt) {
  case Dtype::F32:
    return "F32";
  }
}

std::size_t size_in_bytes(Dtype dt) {
  switch (dt) {
  case Dtype::F32:
    return 4;
  }
}
} // namespace tcr::ir