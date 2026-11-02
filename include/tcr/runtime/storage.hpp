#pragma once

#include <cstddef>
#include <vector>

namespace tcr::runtime {

struct RuntimeStorage {
  std::vector<std::byte> bytes;
};

} // namespace tcr::runtime