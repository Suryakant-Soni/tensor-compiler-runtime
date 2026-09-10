#pragma once
#include <cstddef>

namespace tcr::lir {

struct BufferId {
  std::size_t value;
  bool operator==(const BufferId &) const = default;
};

struct ScalarId {
  std::size_t value;
  bool operator==(const ScalarId &) const = default;
};

struct IndexId {
  std::size_t value;
  bool operator==(const IndexId &) const = default;
};

} // namespace tcr::lir