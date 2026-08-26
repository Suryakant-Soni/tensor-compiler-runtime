#pragma once
// documents the id type given to nodes and tensor-input/output values
#include <cstdint>
#include <ostream>
namespace tcr::ir {
struct ValueId {
  std::int32_t value;
  // equality implemented
  bool operator==(const ValueId &) const = default;
};

struct NodeId {
  std::int32_t value;
  // equality implemented
  bool operator==(const NodeId &) const = default;
};

// for print/debug
inline std::ostream &operator<<(std::ostream &os, ValueId id) {
  return os << "ValueId(" << id.value << ")";
}

inline std::ostream &operator<<(std::ostream &os, NodeId id) {
  return os << "NodeId(" << id.value << ")";
}
} // namespace tcr::ir