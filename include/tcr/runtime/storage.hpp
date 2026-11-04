#pragma once

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace tcr::runtime {

struct RuntimeStorage {
  std::vector<std::byte> bytes;
  float load_f32(std::size_t offset) const {
    if (offset >= bytes.size() / sizeof(float)) {
      throw std::out_of_range("offset out of bounds");
    }
    float value;
    std::memcpy(&value, bytes.data() + offset * sizeof(float), sizeof(float));
    return value;
  }

  void store_f32(std::size_t offset, float value) {
    if (offset >= bytes.size() / sizeof(float)) {
      throw std::out_of_range("offset out of bounds");
    }
    std::memcpy(bytes.data() + offset * sizeof(float), &value, sizeof(float));
  }
};

} // namespace tcr::runtime