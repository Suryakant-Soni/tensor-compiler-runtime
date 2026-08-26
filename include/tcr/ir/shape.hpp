#pragma once
#include <cstddef>
#include <initializer_list>
#include <vector>
namespace tcr::ir {

// declaring a class for shape management
class Shape {
public:
  // public attributes declared
  // constructor to create a shape object
  Shape(std::initializer_list<std::size_t> dims);

  // find rank
  std::size_t rank() const;

/**
  @brief Returns the size of the dimension at the input axis

  @param axis Zero-based index for dimension queried

  @return Size along the given dimension

  @note axis must be less than rank(); currently unchecked in release builds
*/
  std::size_t dim(std::size_t axis) const;
  // find the no. of elements in tensor
  std::size_t num_elements() const;

  bool operator==(const Shape &) const = default;

private:
  std::vector<std::size_t> dims_;
};

} // namespace tcr::ir