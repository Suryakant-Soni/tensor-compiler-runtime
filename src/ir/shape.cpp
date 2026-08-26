// implementing shape class


#include "tcr/ir/shape.hpp"
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <vector>

namespace tcr::ir {

  // construct the Shape-class member variable dims_ directly with contructor parameter dims 
  // member initializer design
Shape::Shape(std::initializer_list<std::size_t> dims) : dims_(dims) {}

std::size_t Shape::rank() const {
  return dims_.size();
}

std::size_t Shape::dim(std::size_t axis) const {
  assert(axis < dims_.size() && "input axis is out of bounds");
  return dims_[axis];
}

std::size_t Shape::num_elements() const {
    std::size_t total = 1;
    for (std::size_t d : dims_) {
      total *= d;
    }
  return total; 
}

}