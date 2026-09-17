#include "tcr/lir/program.hpp"
#include "tcr/ir/dtype.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/lir/ids.hpp"
#include "tcr/lir/instruction.hpp"
#include <cstddef>
#include <utility>
#include <vector>
namespace tcr::lir {
using ir::Dtype;
using ir::Shape;

BufferId Program::create_buffer(const Shape &shape, Dtype dtype) {
  const BufferId id = BufferId{buffer_.size()};
  buffer_.push_back(Buffer{
      id,
      shape,
      dtype,
  });
  return id;
}

ScalarId Program::create_scalar() { return ScalarId{next_scalar_++}; }

IndexId Program::create_index() { return IndexId{next_index_++}; }

void Program::add_input(ValueId value, BufferId buffer) {
  inputs_.push_back(Input{value, buffer});
}

void Program::add_output(BufferId buffer) { outputs_.push_back(buffer); }

void Program::add_instruction(Instruction inst) {
  instructions_.push_back(std::move(inst));
}

const std::vector<Buffer> &Program::buffers() const { return buffer_; }

const std::vector<Input> &Program::inputs() const { return inputs_; }

const std::vector<BufferId> &Program::outputs() const { return outputs_; }

const std::vector<Instruction> &Program::instructions() const {
  return instructions_;
}

std::size_t Program::scalar_count() const { return next_scalar_; }
std::size_t Program::buffer_count() const { return buffer_.size(); }
std::size_t Program::index_count() const { return next_index_; }

} // namespace tcr::lir