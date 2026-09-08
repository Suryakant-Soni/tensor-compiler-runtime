// structure to buffer, input, create buffer, scalar,index, add new
// input,output, instructions what are the program's buffers(), inputs(),
// outputs(), intructions(), scalar_count() and index_count and all private
// members 6
#pragma once
#include "tcr/ir/dtype.hpp"
#include "tcr/ir/ids.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/lir/ids.hpp"
#include "tcr/lir/instruction.hpp"
#include <cstddef>
#include <vector>

namespace tcr::lir {

using tcr::ir::Dtype;
using tcr::ir::Shape;
using tcr::ir::ValueId;

struct Buffer {
  BufferId id;
  Shape shape;
  Dtype dtype;
};

struct Input {
  ValueId value;
  BufferId buffer;
};

class Program {
public:
  BufferId create_buffer(const Shape &shape, Dtype dtype);
  ScalarId create_scalar();
  IndexId create_index();
  void add_input(ValueId value, BufferId buffer);
  void add_output(BufferId buffer);
  void add_instruction(Instruction instruction);

  const std::vector<Buffer> &buffers() const;
  const std::vector<Input> &inputs() const;
  const std::vector<BufferId> &outputs() const;
  const std::vector<Instruction> &instructions() const;

  std::size_t scalar_count() const;
  std::size_t index_count() const;

private:
  std::vector<Buffer> buffer_;
  std::vector<Input> inputs_;
  std::vector<BufferId> outputs_;
  std::vector<Instruction> instructions_;

  std::size_t next_scalar_;
  std::size_t next_index_;
};
} // namespace tcr::lir