// constant, load, store, add , mul, For alisa,, instructions,payload alias,
// payload, 6 constructors,move assignment, move constructor-noexcept,
// copy assignment, copy constructor-delete;desctructor declration, For
// definition full

#pragma once

#include "tcr/lir/ids.hpp"
#include "tcr/lir/index.hpp"
#include <cstddef>
#include <memory>
#include <variant>
#include <vector>
namespace tcr::lir {
using tcr::lir::BufferId;
using tcr::lir::IndexExpr;
using tcr::lir::IndexId;
using tcr::lir::ScalarId;

struct Constant {
  ScalarId dst;
  float value;
};

struct Load {
  ScalarId dst;
  BufferId buffer_src;
  IndexExpr offset;
};

struct Store {
  ScalarId value;
  BufferId buffer_dst;
  IndexExpr offset;
};

struct Add {
  ScalarId lhs;
  ScalarId rhs;
  ScalarId dst;
};

struct Mul {
  ScalarId lhs;
  ScalarId rhs;
  ScalarId dst;
};

struct IndexAdd {
  IndexExpr lhs;
  IndexExpr rhs;
  IndexId dst;
};

struct IndexMul {
  IndexExpr lhs;
  IndexExpr rhs;
  IndexId dst;
};

// only declaring , will define later to break cyclic dependency with
// instruction
struct For;

struct Instruction {
  using Payload = std::variant<Constant, Load, Store, Add, Mul, IndexAdd,
                               IndexMul, std::unique_ptr<For>>;
  Payload payload;
  Instruction(Add value);
  Instruction(Load value);
  Instruction(Store value);
  Instruction(Mul value);
  Instruction(Constant value);
  Instruction(IndexAdd value);
  Instruction(IndexMul value);
  Instruction(std::unique_ptr<For> value);

  // move assignment and move constructor, specifying that these constructors
  // should not throw exception outside so that the std::vector uses them while
  // moving the values from one memory to other
  Instruction(Instruction &&) noexcept = default;
  Instruction &operator=(Instruction &&) noexcept = default;

  // deleting the copy constructor and copy assignment, reason: instructor
  // contain the unique_ptr which should always have single ownership, so
  // compiler should not call copy constructor or assignment func anywhere
  Instruction(const Instruction &) = delete;
  Instruction &operator=(const Instruction &) = delete;

  ~Instruction();
};

struct For {
  IndexId index;
  IndexExpr begin;
  IndexExpr end;
  std::vector<Instruction> body;
};
} // namespace tcr::lir