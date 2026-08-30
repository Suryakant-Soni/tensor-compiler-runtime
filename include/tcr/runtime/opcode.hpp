#pragma once
// it will contain and enum Opcode class which would contain the operations
// which should be finally dispatched to VM and they will be stored as program
// instructions opcode per instruction

namespace tcr::runtime {
enum class Opcode {
  ADD,
  MUL,
  SUM,
  MATMUL,
};
}