cmake_minimum_required(VERSION 3.18)
project(tensor-compiler-runtime LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options(--coverage -O0 -g)
add_link_options(--coverage)

# ------------------------------------------------------------
# Catch2
# ------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.8.1
)

FetchContent_MakeAvailable(Catch2)

# ------------------------------------------------------------
# Main TCR library
# ------------------------------------------------------------
add_library(tcr STATIC
    src/ir/shape.cpp
    src/ir/dtype.cpp
    src/ir/graph.cpp
    src/ir/op.cpp
    src/ir/type_inference.cpp

    src/reference/tensor.cpp
    src/reference/bindings.cpp
    src/reference/interpreter.cpp
    src/reference/ops.cpp

    src/compiler/compiler.cpp
    src/compiler/lowering.cpp
    src/compiler/program_emitter.cpp
    src/compiler/elementwise_fusion.cpp
    src/compiler/scheduler.cpp
    src/compiler/memory_planner.cpp

    src/runtime/tensor.cpp
    src/runtime/bindings.cpp
    src/runtime/program.cpp
    src/runtime/vm.cpp

    src/lir/instruction.cpp
    src/lir/program.cpp
)

# add headers to the main code lib
target_include_directories(tcr PUBLIC
        ${PROJECT_SOURCE_DIR}/include)       

# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------
add_executable(tcr_tests tests/ir/test_types.cpp
                         tests/ir/test_graph.cpp
                         tests/ir/test_shape.cpp
                         tests/ir/test_type_inference.cpp

                         tests/reference/test_interpreter.cpp
                         tests/reference/test_ops.cpp
                         tests/reference/test_bindings.cpp
                         tests/reference/test_tensor.cpp

                         tests/runtime/test_bindings.cpp
                         tests/runtime/test_program.cpp
                         tests/runtime/test_tensor.cpp
                         tests/runtime/test_vm.cpp

                         tests/compiler/test_compiler.cpp
                         tests/compiler/test_lowering.cpp
                         tests/compiler/test_program_emitter.cpp
                         tests/compiler/test_elementwise_fusion.cpp
                         tests/compiler/test_scheduler.cpp

                         tests/lir/test_program.cpp

                         tests/integration/test_compiler_vm_correctness.cpp
                         )

target_link_libraries(tcr_tests
PRIVATE 
tcr
Catch2::Catch2WithMain)

# ------------------------------------------------------------
# Ctest integration
# ------------------------------------------------------------
# turn on Ctest for this project
enable_testing()
# load Catch2 Cmake helpers
include(Catch)
#Register all TEST_CASEs from tct_tests with Ctest(Cmake's test runner)
catch_discover_tests(tcr_tests)