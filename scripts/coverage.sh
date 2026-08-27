#!/bin/bash
set -e
cd build
cmake --build .
ctest
gcovr -r .. --filter '../src/' --gcov-ignore-errors=source_not_found 2>/dev/null