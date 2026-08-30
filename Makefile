gen:
	@test -d build || cmake -B build
	cmake --build build

# Full tests
test:
	ctest --test-dir build --output-on-failure

# Escape hatch for a truly stale/corrupt cache
rebuild:
	rm -rf build && cmake -B build && cmake --build build

build:
	cmake --build build

lines:
	find src include -name '*.cpp' -o -name '*.h' -o -name '*.hpp' | xargs wc -l

clean:
	rm -rf build

cov:
	cd build && cmake --build . && ctest --quiet >/dev/null 2>&1 && \
	gcovr -r .. --filter '../src/' --gcov-ignore-errors=source_not_found 2>/dev/null
