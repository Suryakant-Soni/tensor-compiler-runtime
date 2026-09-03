.PHONY: gen build test rebuild reconfigure lines clean cov

# Configure (if needed) + build.
# If build/ already exists, `cmake --build` auto-reruns configure
# when CMakeLists.txt is newer than the cache — so an edited
# CMakeLists.txt is picked up here without a full wipe.
gen:
	@test -d build || cmake -B build -G Ninja
	cmake --build build

# Incremental build only
build:
	cmake --build build

# Force a reconfigure against the existing build dir.
# Use after editing CMakeLists.txt if you want to be explicit
# rather than relying on the auto-rerun above.
reconfigure:
	cmake -B build -G Ninja
	cmake --build build

# Full tests
test:
	ctest --test-dir build --output-on-failure

# Escape hatch for a truly stale/corrupt cache
rebuild:
	rm -rf build && cmake -B build -G Ninja && cmake --build build

lines:
	find src include -name '*.cpp' -o -name '*.h' -o -name '*.hpp' | xargs wc -l

clean:
	rm -rf build

cov:
	cd build && cmake --build . && ctest --quiet >/dev/null 2>&1 && \
	gcovr -r .. --filter '../src/' --gcov-ignore-errors=source_not_found 2>/dev/null