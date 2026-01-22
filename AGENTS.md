# Repository Guidelines

## Project Structure & Module Organization

- Root: `CMakeLists.txt`, `CMakePresets.json`, `AGENTS.md`.
- Public headers: `include/subsetix/` (modular folders: `geometry/`, `field/`, `io/`, `multilevel/`, `fvd/`)
- Implementation: `include/subsetix/csr_ops/` and `include/subsetix/detail/`
- Tests: `tests/` (standalone executables via CTest)
- Examples: `examples/` (VTK generation and usage demos)
- Benchmarks: `benchmarks/` (lightweight performance checks)
- Experimental: `experimental/` (alternative algorithms, disabled by default)
- Build trees: `build-*` directories from CMake presets (do not hard‑code paths)

## Build, Test, and Development Commands

Use CMake presets - see **CLAUDE.md** for complete reference.

### Common Presets

| Category | Preset | Backend | Build dir |
|----------|--------|---------|-----------|
| **Stable builds** | `serial` | Serial CPU | `build-serial` |
| | `openmp` | Multi-threaded CPU | `build-openmp` |
| | `cuda-gcc12` | NVIDIA GPU | `build-cuda-gcc12` |
| | `serial-asan` | Serial + sanitizers | `build-serial-asan` |
| **Experimental-only** | `experimental-serial` | Serial | `build-experimental-serial` |
| | `experimental-openmp` | OpenMP | `build-experimental-openmp` |
| | `experimental-cuda-gcc12` | CUDA | `build-experimental-cuda-gcc12` |
| | `experimental-asan` | Serial + sanitizers | `build-experimental-asan` |
| **Profiling** | `profiling-cuda-gcc12` | CUDA + debug symbols | `build-profiling-cuda-gcc12` |

### Quick Start

```bash
# Configure + build
cmake --preset serial && cmake --build --preset serial

# Run tests
ctest --preset serial

# Run specific test
./build-serial/tests/subsetix_test_core
```

Prefer presets over direct `make` calls.

## Coding Style & Naming Conventions

- Language: C++20, Kokkos-first for parallel code (no raw CUDA/OpenMP loops).
- Indentation: 2 spaces, no tabs; follow existing header style.
- Namespaces: `subsetix::csr` for geometry/fields, `subsetix::vtk` for export.
- Types in `CamelCase`, free functions in `snake_case`.
- Avoid new third-party dependencies unless discussed.

## Testing Guidelines

### Stable Tests

- Tests use GoogleTest and live in `tests/`, organized into separate executables by domain: `subsetix_test_core`, `subsetix_test_ops`, `subsetix_test_advanced`, `subsetix_test_amr`, `subsetix_test_fvd_api`, `subsetix_test_fvd_execution`, `subsetix_test_fvd_integrators`.
- Keep tests fast and deterministic; they must pass on serial, OpenMP, and CUDA (use preset `cuda-gcc12`).
- Prefer focused `TEST()` cases over large monolithic tests; share common helpers in small headers or `.cpp` files.
- When adding device code, exercise it at least in the serial preset.
- For set‑algebra primitives (e.g. `set_union_device`), add both high‑level tests and focused tests for low‑level building blocks to simplify debugging.

### Running Tests

```bash
# All tests (stable)
ctest --preset serial
ctest --preset openmp
ctest --preset cuda-gcc12

# Specific test executable
./build-serial/tests/subsetix_test_core

# Experimental tests
ctest --preset experimental-serial
./build-experimental-serial/experimental/tests/experimental_v1_unitary_test
```

## Experimental Module Guidelines

The `experimental/` directory is a **playground** for algorithm research and experimentation:

- **No stability guarantees**: APIs may change without notice
- **Isolated**: Completely separate from stable codebase (separate namespace: `experimental::subsetix::csr`)
- **Tests must pass**: Even in the playground, all tests are expected to pass
- **Versioned framework**: v1 (baseline), v2/v3 (research slots)

### Development Workflow

Always use dedicated experimental presets. Manual configuration requires 4 flags and is error-prone.

```bash
cmake --preset experimental-serial && cmake --build --preset experimental-serial
ctest --preset experimental-serial
```

### Contribution Rules

- Experimental code has **no stability guarantees** - APIs may change without notice
- When adding new algorithm versions (v4, v5, etc.):
  1. Create new header in `experimental/include/experimental/subsetix/csr/set_algebra/vN.hpp`
  2. Add corresponding test in `experimental/tests/set_algebra/test_vN_unitary.cpp`
  3. Update `set_algebra.hpp` to include the new version
  4. Add benchmarks to `experimental/benchmarks/set_algebra/unified_comparison_benchmark.cpp`
- **Do not** import experimental code into stable modules
- To promote experimental code to stable: copy and adapt, do not move

## Development Best Practices

### Before Committing

**Always build and test completely** before committing:

1. **Identify your scope**: Are you working in `stable/` or `experimental/`?
2. **Full build for scope**:
   - **Stable changes**: Build and test on `serial`, `openmp`, `cuda-gcc12`
   - **Experimental changes**: Build and test on `experimental-serial`, `experimental-openmp`, `experimental-cuda-gcc12`
3. **All tests must pass**: No exceptions, even in the playground

```bash
# Example: Working in stable/
for preset in serial openmp cuda-gcc12; do
  cmake --preset $preset || exit 1
  cmake --build --preset $preset || exit 1
  ctest --preset $preset --output-on-failure || exit 1
done

# Example: Working in experimental/
for preset in experimental-serial experimental-openmp experimental-cuda-gcc12; do
  cmake --preset $preset || exit 1
  cmake --build --preset $preset || exit 1
  ctest --preset $preset --output-on-failure || exit 1
done
```

### Why This Matters

- **Cross-platform compatibility**: Code that works on Serial may fail on CUDA
- **Portability**: OpenMP threading can expose race conditions not visible in Serial
- **Confidence**: Full testing prevents regressions across all execution spaces

## Commit & Pull Request Guidelines

- Commit messages: short, imperative, and scoped (e.g. `Add CSR fields`, `Fix cuda-clang preset`).
- Keep changes focused; avoid mixing build, API, and formatting changes in one commit.
- Do not commit large temporary artifacts or local build directories; prefer `.gitignore` updates.
- Document new public APIs briefly in comments and, if relevant, in examples or tests.

## Profiling Guidelines

For performance-critical work, especially in the experimental module, profiling is essential:

- **Use Nsight Systems (`nsys`)** for modern GPUs (compute capability 8.0+)
- **Use nvprof** for legacy GPUs (compute capability < 8.0)
- Build with profiling presets: `cmake --preset profiling-cuda-gcc12`
- See **PROFILING.md** for comprehensive profiling guide, scripts, and best practices

Quick profiling example:
```bash
# Quick profiling on 3D SmallConfig
./scripts/profiling/run_nsys_quick.sh

# Profile specific benchmark
./scripts/profiling/run_nsys.sh --benchmark "3D_LargeConfig"
```

## CMake Options and Build Configuration

When modifying CMake configuration or adding new options, refer to **CLAUDE.md** for the complete CMake options reference. Key points:

- **Use presets** instead of manual flags when possible (e.g., `cmake --preset experimental-serial`)
- **Experimental mode** requires disabling stable components to avoid linking errors
- **Execution/Memory space** flags are mutually exclusive (CMake will error if multiple are set)
- See `CLAUDE.md` → "CMake Options Reference" for full documentation

## Agent-Specific Instructions

- Respect this layout when adding new geometry/field features; reuse existing CSR types.
- Prefer adding example usage in `examples/` for new public capabilities.
- When modifying CMake or presets, preserve existing presets and options unless there is a clear reason to change them.

## Communication Guidelines

- Always respond in chat in French.
- Always edit code and comments in English.
- Keep git commit messages concise and in English.
