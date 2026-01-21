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

Use CMake presets - see **CLAUDE.md** for complete reference including experimental-only presets.

Quick start:
```bash
cmake --preset serial && cmake --build --preset serial
ctest --preset serial
```

Prefer presets over direct `make` calls.

## Coding Style & Naming Conventions

- Language: C++20, Kokkos-first for parallel code (no raw CUDA/OpenMP loops).
- Indentation: 2 spaces, no tabs; follow existing header style.
- Namespaces: `subsetix::csr` for geometry/fields, `subsetix::vtk` for export.
- Types in `CamelCase`, free functions in `snake_case`.
- Avoid new third-party dependencies unless discussed.

## Testing Guidelines

- Tests use GoogleTest and live in `tests/`, organized into separate executables by domain: `subsetix_test_core`, `subsetix_test_ops`, `subsetix_test_advanced`, `subsetix_test_amr`, `subsetix_test_fvd_api`, `subsetix_test_fvd_execution`, `subsetix_test_fvd_integrators`.
- Keep tests fast and deterministic; they must pass on serial, OpenMP, and CUDA (use preset `cuda-gcc12`).
- Prefer focused `TEST()` cases over large monolithic tests; share common helpers in small headers or `.cpp` files.
- When adding device code, exercise it at least in the serial preset.
- For set‑algebra primitives (e.g. `set_union_device`), add both high‑level tests and focused tests for low‑level building blocks to simplify debugging.

## Experimental Module Guidelines

The `experimental/` directory provides an isolated research space for alternative algorithm implementations:

### Architecture

- **Isolation**: Completely separate from stable codebase
  - Separate namespace: `experimental::subsetix::csr` (not `subsetix::csr`)
  - Separate library target: `experimental::csr` (depends only on Kokkos)
  - Only built when `SUBSETIX_ENABLE_EXPERIMENTAL=ON` (default: OFF)

- **Versioned framework**: v1, v2, v3 for algorithm comparison
  - v1: Baseline algorithm (port of subsetix_kokkos_2)
  - v2, v3: Research slots for experimentation
  - Cross-version tests ensure all versions produce identical results

### Development Workflow

**IMPORTANT**: Always use dedicated experimental presets (see CLAUDE.md for commands). Manual configuration requires 4 flags and is error-prone.

Presets available: `experimental-serial`, `experimental-openmp`, `experimental-cuda-gcc12`, `experimental-asan`.

### Contribution Rules

- Experimental code has **no stability guarantees** - APIs may change without notice
- When adding new algorithm versions (v4, v5, etc.):
  1. Create new header in `experimental/include/experimental/subsetix/csr/set_algebra/vN.hpp`
  2. Add corresponding test in `experimental/tests/set_algebra/test_vN_unitary.cpp`
  3. Update `set_algebra.hpp` to include the new version
  4. Add benchmarks to `experimental/benchmarks/set_algebra/unified_comparison_benchmark.cpp`
- **Do not** import experimental code into stable modules
- To promote experimental code to stable: copy and adapt, do not move

## Commit & Pull Request Guidelines

- Commit messages: short, imperative, and scoped (e.g. `Add CSR fields`, `Fix cuda-clang preset`).
- Keep changes focused; avoid mixing build, API, and formatting changes in one commit.
- Do not commit large temporary artifacts or local build directories; prefer `.gitignore` updates.
- Document new public APIs briefly in comments and, if relevant, in examples or tests.

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
