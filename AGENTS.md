# Repository Guidelines

## Project Structure & Module Organization

- Root: `CMakeLists.txt`, `CMakePresets.json`, `AGENTS.md`.
- Public headers now live in modular folders: `include/subsetix/geometry/`, `field/`, `io/`, `multilevel/` (legacy root `csr_*.hpp` or `multilevel.hpp` stubs are removed).
- Implementation headers: `include/subsetix/csr_ops/` (Parallel kernels) and `include/subsetix/detail/`.
- Library targets: INTERFACE `subsetix::geometry`, `subsetix::field`, `subsetix::multilevel`, `subsetix::vtk`, and aggregate `subsetix::core` (legacy alias `subsetix_core` remains).
- Tests: `tests/` (standalone executables registered via CTest).
- Examples: `examples/` (VTK generation and usage demos).
- Benchmarks: `benchmarks/` (lightweight performance checks).
- Build trees: `build-*` directories created by CMake presets (do not hard‑code paths).

## Build, Test, and Development Commands

- Configure + build (serial):  
  - `cmake --preset serial`  
  - `cmake --build --preset serial`
- Configure + build (OpenMP): `cmake --build --preset openmp`
- Configure + build (CUDA with GCC 12):  
  - `cmake --preset cuda-gcc12`  
  - `cmake --build --preset cuda-gcc12`
- Run tests via CTest: `ctest --preset <serial|openmp|cuda-gcc12|serial-asan>`
- Prefer presets (Ninja generator) over calling `make` directly.

## Coding Style & Naming Conventions

- Language: C++17, Kokkos-first for parallel code (no raw CUDA/OpenMP loops).
- Indentation: 2 spaces, no tabs; follow existing header style.
- Namespaces: `subsetix::csr` for geometry/fields, `subsetix::vtk` for export.
- Types in `CamelCase`, free functions in `snake_case`.
- Avoid new third-party dependencies unless discussed.

## Testing Guidelines

- Tests use GoogleTest and live in `tests/`, all compiled into the `subsetix_tests` executable.
- Keep tests fast and deterministic; they must pass on serial, OpenMP, and CUDA (use preset `cuda-gcc12`).
- Prefer focused `TEST()` cases over large monolithic tests; share common helpers in small headers or `.cpp` files.
- When adding device code, exercise it at least in the serial preset.
- For set‑algebra primitives (e.g. `set_union_device`), add both high‑level tests and focused tests for low‑level building blocks to simplify debugging.

## Commit & Pull Request Guidelines

- Commit messages: short, imperative, and scoped (e.g. `Add CSR fields`, `Fix cuda-clang preset`).
- Keep changes focused; avoid mixing build, API, and formatting changes in one commit.
- Do not commit large temporary artifacts or local build directories; prefer `.gitignore` updates.
- Document new public APIs briefly in comments and, if relevant, in examples or tests.

## Agent-Specific Instructions

- Respect this layout when adding new geometry/field features; reuse existing CSR types.
- Prefer adding example usage in `examples/` for new public capabilities.
- When modifying CMake or presets, preserve existing presets and options unless there is a clear reason to change them.

## Communication Guidelines

- Always respond in chat in French.
- Always edit code and comments in English.
- Keep git commit messages concise and in English.
