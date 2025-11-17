# Repository Guidelines

## Project Structure & Module Organization

- Root: `CMakeLists.txt`, `CMakePresets.json`, `AGENTS.md`.
- Public headers: `include/subsetix/` (CSR geometry, fields, VTK export).
- Library target: `subsetix_core` in `src/` (INTERFACE, links to Kokkos).
- Tests: `tests/` (standalone executables registered via CTest).
- Examples: `examples/` (VTK generation and usage demos).
- Benchmarks: `benchmarks/` (lightweight performance checks).
- Build trees: `build-*` directories created by CMake presets (do not hard‑code paths).

## Build, Test, and Development Commands

- Configure + build (serial):  
  - `cmake --preset serial`  
  - `cmake --build --preset serial`
- Configure + build (OpenMP): `cmake --build --preset openmp`
- Configure + build (CUDA with clang): `cmake --build --preset cuda-clang`
- Run tests via CTest: `ctest --preset <serial|openmp|cuda-clang|serial-asan>`
- Prefer presets (Ninja generator) over calling `make` directly.

## Coding Style & Naming Conventions

- Language: C++17, Kokkos-first for parallel code (no raw CUDA/OpenMP loops).
- Indentation: 2 spaces, no tabs; follow existing header style.
- Namespaces: `subsetix::csr` for geometry/fields, `subsetix::vtk` for export.
- Types in `CamelCase`, free functions in `snake_case`.
- Avoid new third-party dependencies unless discussed.

## Testing Guidelines

- New tests live in `tests/` and are added as executables in `tests/CMakeLists.txt`.
- Keep tests fast and deterministic; they must pass on serial and OpenMP.  
- Use simple `main(int, char**)` returning non‑zero on failure (no heavy test framework).
- When adding device code, exercise it at least in the serial preset.

## Commit & Pull Request Guidelines

- Commit messages: short, imperative, and scoped (e.g. `Add CSR fields`, `Fix cuda-clang preset`).
- Keep changes focused; avoid mixing build, API, and formatting changes in one commit.
- Do not commit large temporary artifacts or local build directories; prefer `.gitignore` updates.
- Document new public APIs briefly in comments and, if relevant, in examples or tests.

## Agent-Specific Instructions

- Respect this layout when adding new geometry/field features; reuse existing CSR types.
- Prefer adding example usage in `examples/` for new public capabilities.
- When modifying CMake or presets, preserve existing presets and options unless there is a clear reason to change them.

