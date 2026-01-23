<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique
-->
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build, Test, and Development Commands

### Build Configuration

The project uses CMake Presets with Ninja generator:

```bash
# Available presets
cmake --preset serial              # Serial backend
cmake --build --preset serial

cmake --preset openmp              # OpenMP + Serial
cmake --build --preset openmp

cmake --preset cuda                # CUDA + Serial
cmake --build --preset cuda

# Debug with sanitizers
cmake --preset serial-asan         # Address + UB sanitizer
cmake --build --preset serial-asan

# Playground-only builds (algorithm research)
cmake --preset playground-serial      # Serial, playground module only
cmake --build --preset playground-serial
cmake --preset playground-openmp      # OpenMP, playground module only
cmake --build --preset playground-openmp
cmake --preset playground-cuda        # CUDA, playground module only
cmake --build --preset playground-cuda
cmake --preset playground-asan        # Serial + sanitizers, playground only
cmake --build --preset playground-asan

# Profiling builds
cmake --preset playground-perf-serial       # Linux perf, playground only
cmake --build --preset playground-perf-serial
cmake --preset playground-perf-openmp       # Linux perf + OpenMP
cmake --build --preset playground-perf-openmp
cmake --preset playground-serial-profile    # Kokkos profiling tools
cmake --build --preset playground-serial-profile
cmake --preset profiling-nsight-cuda        # Nsight GPU profiling
cmake --build --preset profiling-nsight-cuda
cmake --preset profiling-nsight-cuda-release # Nsight with Release + symbols
cmake --build --preset profiling-nsight-cuda-release
```

### Machine-Specific Overrides

The default presets use the generic `g++` compiler (system default). To override compiler or CUDA settings for your machine:

1. **Copy the example template:**
   ```bash
   cp CMakeUserPresets.json.example CMakeUserPresets.json
   ```

2. **Edit `CMakeUserPresets.json`** with your local configuration:
   ```json
   {
     "configurePresets": [
       {
         "name": "cuda-gcc14",
         "inherits": "cuda",
         "cacheVariables": {
           "CMAKE_CXX_COMPILER": "g++-14"
         }
       }
     ]
   }
   ```

3. **Use your custom preset:**
   ```bash
   cmake --preset cuda-gcc14
   ```

**Note**: `CMakeUserPresets.json` is gitignored and never committed.

### GPU Architecture (CUDA)

**CRITICAL FOR AGENTS**: When running ANY CUDA preset (`cuda`, `playground-cuda`, `*-cuda-*`, `profiling-nsight-cuda`):
1. **FIRST** detect the GPU with `nvidia-smi -L`
2. **THEN** add the appropriate `-DKokkos_ARCH_<NAME>=ON` flag to the cmake command

```bash
# Step 1: Detect GPU
nvidia-smi -L

# Step 2: Use the detected architecture
cmake --preset cuda -DKokkos_ARCH_ADA89=ON  # Ada (RTX 40xx, RTX 1000 Ada)
cmake --preset cuda -DKokkos_ARCH_AMPERE86=ON  # Ampere (RTX 30xx, A100)
cmake --preset cuda -DKokkos_ARCH_HOPPER90=ON  # Hopper (H100)
```

**NEVER run a CUDA preset without the architecture flag - it will fail.**

## CMake Options Reference

All available CMake options and their default values:

### Backend Selection
| Option | Default | Description |
|--------|---------|-------------|
| `SUBSETIX_KOKKOS_OPENMP` | `OFF` | Enable Kokkos OpenMP backend |
| `SUBSETIX_KOKKOS_CUDA` | `OFF` | Enable Kokkos CUDA backend |
| `SUBSETIX_USE_MPI` | `OFF` | Enable MPI support for FVD layer |

### Execution Space Overrides
| Option | Default | Description |
|--------|---------|-------------|
| `SUBSETIX_EXECSPACE_FORCE_CUDA` | `OFF` | Force ExecSpace = Kokkos::Cuda |
| `SUBSETIX_EXECSPACE_FORCE_OPENMP` | `OFF` | Force ExecSpace = Kokkos::OpenMP |
| `SUBSETIX_EXECSPACE_FORCE_SERIAL` | `OFF` | Force ExecSpace = Kokkos::Serial |

**Important**: Only one of `SUBSETIX_EXECSPACE_FORCE_*` can be `ON` at a time (CMake will error if multiple are set).

### Memory Space Overrides
| Option | Default | Description |
|--------|---------|-------------|
| `SUBSETIX_MEMORYSPACE_FORCE_UVM` | `OFF` | Force DeviceMemorySpace = Kokkos::CudaUVMSpace |
| `SUBSETIX_MEMORYSPACE_FORCE_HOSTPINNED` | `OFF` | Force DeviceMemorySpace = Kokkos::HostPinnedSpace |

**Important**: Only one of `SUBSETIX_MEMORYSPACE_FORCE_*` can be `ON` at a time.

### Playground Module
| Option | Default | Description |
|--------|---------|-------------|
| `SUBSETIX_ENABLE_PLAYGROUND` | `OFF` | Enable playground set algebra algorithms |
| `SUBSETIX_BUILD_STABLE_LIBS` | `ON` | Build stable (non-playground) libraries |
| `SUBSETIX_BUILD_STABLE_TESTS` | `ON` | Build stable (non-playground) tests |
| `SUBSETIX_BUILD_STABLE_BENCHMARKS` | `ON` | Build stable (non-playground) benchmarks |

**Critical**: When `SUBSETIX_ENABLE_PLAYGROUND=ON`, you typically want to disable all stable components:
```bash
# Wrong - will cause linking errors
cmake --preset serial -DSUBSETIX_ENABLE_PLAYGROUND=ON

# Correct - disables stable components
cmake --preset serial \
  -DSUBSETIX_ENABLE_PLAYGROUND=ON \
  -DSUBSETIX_BUILD_STABLE_LIBS=OFF \
  -DSUBSETIX_BUILD_STABLE_TESTS=OFF \
  -DSUBSETIX_BUILD_STABLE_BENCHMARKS=OFF

# Best - use dedicated preset (sets all flags automatically)
cmake --preset playground-serial
```

### Code Coverage
| Option | Default | Description |
|--------|---------|-------------|
| `SUBSETIX_ENABLE_COVERAGE` | `OFF` | Enable code coverage analysis (GCC only, forces Debug build) |

### Profiling
| Option | Default | Description |
|--------|---------|-------------|
| `SUBSETIX_ENABLE_PROFILING_PERF` | `OFF` | Enable Linux perf profiling support (adds debug symbols) |
| `SUBSETIX_ENABLE_PROFILING_CUDA` | `OFF` | Enable CUDA profiling support (Nsight ncu/nsys, adds debug symbols) |
| `SUBSETIX_ENABLE_PROFILING_KOKKOS` | `OFF` | Enable Kokkos profiling tools support (fetches and builds kokkos-tools) |

**Important**:
- Only one profiling option can be `ON` at a time
- All profiling options add debug symbols (`-g`) automatically
- `SUBSETIX_ENABLE_PROFILING_KOKKOS` downloads and builds kokkos-tools via FetchContent
- See **PROFILING.md** for comprehensive profiling guide

**Profiling presets:**
| Preset | Tool | Backend |
|--------|------|---------|
| `playground-perf-serial` | Linux perf | Serial |
| `playground-perf-openmp` | Linux perf | OpenMP |
| `playground-serial-profile` | Kokkos tools | Serial |
| `playground-openmp-profile` | Kokkos tools | OpenMP |
| `playground-cuda-profile` | Kokkos tools | CUDA |
| `profiling-nsight-cuda` | Nsight | CUDA |
| `profiling-nsight-cuda-release` | Nsight (Release) | CUDA |

### Running Tests

Tests are organized into 7 separate executables (to avoid ODR/CUDA linking issues):

```bash
# Run all tests
ctest --preset serial
ctest --preset openmp
ctest --preset cuda

# Run specific test executable
./build-serial/tests/subsetix_test_core          # Builders, basic data structures, VTK
./build-serial/tests/subsetix_test_ops           # Set algebra (union, intersection, difference)
./build-serial/tests/subsetix_test_advanced      # Field ops, morphology, threshold
./build-serial/tests/subsetix_test_amr           # Refine/project, remap, multilevel
./build-serial/tests/subsetix_test_fvd_api       # FVD compilation tests
./build-serial/tests/subsetix_test_fvd_execution # FVD numerical validation
./build-serial/tests/subsetix_test_fvd_integrators # Time integrators, AMR
```

### Running Examples

```bash
# From build directory
./build-serial/examples/<example_name>

# Examples accept --output-dir <path> to override default output location
# Most generate .vtk files in examples_output/<example_name>/
```

## Architecture Overview

### Core Concept: CSR Representation

The library implements a novel **Compressed Sparse Row of Intervals** data structure for 2D sparse domains:

```cpp
template <class MemorySpace>
struct IntervalSet2D {
  Kokkos::View<RowKey2D*, MemorySpace> row_keys;     // Y coordinates per row
  Kokkos::View<std::size_t*, MemorySpace> row_ptr;    // CSR row pointers
  Kokkos::View<Interval*, MemorySpace> intervals;     // [begin, end) X intervals
  Kokkos::View<std::size_t*, MemorySpace> cell_offsets; // Cell offset per interval
  std::size_t total_cells;   // Sum of all interval lengths
};
```

This representation enables:
- Memory-efficient storage of irregular 2D geometries
- Fast set algebra operations (union, intersection, difference)
- GPU-compatible via Kokkos::View

### Module Organization

```
include/subsetix/
├── geometry/        # CSR interval set representation (IntervalSet2D, mapping)
├── field/           # Fields on CSR geometries (Field2D<T>)
├── csr_ops/         # Parallel kernels (set_algebra, transform, threshold, field_*)
├── detail/          # Implementation details (scan_utils, memory_utils)
├── multilevel/      # AMR support (MultilevelGeo, MultilevelField)
├── io/              # VTK export (vtk_export.hpp)
└── fvd/             # Finite Volume Discretization layer
    ├── geometry/    # Geometry2D fluent API
    ├── system/      # PDE systems (Euler2D)
    ├── flux/        # Numerical fluxes (Rusanov, HLLC, Roe)
    ├── reconstruction/ # MUSCL, limiters
    ├── solver/      # AdaptiveSolver, solver_aliases
    ├── time/        # RK1, RK2, RK3 integrators
    └── mpi/         # MPI support (optional, stub when disabled)

playground/          # Algorithm research playground (disabled by default)
├── intersection/               # Intersection algorithms playground
│   ├── include/playground/subsetix/csr/intersection/
│   ├── tests/intersection/
│   └── benchmarks/intersection/
├── modal/                      # Shared GPU CI scripts
└── profiling_patches/          # Shared Kokkos tools fixes
```

#### Playground Module

The `playground/` directory provides alternative implementations of set algebra algorithms for research and comparison. It is **completely isolated** from the stable codebase and disabled by default.

**Quick setup with dedicated presets:**
```bash
cmake --preset playground-serial      # Serial backend
cmake --preset playground-openmp      # OpenMP backend
cmake --preset playground-cuda        # CUDA backend
cmake --preset playground-asan        # Serial + sanitizers
```

**Manual setup (not recommended - use presets instead):**
```bash
cmake --preset serial \
  -DSUBSETIX_ENABLE_PLAYGROUND=ON \
  -DSUBSETIX_BUILD_STABLE_LIBS=OFF \
  -DSUBSETIX_BUILD_STABLE_TESTS=OFF \
  -DSUBSETIX_BUILD_STABLE_BENCHMARKS=OFF
```

### Running Playground Tests

Playground tests are separate executables from stable tests:

```bash
# After enabling playground module and building
ctest --preset playground-serial  # Runs all playground tests
ctest --preset playground-openmp  # Runs with OpenMP backend
ctest --preset playground-cuda    # Runs with CUDA backend

# Run specific playground test executables (serial preset)
./build-playground-serial/playground/intersection/tests/playground_intersection_v1_unitary_test
./build-playground-serial/playground/intersection/tests/playground_intersection_v2_unitary_test
./build-playground-serial/playground/intersection/tests/playground_intersection_v3_unitary_test
./build-playground-serial/playground/intersection/tests/playground_intersection_cross_version_test  # Verifies v1/v2/v3 produce identical results
./build-playground-serial/playground/intersection/tests/playground_intersection_overlap_patterns_test
./build-playground-serial/playground/intersection/tests/playground_intersection_large_mesh_test
./build-playground-serial/playground/intersection/tests/playground_intersection_sorted_rows_test
```

### Running Playground Benchmarks

```bash
# Run all playground benchmarks (serial preset)
./build-playground-serial/playground/intersection/benchmarks/playground_intersection_comparison_benchmark

# Run specific size configurations
./build-playground-serial/playground/intersection/benchmarks/playground_intersection_comparison_benchmark --benchmark_filter=SmallConfig
./build-playground-serial/playground/intersection/benchmarks/playground_intersection_comparison_benchmark --benchmark_filter=MediumConfig
./build-playground-serial/playground/intersection/benchmarks/playground_intersection_comparison_benchmark --benchmark_filter=LargeConfig

# Run only 2D benchmarks
./build-playground-serial/playground/intersection/benchmarks/playground_intersection_comparison_benchmark --benchmark_filter="2D"

# Run only 3D benchmarks
./build-playground-serial/playground/intersection/benchmarks/playground_intersection_comparison_benchmark --benchmark_filter="3D"
```

### Execution Space Configuration

Execution/memory spaces are compile-time configurable via CMake defines:

```cpp
namespace subsetix::csr {
#ifdef SUBSETIX_EXECSPACE_CUDA
  using ExecSpace = Kokkos::Cuda;
#elif defined(SUBSETIX_EXECSPACE_OPENMP)
  using ExecSpace = Kokkos::OpenMP;
#else
  using ExecSpace = Kokkos::Serial;
#endif

#ifdef SUBSETIX_MEMORYSPACE_FORCE_UVM
  using DeviceMemorySpace = Kokkos::CudaUVMSpace;
#else
  using DeviceMemorySpace = typename ExecSpace::memory_space;
#endif
}
```

### Host/Device Synchronization

Explicit conversions are required:

```cpp
// Device → Host
auto host_geom = to<HostMemorySpace>(device_geom);
auto host_field = build_host_field_from_device(device_field);

// Host → Device
auto dev_geom = to<DeviceMemorySpace>(host_geom);
auto dev_field = build_device_field_from_host(host_field);
```

### Set Algebra Pattern

All set operations follow this pattern:

```cpp
CsrSetAlgebraContext ctx;  // Reusable workspace (memory pool)
auto result = allocate_interval_set_device(rows, intervals);
set_union_device(A, B, result, ctx);  // or set_intersection, set_difference
compute_cell_offsets_device(result);  // Always call after operations
```

## FVD (Finite Volume Discretization) Layer

The FVD layer provides a high-level API for solving hyperbolic PDEs on sparse geometries.

### System Concept

A PDE system must satisfy the `FiniteVolumeSystem` concept:

```cpp
template<typename System>
concept FiniteVolumeSystem = requires {
  typename System::RealType;
  typename System::Conserved;
  typename System::Primitive;
  typename System::Views;
  { System::to_primitive(...) } -> std::same_as<typename System::Primitive>;
  { System::from_primitive(...) } -> std::same_as<typename System::Conserved>;
  { System::flux_phys_x(...) } -> std::same_as<typename System::Conserved>;
  { System::sound_speed(...) } -> std::convertible_to<typename System::RealType>;
};
```

Example: `Euler2D<float>` in `include/subsetix/fvd/system/euler2d.hpp`

### Solver Design

Policy-based design (no runtime polymorphism for GPU compatibility):

```cpp
template<
  FiniteVolumeSystem System,
  typename Reconstruction,           // NoReconstruction or MUSCL_Reconstruction<Limiter>
  template<typename> class FluxScheme // RusanovFlux, HLLCFlux, or RoeFlux
>
class AdaptiveSolver { /* ... */ };
```

Convenience aliases (defined in `solver_aliases.hpp`):

```cpp
template<typename Real = float>
using EulerSolver2ndHLLC = AdaptiveSolver<
  Euler2D<Real>,
  reconstruction::MUSCL_Reconstruction<reconstruction::MinmodLimiter>,
  flux::HLLCFlux
>;

// Usage
EulerSolver2ndHLLC<> solver(fluid, domain, cfg);
solver.step(dt);
```

### Geometry Builder

Fluent API for constructing complex geometries:

```cpp
auto fluid_geom = Geometry2D<float>::build_box(nx, ny, dx, dy)
                      .add_cylinder(cx, cy, radius, /* inside= */ true)
                      .add_box(x0, x1, y0, y1, /* inside= */ false)
                      .build();
```

## GPU Compatibility Rules

**Critical for device code:**

- **NO runtime polymorphism** (virtual functions) in device code - use templates instead
- All kernel lambdas must be marked with `KOKKOS_INLINE_FUNCTION`
- Avoid `std::function`, `std::map`, `std::string` on device
- Use fixed-size arrays or compile-time strings for configuration
- Fence after parallel operations if host needs results: `ExecSpace().fence();`

## Library Targets

CMake INTERFACE targets (link via `target_link_libraries`):

- `subsetix::geometry` - CSR geometry primitives
- `subsetix::field` - Field operations
- `subsetix::multilevel` - AMR multilevel support
- `subsetix::vtk` - VTK export
- `subsetix::core` - Aggregate target (includes all above)

## Additional Guidelines

### Copyright & License Headers

All source files (`.hpp`, `.cpp`) must include the following header:

```cpp
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique
```

This SPDX identifier replaces the full Apache-2.0 license text; see `LICENSE` in the root directory for complete terms.

### Other Documentation

See **AGENTS.md** for:
- Coding style (C++20, 2-space indentation, naming conventions)
- Testing guidelines
- Commit message conventions
- Communication rules (chat in French, code/comments/commits in English)

## Key Documentation Files

- `/AGENTS.md` - Project guidelines (READ THIS FIRST)
- `/PROFILING.md` - Profiling guide (Linux perf, Nsight, Kokkos tools)
- `/3D.md` - Design notes for 3D extension
- `/FIELD_UPGRADE_SUMMARY.md` - Field system evolution
- `/docs/fvd_layer_proposal_v3.1.md` - Detailed FVD design specification
- `/docs/MODAL.md` - Modal GPU CI guide
- `/docs/PERF_PROFILING.md` - Detailed Linux perf profiling guide
