# Subsetix Kokkos

> **Sparse Finite Volume Discretization on Complex 2D Geometries with GPU Acceleration**

Subsetix is a modern C++20 library for solving hyperbolic PDEs on sparse 2D domains using a novel **Compressed Sparse Row (CSR) of intervals** data structure. Built on Kokkos for portable performance across CPUs and GPUs.

---

## Features

- **Sparse-first design**: Memory-efficient representation of complex 2D geometries
- **GPU-native**: Zero-copy data structures via Kokkos — same code runs on CPU and GPU
- **Set algebra**: Compose geometries with boolean operations (union, intersection, difference)
- **AMR-ready**: Block-structured adaptive mesh refinement built into the core representation
- **High-level FVD API**: Solve Euler equations in ~10 lines of code
- **Header-only**: Easy integration with no build-time dependencies beyond Kokkos

---

## Quick Start

### Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| C++ compiler | C++20 compliant | GCC 12+ |
| CMake | 3.16 | 3.23+ (for presets) |
| Kokkos | 4.5.00 (auto-fetched) | — |
| CUDA (optional) | CUDA 12.x | — |

### Build

```bash
# Clone the repository
git clone https://github.com/sbstndb/subsetix_kokkos.git
cd subsetix_kokkos

# Configure (choose one preset)
cmake --preset serial          # CPU only
cmake --preset openmp          # CPU with OpenMP
cmake --preset cuda-gcc12      # NVIDIA GPU

# Build
cmake --build --preset serial

# Run tests
ctest --preset serial
```

### Available Presets

| Preset | Backend | Description |
|--------|---------|-------------|
| `serial` | Serial | Basic CPU execution |
| `openmp` | OpenMP | Multi-threaded CPU |
| `cuda-gcc12` | CUDA | NVIDIA GPU (requires GCC 12) |
| `serial-asan` | Serial + sanitizers | Debug mode with AddressSanitizer |

---

## Usage Examples

### Hello World: Advection in 10 Lines

```cpp
#include <subsetix/fvd/solver/adaptive_solver.hpp>
#include <subsetix/fvd/system/advection2d.hpp>
#include <subsetix/fvd/solver/solver_aliases.hpp>

using namespace subsetix::fvd;

// Define solver: 2nd order MUSCL + HLLC flux
using MySolver = EulerSolver2ndHLLC<float>;

int main() {
    // Build geometry: box with cylinder obstacle
    auto fluid = Geometry2D<float>::build_box(400, 160, 0.005f, 0.005f)
                     .add_cylinder(0.5f, 0.4f, 0.1f, /* inside= */ true)
                     .build();

    // Configure solver
    MySolver::Config cfg = MySolver::Config::from_cfl(0.45f);
    cfg.gamma = 1.4f;
    cfg.refine_fraction = 0.1f;

    // Create and initialize
    MySolver solver(fluid, {0, 400, 0, 160}, cfg);
    solver.initialize(Euler2D<float>::Primitive{1.0f, 2.0f, 0.0f, 1.0f/1.4f});

    // Time march
    while (solver.time() < 1.0f) {
        solver.step();
        solver.write_vtk("output_" + std::to_string(solver.step()) + ".vtk");
    }
}
```

### Low-Level CSR Operations

```cpp
#include <subsetix/geometry/csr_set_ops.hpp>
#include <subsetix/geometry/csr_generators.hpp>

using namespace subsetix::csr;

// Create geometries
auto box = make_box_device(Box2D{0, 100, 0, 100});
auto disk = make_disk_device(Disk2D{50, 50, 20});

// Compute difference (box with hole)
CsrSetAlgebraContext ctx;
auto result = allocate_interval_set_device(100, 10);
set_difference_device(box, disk, result, ctx);
compute_cell_offsets_device(result);

// Export to VTK
vtk_export_device(result, "geometry.vtk");
```

### More Examples

See the `examples/` directory for complete demos:

| Example | Description |
|---------|-------------|
| `fvd_simulation_examples.cpp` | 16+ usage patterns (AMR, BCs, checkpoint, multi-physics) |
| `fvd_mach2_cylinder_example.cpp` | Mach 2 flow over cylinder |
| `smoke_plume/smoke_plume.cpp` | Full buoyancy-driven flow with Poisson solver |
| `amr_advection/amr_advection_2d.cpp` | AMR on 8192×8192 sparse grid |

---

## Documentation

### Core Concepts

#### CSR of Intervals

Subsetix represents 2D domains as rows of X-intervals:

```cpp
template <class MemorySpace>
struct IntervalSet2D {
  Kokkos::View<RowKey2D*, MemorySpace>   row_keys;     // Y coordinates
  Kokkos::View<std::size_t*, MemorySpace> row_ptr;      // CSR pointers
  Kokkos::View<Interval*, MemorySpace>    intervals;    // [begin, end) X intervals
  std::size_t total_cells;                              // Sum of interval lengths
};
```

**Why this matters:**
- **O(n) memory** where n = active cells (not domain size)
- **GPU-friendly**: All data in `Kokkos::View`
- **AMR-native**: Refinement is simple interval arithmetic (×2 or ÷2)

#### Module Organization

```
include/subsetix/
├── geometry/        # CSR interval sets, set algebra
├── field/           # Fields on sparse geometries
├── csr_ops/         # Parallel kernels (stencil, transform, AMR)
├── multilevel/      # AMR hierarchies
├── io/              # VTK export
└── fvd/             # Finite Volume Discretization layer
    ├── geometry/    # Fluent geometry builder
    ├── system/      # PDE systems (Euler2D, Advection2D)
    ├── flux/        # Numerical fluxes (Rusanov, HLLC, Roe)
    ├── reconstruction/ # MUSCL with limiters
    ├── solver/      # Adaptive solver
    └── time/        # Runge-Kutta integrators
```

### Solver Configuration

#### Available Solvers (Type Aliases)

```cpp
// 1st order schemes
using EulerSolver1st = EulerSolver1stRusanov<float>;

// 2nd order schemes
using EulerSolver2ndHLLC = EulerSolver2ndHLLC<float>;
using EulerSolver2ndRoe  = EulerSolver2ndRoe<float>;

// High-order
using EulerSolverRK3 = EulerSolverRK3<float>;  // SSPRK3 + HLLC
```

#### Flux Schemes

| Scheme | Accuracy | Robustness | Use Case |
|--------|----------|------------|----------|
| Rusanov | 1st order | Very robust | Quick prototyping |
| HLLC | 2nd order | Robust | Shocks, contacts |
| Roe | 2nd order | Sensitive | Accurate smooth flows |

#### Reconstruction

```cpp
// No reconstruction (1st order)
reconstruction::NoReconstruction

// MUSCL with limiters (2nd order)
reconstruction::MUSCL_Reconstruction<
    reconstruction::MinmodLimiter   // Most diffusive
    // reconstruction::MCLimiter      // Moderate
    // reconstruction::VanLeerLimiter  // Less diffusive
    // reconstruction::SuperbeeLimiter // Least diffusive
>
```

### API Reference

#### Geometry Builder

```cpp
auto geom = Geometry2D<float>::build_box(nx, ny, dx, dy)
    .add_cylinder(cx, cy, radius, inside)
    .add_rectangle(x0, x1, y0, y1, inside)
    .add_bitmap("obstacle.pbm", inside)
    .build();
```

#### Solver Configuration

```cpp
MySolver::Config cfg;
cfg.cfl = 0.45f;              // CFL number
cfg.gamma = 1.4f;             // Heat capacity ratio
cfg.refine_fraction = 0.1f;   // AMR: fraction of cells to refine
cfg.coarsen_fraction = 0.05f; // AMR: fraction to coarsen
cfg.min_level = 0;            // AMR: minimum refinement level
cfg.max_level = 2;            // AMR: maximum refinement level
```

#### Boundary Conditions

```cpp
// Inflow/Outflow
auto bc = BoundaryConfigBuilder<Euler2D<float>>::inflow_outflow(inflow, gamma);
solver.set_boundary_conditions(bc);

// Time-dependent
solver.set_time_dependent_bc("left", [](double t) {
    return Primitive{1.0f, 0.5f * sin(t), 0.0f, 1.0f};
});
```

#### Source Terms

```cpp
// Gravity
solver.add_gravity(-9.81f);

// Custom source
solver.add_source([](const auto& U, const auto& q, Real x, Real y, Real t) {
    return Conserved{0, 0, 0, heating_rate(x, y, t)};
});
```

---

## Testing

```bash
# Run all tests
ctest --preset serial

# Run specific test suite
./build-serial/tests/subsetix_test_core
./build-serial/tests/subsetix_test_ops
./build-serial/tests/subsetix_test_advanced
./build-serial/tests/subsetix_test_amr
./build-serial/tests/subsetix_test_fvd_api
./build-serial/tests/subsetix_test_fvd_execution
./build-serial/tests/subsetix_test_fvd_integrators
```

---

## Performance Tips

### GPU Optimization

```cpp
// Enable CUDA graph for 10-100x speedup
// (reduces CPU-GPU synchronization)
MySolver solver(...);
solver.enable_cuda_graph(true);
```

### Memory Management

```cpp
// Reuse workspace for multiple operations
CsrSetAlgebraContext ctx;  // Allocate once
set_union_device(A, B, result, ctx);
set_intersection_device(C, D, result2, ctx);  // Reuses memory
```

---

## Contributing

See `AGENTS.md` for coding guidelines and `CLAUDE.md` for architecture details.

---

## License

[Your License Here]

---

## Links

- **Architecture Details**: See `CLAUDE.md`
- **Development Guidelines**: See `AGENTS.md`
- **3D Design Notes**: See `3D.md`
- **FVD Specification**: See `docs/fvd_layer_proposal_v3.1.md`
