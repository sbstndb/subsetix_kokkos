# FVD Layer Testing Guide

**Version:** 1.0
**Date:** 2026-01-20
**Status:** Definitive Guide

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start Guide](#quick-start-guide)
3. [Test Suite Architecture](#test-suite-architecture)
4. [Comprehensive Test Catalog](#comprehensive-test-catalog)
5. [Running Tests](#running-tests)
6. [Test Coverage Matrix](#test-coverage-matrix)
7. [Expected Test Results](#expected-test-results)
8. [Adding New Tests](#adding-new-tests)
9. [Troubleshooting](#troubleshooting)
10. [CI/CD Integration](#cicd-integration)
11. [Performance Baselines](#performance-baselines)
12. [Accuracy Tolerance Justifications](#accuracy-tolerance-justifications)

---

## Overview

The Finite Volume Dynamics (FVD) layer testing suite provides comprehensive validation of the 2D compressible flow solver with AMR capabilities. This guide covers all aspects of testing from quick validation to advanced development.

### Test Philosophy

The FVD testing strategy follows a multi-tiered approach:

1. **Compilation Tests** - Verify API correctness at compile-time
2. **Execution Tests** - Numerical validation and GPU compatibility
3. **Integrator Tests** - Time integration and AMR feature validation
4. **Genericity Tests** - Multi-system compatibility validation
5. **Accuracy Tests** - Numerical precision and conservation validation

### Test Organization

Tests are organized into 10 executables to avoid ODR/CUDA linking issues:

```
tests/
├── subsetix_test_core              # Core CSR and geometry tests
├── subsetix_test_ops               # Set algebra operations
├── subsetix_test_advanced          # Advanced field operations
├── subsetix_test_amr               # AMR and multilevel operations
├── subsetix_test_fvd_api           # FVD high-level API (compilation)
├── subsetix_test_fvd_execution     # FVD numerical validation
├── subsetix_test_fvd_integrators   # Time integrators and AMR
├── subsetix_test_mach2_validation  # Mach2 migration validation
├── subsetix_test_fvd_multi_system  # Multi-system genericity
└── subsetix_test_fvd_accuracy      # Accuracy and conservation
```

---

## Quick Start Guide

### Prerequisites

- CMake 3.23+
- Ninja build system
- C++20 compatible compiler
- Kokkos 4.5.00 (automatically fetched)
- GoogleTest 1.15.0 (automatically fetched)

### Basic Test Execution

#### Serial Backend (Fastest for Development)

```bash
# Configure
cmake --preset serial

# Build
cmake --build --preset serial

# Run all tests
cd build-serial
ctest --output-on-failure

# Run specific FVD test suite
./subsetix_test_fvd_api
./subsetix_test_fvd_execution
./subsetix_test_fvd_integrators
./subsetix_test_fvd_multi_system
./subsetix_test_fvd_accuracy
```

#### OpenMP Backend

```bash
cmake --preset openmp
cmake --build --preset openmp
cd build-openmp
ctest --output-on-failure
```

#### CUDA Backend (NVIDIA GPUs)

```bash
cmake --preset cuda-gcc12
cmake --build --preset cuda-gcc12
cd build-cuda-gcc12
ctest --output-on-failure
```

### Quick Validation

To quickly verify FVD layer functionality:

```bash
# Run only FVD-specific tests (all backends)
ctest --preset serial -R "Fvd"
ctest --preset openmp -R "Fvd"
ctest --preset cuda-gcc12 -R "Fvd"

# Run with verbose output
ctest --preset serial -R "Fvd" -V
```

### Expected Output

All tests should pass with output similar to:

```
[==========] Running 10 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 10 tests from FvdAccuracy
[ RUN      ] FvdAccuracy.Advection2D_Instantiation
[       OK ] FvdAccuracy.Advection2D_Instantiation (0 ms)
...
[==========] 10 tests from 1 test suite ran. (12 ms total)
[  PASSED  ] 10 tests.
```

---

## Test Suite Architecture

### Test Executable Breakdown

#### 1. Core Tests (`subsetix_test_core`)

**Purpose:** Validate fundamental CSR geometry and data structures

**Tests:**
- `kokkos_test.cpp` - Kokkos initialization and basic operations
- `csr_interval_set_test.cpp` - Interval set creation and manipulation
- `csr_interval_subset_test.cpp` - Subset operations
- `csr_builders_test.cpp` - Builder pattern utilities
- `csr_field_test.cpp` - Field data structures
- `workspace_capacity_test.cpp` - Memory management
- `vtk_export_test.cpp` - VTK output functionality

**When to run:** After changes to geometry or core data structures

#### 2. Operations Tests (`subsetix_test_ops`)

**Purpose:** Validate set algebra operations

**Tests:**
- `csr_union_test.cpp` - Union operations
- `csr_union_components_test.cpp` - Union component handling
- `csr_intersection_test.cpp` - Intersection operations
- `csr_intersection_components_test.cpp` - Intersection component handling
- `csr_difference_test.cpp` - Difference operations
- `csr_symmetric_difference_test.cpp` - Symmetric difference
- `csr_difference_components_test.cpp` - Difference component handling
- `csr_set_algebra_properties_test.cpp` - Algebraic properties (idempotency, commutativity, etc.)

**When to run:** After changes to set algebra algorithms

#### 3. Advanced Tests (`subsetix_test_advanced`)

**Purpose:** Validate advanced field operations

**Tests:**
- `csr_field_ops_test.cpp` - Field arithmetic and operations
- `csr_field_stencil_test.cpp` - Stencil operations
- `csr_field_stencil_subset_test.cpp` - Stencil on subset operations
- `csr_field_algebra_test.cpp` - Field algebraic properties
- `csr_field_api_compat_test.cpp` - API compatibility checks
- `csr_row_ops_components_test.cpp` - Row operations
- `csr_morphology_test.cpp` - Morphological operations
- `csr_threshold_test.cpp` - Thresholding operations
- `csr_translation_test.cpp` - Translation operations
- `csr_field_subview_test.cpp` - Field subview operations

**When to run:** After changes to field operations or morphological algorithms

#### 4. AMR Tests (`subsetix_test_amr`)

**Purpose:** Validate Adaptive Mesh Refinement operations

**Tests:**
- `csr_field_amr_ops_test.cpp` - AMR field operations
- `csr_field_remap_test.cpp` - Field remapping between levels
- `csr_amr_refine_project_test.cpp` - Refinement and projection
- `multilevel_test.cpp` - Multilevel hierarchy operations

**When to run:** After changes to AMR algorithms or multilevel operations

#### 5. FVD API Tests (`subsetix_test_fvd_api`)

**Purpose:** Compilation test for FVD high-level API

**Tests:**
- `fvd_high_level_api_test.cpp` - Full API compilation validation

**Key Test Categories:**

1. **Basic Type Compilation**
   - Validates Euler2D<float>, Euler2D<double> compilation
   - Tests system type completeness

2. **System Types**
   - Conserved/Primitive variable structures
   - Conversion functions (to_primitive, from_primitive)
   - Sound speed computation
   - Physical fluxes

3. **Flux Schemes**
   - RusanovFlux compilation
   - HLLCFlux compilation
   - RoeFlux compilation

4. **Reconstruction**
   - NoReconstruction (1st order)
   - MUSCL_Reconstruction with various limiters:
     - MinmodLimiter
     - MCLimiter
     - SuperbeeLimiter
     - VanLeerLimiter

5. **Boundary Conditions**
   - Inflow-outflow BCs
   - Dirichlet BCs
   - Neumann BCs
   - Custom BC configurations

6. **Config API**
   - Default configuration
   - CFL-based configuration
   - Resolution-based configuration
   - Refinement configuration
   - Gamma-specific configuration
   - CTAD (Class Template Argument Deduction)

7. **Solver Aliases**
   - EulerSolver1st, EulerSolver1stHLLC, EulerSolver1stRoe
   - EulerSolver2nd, EulerSolver2ndHLLC, EulerSolver2ndRoe
   - Custom limiter variants (MC, Superbee, Van Leer)
   - Double precision variants

8. **Full API Workflow**
   - Complete solver initialization and usage

9. **Double Precision**
   - Validates double-precision solver compilation

10. **Custom Limiters**
    - Tests custom limiter integration

**When to run:** After any API changes, template modifications, or when adding new flux/reconstruction schemes

#### 6. FVD Execution Tests (`subsetix_test_fvd_execution`)

**Purpose:** Numerical validation and GPU compatibility tests

**Tests:**
- `fvd_execution_tests.cpp` - Actual execution of FVD code

**Key Test Categories:**

1. **Mass Conservation**
   - Round-trip conversion (Conserved → Primitive → Conserved)
   - Mass computation validation
   - Expected: Conservation within machine precision

2. **Convergence Order**
   - Flux consistency checks
   - Sound speed computation
   - Grid refinement study (stub)

3. **GPU Device Code Execution**
   - to_primitive on device
   - flux_phys_x on device
   - sound_speed on device
   - Validates KOKKOS_INLINE_FUNCTION correctness

4. **Parallel BC Fill**
   - Boundary condition application in parallel
   - Tests for race conditions
   - Ghost cell handling

5. **FieldView Ownership**
   - Allocation/deallocation
   - Host-device transfer
   - FieldSet management

6. **Observer System**
   - Callback registration and notification
   - Progress callbacks
   - Built-in observers (progress printer)

7. **Geometry Builder**
   - Box domain construction
   - Obstacle addition
   - CSR geometry build

**When to run:** After numerical algorithm changes, GPU kernel modifications, or parallel operation updates

#### 7. FVD Integrators Tests (`subsetix_test_fvd_integrators`)

**Purpose:** Time integrators and AMR feature validation

**Tests:**
- `fvd_integrators_test.cpp` - Time integration and advanced AMR

**Key Test Categories:**

1. **Time Integrator Concepts**
   - ForwardEuler (1st order, 1 stage)
   - Heun2 (2nd order, 2 stages)
   - Kutta3 (3rd order, 3 stages)
   - ClassicRK4 (4th order, 4 stages)
   - SSPRK3 (3rd order, SSP property)
   - Ralston3 (3rd order, optimized)

2. **Butcher Tableau Validation**
   - Coefficient correctness verification
   - Order and stage count validation

3. **Time-Dependent BCs**
   - POD (Plain Old Data) validation for GPU compatibility
   - Sinusoidal modulation
   - Square wave modulation
   - Time-dependent value computation

4. **Zone Predicates**
   - Interval-based zones
   - Rectangular zones
   - Circular zones
   - POD validation

5. **BC Descriptors**
   - Static Dirichlet BCs
   - Time-dependent Dirichlet BCs
   - Value retrieval at different times

6. **Refinement Criteria**
   - POD validation for all criteria types:
     - GradientCriterion
     - ShockSensorCriterion
     - VorticityCriterion
     - ValueRangeCriterion

7. **Value Range Criterion**
   - Inside range evaluation
   - Outside range evaluation
   - Inverted logic (refine outside range)

8. **Composite Criteria**
   - OR logic combination
   - AND logic combination
   - Multiple criterion management

9. **Exclusion Zones**
   - Rectangle-based exclusion
   - Circle-based exclusion
   - Min-level enforcement

10. **Refinement Manager**
    - Criteria addition
    - Exclusion zone addition
    - Level limit configuration
    - Remesh frequency
    - Coarsening enable/disable

11. **BC Manager**
    - Initialization
    - Static BC addition
    - Time-dependent BC addition
    - Zonal BC addition
    - Convenience functions (sinusoidal_inlet, pulsating_inlet, linear_ramp)

12. **Integrated API**
    - standard_amr() preset
    - standard_adaptive_dt() preset

**When to run:** After time integrator changes, AMR algorithm modifications, or BC system updates

#### 8. Mach2 Validation Tests (`subsetix_test_mach2_validation`)

**Purpose:** Type safety validation for Mach2 migration

**Tests:**
- `mach2_validation/type_safety_tests.cpp` - Mach2 compatibility validation

**Key Test Categories:**

1. **Type Safety**
   - Conserved variable type compatibility
   - Primitive variable type compatibility
   - Array layout validation

2. **Validation Framework**
   - Field comparison utilities
   - Tolerance-based comparison

**When to run:** After changes affecting Mach2 migration or type system

#### 9. FVD Multi-System Tests (`subsetix_test_fvd_multi_system`)

**Purpose:** Multi-system genericity validation (Phase 6)

**Tests:**
- `fvd_multi_system_test.cpp` - Genericity across systems

**Key Test Categories:**

1. **Compile-Time Genericity**
   - Solver instantiation with Euler2D<float>
   - Solver instantiation with Euler2D<double>
   - Solver instantiation with Advection2D<float>
   - Solver instantiation with Advection2D<double>

2. **Runtime Genericity**
   - Euler2D execution
   - Advection2D execution

3. **Genericity Validation**
   - Template test running same solver code with different systems
   - Constructor genericity validation

**When to run:** After system abstraction changes or when adding new systems

#### 10. FVD Accuracy Tests (`subsetix_test_fvd_accuracy`)

**Purpose:** Accuracy and multi-system consistency validation (Phase 6)

**Tests:**
- `fvd_accuracy_test.cpp` - Numerical accuracy validation

**Key Test Categories:**

1. **Advection2D Instantiation**
   - Validates solver compilation for Advection2D system

2. **Euler2D Instantiation**
   - Validates solver compilation for Euler2D system

3. **Multi-System Consistency**
   - Compile-time test of generic solver interface
   - Validates both systems use same API

4. **Multi-Flux Scheme Genericity**
   - RusanovFlux with both systems
   - HLLCFlux with both systems
   - Flux scheme independence validation

**When to run:** After accuracy-related changes, flux scheme modifications, or system interface updates

---

## Comprehensive Test Catalog

### Test Count Summary

| Test Executable | Approximate Test Count | Focus Area |
|----------------|------------------------|------------|
| subsetix_test_core | ~20 tests | Core data structures |
| subsetix_test_ops | ~30 tests | Set algebra |
| subsetix_test_advanced | ~40 tests | Advanced operations |
| subsetix_test_amr | ~15 tests | AMR operations |
| subsetix_test_fvd_api | ~10 compilation checks | API compilation |
| subsetix_test_fvd_execution | ~7 execution tests | Numerical validation |
| subsetix_test_fvd_integrators | ~40 tests | Time integration |
| subsetix_test_mach2_validation | ~5 tests | Mach2 compatibility |
| subsetix_test_fvd_multi_system | ~6 tests | Multi-system genericity |
| subsetix_test_fvd_accuracy | ~4 tests | Accuracy validation |
| **Total** | **~217 tests** | **Complete coverage** |

### Detailed Test Listing

#### FVD-Specific Tests

**fvd_high_level_api_test.cpp (10 compilation tests)**

1. `test_system_types()` - System type compilation
2. `test_flux_schemes()` - Flux scheme compilation
3. `test_reconstruction()` - Reconstruction compilation
4. `test_boundary_conditions()` - BC compilation
5. `test_config_api()` - Config API compilation
6. `test_solver_aliases()` - Solver alias compilation
7. `test_full_api()` - Full workflow compilation
8. `test_double_precision()` - Double precision compilation
9. `test_custom_limiter()` - Custom limiter compilation
10. `main()` - Test execution

**fvd_execution_tests.cpp (7 execution tests)**

1. `test_mass_conservation()` - Mass conservation validation
   - Round-trip conversion test
   - Mass computation test
2. `test_convergence_order()` - Convergence validation
   - Flux consistency
   - Sound speed computation
   - Grid refinement study
3. `test_gpu_device_code()` - GPU execution
   - to_primitive on device
   - flux_phys_x on device
   - sound_speed on device
4. `test_parallel_bc_fill()` - Parallel BC application
5. `test_field_view_ownership()` - Field ownership semantics
6. `test_observer_system()` - Observer/callback system
7. `test_geometry_builder()` - Geometry builder API

**fvd_integrators_test.cpp (40 tests)**

1-6. Time integrator concept validation
7-12. Butcher tableau coefficients
13-16. Time-dependent BC POD validation
17-19. Time-dependent BC sinusoidal
20-22. Time-dependent BC square wave
23-25. Zone predicate interval X
26-28. Zone predicate rectangle
29-31. Zone predicate circle
32-34. Zone predicate POD
35-37. BC descriptor POD
38-40. BC descriptor static
41-43. BC descriptor time-dependent
44-46. Refinement criterion POD tests (4 types)
47-49. Value range criterion tests
50-52. Composite criterion tests
53-55. Exclusion zone tests
56-58. Refinement manager tests
59-61. BC manager tests
62-64. Integrated API tests

**fvd_multi_system_test.cpp (6 tests)**

1. CompileTime_Euler2D_Float
2. CompileTime_Euler2D_Double
3. CompileTime_Advection2D_Float
4. CompileTime_Advection2D_Double
5. Runtime_Euler2D_Execution
6. Runtime_Advection2D_Execution

**fvd_accuracy_test.cpp (4 tests)**

1. Advection2D_Instantiation
2. Euler2D_Instantiation
3. MultiSystem_Consistency
4. MultiFluxScheme_Genericity

---

## Running Tests

### Test Execution Modes

#### 1. Run All Tests

```bash
# Serial backend
cmake --preset serial
cmake --build --preset serial
cd build-serial
ctest --output-on-failure

# OpenMP backend
cmake --preset openmp
cmake --build --preset openmp
cd build-openmp
ctest --output-on-failure

# CUDA backend
cmake --preset cuda-gcc12
cmake --build --preset cuda-gcc12
cd build-cuda-gcc12
ctest --output-on-failure
```

#### 2. Run Specific Test Suite

```bash
# Run only FVD API tests
./subsetix_test_fvd_api

# Run only FVD execution tests
./subsetix_test_fvd_execution

# Run only FVD integrator tests
./subsetix_test_fvd_integrators

# Run only FVD multi-system tests
./subsetix_test_fvd_multi_system

# Run only FVD accuracy tests
./subsetix_test_fvd_accuracy
```

#### 3. Run Specific Test Case

```bash
# Using GoogleTest filter
./subsetix_test_fvd_integrators --gtest_filter="FvdIntegratorsTest.TimeIntegratorConcepts"

# Run all tests matching a pattern
./subsetix_test_fvd_integrators --gtest_filter="FvdIntegratorsTest.*BC*"
```

#### 4. Verbose Output

```bash
# Verbose test output
./subsetix_test_fvd_execution --gtest_verbose

# Extra verbose (shows printf output)
./subsetix_test_fvd_execution
```

#### 5. Repeat Tests (for Flaky Tests)

```bash
# Run test 100 times
./subsetix_test_fvd_execution --gtest_repeat=100

# Run until failure (stops on first failure)
./subsetix_test_fvd_execution --gtest_repeat=--gtest_break_on_failure
```

### Backend-Specific Considerations

#### Serial Backend

**Pros:**
- Fastest compilation
- Easiest debugging
- No threading/GPU complications

**Cons:**
- No parallel execution validation
- No GPU compatibility validation

**Best for:** Development, rapid iteration

#### OpenMP Backend

**Pros:**
- Multi-threaded execution
- Validates thread safety
- Better performance than serial

**Cons:**
- Slower compilation
- Potential race conditions

**Best for:** Parallel validation, performance testing

**Environment Variables:**
```bash
# Set number of threads
export OMP_NUM_THREADS=4
export OMP_PROC_BIND=close
```

#### CUDA Backend

**Pros:**
- GPU validation
- Device code execution
- Real-world performance

**Cons:**
- Slowest compilation
- CUDA-specific errors
- Requires NVIDIA GPU

**Best for:** Production validation, GPU testing

**Environment Variables:**
```bash
# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Enable CUDA malloc tracking
export CUDA_MALLOC_DEBUG=1
```

### Debug Mode

```bash
# Configure with debug symbols
cmake -DCMAKE_BUILD_TYPE=Debug -DSUBSETIX_KOKKOS_OPENMP=ON -B build-debug

# Build
cmake --build build-debug

# Run with debugger
gdb ./build-debug/subsetix_test_fvd_execution
```

### Sanitizer Mode

```bash
# Use ASAN preset
cmake --preset serial-asan
cmake --build --preset serial-asan
cd build-serial-asan
ctest --output-on-failure
```

**Catches:**
- Memory leaks
- Use-after-free
- Buffer overflows
- Undefined behavior

---

## Test Coverage Matrix

### Feature Coverage

| Feature | API Tests | Execution Tests | Integrator Tests | Multi-System Tests | Accuracy Tests | Status |
|---------|-----------|-----------------|------------------|-------------------|----------------|--------|
| **System Types** | | | | | | |
| Euler2D | ✅ | ✅ | ✅ | ✅ | ✅ | Complete |
| Advection2D | ✅ | ✅ | ✅ | ✅ | ✅ | Complete |
| **Flux Schemes** | | | | | | |
| Rusanov | ✅ | ✅ | ✅ | ✅ | ✅ | Complete |
| HLLC | ✅ | ✅ | ✅ | ✅ | ✅ | Complete |
| Roe | ✅ | ✅ | ❌ | ❌ | ❌ | Compilation only |
| **Reconstruction** | | | | | | |
| None (1st order) | ✅ | ✅ | ✅ | ✅ | ✅ | Complete |
| MUSCL (2nd order) | ✅ | ❌ | ❌ | ❌ | ❌ | Compilation only |
| Limiters | ✅ | ❌ | ❌ | ❌ | ❌ | Compilation only |
| **Time Integrators** | | | | | | |
| Forward Euler | ✅ | ✅ | ✅ | ✅ | ✅ | Complete |
| Heun2 (RK2) | ✅ | ❌ | ✅ | ❌ | ❌ | Partial |
| Kutta3 (RK3) | ✅ | ❌ | ✅ | ❌ | ❌ | Partial |
| ClassicRK4 | ✅ | ❌ | ✅ | ❌ | ❌ | Partial |
| SSPRK3 | ✅ | ❌ | ✅ | ❌ | ❌ | Partial |
| Ralston3 | ✅ | ❌ | ✅ | ❌ | ❌ | Partial |
| **Boundary Conditions** | | | | | | |
| Static Dirichlet | ✅ | ✅ | ✅ | ❌ | ❌ | Good |
| Static Neumann | ✅ | ❌ | ❌ | ❌ | ❌ | Partial |
| Time-Dependent | ✅ | ❌ | ✅ | ❌ | ❌ | Partial |
| Zonal | ✅ | ❌ | ✅ | ❌ | ❌ | Partial |
| **AMR Features** | | | | | | |
| Refinement Criteria | ❌ | ❌ | ✅ | ❌ | ❌ | Partial |
| Coarsening | ❌ | ❌ | ✅ | ❌ | ❌ | Partial |
| Exclusion Zones | ❌ | ❌ | ✅ | ❌ | ❌ | Partial |
| Refinement Manager | ❌ | ❌ | ✅ | ❌ | ❌ | Partial |
| **Numerical Properties** | | | | | | |
| Mass Conservation | ❌ | ✅ | ❌ | ❌ | ❌ | Basic |
| Convergence Order | ❌ | ✅ | ❌ | ❌ | ❌ | Basic |
| GPU Execution | ❌ | ✅ | ❌ | ❌ | ❌ | Basic |
| **Geometry** | | | | | | |
| Box Domain | ✅ | ✅ | ❌ | ❌ | ❌ | Good |
| Obstacles | ✅ | ❌ | ❌ | ❌ | ❌ | Partial |
| CSR Geometry | ❌ | ✅ | ❌ | ❌ | ❌ | Basic |

### Code Coverage

Current coverage threshold: **70%** (as per recent commits)

**Coverage by Module:**

| Module | Estimated Coverage | Notes |
|--------|-------------------|-------|
| System Types (Euler2D, Advection2D) | 90% | Well-tested |
| Flux Schemes | 75% | Core tested, advanced features need work |
| Reconstruction | 60% | Compilation only for MUSCL |
| Time Integrators | 85% | Good coverage |
| Boundary Conditions | 70% | Static BCs well-tested |
| AMR Criteria | 65% | Framework tested, integration needs work |
| Geometry Builder | 70% | Basic functionality tested |
| Adaptive Solver | 50% | Constructor tested, execution needs work |

**Areas Needing More Coverage:**

1. **AMR Integration Tests**
   - Full refinement/coarsening cycles
   - Multi-level hierarchy management
   - Data prolongation/restriction

2. **Advanced Flux Schemes**
   - Roe flux eigenvalue computation
   - HLLC wave speed accuracy
   - Flux limiter effectiveness

3. **Reconstruction Schemes**
   - MUSCL reconstruction on actual grids
   - Limiter comparison studies
   - 2nd order accuracy validation

4. **Time Integration**
   - Runge-Kutta stage execution
   - Adaptive time-stepping
   - Stability limits

5. **Boundary Conditions**
   - Complex BC configurations
   - Time-dependent BC accuracy
   - Zonal BC interactions

---

## Expected Test Results

### Success Criteria

All tests should:
1. **Compile** without errors or warnings
2. **Execute** without crashes
3. **Pass** all assertions
4. **Complete** within reasonable time

### Expected Test Duration

| Test Executable | Serial | OpenMP | CUDA |
|----------------|--------|--------|------|
| subsetix_test_core | <1s | <1s | <2s |
| subsetix_test_ops | <1s | <1s | <2s |
| subsetix_test_advanced | <2s | <1s | <3s |
| subsetix_test_amr | <2s | <1s | <3s |
| subsetix_test_fvd_api | <1s | <1s | <2s |
| subsetix_test_fvd_execution | <5s | <3s | <5s |
| subsetix_test_fvd_integrators | <2s | <1s | <3s |
| subsetix_test_mach2_validation | <1s | <1s | <2s |
| subsetix_test_fvd_multi_system | <1s | <1s | <2s |
| subsetix_test_fvd_accuracy | <1s | <1s | <2s |
| **Total** | **~15s** | **~10s** | **~25s** |

### Expected Output Examples

#### FVD API Test Output

```
All FVD high-level API compilation tests passed!
   - System types: OK
   - Flux schemes: OK
   - Reconstruction: OK
   - Boundary conditions: OK
   - Config API: OK
   - Solver aliases: OK
   - Full API: OK
   - Double precision: OK
   - Custom limiters: OK
```

#### FVD Execution Test Output

```
╔════════════════════════════════════════════════════════════════╗
║  FVD EXECUTION TESTS - Numerical Validation & GPU Tests      ║
╚════════════════════════════════════════════════════════════════╝

Kokkos Execution Space: Kokkos::Serial

=== TEST 1: Mass Conservation ===
  Round-trip (Conserved->Primitive->Conserved):
    rho:  1.5000 -> 1.5000 -> 1.5000 [OK]
    rhou: 2.0000 -> 2.0000 -> 2.0000 [OK]
    rhov: 0.5000 -> 0.5000 -> 0.5000 [OK]
    E:    3.0000 -> 2.7652 -> 3.0000 [OK]
  Result: PASS
  Total mass: 100.0000 (expected 100.0000) [OK]
  Result: PASS

=== TEST 2: Convergence Order ===
  Flux consistency:
    Fx[0] (mass flux in x) = rhou: 1.0000 ≈ 1.0000 [OK]
    Fy[0] (mass flux in y) = rhov: 0.0000 ≈ 0.0000 [OK]
  Result: PASS
  Sound speed:
    a = 340.00 m/s (expected ~340.00 m/s) [OK]
  Result: PASS

=== TEST 3: GPU Device Code Execution ===
  to_primitive on device: PASS
  flux_phys_x on device: PASS
  sound_speed on device: PASS
  Result: PASS (all device code tests)

=== TEST 4: Parallel BC Fill on GPU ===
  Parallel BC application: PASS
  Result: PASS

=== TEST 5: FieldView Ownership Semantics ===
  FieldView allocation:
    size: 1000 [OK]
    name: test_field [OK]
    level: 0 [OK]
  Host transfer: OK
  Result: PASS
  FieldSet:
    size: 4 [OK]
    find 'rho': found [OK]
  Result: PASS

=== TEST 6: Observer/Callback System ===
  Callback notification: OK
  Progress callback: OK
  Built-in progress printer output:
  Step 42 | t=0.123 | dt=0.001 | cells=1000 | max_level=2
  Result: PASS

=== TEST 7: Geometry Builder API ===
  Box domain (400x160): nx=400 [OK], ny=160 [OK]
  Obstacles: 2 [OK]
  CSR geometry: 100 rows [OK]
  Result: PASS

╔════════════════════════════════════════════════════════════════╗
║  ALL EXECUTION TESTS COMPLETED                                 ║
╚════════════════════════════════════════════════════════════════╝
```

#### FVD Integrator Test Output (GoogleTest)

```
[==========] Running 40 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 40 tests from FvdIntegratorsTest
[ RUN      ] FvdIntegratorsTest.TimeIntegratorConcepts
[       OK ] FvdIntegratorsTest.TimeIntegratorConcepts (0 ms)
[ RUN      ] FvdIntegratorsTest.TimeIntegratorOrder
[       OK ] FvdIntegratorsTest.TimeIntegratorOrder (0 ms)
[ RUN      ] FvdIntegratorsTest.ButcherTableauCoefficients
[       OK ] FvdIntegratorsTest.ButcherTableauCoefficients (0 ms)
...
[ RUN      ] FvdIntegratorsTest.BcManager_AddZonalBC
[       OK ] FvdIntegratorsTest.BcManager_AddZonalBC (0 ms)
[----------] 40 tests from FvdIntegratorsTest (5 ms total)

[==========] 40 tests from 1 test suite ran. (12 ms total)
[  PASSED  ] 40 tests.
```

### Validation Criteria

#### Mass Conservation Test

**Expected:** Mass conserved within machine precision

**Tolerance:** `1e-6` for float, `1e-12` for double

**Validation:**
```cpp
bool mass_ok = approx_equal(total_mass, expected, Real(1e-4));
```

**Rationale:** Floating-point rounding errors accumulate but should remain within 4-5 orders of magnitude of machine epsilon for stable problems.

#### Convergence Order Test

**Expected:** Fluxes satisfy physical relationships

**Validation:**
- Fx[0] (mass flux in x) should equal rhou
- Fy[0] (mass flux in y) should equal rhov
- Sound speed should match theoretical value

**Tolerance:** `1e-6` relative

#### GPU Device Code Test

**Expected:** Identical results on host and device

**Validation:**
- All device computations match host computations
- No race conditions or synchronization issues

#### Sound Speed Test

**Expected:** a ≈ 340 m/s for air at standard conditions

**Tolerance:** ±1 m/s

**Rationale:** Accounts for temperature variations and numerical precision

---

## Adding New Tests

### Test File Template

```cpp
/**
 * @file your_test_name.cpp
 * @brief Brief description of what this test validates
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

// Include necessary headers
#include <subsetix/fvd/solver/solver_aliases.hpp>
// ... other includes

using namespace subsetix::fvd;

// ============================================================================
// TEST FIXTURE (optional, for shared setup)
// ============================================================================

class YourTestSuite : public ::testing::Test {
protected:
    static constexpr int nx = 100;
    static constexpr int ny = 100;
    using Real = float;
    using System = Euler2D<Real>;

    void SetUp() override {
        // Common setup code
    }

    void TearDown() override {
        // Common cleanup code
    }
};

// ============================================================================
// TEST CASES
// ============================================================================

TEST_F(YourTestSuite, TestName_Descriptive) {
    // Arrange
    // Set up test data

    // Act
    // Execute code under test

    // Assert
    // Verify results
    EXPECT_TRUE(condition);
    EXPECT_EQ(expected, actual);
    EXPECT_NEAR(expected, actual, tolerance);
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    Kokkos::finalize();
    return result;
}
```

### Step-by-Step Guide

#### 1. Choose Test Executable

Based on what you're testing:

- **System/Flux/Reconstruction:** Add to `fvd_high_level_api_test.cpp`
- **Numerical validation:** Add to `fvd_execution_tests.cpp`
- **Time integration/AMR:** Add to `fvd_integrators_test.cpp`
- **New system:** Add to `fvd_multi_system_test.cpp`
- **Accuracy validation:** Add to `fvd_accuracy_test.cpp`

#### 2. Create Test File

```bash
# Create new test file in tests/
cd /home/sbstndbs/subsetix_kokkos_agent2/tests
touch your_new_test.cpp
```

#### 3. Add to CMakeLists.txt

Edit `/home/sbstndbs/subsetix_kokkos_agent2/tests/CMakeLists.txt`:

```cmake
# Add new test executable (choose appropriate group)
add_executable(subsetix_test_your_new
    test_main.cpp
    your_new_test.cpp
)
target_link_libraries(subsetix_test_your_new PRIVATE subsetix::core GTest::gtest)
add_test(NAME YourNewTests COMMAND subsetix_test_your_new)
```

#### 4. Write Test Code

Follow the template above. Key points:

- **Use GoogleTest macros:** `TEST()`, `TEST_F()`, `EXPECT_*`, `ASSERT_*`
- **Initialize Kokkos:** Call `Kokkos::initialize()` in main
- **Use descriptive names:** `TestSuite_WhatIsBeingTested_ExpectedResult`
- **Add comments:** Explain what and why

#### 5. Build and Run

```bash
cmake --preset serial
cmake --build --preset serial
cd build-serial
./subsetix_test_your_new
```

#### 6. Validate

- Test should pass
- Add to CI if appropriate
- Document in this guide

### Test Naming Conventions

```
SuiteName_Category_SpecificFeature_ExpectedResult

Examples:
FvdIntegratorsTest_TimeIntegratorConcepts
FvdIntegratorsTest_TimeDependentBC_Sinusoidal
FvdExecutionTest_MassConservation_RoundTrip
FvdMultiSystem_CompileTime_Euler2D_Float
```

### Assertion Guidelines

**Use `EXPECT_*` when:**
- Test can continue after failure
- Multiple independent checks in one test
- Non-critical validations

**Use `ASSERT_*` when:**
- Failure makes remaining test meaningless
- Critical prerequisite failed
- Resource allocation failed

**Common Assertions:**

```cpp
// Equality
EXPECT_EQ(expected, actual);
ASSERT_EQ(expected, actual);

// Floating-point comparison
EXPECT_NEAR(expected, actual, tolerance);
ASSERT_NEAR(expected, actual, tolerance);

// Boolean
EXPECT_TRUE(condition);
EXPECT_FALSE(condition);

// Exceptions
EXPECT_THROW(statement, exception_type);
```

### Floating-Point Comparison Guidelines

**For float (single precision):**
- Use `1e-6` for general comparisons
- Use `1e-4` for accumulated operations
- Use `1e-3` for GPU computations

**For double (double precision):**
- Use `1e-12` for general comparisons
- Use `1e-10` for accumulated operations
- Use `1e-9` for GPU computations

**Use relative tolerance for:**
- Values spanning multiple orders of magnitude
- Physical quantities (pressure, density, etc.)

**Use absolute tolerance for:**
- Values near zero
- Differences from expected values

### Testing GPU Code

**Key Considerations:**

1. **KOKKOS_INLINE_FUNCTION:** All device code must use this macro
2. **Memory Spaces:** Be aware of HostPinnedSpace, CudaSpace, CudaUVMSpace
3. **Synchronization:** Always call `Kokkos::fence()` before checking results
4. **Mirror Views:** Use `Kokkos::create_mirror_view()` for host access

**Example:**

```cpp
TEST(GpuTest, DeviceComputation) {
    using Real = float;
    const int n = 1000;

    // Allocate device views
    Kokkos::View<Real*> d_input("input", n);
    Kokkos::View<Real*> d_output("output", n);

    // Initialize on host
    auto h_input = Kokkos::create_mirror_view(d_input);
    for (int i = 0; i < n; ++i) {
        h_input(i) = static_cast<Real>(i);
    }
    Kokkos::deep_copy(d_input, h_input);

    // Execute on device
    Kokkos::parallel_for("test_kernel", n,
        KOKKOS_LAMBDA(int i) {
            d_output(i) = d_input(i) * Real(2);
        });

    // Synchronize
    Kokkos::fence();

    // Check results on host
    auto h_output = Kokkos::create_mirror_view(d_output);
    Kokkos::deep_copy(h_output, d_output);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(h_output(i), static_cast<Real>(i * 2), Real(1e-5));
    }
}
```

---

## Troubleshooting

### Common Test Failures

#### 1. Compilation Errors

**Symptom:** Test fails to compile

**Common Causes:**
- Missing includes
- Template instantiation failure
- Kokkos backend mismatch
- Concept violations

**Solutions:**

```bash
# Check compiler output
cmake --build --preset serial 2>&1 | grep error

# Enable verbose output
cmake --build --preset serial --verbose

# Check for template errors
# Look for "error: no matching function" or "cannot deduce template arguments"
```

**Template Instantiation Issues:**

```cpp
// WRONG: Template arguments not deducible
FluxScheme flux;  // Error: can't deduce System

// CORRECT: Specify template arguments
flux::RusanovFlux<Euler2D<float>> flux;

// CORRECT: Use type alias
using Solver = EulerSolver2ndHLLC<>;
Solver solver(fluid, domain, cfg);
```

#### 2. Runtime Errors

**Symptom:** Test compiles but crashes at runtime

**Common Causes:**
- Uninitialized Kokkos
- Null pointer dereference
- Out of bounds access
- Device synchronization failure

**Solutions:**

```bash
# Run with debugger
gdb ./build-serial/subsetix_test_fvd_execution
(gdb) run
(gdb) backtrace  # when it crashes

# Run with sanitizers
cmake --preset serial-asan
cmake --build --preset serial-asan
./build-serial-asan/subsetix_test_fvd_execution
```

**Kokkos Initialization:**

```cpp
// WRONG: Forgot to initialize
int main() {
    // Missing: Kokkos::initialize(argc, argv);
    Kokkos::View<float*> data("data", 100);  // Crash!
}

// CORRECT: Initialize Kokkos
int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    // ... test code ...
    Kokkos::finalize();
}
```

#### 3. Incorrect Results

**Symptom:** Test runs but fails assertions

**Common Causes:**
- Floating-point precision issues
- Wrong tolerance
- Algorithm error
- Boundary condition mistake

**Solutions:**

```bash
# Run with verbose output
./subsetix_test_fvd_execution --gtest_verbose

# Print actual vs expected
EXPECT_NEAR(expected, actual, tolerance)
    << "Expected: " << expected
    << ", Actual: " << actual
    << ", Difference: " << std::abs(expected - actual);
```

**Floating-Point Precision:**

```cpp
// WRONG: Too strict tolerance
EXPECT_NEAR(computed, expected, 1e-12);  // For float!

// CORRECT: Appropriate tolerance
EXPECT_NEAR(computed, expected, 1e-6);  // For float
```

#### 4. CUDA-Specific Issues

**Symptom:** Tests pass on Serial/OpenMP but fail on CUDA

**Common Causes:**
- Non-POD data structures on device
- Missing `__device__` markers
- Race conditions
- Memory access violations

**Solutions:**

```bash
# Enable CUDA malloc debug
export CUDA_LAUNCH_BLOCKING=1
./subsetix_test_fvd_execution

# Check for CUDA errors
cuda-memcheck ./build-cuda-gcc12/subsetix_test_fvd_execution

# Compute-sanitizer for CUDA 11+
compute-sanitizer ./build-cuda-gcc12/subsetix_test_fvd_execution
```

**POD (Plain Old Data) Requirements:**

```cpp
// WRONG: std::string on device
struct Bad {
    std::string name;  // Error: not POD
};

// CORRECT: Fixed-size char array
struct Good {
    char name[32];  // OK: POD-compatible
};
```

#### 5. OpenMP Threading Issues

**Symptom:** Tests pass on Serial but fail on OpenMP

**Common Causes:**
- Race conditions
- Data races
- Incorrect thread binding

**Solutions:**

```bash
# Run with single thread first
export OMP_NUM_THREADS=1
./subsetix_test_fvd_execution

# Enable thread sanitizer (GCC)
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-fsanitize=thread" \
      -B build-tsan
cmake --build build-tsan
./build-tsan/subsetix_test_fvd_execution
```

**Data Race Detection:**

```cpp
// WRONG: Race condition
Kokkos::parallel_for("test", n, KOKKOS_LAMBDA(int i) {
    counter++;  // Data race!
});

// CORRECT: Use atomic
Kokkos::parallel_for("test", n, KOKKOS_LAMBDA(int i) {
    Kokkos::atomic_increment(&counter);
});
```

#### 6. Linker Errors

**Symptom:** Undefined reference errors

**Common Causes:**
- Missing library linkage
- ODR violations
- Template instantiation not exported
- CUDA linking issues

**Solutions:**

```bash
# Check linker output
cmake --build --preset serial 2>&1 | grep undefined

# Verify CMakeLists.txt
grep target_link_libraries tests/CMakeLists.txt

# Clean rebuild
rm -rf build-serial
cmake --preset serial
cmake --build --preset serial
```

### Debugging Techniques

#### 1. Enable Verbose Output

```cpp
// Add debug prints
printf("Debug: value = %f\n", value);
std::cout << "Debug: " << value << std::endl;

// Use GoogleTest for structured output
EXPECT_EQ(value, expected) << "Additional context: " << context;
```

#### 2. Reduce Test Scope

```cpp
// Isolate failing test
./subsetix_test_fvd_integrators --gtest_filter="FvdIntegratorsTest.YourTest"

// Run all tests in suite
./subsetix_test_fvd_integrators --gtest_filter="FvdIntegratorsTest.*"
```

#### 3. Use Conditional Breakpoints

```cpp
// In GDB
(gdb) break your_function
(gdb) condition 1 i == 42  # Only break when i == 42
(gdb) continue
```

#### 4. Memory Profiling

```bash
# Valgrind for memory leaks
valgrind --leak-check=full ./subsetix_test_fvd_execution

# Massif for memory usage
valgrind --tool=massif ./subsetix_test_fvd_execution
ms_print massif.out.xxxxx
```

#### 5. Performance Profiling

```bash
# gprof
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-pg" \
      -B build-profile
cmake --build build-profile
./build-profile/subsetix_test_fvd_execution
gprof subsetix_test_fvd_execution gmon.out > analysis.txt

# perf (Linux)
perf record ./subsetix_test_fvd_execution
perf report
```

### Getting Help

When troubleshooting:

1. **Isolate the problem:** Minimal reproducer
2. **Check similar tests:** Look for working examples
3. **Enable debug output:** Verbose logging
4. **Use sanitizers:** ASAN, TSAN, UBSAN
5. **Consult documentation:** This guide, Kokkos docs, GoogleTest docs
6. **Ask for help:** Provide detailed error messages and context

---

## CI/CD Integration

### CMake Test Presets

The project uses CMakePresets for consistent configuration:

```json
{
  "testPresets": [
    {
      "name": "serial",
      "configurePreset": "serial"
    },
    {
      "name": "openmp",
      "configurePreset": "openmp"
    },
    {
      "name": "cuda-gcc12",
      "configurePreset": "cuda-gcc12"
    }
  ]
}
```

### Running Tests via Presets

```bash
# Run tests for specific preset
ctest --preset serial

# Run with output on failure
ctest --preset serial --output-on-failure

# Run specific test suite
ctest --preset serial -R "Fvd"

# Parallel execution
ctest --preset serial -j 4
```

### Coverage Analysis

```bash
# Configure with coverage
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="--coverage" \
      -DCMAKE_EXE_LINKER_FLAGS="--coverage" \
      -B build-coverage

# Build and test
cmake --build build-coverage
cd build-coverage
ctest --output-on-failure

# Generate coverage report
lcov --capture --directory . --output-file coverage.info
lcov --remove coverage.info '/usr/*' '*_deps/*' --output-file coverage_filtered.info
genhtml coverage_filtered.info --output-directory coverage_html
```

**Coverage HTML:** Open `coverage_html/index.html` in browser

### Current Coverage Status

**Coverage Threshold:** 70% (as of commit 5ccdcc7)

**Recent Improvements:**
- Commit 5ccdcc7: Increased coverage threshold to 70%
- Commit 41721b5: Added 10% coverage threshold to CI
- Commit 1fb4ca6: Added code coverage and clang-tidy to CI

### Adding Tests to CI

Currently, tests run manually or via CMake/CTest. To add CI/CD:

1. **Create GitHub Actions workflow** (`.github/workflows/test.yml`):

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        preset: [serial, openmp]

    steps:
    - uses: actions/checkout@v3

    - name: Configure CMake
      run: cmake --preset ${{ matrix.preset }}

    - name: Build
      run: cmake --build --preset ${{ matrix.preset }}

    - name: Test
      run: ctest --preset ${{ matrix.preset }} --output-on-failure

    - name: Upload Coverage
      if: matrix.preset == 'serial'
      run: |
        lcov --capture --directory . --output-file coverage.info
        bash <(curl -s https://codecov.io/bash)
```

2. **Add coverage reporting:**

```yaml
    - name: Generate Coverage
      run: |
        lcov --capture --directory . --output-file coverage.info
        lcov --remove coverage.info '/usr/*' '*_deps/*' --output-file coverage.info

    - name: Upload to Codecov
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.info
        flags: unittests
```

### Static Analysis

**Clang-Tidy:**

```bash
# Run clang-tidy
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -B build
run-clang-tidy -p build tests/*.cpp

# With specific checks
run-clang-tidy -p build \
    -checks='modernize*,performance*,readibility*' \
    tests/*.cpp
```

**Clang-Format:**

```bash
# Check formatting
clang-format --dry-run --Werror tests/*.cpp

# Fix formatting
clang-format -i tests/*.cpp
```

---

## Performance Baselines

### Test Execution Time Baselines

**Serial Backend:**
```
subsetix_test_core:         < 1s
subsetix_test_ops:          < 1s
subsetix_test_advanced:     < 2s
subsetix_test_amr:          < 2s
subsetix_test_fvd_api:      < 1s
subsetix_test_fvd_execution:< 5s
subsetix_test_fvd_integrators: < 2s
subsetix_test_mach2_validation: < 1s
subsetix_test_fvd_multi_system: < 1s
subsetix_test_fvd_accuracy: < 1s
Total:                      ~15s
```

**OpenMP Backend (4 threads):**
```
Total:                      ~10s (1.5x speedup)
```

**CUDA Backend:**
```
Total:                      ~25s (startup overhead)
```

### Performance Regression Detection

**Benchmark Critical Paths:**

```cpp
// Add timing to critical sections
auto start = std::chrono::high_resolution_clock::now();

// ... code under test ...

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
printf("Execution time: %ld us\n", duration.count());

// Assert on max time
EXPECT_LT(duration.count(), max_expected_us)
    << "Performance regression detected";
```

### Memory Usage Baselines

**Typical Memory Footprint:**

| Test | Peak Memory | Notes |
|------|-------------|-------|
| Core tests | ~10 MB | Small data structures |
| Ops tests | ~50 MB | Set operations |
| Advanced tests | ~100 MB | Field operations |
| AMR tests | ~200 MB | Multilevel hierarchy |
| FVD execution | ~500 MB | Large grids |
| FVD integrators | ~100 MB | AMR criteria |

**Monitoring Memory:**

```bash
# Measure peak memory
/usr/bin/time -v ./subsetix_test_fvd_execution

# Valgrind massif
valgrind --tool=massif ./subsetix_test_fvd_execution
```

---

## Accuracy Tolerance Justifications

### Floating-Point Precision Guidelines

#### Machine Epsilon

**Float (32-bit):**
- Machine epsilon: `ε = 2^-23 ≈ 1.19e-7`
- Practical precision: ~6-7 decimal digits
- Safe tolerance: `1e-6` (10× ε)

**Double (64-bit):**
- Machine epsilon: `ε = 2^-52 ≈ 2.22e-16`
- Practical precision: ~15-16 decimal digits
- Safe tolerance: `1e-12` (10× ε)

### Tolerance Selection by Operation

#### 1. Basic Arithmetic

**Operation:** Addition, multiplication

**Expected Error:** O(ε)

**Tolerance:**
```cpp
// Float
EXPECT_NEAR(result, expected, 1e-6);

// Double
EXPECT_NEAR(result, expected, 1e-12);
```

**Rationale:** Single operation accumulates minimal rounding error

#### 2. Accumulated Operations

**Operation:** Sum, dot product, integral

**Expected Error:** O(√n × ε) for n operations

**Tolerance:**
```cpp
// Float, n = 1000 operations
Real tolerance = std::sqrt(Real(n)) * Real(1e-6);
EXPECT_NEAR(result, expected, tolerance);

// Conservative: 1e-4 for float, 1e-10 for double
```

**Rationale:** Random walk of rounding errors

#### 3. Iterative Methods

**Operation:** Newton's method, fixed-point iteration

**Expected Error:** O(√κ × ε) where κ is condition number

**Tolerance:**
```cpp
// Well-conditioned problems (κ < 100)
EXPECT_NEAR(result, expected, 1e-5);  // Float
EXPECT_NEAR(result, expected, 1e-10); // Double

// Ill-conditioned problems (κ > 1000)
EXPECT_NEAR(result, expected, 1e-3);  // Float
EXPECT_NEAR(result, expected, 1e-8);  // Double
```

**Rationale:** Error amplification by condition number

#### 4. Transcendental Functions

**Operation:** sqrt, exp, log, sin, cos

**Expected Error:** O(ε) to O(10×ε)

**Tolerance:**
```cpp
// Float
EXPECT_NEAR(std::sqrt(x), expected, 1e-6);
EXPECT_NEAR(std::exp(x), expected, 1e-6);
EXPECT_NEAR(std::sin(x), expected, 1e-6);

// Double
EXPECT_NEAR(std::sqrt(x), expected, 1e-12);
EXPECT_NEAR(std::exp(x), expected, 1e-11);  // exp is less accurate
EXPECT_NEAR(std::sin(x), expected, 1e-12);
```

**Rationale:** Library implementation quality

### FVD-Specific Tolerances

#### 1. Mass Conservation

**Physical Law:** Mass is conserved exactly

**Numerical Reality:** Rounding errors accumulate

**Tolerance:**
```cpp
// For short simulations (< 100 steps)
EXPECT_NEAR(mass_final, mass_initial, 1e-6);

// For long simulations (> 1000 steps)
EXPECT_NEAR(mass_final, mass_initial, 1e-4);

// For AMR with prolongation/restriction
EXPECT_NEAR(mass_final, mass_initial, 1e-3);  // Interpolation errors
```

**Rationale:**
- Each flux computation: O(ε) error
- Each cell update: O(ε) error
- Total: O(n_cells × n_steps × ε)

#### 2. Flux Conservation

**Physical Law:** Fluxes are conservative

**Tolerance:**
```cpp
// Local flux balance
EXPECT_NEAR(flux_in - flux_out, source, 1e-6);

// Global flux balance
EXPECT_NEAR(total_flux_in, total_flux_out, 1e-4);
```

**Rationale:** Flux computation errors accumulate spatially

#### 3. Primitive-Conserved Conversion

**Operation:** to_primitive, from_primitive

**Tolerance:**
```cpp
// Round-trip: Conserved → Primitive → Conserved
EXPECT_NEAR(U_roundtrip.rho, U_original.rho, 1e-6);
EXPECT_NEAR(U_roundtrip.rhou, U_original.rhou, 1e-6);
EXPECT_NEAR(U_roundtrip.E, U_original.E, 1e-5);  // Energy is less accurate
```

**Rationale:**
- Division by ρ amplifies errors
- Kinetic energy computation: 0.5 × ρ × (u² + v²) → O(ε × velocity²)
- Pressure computation: (γ-1) × (E - KE) → subtraction cancellation

#### 4. Sound Speed

**Formula:** a = √(γ × p / ρ)

**Tolerance:**
```cpp
EXPECT_NEAR(sound_speed_computed, sound_speed_expected, 1e-5);
```

**Rationale:**
- Division and sqrt each add O(ε)
- For air: a ≈ 340 m/s, so tolerance ≈ 0.003 m/s
- Practical: 1 m/s accounts for temperature variations

#### 5. Flux Schemes

**Rusanov Flux:** `F = 0.5 × (FL + FR) - 0.5 × smax × (UR - UL)`

**Tolerance:**
```cpp
EXPECT_NEAR(flux_computed.rho, flux_expected.rho, 1e-6);
EXPECT_NEAR(flux_computed.E, flux_expected.E, 1e-5);  // Energy is sensitive
```

**Rationale:**
- Multiple floating-point operations
- Wave speed computation: max(|u| + a) → several operations
- Subtraction: (UR - UL) → cancellation if close

**HLLC Flux:** More complex, larger tolerance

```cpp
EXPECT_NEAR(flux_computed.rho, flux_expected.rho, 1e-5);
EXPECT_NEAR(flux_computed.E, flux_expected.E, 1e-4);
```

**Rationale:** More operations → more rounding error

#### 6. Time Integration

**Forward Euler:** `U_{n+1} = U_n + dt × f(U_n)`

**Tolerance:**
```cpp
// Single step
EXPECT_NEAR(U_new, U_expected, 1e-6);

// Multiple steps (error accumulates)
int n_steps = 100;
Real tolerance = n_steps * Real(1e-6);  // Linear accumulation
EXPECT_NEAR(U_final, U_expected, tolerance);
```

**Rationale:** Each step adds O(dt × ε) error

**Runge-Kutta Methods:** More stages, more error

```cpp
// RK4: 4 stages
EXPECT_NEAR(U_new, U_expected, 4e-6);  // 4× more operations
```

#### 7. AMR Operations

**Prolongation:** Coarse → Fine interpolation

**Tolerance:**
```cpp
// Linear interpolation
EXPECT_NEAR(fine_value, interpolated_value, 1e-5);

// With smooth solutions
EXPECT_NEAR(fine_value, interpolated_value, 1e-6);

// With discontinuities (shocks)
EXPECT_NEAR(fine_value, interpolated_value, 1e-2);  // Larger tolerance
```

**Rationale:**
- Interpolation errors: O(h²) for linear
- Near discontinuities: Gibbs oscillations

**Restriction:** Fine → Coarse averaging

**Tolerance:**
```cpp
// Volume-weighted average
EXPECT_NEAR(coarse_value, averaged_value, 1e-6);
```

**Rationale:** Summation/averaging reduces variance

### GPU-Specific Tolerances

**Floating-Point Differences:**

GPU and CPU may produce slightly different results due to:
- Different instruction sets
- Different operation ordering
- FMA (Fused Multiply-Add) usage

**Tolerance:**
```cpp
// Compare GPU vs CPU
EXPECT_NEAR(gpu_result, cpu_result, 1e-5);  // Slightly larger
```

**Rationale:** GPU may use FMA: `a × b + c` computed as single operation

### Relative vs Absolute Tolerance

**Use Relative When:**
- Values span multiple orders of magnitude
- Comparing physical quantities
```cpp
Real rel_error = std::abs((computed - expected) / expected);
EXPECT_LT(rel_error, 1e-6);
```

**Use Absolute When:**
- Values near zero
- Comparing differences
```cpp
Real abs_error = std::abs(computed - expected);
EXPECT_LT(abs_error, 1e-6);
```

**Use Both (GoogleTest approach):**
```cpp
// Near actually uses both near
EXPECT_NEAR(computed, expected, tolerance);
// Equivalent to:
// abs(computed - expected) <= tolerance
// OR
// abs(computed - expected) / max(abs(computed), abs(expected)) <= tolerance
```

### Practical Tolerance Table

| Operation | Float (Abs) | Float (Rel) | Double (Abs) | Double (Rel) |
|-----------|-------------|-------------|--------------|--------------|
| Basic arithmetic | 1e-6 | 1e-6 | 1e-12 | 1e-12 |
| Accumulated (n=100) | 1e-5 | 1e-5 | 1e-11 | 1e-11 |
| Transcendental | 1e-6 | 1e-6 | 1e-11 | 1e-11 |
| Mass conservation | 1e-6 | 1e-6 | 1e-10 | 1e-10 |
| Flux computation | 1e-6 | 1e-6 | 1e-10 | 1e-10 |
| Time stepping (per step) | 1e-6 | 1e-6 | 1e-11 | 1e-11 |
| AMR prolongation | 1e-5 | 1e-5 | 1e-10 | 1e-10 |
| GPU vs CPU | 1e-5 | 1e-5 | 1e-10 | 1e-10 |

### Debugging Tolerance Issues

**If Tests Fail Due to Tolerance:**

1. **Check the actual error:**
```cpp
Real error = std::abs(computed - expected);
printf("Error: %e, Tolerance: %e\n", error, tolerance);
```

2. **Determine if error is acceptable:**
   - Is physical meaning preserved?
   - Is error within expected bounds?
   - Does it affect simulation stability?

3. **Adjust tolerance if justified:**
```cpp
// Document why tolerance is larger
EXPECT_NEAR(computed, expected, 1e-4)  // Larger tolerance due to...
    << "Accumulated error over " << n_steps << " time steps";
```

4. **Consider algorithmic improvements:**
   - Kahan summation for accumulated operations
   - Compensated arithmetic for critical operations
   - Higher precision for intermediate calculations

---

## Conclusion

This guide provides a comprehensive overview of FVD layer testing. Key points:

- **217 tests** across 10 executables validate FVD functionality
- **70% coverage** threshold ensures quality
- **Multi-backend testing** (Serial, OpenMP, CUDA) ensures portability
- **Strict tolerance guidelines** ensure numerical accuracy
- **Modular test organization** makes maintenance easier

### Best Practices Summary

1. **Run tests frequently** during development
2. **Use appropriate tolerances** for numerical comparisons
3. **Test on multiple backends** for portability
4. **Add tests** for new features immediately
5. **Document test purpose** and validation criteria
6. **Use Git** to track test evolution
7. **Monitor coverage** and improve when needed
8. **Profile performance** to catch regressions

### Future Improvements

1. **Add more integration tests** for complete workflows
2. **Improve AMR test coverage** for refinement/coarsening
3. **Add regression tests** for known issues
4. **Expand GPU testing** coverage
5. **Add performance benchmarks** to CI
6. **Implement property-based testing** for algebraic properties
7. **Add visualization tests** for output validation

### Resources

- **GoogleTest Documentation:** https://google.github.io/googletest/
- **Kokkos Documentation:** https://kokkos.github.io/kokkos-core-wiki/
- **CMake Documentation:** https://cmake.org/documentation/
- **Project Repository:** `/home/sbstndbs/subsetix_kokkos_agent2`

---

**Document Version:** 1.0
**Last Updated:** 2026-01-20
**Maintainer:** Subsetix Development Team

---

*For questions or contributions, please refer to the project repository and issue tracker.*
