# FVD High-Level API - Missing Features Implementation Plan

**Version**: 1.0
**Date**: 2025-01-20
**Status**: Ready for Implementation
**Estimated Duration**: 8 weeks (critical path)

---

## Executive Summary

This document provides a comprehensive implementation plan for completing the FVD (Finite Volume Discretization) high-level API by adding five critical missing features:

1. **Multi-level AMR Access** - Expose intermediate levels, prolong/restrict operations
2. **Custom Refinement Criteria API** - User-defined refinement indicators
3. **Multi-level VTK Export** - Visualization support for AMR hierarchies
4. **Extended Stencil for MUSCL** - Proper 2nd order reconstruction
5. **Adaptive Time Stepping with AMR** - Multi-rate time integration

### Key Finding

The foundation is **80% complete**. CSR AMR operations in `/include/subsetix/csr_ops/amr.hpp` and `/include/subsetix/csr_ops/field_amr.hpp` are production-ready. The gap is primarily in high-level API exposure and integration.

---

## Current State Assessment

### Existing Infrastructure (Ready to Use)

| Component | Location | Status | Notes |
|-----------|----------|--------|-------|
| Multilevel containers | `multilevel/multilevel.hpp` | ✅ Complete | `MultilevelGeo`, `MultilevelField` with deep_copy |
| AMR geometry ops | `csr_ops/amr.hpp` | ✅ Production | `refine_level_up_device`, `project_level_down_device` |
| Field AMR ops | `csr_ops/field_amr.hpp` | ✅ Production | `prolong_field_on_subset_device`, `restrict_field_on_subset_device` |
| VTK export | `io/vtk_export.hpp` | ✅ Multi-level | `write_multilevel_field_vtk` exists |
| Refinement criteria | `fvd/amr/refinement_criteria.hpp` | ✅ Complete | Multiple criteria + composite support |
| Adaptive solver | `fvd/solver/adaptive_solver.hpp` | ⚠️ 80% | Has AMR level storage, missing integration |
| Time integrators | `fvd/time/time_integrators.hpp` | ✅ Complete | RK1-RK4, SSPRK3, time step controller |
| Reconstruction | `fvd/reconstruction/reconstruction.hpp` | ⚠️ Partial | Has MUSCL, needs extended stencil |

### What's Missing

1. **Multi-level AMR access API**: Solver has internal `AmrLevel` array but no public accessors
2. **Custom refinement criteria**: Infrastructure exists but no user-extension mechanism
3. **Multi-level VTK integration**: Export function exists but not integrated with solver
4. **Extended MUSCL stencil**: Current MUSCL uses 3-point stencil, needs proper multi-level support
5. **Multi-rate time integration**: TimeStepController exists but not integrated with AMR levels

---

## Phase 1: Multi-level AMR Access (Week 1-2)

**Priority**: HIGH | **Complexity**: LOW | **Dependencies**: None

### Objectives

Expose the AMR hierarchy to users through a clean API, enabling inspection and manipulation of intermediate levels.

### Implementation Tasks

#### 1.1 Add Public Accessors to `AdaptiveSolver`

**Location**: `include/subsetix/fvd/solver/adaptive_solver.hpp`

**Add methods:**
```cpp
// Get geometry at specific level
const csr::IntervalSet2DDevice& get_level_geometry(int level) const;

// Get solution at specific level
const Kokkos::View<Conserved*>& get_level_solution(int level) const;
Kokkos::View<Conserved*>& get_level_solution_mutable(int level);

// Get number of active levels
int get_num_levels() const;

// Get multilevel container (for VTK export)
MultilevelFieldDevice<Conserved> get_multilevel_solution() const;
MultilevelGeoDevice get_multilevel_geometry() const;

// Get cell size (dx) for a specific level
Real level_dx(int level) const;

// Check if a level is active
bool is_level_active(int level) const;
```

#### 1.2 Implement AmrOperations Wrapper

**Location**: New file `include/subsetix/fvd/amr/amr_operations.hpp`

Wrap existing CSR operations in FVD-friendly API:
```cpp
template<typename System>
class AmrOperations {
public:
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;

    // Prolong from coarse to fine level
    static void prolong_level(
        const Kokkos::View<Conserved*>& coarse,
        const csr::IntervalSet2DDevice& coarse_geom,
        Kokkos::View<Conserved*>& fine,
        const csr::IntervalSet2DDevice& fine_geom,
        bool use_linear_prediction = false
    );

    // Restrict from fine to coarse level
    static void restrict_level(
        const Kokkos::View<Conserved*>& fine,
        const csr::IntervalSet2DDevice& fine_geom,
        Kokkos::View<Conserved*>& coarse,
        const csr::IntervalSet2DDevice& coarse_geom
    );
};
```

#### 1.3 Add Multi-level Diagnostics

**Location**: `include/subsetix/fvd/solver/observer.hpp`

Extend observer system for per-level diagnostics:
```cpp
struct AmrLevelDiagnostics {
    int level;
    std::size_t n_cells;
    Real min_rho, max_rho;
    Real total_mass;
    Real refinement_fraction;
};
```

#### 1.4 Testing

**Location**: `tests/amr_level_access_test.cpp`

Unit tests:
- Accessor methods return correct data
- Prolong operation conserves mass within 1e-10
- Restrict operation conserves mass within 1e-10
- Diagnostics report correct cell counts per level
- All backends (Serial, OpenMP, CUDA) compile

### Validation Criteria

- [ ] Can access all levels 0 to finest_level
- [ ] Prolong operation conserves mass within 1e-10
- [ ] Restrict operation conserves mass within 1e-10
- [ ] Diagnostics report correct cell counts per level
- [ ] All backends compile and pass tests

### Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| AmrLevel structure is private | Already public for CUDA compatibility (lines 2083-2095) |
| Level numbering confusion | Document clearly, use consistent "0=coarsest" convention |
| References invalidated after remesh | Document that level references are invalidated by `remesh()` |

---

## Phase 2: Custom Refinement Criteria API (Week 2-3)

**Priority**: MEDIUM | **Complexity**: MEDIUM | **Dependencies**: None

### Objectives

Enable users to define custom refinement indicators beyond the built-in criteria (Gradient, ShockSensor, Vorticity).

### Implementation Tasks

#### 2.1 Design User-Defined Criterion Interface

**Location**: `include/subsetix/fvd/amr/refinement_criteria.hpp`

Add concept and wrapper:
```cpp
// Concept for user-defined refinement functions
template<typename F, typename System>
concept RefinementFunction =
    std::is_trivially_copyable_v<F> &&
    requires(const F& func,
             const typename System::Conserved& U,
             const typename System::Primitive& q,
             const typename System::RealType dx)
    {
        { func(U, q, dx) } -> std::convertible_to<RefinementAction>;
    };

// User-defined criterion wrapper
template<typename System, typename Function>
    requires RefinementFunction<Function, System>
class UserDefinedCriterion {
public:
    Function func;

    KOKKOS_INLINE_FUNCTION
    RefinementAction evaluate(const Conserved& U, const Primitive& q, Real dx) const {
        return func(U, q, dx);
    }
};

// Factory function with CTAD
template<typename System, RefinementFunction<System> Function>
KOKKOS_INLINE_FUNCTION
auto make_refinementCriterion(Function&& func) {
    return UserDefinedCriterion<System, std::decay_t<Function>>(func);
}
```

#### 2.2 Integration with CompositeCriterion

**Location**: `include/subsetix/fvd/amr/refinement_criteria.hpp`

Add user criterion storage to `CompositeCriterion`:
```cpp
template<typename System, int MaxCriteria = 8>
class CompositeCriterion {
public:
    // Add user-defined criterion storage
    struct UserCriterionStorage {
        using EvaluateFn = RefinementAction(*)(const Conserved&, const Primitive&, Real);
        EvaluateFn evaluate_fn = nullptr;
        Real user_data[16] = {0};
    };

    UserCriterionStorage<System> user_criteria[MaxCriteria];
    int8_t num_user_criteria = 0;

    int add_user_criterion(typename UserCriterionStorage<System>::EvaluateFn fn);
};
```

#### 2.3 Example Custom Criteria

**Location**: `examples/amr_custom_criteria.cpp`

Demonstrate:
- Curvature-based refinement
- Vorticity magnitude
- Q-criterion (for vortex cores)
- User-defined scalar field

#### 2.4 Testing

**Location**: `tests/custom_criteria_test.cpp`

- Compile-time tests for concept satisfaction
- Runtime tests with custom criteria
- Validation that refinement tags are correct
- GPU compatibility tests

### Validation Criteria

- [ ] User-defined criteria compile and run on all backends
- [ ] CompositeCriterion can mix built-in and user criteria
- [ ] Example custom criteria refine appropriate regions
- [ ] No performance regression vs built-in criteria

### Usage Example

```cpp
// Define custom criterion: refine where Mach > 0.8
auto mach_criterion = make_refinementCriterion<Euler2D<float>>(
    KOKKOS_LAMBDA(const auto& U, const auto& q, float dx) {
        constexpr float gamma = 1.4f;
        float a = Kokkos::sqrt(gamma * q.p / (q.rho + 1e-10f));
        float mach = Kokkos::sqrt(q.u*q.u + q.v*q.v) / (a + 1e-10f);
        return (mach > 0.8f) ? RefinementAction::Refine
                             : RefinementAction::Keep;
    }
);

// Use with solver
RefinementConfig<Euler2D<float>> config;
config.criterion.add_user_criterion(mach_criterion);
solver.set_refinement(config);
```

### Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| Type erasure breaks GPU compatibility | Use compile-time polymorphism with concepts |
| User criteria crash on device | Compile-time concept checks, clear documentation |

---

## Phase 3: Multi-level VTK Export (Week 3-4)

**Priority**: HIGH | **Complexity**: LOW | **Dependencies**: Phase 1

### Objectives

Integrate VTK export with the AdaptiveSolver for visualization of complete AMR hierarchies.

### Implementation Tasks

#### 3.1 Add VTK Export to AdaptiveSolver

**Location**: `include/subsetix/fvd/solver/adaptive_solver.hpp`

```cpp
void write_multilevel_vtk(
    const std::string& filename_base,
    const std::string& variable_name = "density"
) const;

void write_level_vtk(int level, const std::string& filename) const;
```

#### 3.2 Enhanced VTK Export Implementation

**Location**: New file `include/subsetix/fvd/output/vtk_export.hpp`

Support both legacy and modern formats:
```cpp
enum class VTKFormat {
    LEGACY_BINARY,  // Current .vtk binary format
    VTU_XML         // VTK XML unstructured grid
};

struct VTKExportConfig {
    VTKFormat format = VTKFormat::VTU_XML;
    bool use_physical_coordinates = true;
    bool append_level = true;
    bool append_refinement_ratio = true;
};
```

#### 3.3 Integration with Observer

Auto-export on observer callback:
```cpp
solver.on_progress([](auto state) {
    if (state.step % 100 == 0)
        solver.write_multilevel_vtk("output/frame_" + std::to_string(state.step));
});
```

#### 3.4 Testing

**Location**: `tests/vtk_export_test.cpp`

- Export test case with known output
- Verify Paraview can load files
- Check level boundaries are correct
- Validate field values

### Validation Criteria

- [ ] VTK files load correctly in Paraview/VisIt
- [ ] All levels are present in output
- [ ] Cell counts match solver state
- [ ] Field values match solver within 1e-6
- [ ] Multiple variables export correctly

### Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| Memory pressure from host copy | Stream levels sequentially, use compression |
| VTK format incompatibilities | Support both legacy and XML formats |

---

## Phase 4: Extended Stencil for MUSCL (Week 4-6)

**Priority**: MEDIUM | **Complexity**: HIGH | **Dependencies**: Phase 1

### Objectives

Implement proper 2nd order MUSCL reconstruction with extended stencil support for multi-level AMR.

### Background

Current MUSCL implementation uses 3-point stencil (left, center, right). For proper 2nd order accuracy with AMR, need:
- Extended stencil (5-point or 9-point) near refinement boundaries
- Proper handling of coarse-fine interfaces
- Slope limiting across level boundaries

### Implementation Tasks

#### 4.1 Design Extended Stencil Framework

**Location**: `include/subsetix/fvd/reconstruction/reconstruction.hpp`

```cpp
template<template<typename> class Limiter, int StencilWidth = 3>
struct MUSCL_ReconstructionExtended {
    static constexpr int stencil_width = StencilWidth;

    // 5-point reconstruction
    template<typename Real>
    KOKKOS_INLINE_FUNCTION
    static Real reconstruct_left_extended(
        Real U_left2, Real U_left1, Real U_center,
        Real U_right1, Real U_right2
    );

    // 9-point reconstruction (2D)
    template<typename Real>
    KOKKOS_INLINE_FUNCTION
    static void reconstruct_2d_extended(
        const Primitive& q_ww, const Primitive& q_w,
        const Primitive& q_c,
        const Primitive& q_e, const Primitive& q_ee,
        const Primitive& q_sw, const Primitive& q_s,
        const Primitive& q_n, const Primitive& q_ne,
        Primitive& qL, Primitive& qR
    );
};
```

#### 4.2 Implement Extended CSR Stencil

**Location**: `include/subsetix/csr_ops/field_stencil.hpp`

Add 9-point stencil support:
```cpp
template <typename T>
struct CsrStencilPoint9 {
    std::size_t idx_center;
    std::size_t idx_west, idx_east;
    std::size_t idx_south, idx_north;
    std::size_t idx_west_west, idx_east_east;   // NEW: i±2
    std::size_t idx_south_south, idx_north_north; // NEW: j±2
};
```

#### 4.3 AMR-Aware Reconstruction

Handle coarse-to-fine transitions:
```cpp
template<typename Real>
KOKKOS_INLINE_FUNCTION
static void reconstruct_amr_interface(
    const Primitive& q_coarse,
    const Primitive& q_fine_L,
    const Primitive& q_fine_R,
    Primitive& q_reconstructed,
    Real level_ratio
);
```

#### 4.4 Extended Limiters

Add limiters for wider stencils:
- Minmod5: 5-point minmod
- WENO: Weighted Essentially Non-Oscillatory
- MP: Monotonicity Preserving

#### 4.5 Testing

**Location**: `tests/extended_stencil_test.cpp`

- Convergence tests: verify 2nd order on smooth solutions
- Shock tests: verify non-oscillatory property
- AMR tests: verify accuracy at level boundaries

### Validation Criteria

- [ ] 2nd order convergence on smooth problems (error ∝ h²)
- [ ] No oscillations on discontinuous problems (TVD property)
- [ ] Correct reconstruction at coarse-fine interfaces
- [ ] Conservation maintained
- [ ] Performance < 2x slower than 3-point stencil

### Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| Complexity explosion with level-aware logic | Start with single-level extended stencil |
| Performance degradation | Make extended stencil optional, benchmark |

---

## Phase 5: Adaptive Time Stepping with AMR (Week 6-8)

**Priority**: MEDIUM | **Complexity**: HIGH | **Dependencies**: Phase 1, Phase 4

### Objectives

Implement multi-rate time integration where different AMR levels use different time steps for efficiency.

### Background

Classic AMR time stepping:
- Coarse levels: larger time step
- Fine levels: smaller time step
- Synchronization at time boundaries
- Sub-cycling in time (flux correction)

### Implementation Tasks

#### 5.1 Design Multi-Rate Time Integrator

**Location**: New file `include/subsetix/fvd/time/multirate_integrators.hpp`

```cpp
template<typename System, typename BaseIntegrator>
class MultiRateIntegrator {
public:
    struct LevelConfig {
        int level;
        int subcycles;  // 2^level substeps
        Real dt;
        Real t;         // Current time for this level
    };

    void step(
        MultilevelFieldDevice<Conserved>& U,
        const MultilevelGeoDevice& geometries,
        Real dt_coarse
    );
};
```

#### 5.2 Flux Correction at Coarse-Fine Boundaries

**Location**: New file `include/subsetix/fvd/time/flux_correction.hpp`

```cpp
template<typename System>
class FluxRegister {
public:
    void accumulate_fine_flux(
        int fine_level,
        const Kokkos::View<Conserved**>& fine_flux_x,
        const Kokkos::View<Conserved**>& fine_flux_y,
        Real dt_fine
    );

    void apply_coarse_correction(
        int coarse_level,
        Kokkos::View<Conserved*>& coarse_U
    );
};
```

#### 5.3 Time Step Synchronization

```cpp
void sync_time_steps(
    std::array<Real, MAX_LEVELS>& dt_level,
    Real dt_global
);

Real compute_dt_for_level(int level, Real cfl_target);
```

#### 5.4 Integration with AdaptiveSolver

```cpp
struct TimeStepConfig {
    bool multi_rate = false;
    int max_subcycles = 4;
    bool reflux_correction = true;
};
```

#### 5.5 Testing

**Location**: `tests/multirate_time_test.cpp`

- Conservation tests (mass, momentum, energy)
- Accuracy tests vs single-rate
- Stability tests for various CFL numbers
- Performance benchmarks

### Validation Criteria

- [ ] Global conservation maintained (drift < 1e-10)
- [ ] Accuracy comparable to single-rate (error within 10%)
- [ ] Stable for CFL up to 0.8
- [ ] Speedup > 1.5x for appropriate problems
- [ ] No spurious oscillations at time boundaries

### Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| Conservation violation from flux correction | Careful flux accumulation, conservative refluxing |
| Complexity in time synchronization | Start with 2-level case, generalize |
| Instability from mismatched time steps | Restrict CFL, add sub-cycling limits |

---

## Implementation Order & Dependencies

```
Week 1-2: Phase 1 (Multi-level AMR Access)
    ↓
Week 2-3: Phase 2 (Custom Refinement) [parallel with Phase 1]
    ↓
Week 3-4: Phase 3 (VTK Export) [depends on Phase 1]
    ↓
Week 4-6: Phase 4 (Extended Stencil) [depends on Phase 1]
    ↓
Week 6-8: Phase 5 (Multi-Rate Time) [depends on Phase 1, Phase 4]
```

**Critical Path**: 8 weeks
**Parallelizable**: Phases 2 and 4 can overlap with Phase 3

---

## File Structure Changes

### New Files

```
include/subsetix/fvd/
├── amr/
│   ├── amr_operations.hpp          # Phase 1
│   └── user_criteria.hpp           # Phase 2
├── reconstruction/
│   └── extended_stencil.hpp        # Phase 4
├── time/
│   ├── multirate_integrators.hpp   # Phase 5
│   └── flux_correction.hpp         # Phase 5
└── output/
    └── vtk_export.hpp              # Phase 3

examples/
├── amr_custom_criteria.cpp         # Phase 2
├── multilevel_vtk_demo.cpp         # Phase 3
├── extended_stencil_demo.cpp       # Phase 4
└── multirate_time_demo.cpp         # Phase 5

tests/
├── amr_level_access_test.cpp       # Phase 1
├── custom_criteria_test.cpp        # Phase 2
├── vtk_export_test.cpp             # Phase 3
├── extended_stencil_test.cpp       # Phase 4
└── multirate_time_test.cpp         # Phase 5
```

### Modified Files

```
include/subsetix/fvd/
├── solver/
│   └── adaptive_solver.hpp         # All phases
├── amr/
│   └── refinement_criteria.hpp     # Phase 2
└── reconstruction/
    └── reconstruction.hpp          # Phase 4

include/subsetix/csr_ops/
└── field_stencil.hpp               # Phase 4 (9-point support)
```

---

## Testing Strategy

### Unit Tests (Each Phase)
- Compile-time tests (concepts, static assertions)
- GPU compatibility tests (Serial, OpenMP, CUDA)
- Numerical accuracy tests (known solutions)

### Integration Tests
- 2D advection with AMR
- 2D Euler equations (shock tube)
- Mach 2 cylinder (production case)

### Regression Tests
- Compare against reference solutions
- Validate conservation properties
- Check convergence rates

### Performance Tests
- Measure speedup from multi-rate time stepping
- Compare extended vs 3-point stencil
- Benchmark VTK export overhead

---

## Success Criteria

### Phase 1: Multi-level AMR Access
- [ ] Users can access all AMR levels programmatically
- [ ] Prolong/restrict operations are exposed and tested
- [ ] Per-level diagnostics available
- [ ] All examples compile on all backends

### Phase 2: Custom Refinement Criteria
- [ ] User-defined criteria compile and run
- [ ] Field-based refinement works
- [ ] Example criteria demonstrate utility
- [ ] No performance regression

### Phase 3: Multi-level VTK Export
- [ ] VTK files export complete AMR hierarchy
- [ ] Paraview can visualize multi-level data
- [ ] Multiple variables export correctly
- [ ] Automated export works

### Phase 4: Extended Stencil for MUSCL
- [ ] 5-point stencil achieves 2nd order accuracy
- [ ] AMR-aware reconstruction works
- [ ] WENO limiters available
- [ ] Performance acceptable (< 2x overhead)

### Phase 5: Adaptive Time Stepping with AMR
- [ ] Multi-rate integration conserves mass
- [ ] Sub-cycling algorithm stable
- [ ] Refluxing maintains accuracy
- [ ] Demonstrated speedup > 1.5x

---

## Critical Files for Implementation

| File | Purpose | Lines to Modify |
|------|---------|-----------------|
| `fvd/solver/adaptive_solver.hpp` | Core solver with AMR level storage | 2083-2095 (add accessors) |
| `csr_ops/field_amr.hpp` | Production-ready prolong/restrict | Wrap in AmrOperations |
| `fvd/amr/refinement_criteria.hpp` | Refinement criteria framework | Add user criteria support |
| `io/vtk_export.hpp` | Existing multilevel VTK export | Integrate with solver |
| `fvd/reconstruction/reconstruction.hpp` | MUSCL reconstruction | Extend to 5/9-point |

---

## Conclusion

This implementation plan addresses all five missing features in the FVD high-level API with:
- Clear prioritization based on impact and complexity
- Phased approach with 8-week critical path
- Risk mitigation for identified blockers
- Comprehensive testing strategy
- Documentation and user support

The plan leverages existing production-ready infrastructure and focuses primarily on API design and integration rather than low-level algorithm development.

**Recommendation**: Start with Phase 1 (Multi-level AMR Access) as it enables all subsequent phases and provides immediate user value.
