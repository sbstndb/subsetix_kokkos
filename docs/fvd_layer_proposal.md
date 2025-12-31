# Proposal: Finite Volume Dynamics (FVD) Layer for Subsetix

**Date**: 2025-12-30
**Authors**: Subsetix team
**Status**: Design Proposal

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Analysis of MACH2 Implementation](#analysis-of-mach2-implementation)
3. [Design Goals](#design-goals)
4. [Architecture Overview](#architecture-overview)
5. [Detailed API Design](#detailed-api-design)
6. [Stencil Strategy](#stencil-strategy)
7. [AMR Integration](#amr-integration)
8. [Implementation Roadmap](#implementation-roadmap)
9. [Code Examples](#code-examples)
10. [Comparison Matrix](#comparison-matrix)

---

## Executive Summary

This proposal outlines the design of a **Finite Volume Dynamics (FVD)** layer for the Subsetix library. The goal is to extract and generalize the CFD mechanisms currently hardcoded in the MACH2 example, creating a reusable, configurable framework for 2D compressible flow simulations on sparse AMR grids.

### Key Objectives

- **Multi-physics support**: Generic system for Euler, Navier-Stokes, Advection, etc.
- **Template-based**: Compile-time configuration for N conserved variables
- **Complete solver**: Time stepping, fluxes, BCs, and AMR integration
- **Separate module**: New `include/subsetix/fvd/` directory
- **Performance**: SoA (Structure of Arrays) storage for optimal cache usage

### What Subsetix Already Provides

| Component | Description |
|-----------|-------------|
| `IntervalSet2D` | CSR-based sparse 2D geometry |
| `Field2D<T>` | Fields defined on sparse geometries |
| `apply_csr_stencil_on_set_device` | 5-point stencil operations |
| AMR operations | `refine_level_up_device`, `prolong_field_on_set_device`, `restrict_field_on_set_device` |
| VTK export | `write_legacy_quads` |

---

## Analysis of MACH2 Implementation

### Location
`examples/mach2_cylinder/mach2_cylinder.cpp` (~2000 lines)

### What is Hardcoded

#### 1. Physics-Specific Structures (Lines 72-323)

```cpp
// Hardcoded Euler variables
struct Conserved { Real rho, rhou, rhov, E; };
struct Primitive { Real rho, u, v, p; };

// Hardcoded ideal gas EOS
KOKKOS_INLINE_FUNCTION
Primitive cons_to_prim(const Conserved& U, Real gamma);

KOKKOS_INLINE_FUNCTION
Real sound_speed(const Primitive& q, Real gamma);

// Hardcoded Rusanov flux
KOKKOS_INLINE_FUNCTION
Conserved rusanov_flux_x(const Conserved& UL, const Conserved& UR,
                         const Primitive& qL, const Primitive& qR,
                         Real gamma);
```

#### 2. Boundary Conditions (Lines 426-552)

```cpp
// Hardcoded BC types
enum class BcKind { Inlet, Outlet, WallBottom, WallTop };

// Monolithic fill_ghost_cells function
void fill_ghost_cells(ConservedFields& field,
                      const IntervalSet2DDevice& ghost_mask,
                      const IntervalSet2DDevice& base_mask,
                      const Box2D& domain,
                      const Conserved& inflow,
                      Real gamma,
                      bool no_slip);
```

#### 3. Numerical Scheme (Lines 1057-1119)

```cpp
// Monolithic stencil with inline flux computation
struct EulerStencilSoA {
  Real gamma;
  Real dt;
  Real dx, dy;
  ConservedViews U_in, U_out;

  KOKKOS_INLINE_FUNCTION
  Real operator()(Coord, Coord, const CsrStencilPoint<Real>& p) const {
    // Everything hardcoded: Rusanov flux, Euler update, etc.
  }
};
```

#### 4. Time Stepping (Lines 550-589)

```cpp
// Hardcoded CFL computation
Real compute_dt(const ConservedFields& U, Real gamma, Real cfl,
                Real dx, Real dy);
```

#### 5. AMR/Remeshing (Lines 606-869)

```cpp
// Hardcoded gradient-based refinement indicator
struct IndicatorStencil {
  Real inv_dx, inv_dy;
  // Computes |grad(rho)|
};

// Problem-specific mask building
IntervalSet2DDevice build_refine_mask(const ConservedFields& U,
                                      const IntervalSet2DDevice& fluid_dev,
                                      const Box2D& domain,
                                      const RunConfig& cfg,
                                      CsrSetAlgebraContext& ctx);
```

### Summary of Issues

| Issue | Impact |
|-------|--------|
| Physics and numerical scheme are tightly coupled | Cannot reuse for other systems |
| BCs are hardcoded | Difficult to add new boundary conditions |
| No abstraction for flux schemes | Cannot switch between Rusanov, HLLC, etc. |
| AMR is mixed with Euler-specific code | Cannot use for other physics |

---

## Design Goals

Based on user requirements:

1. **Multi-physics support**: Support Euler, Navier-Stokes, Advection-Diffusion, etc.
2. **Generic system**: Template on N conserved variables
3. **Complete solver + building blocks**: Both high-level solver and low-level components
4. **AMR integration**: Seamless integration with Subsetix AMR
5. **Separate module**: New `include/subsetix/fvd/` directory
6. **2D compressible only**: Focus on 2D compressible flow (Euler/NS)
7. **SoA storage**: Keep the 4 separate fields approach for performance

---

## Architecture Overview

### Directory Structure

```
include/subsetix/
├── fvd/                                    # Finite Volume Dynamics module
│   ├── config.hpp                          # Configuration, basic types
│   ├── system.hpp                          # SystemDescription concept
│   ├── systems/
│   │   ├── euler2d.hpp                     # Euler2D system implementation
│   │   ├── advection2d.hpp                 # Scalar advection (future)
│   │   └── navier_stokes2d.hpp             # Navier-Stokes (future)
│   ├── flux.hpp                            # NumericalFlux concept + implementations
│   │   # (included: Rusanov, HLLC, Central)
│   ├── boundary.hpp                        # Boundary conditions
│   ├── fields.hpp                          # SoA multi-variable fields
│   ├── solver.hpp                          # Main FiniteVolumeSolver class
│   ├── time_stepper.hpp                    # Time integration methods
│   └── amr/
│       ├── error_indicator.hpp             # Error estimators
│       ├── refinement_strategy.hpp         # Refinement criteria
│       └── adaptive_solver.hpp             # AMR-integrated solver
```

### Component Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                     FiniteVolumeSolver                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   System    │  │   Flux      │  │   BoundaryConfig    │ │
│  │ (Euler2D)   │  │  (Rusanov)  │  │  (Inlet/Outlet/etc) │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                     │             │
│         └────────────────┴─────────────────────┘             │
│                            │                                 │
│                            ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              MultiVariableField (SoA)                │   │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐               │   │
│  │  │ rho  │ │ rhou │ │ rhov │ │  E   │  Field2D<Real> │   │
│  │  └──────┘ └──────┘ └──────┘ └──────┘               │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                 │
│                            ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              apply_fv_stencil_on_set_device          │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Subsetix Core                          │
│  IntervalSet2D | Field2D | AMR ops | Stencil ops | VTK     │
└─────────────────────────────────────────────────────────────┘
```

---

## Detailed API Design

### 1. System Description Concept

```cpp
// fvd/system.hpp
namespace subsetix::fvd {

/// Concept for a physical system description
template<typename T, int N>
concept SystemDescription = requires(T t) {
  // Number of conserved variables
  { T::num_vars } -> std::convertible_to<int>;

  // Conserved variable type
  typename T::Conserved;

  // Primitive variable type (optional)
  typename T::Primitive;

  // Conversion: conserved -> primitive
  { T::to_primitive(std::declval<typename T::Conserved>()) }
    -> std::same_as<typename T::Primitive>;

  // Conversion: primitive -> conserved
  { T::from_primitive(std::declval<typename T::Primitive>()) }
    -> std::same_as<typename T::Conserved>;

  // Equation of state: pressure from conserved
  { T::pressure(std::declval<typename T::Conserved>()) }
    -> std::same_as<Real>;

  // Sound speed from primitive
  { T::sound_speed(std::declval<typename T::Primitive>()) }
    -> std::same_as<Real>;

  // Physical flux in x-direction
  { T::flux_x(std::declval<typename T::Conserved>(),
              std::declval<typename T::Primitive>()) }
    -> std::same_as<typename T::Conserved>;

  // Physical flux in y-direction
  { T::flux_y(std::declval<typename T::Conserved>(),
              std::declval<typename T::Primitive>()) }
    -> std::same_as<typename T::Conserved>;
};

} // namespace subsetix::fvd
```

### 2. Euler2D System Implementation

```cpp
// fvd/systems/euler2d.hpp
#pragma once

#include <subsetix/fvd/system.hpp>
#include <cmath>

namespace subsetix::fvd {

/// 2D compressible Euler equations
struct Euler2D {
  static constexpr int num_vars = 4;
  static constexpr Real default_gamma = 1.4;

  // Conserved variables: {rho, rhou, rhov, E}
  struct Conserved {
    Real rho, rhou, rhov, E;
  };

  // Primitive variables: {rho, u, v, p}
  struct Primitive {
    Real rho, u, v, p;
  };

  Real gamma = default_gamma;

  // Convert conserved to primitive
  KOKKOS_INLINE_FUNCTION
  static Primitive to_primitive(const Conserved& U, Real g = default_gamma) {
    constexpr Real eps = 1e-12;
    Primitive q;
    q.rho = U.rho;
    Real inv_rho = 1.0 / (U.rho + eps);
    q.u = U.rhou * inv_rho;
    q.v = U.rhov * inv_rho;
    Real kinetic = 0.5 * (q.u * q.u + q.v * q.v);
    Real pressure = (g - 1.0) * (U.E - U.rho * kinetic);
    q.p = (pressure > eps) ? pressure : eps;
    return q;
  }

  // Convert primitive to conserved
  KOKKOS_INLINE_FUNCTION
  static Conserved from_primitive(const Primitive& q, Real g = default_gamma) {
    Conserved U;
    Real kinetic = 0.5 * q.rho * (q.u * q.u + q.v * q.v);
    U.rho = q.rho;
    U.rhou = q.rho * q.u;
    U.rhov = q.rho * q.v;
    U.E = q.p / (g - 1.0) + kinetic;
    return U;
  }

  // Pressure from conserved
  KOKKOS_INLINE_FUNCTION
  static Real pressure(const Conserved& U, Real g = default_gamma) {
    Primitive q = to_primitive(U, g);
    return q.p;
  }

  // Sound speed from primitive
  KOKKOS_INLINE_FUNCTION
  static Real sound_speed(const Primitive& q, Real g = default_gamma) {
    constexpr Real eps = 1e-12;
    return std::sqrt(g * q.p / (q.rho + eps));
  }

  // Physical flux in x-direction
  KOKKOS_INLINE_FUNCTION
  static Conserved flux_x(const Conserved& U, const Primitive& q) {
    return {
      U.rhou,                          // mass flux
      U.rho * q.u * q.u + q.p,         // x-momentum flux
      U.rho * q.u * q.v,               // y-momentum flux
      (U.E + q.p) * q.u                // energy flux
    };
  }

  // Physical flux in y-direction
  KOKKOS_INLINE_FUNCTION
  static Conserved flux_y(const Conserved& U, const Primitive& q) {
    return {
      U.rhov,                          // mass flux
      U.rho * q.u * q.v,               // x-momentum flux
      U.rho * q.v * q.v + q.p,         // y-momentum flux
      (U.E + q.p) * q.v                // energy flux
    };
  }
};

// Concept check
static_assert(SystemDescription<Euler2D, 4>);

} // namespace subsetix::fvd
```

### 3. Numerical Flux Concept

```cpp
// fvd/flux.hpp
namespace subsetix::fvd {

/// Concept for a numerical flux scheme
template<typename Flux, typename System>
concept NumericalFlux = requires(Flux flux,
                                  typename System::Conserved UL, UR,
                                  typename System::Primitive qL, qR) {
  // Compute numerical flux at x-interface
  { flux.flux_x(UL, UR, qL, qR) }
    -> std::same_as<typename System::Conserved>;

  // Compute numerical flux at y-interface
  { flux.flux_y(UL, UR, qL, qR) }
    -> std::same_as<typename System::Conserved>;
};

// ============================================================================
// Rusanov (Local Lax-Friedrichs) Flux - Generic Implementation
// ============================================================================

template <SystemDescription System>
class RusanovFlux {
public:
  using Conserved = typename System::Conserved;
  using Primitive = typename System::Primitive;

  Real gamma = System::default_gamma;

  KOKKOS_INLINE_FUNCTION
  Conserved flux_x(const Conserved& UL, const Conserved& UR,
                   const Primitive& qL, const Primitive& qR) const {
    // Maximum wave speed
    Real aL = System::sound_speed(qL, gamma);
    Real aR = System::sound_speed(qR, gamma);
    Real smax = std::fmax(std::fabs(qL.u) + aL,
                          std::fabs(qR.u) + aR);

    // Physical fluxes
    Conserved FL = System::flux_x(UL, qL);
    Conserved FR = System::flux_x(UR, qR);

    // Rusanov numerical flux
    return {
      0.5 * (FL.rho + FR.rho) - 0.5 * smax * (UR.rho - UL.rho),
      0.5 * (FL.rhou + FR.rhou) - 0.5 * smax * (UR.rhou - UL.rhou),
      0.5 * (FL.rhov + FR.rhov) - 0.5 * smax * (UR.rhov - UL.rhov),
      0.5 * (FL.E + FR.E) - 0.5 * smax * (UR.E - UL.E)
    };
  }

  KOKKOS_INLINE_FUNCTION
  Conserved flux_y(const Conserved& UL, const Conserved& UR,
                   const Primitive& qL, const Primitive& qR) const {
    Real aL = System::sound_speed(qL, gamma);
    Real aR = System::sound_speed(qR, gamma);
    Real smax = std::fmax(std::fabs(qL.v) + aL,
                          std::fabs(qR.v) + aR);

    Conserved FL = System::flux_y(UL, qL);
    Conserved FR = System::flux_y(UR, qR);

    return {
      0.5 * (FL.rho + FR.rho) - 0.5 * smax * (UR.rho - UL.rho),
      0.5 * (FL.rhou + FR.rhou) - 0.5 * smax * (UR.rhou - UL.rhou),
      0.5 * (FL.rhov + FR.rhov) - 0.5 * smax * (UR.rhov - UL.rhov),
      0.5 * (FL.E + FR.E) - 0.5 * smax * (UR.E - UL.E)
    };
  }
};

// Concept check
static_assert(NumericalFlux<RusanovFlux<Euler2D>, Euler2D>);

} // namespace subsetix::fvd
```

### 4. Boundary Conditions

```cpp
// fvd/boundary.hpp
namespace subsetix::fvd {

enum class BcLocation { Left, Right, Bottom, Top, Obstacle };
enum class BcType {
  Dirichlet,      // Fixed value
  Neumann,        // Zero gradient
  SlipWall,       // Reflect normal velocity
  NoSlipWall,     // Zero velocity
  Outflow,        // Extrapolation
  Inflow          // Fixed state
};

/// Concept for a boundary condition
template<typename BC, typename System>
concept BoundaryCondition = requires(BC bc,
                                      typename System::Conserved interior,
                                      BcLocation loc) {
  { bc.get_ghost_state(interior, loc) }
    -> std::same_as<typename System::Conserved>;
};

// ============================================================================
// Generic Boundary Condition Implementation
// ============================================================================

template <SystemDescription System>
class BoundaryCondition {
public:
  using Conserved = typename System::Conserved;
  using Primitive = typename System::Primitive;

  BcType type;
  Conserved prescribed_value;
  Real gamma = System::default_gamma;

  KOKKOS_INLINE_FUNCTION
  Conserved get_ghost_state(const Conserved& interior,
                            BcLocation loc) const {
    switch (type) {
      case BcType::Dirichlet:
      case BcType::Inflow:
        return prescribed_value;

      case BcType::Neumann:
      case BcType::Outflow:
        return interior;  // Zero-gradient / extrapolation

      case BcType::SlipWall: {
        Primitive q = System::to_primitive(interior, gamma);
        // Reflect normal velocity
        if (loc == BcLocation::Left || loc == BcLocation::Right) {
          q.u = -q.u;
        } else {
          q.v = -q.v;
        }
        return System::from_primitive(q, gamma);
      }

      case BcType::NoSlipWall: {
        Primitive q = System::to_primitive(interior, gamma);
        q.u = 0.0;
        q.v = 0.0;
        return System::from_primitive(q, gamma);
      }
    }
    return interior;
  }
};

// ============================================================================
// Boundary Configuration for a Domain
// ============================================================================

template <SystemDescription System>
class BoundaryConfig {
public:
  using BC = BoundaryCondition<System>;

  BC left, right, bottom, top;

  KOKKOS_INLINE_FUNCTION
  const BC& get_bc(BcLocation loc) const {
    switch (loc) {
      case BcLocation::Left:   return left;
      case BcLocation::Right:  return right;
      case BcLocation::Bottom: return bottom;
      case BcLocation::Top:    return top;
      default:                 return left;
    }
  }

  /// Factory: Supersonic inflow (left) / outflow (right) / slip walls
  static BoundaryConfig euler_supersonic(
      const typename System::Primitive& inflow,
      Real gamma = System::default_gamma) {
    BoundaryConfig cfg;
    cfg.left.type = BcType::Inflow;
    cfg.left.prescribed_value = System::from_primitive(inflow, gamma);
    cfg.left.gamma = gamma;

    cfg.right.type = BcType::Outflow;
    cfg.right.gamma = gamma;

    cfg.bottom.type = BcType::SlipWall;
    cfg.bottom.gamma = gamma;

    cfg.top.type = BcType::SlipWall;
    cfg.top.gamma = gamma;

    return cfg;
  }
};

} // namespace subsetix::fvd
```

### 5. Multi-Variable Fields (SoA)

```cpp
// fvd/fields.hpp
#pragma once

#include <subsetix/field/csr_field.hpp>
#include <subsetix/csr_ops/field_stencil.hpp>
#include <array>

namespace subsetix::fvd {

/// Structure-of-Arrays storage for N conserved variables
template <typename Real, int N>
class MultiVariableField {
public:
  std::array<Field2DDevice<Real>, N> components;

  /// Default constructor
  MultiVariableField() = default;

  /// Construct from geometry and label
  MultiVariableField(const IntervalSet2DDevice& geom,
                     const std::string& base_label) {
    for (int i = 0; i < N; ++i) {
      components[i] = Field2DDevice<Real>(geom,
          base_label + "_var" + std::to_string(i));
    }
  }

  /// Shared geometry
  const IntervalSet2DDevice& geometry() const {
    return components[0].geometry;
  }

  /// Total number of cells
  std::size_t total_cells() const {
    return geometry().total_cells;
  }

  /// Access component by index
  Field2DDevice<Real>& operator[](int i) { return components[i]; }
  const Field2DDevice<Real>& operator[](int i) const { return components[i]; }

  /// Swap with another field (for double buffering)
  void swap(MultiVariableField& other) {
    for (int i = 0; i < N; ++i) {
      std::swap(components[i].values, other.components[i].values);
    }
  }
};

/// Specialization for Euler2D with named access
template <typename Real>
class SoAConservedFields {
public:
  Field2DDevice<Real> rho, rhou, rhov, E;

  SoAConservedFields() = default;

  SoAConservedFields(const IntervalSet2DDevice& geom,
                     const std::string& label)
    : rho(geom, label + "_rho")
    , rhou(geom, label + "_rhou")
    , rhov(geom, label + "_rhov")
    , E(geom, label + "_E")
  {}

  const IntervalSet2DDevice& geometry() const { return rho.geometry; }

  std::size_t size() const { return rho.size(); }

  /// Access as array for generic algorithms
  std::array<Kokkos::View<Real*, DeviceMemorySpace>, 4> views() const {
    return {rho.values, rhou.values, rhov.values, E.values};
  }

  /// Swap
  void swap(SoAConservedFields& other) {
    std::swap(rho.values, other.rho.values);
    std::swap(rhou.values, other.rhou.values);
    std::swap(rhov.values, other.rhov.values);
    std::swap(E.values, other.E.values);
  }
};

} // namespace subsetix::fvd
```

### 6. Main Solver API

```cpp
// fvd/solver.hpp
#pragma once

#include <subsetix/fvd/config.hpp>
#include <subsetix/fvd/system.hpp>
#include <subsetix/fvd/flux.hpp>
#include <subsetix/fvd/boundary.hpp>
#include <subsetix/fvd/fields.hpp>
#include <subsetix/geometry/csr_mapping.hpp>
#include <subsetix/csr_ops/field_mapping.hpp>

namespace subsetix::fvd {

/// Configuration for FiniteVolumeSolver
template <SystemDescription System>
struct SolverConfig {
  Real dx = 1.0;
  Real dy = 1.0;
  Real cfl = 0.45;
  Real gamma = System::default_gamma;
  int ghost_layers = 1;
};

// ============================================================================
// Main Finite Volume Solver
// ============================================================================

template <SystemDescription System,
          template<typename> class NumericalFlux = RusanovFlux>
class FiniteVolumeSolver {
public:
  using Conserved = typename System::Conserved;
  using Primitive = typename System::Primitive;
  using StateField = SoAConservedFields<Real>;

  struct Config : public SolverConfig<System> {
    std::string label = "fv_solver";
  };

private:
  Config cfg_;
  System system_;
  NumericalFlux<System> flux_;
  BoundaryConfig<System> boundaries_;

  // Geometry
  IntervalSet2DDevice fluid_geometry_;
  IntervalSet2DDevice field_geometry_;    // fluid + ghosts
  IntervalSet2DDevice ghost_geometry_;    // ghost cells only
  Box2D domain_;

  // State fields (double buffered)
  StateField U_;
  StateField U_next_;

  // Stencil mappings (precomputed)
  subsetix::csr::FieldMaskMapping stencil_mapping_;
  subsetix::csr::detail::SubsetStencilVerticalMapping<Real> vertical_mapping_;

public:
  FiniteVolumeSolver(
      const IntervalSet2DDevice& fluid,
      const Box2D& domain,
      const System& system,
      const NumericalFlux<System>& flux,
      const BoundaryConfig<System>& boundaries,
      const Config& cfg = Config{})
    : cfg_(cfg)
    , system_(system)
    , flux_(flux)
    , boundaries_(boundaries)
    , fluid_geometry_(fluid)
    , domain_(domain)
  {
    setup_geometry();
    allocate_fields();
    build_mappings();
  }

  /// Initialize with a uniform primitive state
  void initialize(const Primitive& initial) {
    Conserved U_init = System::from_primitive(initial, cfg_.gamma);

    // Fill fluid and ghost regions
    auto U_views = U_.views();
    for (int i = 0; i < System::num_vars; ++i) {
      fill_on_set_device(U_[i], field_geometry_, U_init.data[i]);
    }

    fill_ghost_cells();
  }

  /// Fill ghost cells based on BCs
  void fill_ghost_cells() {
    if (ghost_geometry_.num_intervals == 0) return;

    auto acc = make_accessor(U_);

    apply_on_set_device(U_.rho, ghost_geometry_,
      KOKKOS_LAMBDA(Coord x, Coord y, Real& out, std::size_t idx) {
        // Determine BC location
        BcLocation loc = detect_location(x, y, domain_);

        // Find interior neighbor
        Conserved interior;
        bool found = find_interior_neighbor(acc, x, y, domain_,
                                            fluid_geometry_, interior);

        if (found) {
          // Apply boundary condition
          auto ghost = boundaries_.get_bc(loc).get_ghost_state(interior, loc);
          out = ghost.rho;

          // Write all components (need full accessor)
        } else {
          out = 0.0;  // Fallback
        }
      });
  }

  /// Compute adaptive time step (CFL condition)
  Real compute_dt() {
    Real max_rate = 0.0;
    auto rho = U_.rho.values;
    auto rhou = U_.rhou.values;
    auto rhov = U_.rhov.values;
    auto E = U_.E.values;
    Real gamma = cfg_.gamma;
    Real inv_dx = 1.0 / cfg_.dx;
    Real inv_dy = 1.0 / cfg_.dy;

    Kokkos::parallel_reduce("compute_dt",
      Kokkos::RangePolicy<ExecSpace>(0, U_.size()),
      KOKKOS_LAMBDA(int i, Real& lmax) {
        Conserved U;
        U.rho = rho(i);
        U.rhou = rhou(i);
        U.rhov = rhov(i);
        U.E = E(i);

        Primitive q = System::to_primitive(U, gamma);
        Real a = System::sound_speed(q, gamma);

        Real rate = std::fabs(q.u) * inv_dx
                  + std::fabs(q.v) * inv_dy
                  + a * (inv_dx + inv_dy);
        lmax = fmax(lmax, rate);
      }, Kokkos::Max<Real>(max_rate));

    if (max_rate <= 0.0) {
      return cfg_.cfl * std::min(cfg_.dx, cfg_.dy);
    }
    return cfg_.cfl / max_rate;
  }

  /// Perform one time step
  Real step() {
    // 1. Fill ghost cells
    fill_ghost_cells();

    // 2. Compute time step
    Real dt = compute_dt();

    // 3. Apply FV stencil
    apply_fv_stencil(dt);

    // 4. Swap buffers
    U_.swap(U_next_);

    return dt;
  }

  /// Access current state
  StateField& state() { return U_; }
  const StateField& state() const { return U_; }

  /// Access geometry
  const IntervalSet2DDevice& geometry() const { return fluid_geometry_; }
  const IntervalSet2DDevice& full_geometry() const { return field_geometry_; }

private:
  void setup_geometry() {
    // Expand fluid to include ghost layer
    CsrSetAlgebraContext ctx;
    IntervalSet2DDevice expanded;
    csr::expand_device(fluid_geometry_, cfg_.ghost_layers,
                       cfg_.ghost_layers, expanded, ctx);
    csr::compute_cell_offsets_device(expanded);

    field_geometry_ = expanded;

    // Ghost = expanded - fluid
    ghost_geometry_ = csr::allocate_interval_set_device(
        expanded.num_rows,
        expanded.num_intervals + fluid_geometry_.num_intervals);
    csr::set_difference_device(expanded, fluid_geometry_,
                               ghost_geometry_, ctx);
    csr::compute_cell_offsets_device(ghost_geometry_);
  }

  void allocate_fields() {
    U_ = StateField(field_geometry_, cfg_.label + "_U");
    U_next_ = StateField(field_geometry_, cfg_.label + "_U_next");
  }

  void build_mappings() {
    stencil_mapping_ = csr::build_field_mask_mapping(
        U_.rho, fluid_geometry_);

    vertical_mapping_ = csr::detail::build_subset_stencil_vertical_mapping(
        U_.rho, fluid_geometry_, stencil_mapping_);
  }

  void apply_fv_stencil(Real dt) {
    Real dt_over_dx = dt / cfg_.dx;
    Real dt_over_dy = dt / cfg_.dy;

    apply_fv_stencil_on_set_device(
        U_next_, U_, fluid_geometry_,
        stencil_mapping_, vertical_mapping_,
        system_, flux_, cfg_.gamma,
        dt_over_dx, dt_over_dy);
  }

  KOKKOS_INLINE_FUNCTION
  static BcLocation detect_location(Coord x, Coord y, const Box2D& domain) {
    if (x < domain.x_min) return BcLocation::Left;
    if (x >= domain.x_max) return BcLocation::Right;
    if (y < domain.y_min) return BcLocation::Bottom;
    if (y >= domain.y_max) return BcLocation::Top;
    return BcLocation::Obstacle;
  }
};

// ============================================================================
// Convenience: Euler2D Solver
// ============================================================================

using EulerSolver2D = FiniteVolumeSolver<Euler2D, RusanovFlux>;

} // namespace subsetix::fvd
```

---

## Stencil Strategy

### The Multi-Variable Problem

The existing `apply_csr_stencil_on_set_device` works on a **single field** at a time.
For Euler equations with 4 conserved variables (rho, rhou, rhov, E), we need to:

1. Read all 4 variables at 5 points (center, W, E, S, N) = 20 reads
2. Compute fluxes between neighbors
3. Write 4 updated values

### Options

| Option | Pros | Cons |
|--------|------|------|
| Call stencil 4 times | Simple, reuse existing code | 4x kernel launch overhead |
| Single kernel with manual indexing | Optimal, single pass | More complex code |
| Create wrapper abstraction | Clean API, efficient | Additional layer |

### Chosen Solution: `apply_fv_stencil_on_set_device`

A new specialized function that handles N variables in a single kernel pass:

```cpp
// fvd/stencil.hpp
namespace subsetix::fvd {

/// Stencil point for multi-variable access
template <typename Real, int N>
struct MultiStencilPoint {
  // Values for each variable
  std::array<Kokkos::View<Real*, DeviceMemorySpace>, N> values;

  // Neighbor indices
  struct Indices {
    std::size_t center, west, east, south, north;
  } idx;

  KOKKOS_INLINE_FUNCTION Real center(int var) const {
    return values[var](idx.center);
  }
  KOKKOS_INLINE_FUNCTION Real west(int var) const {
    return values[var](idx.west);
  }
  KOKKOS_INLINE_FUNCTION Real east(int var) const {
    return values[var](idx.east);
  }
  KOKKOS_INLINE_FUNCTION Real south(int var) const {
    return values[var](idx.south);
  }
  KOKKOS_INLINE_FUNCTION Real north(int var) const {
    return values[var](idx.north);
  }
};

/// Apply finite volume stencil on N variables simultaneously
template <typename Real, int N, class Functor>
void apply_fv_stencil_on_set_device(
    SoAConservedFields<Real>& out,
    const SoAConservedFields<Real>& in,
    const IntervalSet2DDevice& mask,
    const subsetix::csr::FieldMaskMapping& mapping,
    const subsetix::csr::detail::SubsetStencilVerticalMapping<Real>& vertical,
    Functor&& functor) {

  using namespace subsetix::csr;

  if (mask.num_rows == 0 || mask.num_intervals == 0) return;

  // Build stencil point structure
  MultiStencilPoint<Real, N> stencil;
  stencil.values = in.views();

  // Get kernel parameters
  auto interval_to_row = mapping.interval_to_row;
  auto mask_intervals = mask.intervals;
  auto mask_row_keys = mask.row_keys;

  auto in_intervals = in.rho.geometry.intervals;
  auto in_offsets = in.rho.geometry.cell_offsets;
  auto out_intervals = out.rho.geometry.intervals;
  auto out_offsets = out.rho.geometry.cell_offsets;

  auto out_views = out.views();

  auto north_interval = vertical.north_interval;
  auto south_interval = vertical.south_interval;

  // Launch kernel
  const int num_intervals = mask.num_intervals;
  Kokkos::parallel_for("apply_fv_stencil_multi",
    Kokkos::RangePolicy<ExecSpace>(0, num_intervals),
    KOKKOS_LAMBDA(const int interval_idx) {
      const int row_idx = interval_to_row(interval_idx);
      if (row_idx < 0) return;

      const Coord y = mask_row_keys(row_idx).y;
      const auto mask_iv = mask_intervals(interval_idx);
      const auto in_iv = in_intervals(interval_idx);
      const auto out_iv = out_intervals(interval_idx);

      const std::size_t in_base = in_offsets(interval_idx);
      const std::size_t out_base = out_offsets(interval_idx);

      // For each cell in interval
      for (Coord x = mask_iv.begin; x < mask_iv.end; ++x) {
        const std::size_t linear_idx =
            out_base + static_cast<std::size_t>(x - out_iv.begin);

        // Compute neighbor indices
        const std::size_t idx_c = in_base + (x - in_iv.begin);
        const std::size_t idx_w = idx_c - 1;
        const std::size_t idx_e = idx_c + 1;

        // North/South indices (using vertical mapping)
        const int n_iv = north_interval(interval_idx);
        const int s_iv = south_interval(interval_idx);

        std::size_t idx_n = idx_c, idx_s = idx_c;
        if (n_iv >= 0) {
          const auto n_iv_struct = in_intervals(n_iv);
          idx_n = in_offsets(n_iv) + (x - n_iv_struct.begin);
        }
        if (s_iv >= 0) {
          const auto s_iv_struct = in_intervals(s_iv);
          idx_s = in_offsets(s_iv) + (x - s_iv_struct.begin);
        }

        stencil.idx = {idx_c, idx_w, idx_e, idx_s, idx_n};

        // Call functor - returns array of N values
        auto result = functor(x, y, stencil);

        // Write output
        for (int ivar = 0; ivar < N; ++ivar) {
          out_views[ivar](linear_idx) = result[ivar];
        }
      }
    });

  ExecSpace().fence();
}

} // namespace subsetix::fvd
```

### Flux Functor for Euler2D

```cpp
// Functor that works with MultiStencilPoint
struct EulerRusanovFunctor {
  Real gamma;
  Real dt_over_dx, dt_over_dy;

  KOKKOS_INLINE_FUNCTION
  std::array<Real, 4> operator()(Coord, Coord,
                                 const MultiStencilPoint<Real, 4>& p) const {
    // Gather conserved states at 5 points
    auto gather = [&](const std::size_t& idx) -> Euler2D::Conserved {
      return {p.center(0, idx), p.center(1, idx),
              p.center(2, idx), p.center(3, idx)};
    };

    const auto U_c = gather(p.idx.center);
    const auto U_l = gather(p.idx.west);
    const auto U_r = gather(p.idx.east);
    const auto U_d = gather(p.idx.south);
    const auto U_u = gather(p.idx.north);

    // Convert to primitives
    const auto q_c = Euler2D::to_primitive(U_c, gamma);
    const auto q_l = Euler2D::to_primitive(U_l, gamma);
    const auto q_r = Euler2D::to_primitive(U_r, gamma);
    const auto q_d = Euler2D::to_primitive(U_d, gamma);
    const auto q_u = Euler2D::to_primitive(U_u, gamma);

    // Rusanov fluxes
    auto rusanov_x = [&](const auto& UL, const auto& UR,
                         const auto& qL, const auto& qR) {
      Real aL = Euler2D::sound_speed(qL, gamma);
      Real aR = Euler2D::sound_speed(qR, gamma);
      Real smax = fmax(fabs(qL.u) + aL, fabs(qR.u) + aR);

      auto FL = Euler2D::flux_x(UL, qL);
      auto FR = Euler2D::flux_x(UR, qR);

      return std::array<Real, 4> {
        0.5*(FL.rho + FR.rho) - 0.5*smax*(UR.rho - UL.rho),
        0.5*(FL.rhou + FR.rhou) - 0.5*smax*(UR.rhou - UL.rhou),
        0.5*(FL.rhov + FR.rhov) - 0.5*smax*(UR.rhov - UL.rhov),
        0.5*(FL.E + FR.E) - 0.5*smax*(UR.E - UL.E)
      };
    };

    auto F_w = rusanov_x(U_l, U_c, q_l, q_c);
    auto F_e = rusanov_x(U_c, U_r, q_c, q_r);

    // Flux Y
    auto rusanov_y = [&](const auto& UL, const auto& UR,
                         const auto& qL, const auto& qR) {
      Real aL = Euler2D::sound_speed(qL, gamma);
      Real aR = Euler2D::sound_speed(qR, gamma);
      Real smax = fmax(fabs(qL.v) + aL, fabs(qR.v) + aR);

      auto FL = Euler2D::flux_y(UL, qL);
      auto FR = Euler2D::flux_y(UR, qR);

      return std::array<Real, 4> {
        0.5*(FL.rho + FR.rho) - 0.5*smax*(UR.rho - UL.rho),
        0.5*(FL.rhou + FR.rhou) - 0.5*smax*(UR.rhou - UL.rhou),
        0.5*(FL.rhov + FR.rhov) - 0.5*smax*(UR.rhov - UL.rhov),
        0.5*(FL.E + FR.E) - 0.5*smax*(UR.E - UL.E)
      };
    };

    auto F_s = rusanov_y(U_d, U_c, q_d, q_c);
    auto F_n = rusanov_y(U_c, U_u, q_c, q_u);

    // Godunov update
    return {
      U_c.rho - dt_over_dx * (F_e[0] - F_w[0]) - dt_over_dy * (F_n[0] - F_s[0]),
      U_c.rhou - dt_over_dx * (F_e[1] - F_w[1]) - dt_over_dy * (F_n[1] - F_s[1]),
      U_c.rhov - dt_over_dx * (F_e[2] - F_w[2]) - dt_over_dy * (F_n[2] - F_s[2]),
      U_c.E - dt_over_dx * (F_e[3] - F_w[3]) - dt_over_dy * (F_n[3] - F_s[3])
    };
  }
};
```

---

## AMR Integration

### Error Indicators

```cpp
// fvd/amr/error_indicator.hpp
namespace subsetix::fvd::amr {

/// Concept for error estimator
template <typename Estimator, typename System>
concept ErrorEstimator = requires(Estimator e,
                                   const SoAConservedFields<Real>& U,
                                   const IntervalSet2DDevice& mask) {
  { e.estimate(U, mask) } -> std::same_as<Field2DDevice<Real>>;
};

// ============================================================================
// Gradient-based Error Indicator
// ============================================================================

class GradientErrorIndicator {
  Real dx_, dy_;

public:
  GradientErrorIndicator(Real dx, Real dy) : dx_(dx), dy_(dy) {}

  /// Compute gradient magnitude of density
  Field2DDevice<Real> estimate(
      const SoAConservedFields<Real>& U,
      const IntervalSet2DDevice& mask) const {

    // Create indicator field on same geometry as U.rho
    Field2DDevice<Real> indicator(U.rho.geometry, "error_indicator");

    // Apply gradient stencil
    IndicatorStencil stencil{1.0/dx_, 1.0/dy_};
    apply_csr_stencil_on_set_device(indicator, U.rho, mask, stencil);

    return indicator;
  }

private:
  struct IndicatorStencil {
    Real inv_dx, inv_dy;

    KOKKOS_INLINE_FUNCTION
    Real operator()(Coord, Coord,
                   const subsetix::csr::CsrStencilPoint<Real>& p) const {
      Real gx = 0.5 * (p.east() - p.west()) * inv_dx;
      Real gy = 0.5 * (p.north() - p.south()) * inv_dy;
      return fabs(gx) + fabs(gy);
    }
  };
};

// ============================================================================
// Wave-based Error Indicator (for shocks)
// ============================================================================

class WaveSensorIndicator {
  Real gamma_;

public:
  WaveSensorIndicator(Real gamma = 1.4) : gamma_(gamma) {}

  Field2DDevice<Real> estimate(
      const SoAConservedFields<Real>& U,
      const IntervalSet2DDevice& mask) const {

    Field2DDevice<Real> sensor(U.rho.geometry, "wave_sensor");

    // Compute pressure gradient + vorticity indicator
    // (detects shocks and shear layers)

    apply_on_set_device(sensor, mask,
      KOKKOS_LAMBDA(Coord x, Coord y, Real& out, std::size_t idx) {
        // Read primitive variables
        Real rho = U.rho.values(idx);
        Real rhou = U.rhou.values(idx);
        Real rhov = U.rhov.values(idx);
        Real E = U.E.values(idx);

        // Convert to primitive
        Real inv_rho = 1.0 / (rho + 1e-12);
        Real u = rhou * inv_rho;
        Real v = rhov * inv_rho;
        Real kinetic = 0.5 * (u*u + v*v);
        Real p = (gamma_ - 1.0) * (E - rho * kinetic);

        out = p;  // Simplified - would compute gradients
      });

    return sensor;
  }
};

} // namespace subsetix::fvd::amr
```

### Adaptive Solver with AMR

```cpp
// fvd/amr/adaptive_solver.hpp
namespace subsetix::fvd::amr {

/// Adaptive finite volume solver with AMR
template <SystemDescription System,
          template<typename> class NumericalFlux = RusanovFlux>
class AdaptiveFiniteVolumeSolver : public FiniteVolumeSolver<System, NumericalFlux> {
public:
  using Base = FiniteVolumeSolver<System, NumericalFlux>;
  using Config = typename Base::Config;

  struct AdaptiveConfig : public Config {
    Real refine_threshold = 0.1;
    Real coarsen_threshold = 0.01;
    int min_levels = 1;
    int max_levels = 4;
    int remesh_stride = 10;
    int guard_cells = 2;
  };

private:
  AdaptiveConfig cfg_;

  // Multi-level data
  static constexpr int MAX_LEVELS = 6;
  std::array<bool, MAX_LEVELS> has_level_;
  std::array<IntervalSet2DDevice, MAX_LEVELS> fluid_geom_;
  std::array<IntervalSet2DDevice, MAX_LEVELS> active_mask_;
  std::array<SoAConservedFields<Real>, MAX_LEVELS> U_levels_;
  std::array<SoAConservedFields<Real>, MAX_LEVELS> U_next_levels_;
  std::array<Box2D, MAX_LEVELS> domains_;
  std::array<Real, MAX_LEVELS> dx_, dy_;

  ErrorEstimator auto& error_estimator_;

public:
  AdaptiveFiniteVolumeSolver(
      const IntervalSet2DDevice& fluid,
      const Box2D& domain,
      const System& system,
      const NumericalFlux<System>& flux,
      const BoundaryConfig<System>& boundaries,
      const ErrorEstimator auto& error_estimator,
      const AdaptiveConfig& cfg = AdaptiveConfig{})
    : Base(fluid, domain, system, flux, boundaries, cfg)
    , cfg_(cfg)
    , error_estimator_(error_estimator)
  {
    initialize_levels();
  }

  /// Initialize AMR hierarchy
  void initialize_levels() {
    has_level_.fill(false);
    has_level_[0] = true;

    fluid_geom_[0] = this->geometry();
    active_mask_[0] = this->geometry();
    U_levels_[0] = this->state();
    domains_[0] = domain_;
    dx_[0] = cfg_.dx;
    dy_[0] = cfg_.dy;

    // Build initial hierarchy
    build_hierarchy();
  }

  /// Perform one time step with subcycling
  Real step() {
    int finest = find_finest_level();
    Real dt_global = compute_global_dt(finest);

    // Step from finest to coarsest
    for (int lvl = finest; lvl >= 0; --lvl) {
      if (!has_level_[lvl]) continue;

      // Prolong ghost regions from coarse
      if (lvl > 0) {
        prolong_ghost_regions(lvl);
      }

      // Fill boundary ghosts
      this->fill_ghost_cells();

      // Perform level step
      int substeps = (lvl == finest) ? 1 : 2;
      Real dt_lvl = dt_global / substeps;

      for (int s = 0; s < substeps; ++s) {
        apply_level_step(lvl, dt_lvl);
      }

      // Restrict to coarse
      if (lvl > 0) {
        restrict_to_coarse(lvl);
      }
    }

    // Adaptive remeshing
    if (cfg_.remesh_stride > 0 &&
        step_count_ % cfg_.remesh_stride == 0) {
      adapt_mesh();
    }

    ++step_count_;
    return dt_global;
  }

private:
  void build_hierarchy() {
    CsrSetAlgebraContext ctx;

    for (int lvl = 1; lvl < cfg_.max_levels; ++lvl) {
      if (!has_level_[lvl - 1]) break;

      // Estimate error on coarse level
      auto error = error_estimator_.estimate(
          U_levels_[lvl - 1], active_mask_[lvl - 1]);

      // Threshold to get refinement mask
      auto refine_mask = subsetix::csr::threshold_field(
          error, cfg_.refine_threshold);

      // Build fine geometry
      IntervalSet2DDevice fine_geom;
      subsetix::csr::refine_level_up_device(
          fluid_geom_[lvl - 1], fine_geom, ctx);
      subsetix::csr::compute_cell_offsets_device(fine_geom);

      // Intersect with refinement region
      IntervalSet2DDevice refined_mask;
      subsetix::csr::refine_level_up_device(
          refine_mask, refined_mask, ctx);
      subsetix::csr::compute_cell_offsets_device(refined_mask);

      IntervalSet2DDevice active;
      subsetix::csr::set_intersection_device(
          fine_geom, refined_mask, active, ctx);
      subsetix::csr::compute_cell_offsets_device(active);

      if (active.total_cells > 0) {
        has_level_[lvl] = true;
        fluid_geom_[lvl] = fine_geom;
        active_mask_[lvl] = active;
        domains_[lvl] = {
          domains_[lvl - 1].x_min * 2,
          domains_[lvl - 1].x_max * 2,
          domains_[lvl - 1].y_min * 2,
          domains_[lvl - 1].y_max * 2
        };
        dx_[lvl] = 0.5 * dx_[lvl - 1];
        dy_[lvl] = 0.5 * dy_[lvl - 1];

        // Allocate fields and prolong from coarse
        U_levels_[lvl] = SoAConservedFields<Real>(
            fine_geom, "U_lvl" + std::to_string(lvl));
        U_next_levels_[lvl] = SoAConservedFields<Real>(
            fine_geom, "U_next_lvl" + std::to_string(lvl));

        prolong_full(U_levels_[lvl], active,
                     U_levels_[lvl - 1], ctx);
      } else {
        break;
      }
    }
  }

  void adapt_mesh() {
    // Save current state
    auto old_has_level = has_level_;
    auto old_U = U_levels_;

    // Rebuild hierarchy
    for (int lvl = 1; lvl < MAX_LEVELS; ++lvl) {
      has_level_[lvl] = false;
    }
    build_hierarchy();

    // Copy overlapping regions from old hierarchy
    CsrSetAlgebraContext ctx;
    for (int lvl = 1; lvl < MAX_LEVELS; ++lvl) {
      if (!has_level_[lvl] || !old_has_level[lvl]) continue;

      // Find overlap
      IntervalSet2DDevice overlap;
      subsetix::csr::set_intersection_device(
          active_mask_[lvl], old_U[lvl].geometry(),
          overlap, ctx);
      subsetix::csr::compute_cell_offsets_device(overlap);

      if (overlap.total_cells > 0) {
        // Copy from old to new
        copy_field_on_subset(U_levels_[lvl], old_U[lvl], overlap);
      }
    }
  }

  int find_finest_level() const {
    for (int lvl = MAX_LEVELS - 1; lvl >= 0; --lvl) {
      if (has_level_[lvl]) return lvl;
    }
    return 0;
  }

  Real compute_global_dt(int finest) {
    Real dt = std::numeric_limits<Real>::max();
    for (int lvl = 0; lvl <= finest; ++lvl) {
      if (!has_level_[lvl]) continue;
      // Compute CFL on each level
      // Use min across all levels
    }
    return dt;
  }

  void apply_level_step(int lvl, Real dt) {
    // Apply FV stencil on this level
    // Similar to Base::apply_fv_stencil but using level-specific fields
  }

  void prolong_ghost_regions(int lvl) {
    // Prolong from coarse to fine ghost regions
  }

  void restrict_to_coarse(int lvl) {
    // Restrict fine correction to coarse
  }

  void prolong_full(SoAConservedFields<Real>& fine,
                    const IntervalSet2DDevice& fine_mask,
                    const SoAConservedFields<Real>& coarse,
                    CsrSetAlgebraContext& ctx) {
    // Full prolongation using subsetix operations
    subsetix::csr::prolong_field_on_set_coords_device(
        fine.rho, coarse.rho, fine_mask);
    subsetix::csr::prolong_field_on_set_coords_device(
        fine.rhou, coarse.rhou, fine_mask);
    subsetix::csr::prolong_field_on_set_coords_device(
        fine.rhov, coarse.rhov, fine_mask);
    subsetix::csr::prolong_field_on_set_coords_device(
        fine.E, coarse.E, fine_mask);
  }

  int step_count_ = 0;
};

} // namespace subsetix::fvd::amr
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

**Goal**: Basic solver without AMR, single-level Euler2D

```
include/subsetix/fvd/
├── config.hpp              ✓ Types, concepts
├── system.hpp              ✓ SystemDescription concept
├── systems/
│   └── euler2d.hpp         ✓ Euler2D implementation
├── flux.hpp                ✓ RusanovFlux
├── boundary.hpp            ✓ BoundaryCondition, BoundaryConfig
├── fields.hpp              ✓ SoAConservedFields
├── solver.hpp              ✓ FiniteVolumeSolver basic
└── stencil.hpp             ✓ apply_fv_stencil_on_set_device
```

**Milestone**: Run simple Euler2D test case (e.g., advection of density pulse)

### Phase 2: Subsetix Integration (Week 2-3)

**Goal**: Refactor MACH2 to use new API

```
examples/mach2_cylinder_v2/
└── mach2_cylinder_v2.cpp   ✓ Using new FVD API
```

**Milestone**: mach2_cylinder_v2 produces same results as original

### Phase 3: Multi-Physics (Week 3-4)

**Goal**: Add other systems and flux schemes

```
include/subsetix/fvd/
├── systems/
│   ├── advection2d.hpp     ✓ Scalar advection
│   └── navier_stokes2d.hpp (future)
└── fluxes/
    ├── rusanov.hpp         (moved from flux.hpp)
    ├── hllc.hpp            ✓ HLLC flux
    └── central.hpp         ✓ Central flux (diffusion)
```

**Milestone**: Run advection2d test, compare HLLC vs Rusanov

### Phase 4: AMR Integration (Week 4-6)

**Goal**: Adaptive mesh refinement

```
include/subsetix/fvd/amr/
├── error_indicator.hpp     ✓ Gradient, WaveSensor
├── refinement_strategy.hpp ✓ Threshold-based
└── adaptive_solver.hpp     ✓ AdaptiveFiniteVolumeSolver
```

**Milestone**: AdaptiveEulerSolver reproduces MACH2 AMR results

### Phase 5: Advanced Features (Week 6+)

**Goal**: Higher-order methods, diagnostics

```
include/subsetix/fvd/
├── reconstruction.hpp       ✓ MUSCL reconstruction
├── limiters.hpp             ✓ Minmod, MC, van Leer
├── diagnostics.hpp          ✓ Vorticity, divergence, etc.
└── time_stepper.hpp         ✓ RK2, RK3
```

---

## Code Examples

### Example 1: Simple Advection Test

```cpp
// examples/fvd_advection2d.cpp
#include <subsetix/fvd/solver.hpp>
#include <subsetix/fvd/systems/advection2d.hpp>
#include <subsetix/io/vtk_export.hpp>

using namespace subsetix;
using namespace subsetix::fvd;

int main() {
  Kokkos::ScopeGuard guard(0, nullptr);

  // Create a simple box domain
  const int nx = 100, ny = 100;
  const Box2D domain{0, nx, 0, ny};
  auto fluid = csr::make_box_device(domain);
  csr::compute_cell_offsets_device(fluid);

  // Configure advection system
  Advection2D system{.vx = 1.0, .vy = 0.5};

  // Configure solver
  FiniteVolumeSolver<Advection2D, RusanovFlux>::Config cfg;
  cfg.dx = 1.0; cfg.dy = 1.0;
  cfg.cfl = 0.5;

  // Boundary conditions: periodic would be nice, but for now:
  BoundaryConfig<Advection2D> bc;
  bc.left.type = BcType::Dirichlet;
  bc.right.type = BcType::Outflow;
  bc.bottom.type = BcType::SlipWall;
  bc.top.type = BcType::SlipWall;

  // Create solver
  FiniteVolumeSolver<Advection2D, RusanovFlux> solver(
      fluid, domain, system, RusanovFlux<Advection2D>{}, bc, cfg);

  // Initialize with Gaussian pulse
  Advection2D::Primitive initial{.value = 0.0};
  solver.initialize(initial);

  // Add pulse
  auto& U = solver.state();
  apply_on_set_device(U.value, U.geometry(),
    KOKKOS_LAMBDA(Coord x, Coord y, Real& out, std::size_t) {
      Real dx = x - 50.0, dy = y - 50.0;
      Real r2 = dx*dx + dy*dy;
      out = std::exp(-r2 / 100.0);
    });

  // Time loop
  Real t = 0.0;
  const Real t_final = 50.0;
  int step = 0;

  while (t < t_final) {
    Real dt = solver.step();
    t += dt;
    ++step;

    if (step % 10 == 0) {
      vtk::write_legacy_quads(
          csr::toHost(fluid), U.value,
          "output/advection_step_" + std::to_string(step) + ".vtk");
      std::cout << "Step " << step << ", t = " << t << std::endl;
    }
  }

  return 0;
}
```

### Example 2: MACH2 Refactored

```cpp
// examples/mach2_cylinder_v2/mach2_cylinder_v2.cpp
#include <subsetix/fvd/solver.hpp>
#include <subsetix/fvd/systems/euler2d.hpp>
#include <subsetix/io/vtk_export.hpp>

using namespace subsetix;
using namespace subsetix::fvd;

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);

  // Configuration
  const int nx = 400, ny = 160;
  const Real mach_inlet = 2.0;
  const Real rho = 1.0, p = 1.0;
  const Real gamma = 1.4;
  const Real cfl = 0.45;

  // Geometry: domain minus cylinder
  const Box2D domain{0, nx, 0, ny};
  auto domain_box = csr::make_box_device(domain);
  auto cylinder = csr::make_disk_device(csr::Disk2D{nx/4, ny/2, 20});

  CsrSetAlgebraContext ctx;
  IntervalSet2DDevice fluid;
  csr::set_difference_device(domain_box, cylinder, fluid, ctx);
  csr::compute_cell_offsets_device(fluid);

  // System
  Euler2D system{.gamma = gamma};

  // Flux
  RusanovFlux<Euler2D> flux{.gamma = gamma};

  // Boundary conditions
  const Real a = std::sqrt(gamma * p / rho);
  const Real u_inlet = mach_inlet * a;
  Euler2D::Primitive inflow{.rho = rho, .u = u_inlet, .v = 0.0, .p = p};

  auto bc = BoundaryConfig<Euler2D>::euler_supersonic(inflow, gamma);

  // Solver
  EulerSolver2D::Config cfg;
  cfg.dx = 1.0; cfg.dy = 1.0;
  cfg.cfl = cfl;
  cfg.gamma = gamma;
  cfg.label = "mach2";

  EulerSolver2D solver(fluid, domain, system, flux, bc, cfg);

  // Initialize
  solver.initialize(inflow);

  // Main loop
  Real t = 0.0;
  const Real t_final = 0.01;
  int step = 0;

  while (t < t_final && step < 5000) {
    Real dt = solver.step();
    t += dt;
    ++step;

    if (step % 50 == 0) {
      // Export diagnostics
      auto& U = solver.state();

      // Compute pressure and Mach number
      Field2DDevice<Real> pressure(fluid, "pressure");
      Field2DDevice<Real> mach(fluid, "mach");

      compute_diagnostics(U, pressure, mach, gamma);

      vtk::write_legacy_quads(
          csr::toHost(fluid), U.rho,
          "output/step_" + std::to_string(step) + "_density.vtk");
      vtk::write_legacy_quads(
          csr::toHost(fluid), pressure,
          "output/step_" + std::to_string(step) + "_pressure.vtk");
      vtk::write_legacy_quads(
          csr::toHost(fluid), mach,
          "output/step_" + std::to_string(step) + "_mach.vtk");

      std::cout << "Step " << step << ", t = " << t
                << ", dt = " << dt << std::endl;
    }
  }

  return 0;
}
```

### Example 3: Adaptive Solver

```cpp
// examples/fvd_adaptive_euler.cpp
#include <subsetix/fvd/amr/adaptive_solver.hpp>
#include <subsetix/fvd/systems/euler2d.hpp>
#include <subsetix/fvd/amr/error_indicator.hpp>

using namespace subsetix;
using namespace subsetix::fvd;
using namespace subsetix::fvd::amr;

int main() {
  Kokkos::ScopeGuard guard(0, nullptr);

  // Domain
  const int nx = 200, ny = 100;
  const Box2D domain{0, nx, 0, ny};
  auto fluid = csr::make_box_device(domain);
  csr::compute_cell_offsets_device(fluid);

  // System
  Euler2D system{.gamma = 1.4};

  // Error estimator
  GradientErrorIndicator error_estimator{1.0, 1.0};

  // Adaptive config
  AdaptiveFiniteVolumeSolver<Euler2D, RusanovFlux>::AdaptiveConfig cfg;
  cfg.dx = 1.0; cfg.dy = 1.0;
  cfg.cfl = 0.45;
  cfg.gamma = 1.4;
  cfg.refine_threshold = 0.05;
  cfg.coarsen_threshold = 0.01;
  cfg.max_levels = 4;
  cfg.remesh_stride = 20;
  cfg.guard_cells = 2;

  // BCs
  Euler2D::Primitive inflow{.rho = 1.0, .u = 1.5, .v = 0.0, .p = 1.0};
  auto bc = BoundaryConfig<Euler2D>::euler_supersonic(inflow, 1.4);

  // Create adaptive solver
  AdaptiveFiniteVolumeSolver<Euler2D, RusanovFlux> solver(
      fluid, domain, system, RusanovFlux<Euler2D>{}, bc,
      error_estimator, cfg);

  // Initialize
  solver.initialize(inflow);

  // Add perturbation to trigger refinement
  // ...

  // Time loop
  Real t = 0.0;
  const Real t_final = 1.0;
  int step = 0;

  while (t < t_final) {
    Real dt = solver.step();
    t += dt;
    ++step;

    if (step % 50 == 0) {
      std::cout << "Step " << step << ", t = " << t
                << ", levels: " << solver.num_active_levels()
                << ", cells: " << solver.total_cells()
                << std::endl;

      // Export multilevel VTK
      solver.write_multilevel_vtk(
          "output/adaptive_step_" + std::to_string(step) + ".vtk");
    }
  }

  return 0;
}
```

---

## Comparison Matrix

### Code Reduction

| Task | MACH2 Original | With FVD Module | Reduction |
|------|----------------|-----------------|-----------|
| New Euler case | ~2000 lines copy | ~50 lines | **97.5%** |
| Change flux (Rusanov → HLLC) | Modify inline function | Change template parameter | **N/A** |
| Add new BC | Modify `fill_ghost_cells` | Create BC class (~20 lines) | **Reusable** |
| Add AMR to new system | Rewrite ~500 lines | Use `AdaptiveFiniteVolumeSolver` | **90%** |
| Test single component | Need full solver | Test component independently | **Modular** |

### Performance Considerations

| Aspect | MACH2 | FVD Module | Notes |
|--------|-------|------------|-------|
| Storage | SoA (4 fields) | SoA (same) | ✓ Same |
| Kernel launches | 1 multi-var kernel | 1 multi-var kernel | ✓ Same |
| Memory access | Coalesced | Coalesced | ✓ Same |
| Ghost cell fill | Manual loop | Generic functor | Slight overhead |
| Compile time | Fast | More templates | Slower |
| Runtime | Optimized | Optimized | ✓ Same |

### Extensibility

| New Feature | MACH2 Effort | FVD Module Effort |
|-------------|--------------|-------------------|
| Navier-Stokes | Rewrite solver | Implement `System` + viscous flux |
| 3D | Rewrite everything | Extend to 3D types |
| New BC (periodic) | Modify `fill_ghost_cells` | New `PeriodicBC` class |
| Higher-order (MUSCL) | Modify stencil | New reconstruction functor |
| Different time stepping | Modify main loop | New `TimeStepper` class |

---

## Summary

The proposed **Finite Volume Dynamics (FVD)** layer for Subsetix provides:

1. **Generic system abstraction** via `SystemDescription` concept
2. **Modular numerical fluxes** that work with any system
3. **Configurable boundary conditions** without code modification
4. **Optimized SoA storage** matching MACH2's performance
5. **Seamless AMR integration** using Subsetix operations
6. **Clean API** that reduces new case development from ~2000 to ~50 lines

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| SoA storage | Proven performance in MACH2, cache-friendly |
| 2D only | Focus on core functionality, 3D can be added later |
| Template-based | Compile-time optimization, no runtime overhead |
| Separate module | Clean separation from core Subsetix |
| New stencil function | Single-kernel multi-variable processing |

### Next Steps

1. **Review and approve** this proposal
2. **Create Phase 1** skeleton (`fvd/` directory structure)
3. **Implement `Euler2D`** system and `RusanovFlux`
4. **Create `apply_fv_stencil_on_set_device`** for multi-variable kernels
5. **Implement basic `FiniteVolumeSolver`**
6. **Test with simple case** (advection pulse)
7. **Refactor MACH2** to use new API
8. **Add AMR integration** in Phase 4

---

## Appendix: MACH2 Code Reference

### Key Functions to Extract

| Function | Lines | Purpose | FVD Equivalent |
|----------|-------|---------|----------------|
| `cons_to_prim` | 275-286 | Convert conserved to primitive | `Euler2D::to_primitive` |
| `prim_to_cons` | 289-297 | Convert primitive to conserved | `Euler2D::from_primitive` |
| `sound_speed` | 300-303 | Compute sound speed | `Euler2D::sound_speed` |
| `flux_x`, `flux_y` | 306-323 | Physical fluxes | `Euler2D::flux_x/y` |
| `rusanov_flux_x/y` | 326-367 | Numerical flux | `RusanovFlux<Euler2D>` |
| `make_wall_ghost` | 392-407 | Wall BC | `BoundaryCondition::get_ghost_state` |
| `fill_ghost_cells` | 454-548 | Apply BCs | `FiniteVolumeSolver::fill_ghost_cells` |
| `compute_dt` | 550-589 | CFL time step | `FiniteVolumeSolver::compute_dt` |
| `EulerStencilSoA` | 1057-1119 | FV update | `apply_fv_stencil_on_set_device` |
| `build_refine_mask` | 606-759 | AMR indicator | `GradientErrorIndicator::estimate` |

### Data Structures to Replace

| MACH2 | FVD Module |
|-------|------------|
| `struct Conserved { Real rho, rhou, rhov, E; }` | `Euler2D::Conserved` |
| `struct Primitive { Real rho, u, v, p; }` | `Euler2D::Primitive` |
| `struct ConservedFields` | `SoAConservedFields<Real>` |
| `struct ConservedViews` | `MultiStencilPoint<Real, 4>` |
| `enum class BcKind` | `enum class BcType` |
| `struct RunConfig` | `EulerSolver2D::Config` |
| `IndicatorStencil` | `GradientErrorIndicator` |

---

*End of Proposal*
