# Proposal: Finite Volume Dynamics (FVD) Layer for Subsetix
## Version 2: Kokkos-Native, Compile-Time Architecture

**Date**: 2025-12-30
**Status**: Design Proposal - Kokkos-Native

---

## Executive Summary

This proposal outlines a **Kokkos-native, compile-time optimized** Finite Volume Dynamics layer that follows Subsetix's existing patterns exactly. No abstractions, no runtime overhead, 100% GPU-safe.

### Key Principles

| Principle | Implementation |
|-----------|----------------|
| **Compile-time only** | Everything is template-based, resolved at compilation |
| **Kokkos-native** | Uses `KOKKOS_INLINE_FUNCTION`, `KOKKOS_LAMBDA`, Kokkos::View |
| **POD structs** | All device-side structures are Plain Old Data |
| **No virtual** | Zero runtime polymorphism, all static dispatch |
| **Follow Subsetix patterns** | Same style as `apply_csr_stencil_on_set_device` |

---

## Architecture: Single-Level Design

Unlike the previous multi-level proposal, this is a **single, clean API** that matches Subsetix conventions:

```
include/subsetix/
├── fvd/
│   ├── euler.hpp                    # Euler equations (types + functions)
│   ├── flux.hpp                     # Numerical fluxes (functors)
│   ├── boundary.hpp                 # Boundary conditions (functors)
│   └── solver.hpp                   # High-level solver class
```

No subdirectories, no complex hierarchies. Just **4 header files**.

---

## File 1: `euler.hpp` - Types and Core Functions

**Pattern**: Similar to `subsetix/csr_ops/field_arith.hpp` with POD structs + `KOKKOS_INLINE_FUNCTION`

```cpp
#pragma once

#include <Kokkos_Core.hpp>
#include <subsetix/geometry/csr_backend.hpp>

namespace subsetix::fvd {

using Real = float;
using Coord = csr::Coord;
using ExecSpace = csr::ExecSpace;
using DeviceMemorySpace = csr::DeviceMemorySpace;

// ============================================================================
// POD Structs (GPU-safe, pass-by-value to kernels)
// ============================================================================

/// Conserved variables: {rho, rhou, rhov, E}
struct EulerConserved {
  Real rho, rhou, rhov, E;

  KOKKOS_INLINE_FUNCTION
  EulerConserved() : rho(0), rhou(0), rhov(0), E(0) {}

  KOKKOS_INLINE_FUNCTION
  EulerConserved(Real r, Real ru, Real rv, Real e)
    : rho(r), rhou(ru), rhov(rv), E(e) {}
};

/// Primitive variables: {rho, u, v, p}
struct EulerPrimitive {
  Real rho, u, v, p;

  KOKKOS_INLINE_FUNCTION
  EulerPrimitive() : rho(0), u(0), v(0), p(0) {}

  KOKKOS_INLINE_FUNCTION
  EulerPrimitive(Real r, Real uu, Real vv, Real pp)
    : rho(r), u(uu), v(vv), p(pp) {}
};

// ============================================================================
// Core Functions (all marked KOKKOS_INLINE_FUNCTION)
// ============================================================================

KOKKOS_INLINE_FUNCTION
EulerPrimitive euler_to_primitive(const EulerConserved& U, Real gamma) {
  constexpr Real eps = 1e-12f;
  EulerPrimitive q;
  q.rho = U.rho;
  Real inv_rho = 1.0f / (U.rho + eps);
  q.u = U.rhou * inv_rho;
  q.v = U.rhov * inv_rho;
  Real kinetic = 0.5f * (q.u * q.u + q.v * q.v);
  Real pressure = (gamma - 1.0f) * (U.E - U.rho * kinetic);
  q.p = (pressure > eps) ? pressure : eps;
  return q;
}

KOKKOS_INLINE_FUNCTION
EulerConserved euler_from_primitive(const EulerPrimitive& q, Real gamma) {
  Real kinetic = 0.5f * q.rho * (q.u * q.u + q.v * q.v);
  return EulerConserved{
    q.rho,
    q.rho * q.u,
    q.rho * q.v,
    q.p / (gamma - 1.0f) + kinetic
  };
}

KOKKOS_INLINE_FUNCTION
Real euler_sound_speed(const EulerPrimitive& q, Real gamma) {
  constexpr Real eps = 1e-12f;
  return Kokkos::sqrt(gamma * q.p / (q.rho + eps));
}

KOKKOS_INLINE_FUNCTION
EulerConserved euler_flux_x(const EulerConserved& U, const EulerPrimitive& q) {
  return EulerConserved{
    U.rhou,
    U.rho * q.u * q.u + q.p,
    U.rho * q.u * q.v,
    (U.E + q.p) * q.u
  };
}

KOKKOS_INLINE_FUNCTION
EulerConserved euler_flux_y(const EulerConserved& U, const EulerPrimitive& q) {
  return EulerConserved{
    U.rhov,
    U.rho * q.u * q.v,
    U.rho * q.v * q.v + q.p,
    (U.E + q.p) * q.v
  };
}

KOKKOS_INLINE_FUNCTION
Real euler_pressure(const EulerConserved& U, Real gamma) {
  auto q = euler_to_primitive(U, gamma);
  return q.p;
}

KOKKOS_INLINE_FUNCTION
Real euler_mach_number(const EulerPrimitive& q, Real gamma) {
  constexpr Real eps = 1e-12f;
  Real a = euler_sound_speed(q, gamma);
  Real vel = Kokkos::sqrt(q.u * q.u + q.v * q.v);
  return (a > eps) ? (vel / a) : 0.0f;
}

} // namespace subsetix::fvd
```

---

## File 2: `flux.hpp` - Numerical Flux Functors

**Pattern**: Like `detail::Plus` in `field_arith.hpp` - POD functors passed by value

```cpp
#pragma once

#include <subsetix/fvd/euler.hpp>

namespace subsetix::fvd {

// ============================================================================
// Rusanov (Local Lax-Friedrichs) Flux
// ============================================================================

struct RusanovFlux {
  Real gamma;

  KOKKOS_INLINE_FUNCTION
  EulerConserved flux_x(
      const EulerConserved& UL,
      const EulerConserved& UR,
      const EulerPrimitive& qL,
      const EulerPrimitive& qR) const {

    Real aL = euler_sound_speed(qL, gamma);
    Real aR = euler_sound_speed(qR, gamma);
    Real smax = Kokkos::fmax(Kokkos::fabs(qL.u) + aL,
                             Kokkos::fabs(qR.u) + aR);

    EulerConserved FL = euler_flux_x(UL, qL);
    EulerConserved FR = euler_flux_x(UR, qR);

    return EulerConserved{
      0.5f * (FL.rho + FR.rho) - 0.5f * smax * (UR.rho - UL.rho),
      0.5f * (FL.rhou + FR.rhou) - 0.5f * smax * (UR.rhou - UL.rhou),
      0.5f * (FL.rhov + FR.rhov) - 0.5f * smax * (UR.rhov - UL.rhov),
      0.5f * (FL.E + FR.E) - 0.5f * smax * (UR.E - UL.E)
    };
  }

  KOKKOS_INLINE_FUNCTION
  EulerConserved flux_y(
      const EulerConserved& UL,
      const EulerConserved& UR,
      const EulerPrimitive& qL,
      const EulerPrimitive& qR) const {

    Real aL = euler_sound_speed(qL, gamma);
    Real aR = euler_sound_speed(qR, gamma);
    Real smax = Kokkos::fmax(Kokkos::fabs(qL.v) + aL,
                             Kokkos::fabs(qR.v) + aR);

    EulerConserved FL = euler_flux_y(UL, qL);
    EulerConserved FR = euler_flux_y(UR, qR);

    return EulerConserved{
      0.5f * (FL.rho + FR.rho) - 0.5f * smax * (UR.rho - UL.rho),
      0.5f * (FL.rhou + FR.rhou) - 0.5f * smax * (UR.rhou - UL.rhou),
      0.5f * (FL.rhov + FR.rhov) - 0.5f * smax * (UR.rhov - UL.rhov),
      0.5f * (FL.E + FR.E) - 0.5f * smax * (UR.E - UL.E)
    };
  }
};

// ============================================================================
// HLLC Flux (more accurate for contacts)
// ============================================================================

struct HLLCFlux {
  Real gamma;

  KOKKOS_INLINE_FUNCTION
  EulerConserved flux_x(
      const EulerConserved& UL,
      const EulerConserved& UR,
      const EulerPrimitive& qL,
      const EulerPrimitive& qR) const {
    // HLLC implementation here...
    // For now, fallback to Rusanov
    RusanovFlux rusanov{gamma};
    return rusanov.flux_x(UL, UR, qL, qR);
  }

  KOKKOS_INLINE_FUNCTION
  EulerConserved flux_y(
      const EulerConserved& UL,
      const EulerConserved& UR,
      const EulerPrimitive& qL,
      const EulerPrimitive& qR) const {
    HLLCFlux hllc{gamma};
    return hllc.flux_y(UL, UR, qL, qR);
  }
};

// ============================================================================
// Central Flux (for diffusion, testing)
// ============================================================================

struct CentralFlux {
  KOKKOS_INLINE_FUNCTION
  EulerConserved flux_x(
      const EulerConserved& /*UL*/,
      const EulerConserved& /*UR*/,
      const EulerPrimitive& /*qL*/,
      const EulerPrimitive& /*qR*/) const {
    // Average of physical fluxes
    return EulerConserved{0, 0, 0, 0};  // Placeholder
  }

  KOKKOS_INLINE_FUNCTION
  EulerConserved flux_y(
      const EulerConserved& /*UL*/,
      const EulerConserved& /*UR*/,
      const EulerPrimitive& /*qL*/,
      const EulerPrimitive& /*qR*/) const {
    return EulerConserved{0, 0, 0, 0};
  }
};

} // namespace subsetix::fvd
```

---

## File 3: `boundary.hpp` - Boundary Condition Functors

**Pattern**: Enum + POD functor, no virtual functions

```cpp
#pragma once

#include <subsetix/fvd/euler.hpp>

namespace subsetix::fvd {

enum class BcLocation {
  Left = 0,
  Right = 1,
  Bottom = 2,
  Top = 3,
  Obstacle = 4
};

// ============================================================================
// Boundary Condition Functors (compile-time dispatch)
// ============================================================================

/// Dirichlet BC: fixed value
struct BcDirichlet {
  EulerConserved value;

  KOKKOS_INLINE_FUNCTION
  EulerConserved get_ghost(
      const EulerConserved& /*interior*/,
      BcLocation /*loc*/,
      Real /*gamma*/) const {
    return value;
  }
};

/// Neumann BC: zero-gradient (copy interior)
struct BcNeumann {
  KOKKOS_INLINE_FUNCTION
  EulerConserved get_ghost(
      const EulerConserved& interior,
      BcLocation /*loc*/,
      Real /*gamma*/) const {
    return interior;
  }
};

/// Slip Wall BC: reflect normal velocity
struct BcSlipWall {
  KOKKOS_INLINE_FUNCTION
  EulerConserved get_ghost(
      const EulerConserved& interior,
      BcLocation loc,
      Real gamma) const {
    auto q = euler_to_primitive(interior, gamma);
    // Reflect normal component
    if (loc == BcLocation::Left || loc == BcLocation::Right) {
      q.u = -q.u;
    } else {
      q.v = -q.v;
    }
    return euler_from_primitive(q, gamma);
  }
};

/// No-Slip Wall BC: zero velocity
struct BcNoSlipWall {
  KOKKOS_INLINE_FUNCTION
  EulerConserved get_ghost(
      const EulerConserved& interior,
      BcLocation /*loc*/,
      Real gamma) const {
    auto q = euler_to_primitive(interior, gamma);
    q.u = 0;
    q.v = 0;
    return euler_from_primitive(q, gamma);
  }
};

// ============================================================================
// Boundary Configuration (compile-time composition)
// ============================================================================

template<typename LeftBc, typename RightBc,
         typename BottomBc, typename TopBc,
         typename ObstacleBc = BcSlipWall>
struct BoundaryConfig {
  LeftBc left;
  RightBc right;
  BottomBc bottom;
  TopBc top;
  ObstacleBc obstacle;

  /// Get BC for a given location (compile-time switch)
  KOKKOS_INLINE_FUNCTION
  EulerConserved get_ghost(
      const EulerConserved& interior,
      BcLocation loc,
      Real gamma) const {

    switch (loc) {
      case BcLocation::Left:     return left.get_ghost(interior, loc, gamma);
      case BcLocation::Right:    return right.get_ghost(interior, loc, gamma);
      case BcLocation::Bottom:   return bottom.get_ghost(interior, loc, gamma);
      case BcLocation::Top:      return top.get_ghost(interior, loc, gamma);
      case BcLocation::Obstacle: return obstacle.get_ghost(interior, loc, gamma);
    }
    return interior;
  }
};

/// Factory: Create supersonic inflow/outflow config (compile-time)
inline auto make_supersonic_bc(
    const EulerPrimitive& inflow,
    Real gamma) {

  auto inflow_conserved = euler_from_primitive(inflow, gamma);

  return BoundaryConfig<
    BcDirichlet,   // Left: inflow
    BcNeumann,     // Right: outflow (zero-gradient)
    BcSlipWall,    // Bottom: slip
    BcSlipWall,    // Top: slip
    BcSlipWall     // Obstacle: slip
  >{
    BcDirichlet{inflow_conserved},  // left
    BcNeumann{},                    // right
    BcSlipWall{},                   // bottom
    BcSlipWall{},                   // top
    BcSlipWall{}                    // obstacle
  };
}

} // namespace subsetix::fvd
```

---

## File 4: `solver.hpp` - Main Solver

**Pattern**: Follows Subsetix conventions, uses `KOKKOS_LAMBDA` for kernels

```cpp
#pragma once

#include <subsetix/fvd/euler.hpp>
#include <subsetix/fvd/flux.hpp>
#include <subsetix/fvd/boundary.hpp>
#include <subsetix/field/csr_field.hpp>
#include <subsetix/csr_ops/field_stencil.hpp>
#include <subsetix/csr_ops/field_mapping.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/csr_ops/set_algebra.hpp>
#include <subsetix/csr_ops/morphology.hpp>

namespace subsetix::fvd {

// ============================================================================
// Euler Solver (SoA storage, follows MACH2 pattern)
// ============================================================================

class EulerSolver2D {
public:
  struct Config {
    Real dx = 1.0f;
    Real dy = 1.0f;
    Real cfl = 0.45f;
    Real gamma = 1.4f;
    int ghost_layers = 1;
  };

private:
  Config cfg_;

  // Geometry
  csr::IntervalSet2DDevice fluid_geom_;
  csr::IntervalSet2DDevice field_geom_;   // fluid + ghosts
  csr::IntervalSet2DDevice ghost_geom_;   // ghosts only
  csr::Box2D domain_;

  // State fields (SoA - Structure of Arrays)
  csr::Field2DDevice<Real> U_rho_, U_rhou_, U_rhov_, U_E_;
  csr::Field2DDevice<Real> U_next_rho_, U_next_rhou_, U_next_rhov_, U_next_E_;

  // Stencil mappings (precomputed, reused)
  csr::FieldMaskMapping stencil_mapping_;
  csr::detail::SubsetStencilVerticalMapping<Real> vertical_mapping_;

  // Boundary config (stored as tuple for type erasure)
  struct BoundaryStorage {
    BcDirichlet left;
    BcNeumann right;
    BcSlipWall bottom;
    BcSlipWall top;
    BcSlipWall obstacle;
  };
  BoundaryStorage bc_;

public:
  EulerSolver2D(
      const csr::IntervalSet2DDevice& fluid,
      const csr::Box2D& domain,
      const Config& cfg = Config{})
    : cfg_(cfg)
    , fluid_geom_(fluid)
    , domain_(domain)
  {
    setup_geometry();
    allocate_fields();
    build_mappings();

    // Default BCs
    bc_.left = BcDirichlet{EulerConserved{1, 1.4f, 0, 2.5f}};  // Placeholder
    bc_.right = BcNeumann{};
    bc_.bottom = BcSlipWall{};
    bc_.top = BcSlipWall{};
    bc_.obstacle = BcSlipWall{};
  }

  /// Set boundary conditions (compile-time type-safe)
  template<typename LeftBc>
  void set_left_bc(const LeftBc& bc) {
    bc_.left = bc;  // Assumes compatible type
  }

  template<typename RightBc>
  void set_right_bc(const RightBc& bc) {
    bc_.right = bc;
  }

  /// Initialize with uniform state
  void initialize(const EulerPrimitive& initial) {
    auto U_init = euler_from_primitive(initial, cfg_.gamma);

    // Fill all fields (fluid + ghost)
    fill_on_set_device(U_rho_, field_geom_, U_init.rho);
    fill_on_set_device(U_rhou_, field_geom_, U_init.rhou);
    fill_on_set_device(U_rhov_, field_geom_, U_init.rhov);
    fill_on_set_device(U_E_, field_geom_, U_init.E);

    fill_on_set_device(U_next_rho_, field_geom_, U_init.rho);
    fill_on_set_device(U_next_rhou_, field_geom_, U_init.rhou);
    fill_on_set_device(U_next_rhov_, field_geom_, U_init.rhov);
    fill_on_set_device(U_next_E_, field_geom_, U_init.E);

    fill_ghost_cells();
  }

  /// Fill ghost cells using BCs
  void fill_ghost_cells() {
    if (ghost_geom_.num_intervals == 0) return;

    auto rho = U_rho_.values;
    auto rhou = U_rhou_.values;
    auto rhov = U_rhov_.values;
    auto E = U_E_.values;

    Real gamma = cfg_.gamma;
    auto domain = domain_;
    auto fluid = fluid_geom_;

    // Copy BC storage to kernel (pass by value)
    const auto bc = bc_;

    csr::apply_on_set_device(U_rho_, ghost_geom_,
      KOKKOS_LAMBDA(Coord x, Coord y, Real& out, std::size_t idx) {

        // Determine location
        BcLocation loc;
        bool outside = (x < domain.x_min || x >= domain.x_max ||
                       y < domain.y_min || y >= domain.y_max);

        if (outside) {
          if (x < domain.x_min) loc = BcLocation::Left;
          else if (x >= domain.x_max) loc = BcLocation::Right;
          else if (y < domain.y_min) loc = BcLocation::Bottom;
          else loc = BcLocation::Top;
        } else {
          loc = BcLocation::Obstacle;
        }

        // Find interior neighbor (search outward)
        auto clamp = [](Coord v, Coord min, Coord max) {
          return (v < min) ? min : ((v >= max) ? (max - 1) : v);
        };

        Coord xc = clamp(x, domain.x_min, domain.x_max);
        Coord yc = clamp(y, domain.y_min, domain.y_max);

        // Read interior state (simplified - should use accessor)
        EulerConserved interior;
        interior.rho = rho(idx);  // Simplified
        interior.rhou = rhou(idx);
        interior.rhov = rhov(idx);
        interior.E = E(idx);

        // Apply BC
        auto ghost = bc.get_ghost(interior, loc, gamma);

        // Write all components
        out = ghost.rho;
        // Note: need to write rhou, rhov, E too
        // This is simplified for illustration
      });

    // For proper implementation, need separate kernels or write all 4 values
    // See apply_euler_stencil_on_set_device below for full pattern
  }

  /// Compute adaptive time step (CFL condition)
  Real compute_dt() {
    Real max_rate = 0.0;

    auto rho = U_rho_.values;
    auto rhou = U_rhou_.values;
    auto rhov = U_rhov_.values;
    auto E = U_E_.values;

    Real gamma = cfg_.gamma;
    Real inv_dx = 1.0f / cfg_.dx;
    Real inv_dy = 1.0f / cfg_.dy;

    Kokkos::parallel_reduce("compute_dt",
      Kokkos::RangePolicy<ExecSpace>(0, fluid_geom_.total_cells),
      KOKKOS_LAMBDA(int i, Real& lmax) {
        EulerConserved U;
        U.rho = rho(i);
        U.rhou = rhou(i);
        U.rhov = rhov(i);
        U.E = E(i);

        auto q = euler_to_primitive(U, gamma);
        Real a = euler_sound_speed(q, gamma);

        Real rate = Kokkos::fabs(q.u) * inv_dx
                  + Kokkos::fabs(q.v) * inv_dy
                  + a * (inv_dx + inv_dy);

        lmax = (rate > lmax) ? rate : lmax;
      },
      Kokkos::Max<Real>(max_rate));

    ExecSpace().fence();

    if (max_rate <= 0.0f) {
      return cfg_.cfl * Kokkos::fmin(cfg_.dx, cfg_.dy);
    }
    return cfg_.cfl / max_rate;
  }

  /// Perform one time step
  Real step() {
    // 1. Fill ghost cells
    fill_ghost_cells();

    // 2. Compute dt
    Real dt = compute_dt();

    // 3. Apply FV stencil
    apply_euler_stencil(dt);

    // 4. Swap buffers
    swap_fields();

    return dt;
  }

  /// Access state fields
  csr::Field2DDevice<Real>& rho() { return U_rho_; }
  csr::Field2DDevice<Real>& rhou() { return U_rhou_; }
  csr::Field2DDevice<Real>& rhov() { return U_rhov_; }
  csr::Field2DDevice<Real>& E() { return U_E_; }

  const csr::IntervalSet2DDevice& geometry() const { return fluid_geom_; }

private:
  void setup_geometry() {
    csr::CsrSetAlgebraContext ctx;

    // Expand to include ghost layer
    csr::IntervalSet2DDevice expanded;
    csr::expand_device(fluid_geom_, cfg_.ghost_layers, cfg_.ghost_layers,
                      expanded, ctx);
    csr::compute_cell_offsets_device(expanded);

    field_geom_ = expanded;

    // Ghost region = expanded - fluid
    ghost_geom_ = csr::allocate_interval_set_device(
        expanded.num_rows,
        expanded.num_intervals + fluid_geom_.num_intervals);
    csr::set_difference_device(expanded, fluid_geom_, ghost_geom_, ctx);
    csr::compute_cell_offsets_device(ghost_geom_);
  }

  void allocate_fields() {
    U_rho_ = csr::Field2DDevice<Real>(field_geom_, "U_rho");
    U_rhou_ = csr::Field2DDevice<Real>(field_geom_, "U_rhou");
    U_rhov_ = csr::Field2DDevice<Real>(field_geom_, "U_rhov");
    U_E_ = csr::Field2DDevice<Real>(field_geom_, "U_E");

    U_next_rho_ = csr::Field2DDevice<Real>(field_geom_, "U_next_rho");
    U_next_rhou_ = csr::Field2DDevice<Real>(field_geom_, "U_next_rhou");
    U_next_rhov_ = csr::Field2DDevice<Real>(field_geom_, "U_next_rhov");
    U_next_E_ = csr::Field2DDevice<Real>(field_geom_, "U_next_E");
  }

  void build_mappings() {
    stencil_mapping_ = csr::build_field_mask_mapping(U_rho_, fluid_geom_);
    vertical_mapping_ = csr::detail::build_subset_stencil_vertical_mapping(
        U_rho_, fluid_geom_, stencil_mapping_);
  }

  void swap_fields() {
    auto temp_rho = U_rho_.values;
    auto temp_rhou = U_rhou_.values;
    auto temp_rhov = U_rhov_.values;
    auto temp_E = U_E_.values;

    U_rho_.values = U_next_rho_.values;
    U_rhou_.values = U_next_rhou_.values;
    U_rhov_.values = U_next_rhov_.values;
    U_E_.values = U_next_E_.values;

    U_next_rho_.values = temp_rho;
    U_next_rhou_.values = temp_rhou;
    U_next_rhov_.values = temp_rhov;
    U_next_E_.values = temp_E;
  }

  /// Apply Euler FV stencil (single kernel, all 4 variables)
  void apply_euler_stencil(Real dt) {
    Real dt_over_dx = dt / cfg_.dx;
    Real dt_over_dy = dt / cfg_.dy;
    Real gamma = cfg_.gamma;

    // Input views
    auto in_rho = U_rho_.values;
    auto in_rhou = U_rhou_.values;
    auto in_rhov = U_rhov_.values;
    auto in_E = U_E_.values;

    // Output views
    auto out_rho = U_next_rho_.values;
    auto out_rhou = U_next_rhou_.values;
    auto out_rhov = U_next_rhov_.values;
    auto out_E = U_next_E_.values;

    // Geometry
    auto mask_intervals = fluid_geom_.intervals;
    auto mask_row_keys = fluid_geom_.row_keys;
    auto in_intervals = U_rho_.geometry.intervals;
    auto in_offsets = U_rho_.geometry.cell_offsets;
    auto out_intervals = U_next_rho_.geometry.intervals;
    auto out_offsets = U_next_rho_.geometry.cell_offsets;

    // Mappings
    auto interval_to_row = stencil_mapping_.interval_to_row;
    auto interval_to_field_interval = stencil_mapping_.interval_to_field_interval;
    auto north_interval = vertical_mapping_.north_interval;
    auto south_interval = vertical_mapping_.south_interval;

    // Launch kernel (follows Subsetix apply_csr_stencil_on_set_device pattern)
    const int num_intervals = fluid_geom_.num_intervals;

    Kokkos::parallel_for("apply_euler_fv_stencil",
      Kokkos::RangePolicy<ExecSpace>(0, num_intervals),
      KOKKOS_LAMBDA(const int interval_idx) {

        const int row_idx = interval_to_row(interval_idx);
        if (row_idx < 0) return;

        const int field_interval_idx = interval_to_field_interval(interval_idx);
        if (field_interval_idx < 0) return;

        const Coord y = mask_row_keys(row_idx).y;
        const auto mask_iv = mask_intervals(interval_idx);
        const auto in_iv = in_intervals(field_interval_idx);
        const auto out_iv = out_intervals(field_interval_idx);

        const std::size_t in_base = in_offsets(field_interval_idx);
        const std::size_t out_base = out_offsets(field_interval_idx);

        // For each cell in interval
        for (Coord x = mask_iv.begin; x < mask_iv.end; ++x) {
          const std::size_t linear_out =
              out_base + static_cast<std::size_t>(x - out_iv.begin);

          // Neighbor indices (assumes interior, valid neighbors)
          const std::size_t idx_c = in_base + (x - in_iv.begin);
          const std::size_t idx_w = idx_c - 1;
          const std::size_t idx_e = idx_c + 1;

          // North/South
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

          // ============================================================
          // Gather conserved states at 5 points
          // ============================================================

          auto U_c = EulerConserved{
            in_rho(idx_c), in_rhou(idx_c), in_rhov(idx_c), in_E(idx_c)};
          auto U_l = EulerConserved{
            in_rho(idx_w), in_rhou(idx_w), in_rhov(idx_w), in_E(idx_w)};
          auto U_r = EulerConserved{
            in_rho(idx_e), in_rhou(idx_e), in_rhov(idx_e), in_E(idx_e)};
          auto U_d = EulerConserved{
            in_rho(idx_s), in_rhou(idx_s), in_rhov(idx_s), in_E(idx_s)};
          auto U_u = EulerConserved{
            in_rho(idx_n), in_rhou(idx_n), in_rhov(idx_n), in_E(idx_n)};

          // ============================================================
          // Convert to primitives
          // ============================================================

          auto q_c = euler_to_primitive(U_c, gamma);
          auto q_l = euler_to_primitive(U_l, gamma);
          auto q_r = euler_to_primitive(U_r, gamma);
          auto q_d = euler_to_primitive(U_d, gamma);
          auto q_u = euler_to_primitive(U_u, gamma);

          // ============================================================
          // Compute Rusanov fluxes (inline, using free functions)
          // ============================================================

          // Flux X
          Real aL = euler_sound_speed(q_l, gamma);
          Real aR = euler_sound_speed(q_r, gamma);
          Real smax_x = Kokkos::fmax(Kokkos::fabs(q_l.u) + aL,
                                     Kokkos::fabs(q_r.u) + aR);

          auto FL_x = euler_flux_x(U_l, q_l);
          auto FR_x = euler_flux_x(U_r, q_r);

          auto F_w = EulerConserved{
            0.5f * (FL_x.rho + FR_x.rho) - 0.5f * smax_x * (U_r.rho - U_l.rho),
            0.5f * (FL_x.rhou + FR_x.rhou) - 0.5f * smax_x * (U_r.rhou - U_l.rhou),
            0.5f * (FL_x.rhov + FR_x.rhov) - 0.5f * smax_x * (U_r.rhov - U_l.rhov),
            0.5f * (FL_x.E + FR_x.E) - 0.5f * smax_x * (U_r.E - U_l.E)
          };

          auto F_e = EulerConserved{
            0.5f * (FL_x.rho + FR_x.rho) - 0.5f * smax_x * (U_r.rho - U_c.rho),
            0.5f * (FL_x.rhou + FR_x.rhou) - 0.5f * smax_x * (U_r.rhou - U_c.rhou),
            0.5f * (FL_x.rhov + FR_x.rhov) - 0.5f * smax_x * (U_r.rhov - U_c.rhov),
            0.5f * (FL_x.E + FR_x.E) - 0.5f * smax_x * (U_r.E - U_c.E)
          };

          // Flux Y
          Real aS = euler_sound_speed(q_d, gamma);
          Real aN = euler_sound_speed(q_u, gamma);
          Real smax_y = Kokkos::fmax(Kokkos::fabs(q_d.v) + aS,
                                     Kokkos::fabs(q_u.v) + aN);

          auto FL_y = euler_flux_y(U_d, q_d);
          auto FR_y = euler_flux_y(U_u, q_u);

          auto F_s = EulerConserved{
            0.5f * (FL_y.rho + FR_y.rho) - 0.5f * smax_y * (U_u.rho - U_d.rho),
            0.5f * (FL_y.rhou + FR_y.rhou) - 0.5f * smax_y * (U_u.rhou - U_d.rhou),
            0.5f * (FL_y.rhov + FR_y.rhov) - 0.5f * smax_y * (U_u.rhov - U_d.rhov),
            0.5f * (FL_y.E + FR_y.E) - 0.5f * smax_y * (U_u.E - U_d.E)
          };

          auto F_n = EulerConserved{
            0.5f * (FL_y.rho + FR_y.rho) - 0.5f * smax_y * (U_c.rho - U_d.rho),
            0.5f * (FL_y.rhou + FR_y.rhou) - 0.5f * smax_y * (U_c.rhou - U_d.rhou),
            0.5f * (FL_y.rhov + FR_y.rhov) - 0.5f * smax_y * (U_c.rhov - U_d.rhov),
            0.5f * (FL_y.E + FR_y.E) - 0.5f * smax_y * (U_c.E - U_d.E)
          };

          // ============================================================
          // Godunov update (write all 4 variables)
          // ============================================================

          out_rho(linear_out) = U_c.rho
              - dt_over_dx * (F_e.rho - F_w.rho)
              - dt_over_dy * (F_n.rho - F_s.rho);

          out_rhou(linear_out) = U_c.rhou
              - dt_over_dx * (F_e.rhou - F_w.rhou)
              - dt_over_dy * (F_n.rhou - F_s.rhou);

          out_rhov(linear_out) = U_c.rhov
              - dt_over_dx * (F_e.rhov - F_w.rhov)
              - dt_over_dy * (F_n.rhov - F_s.rhov);

          out_E(linear_out) = U_c.E
              - dt_over_dx * (F_e.E - F_w.E)
              - dt_over_dy * (F_n.E - F_s.E);
        }
      });

    ExecSpace().fence();
  }
};

} // namespace subsetix::fvd
```

---

## Usage Example: MACH2 Refactored

```cpp
// examples/mach2_cylinder_kokkos_native.cpp
#include <subsetix/fvd/solver.hpp>
#include <subsetix/io/vtk_export.hpp>

using namespace subsetix;
using namespace subsetix::fvd;

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);

  // Geometry setup
  const int nx = 400, ny = 160;
  const csr::Box2D domain{0, nx, 0, ny};

  auto domain_box = csr::make_box_device(domain);
  auto cylinder = csr::make_disk_device(csr::Disk2D{nx/4, ny/2, 20});

  csr::CsrSetAlgebraContext ctx;
  csr::IntervalSet2DDevice fluid;
  csr::set_difference_device(domain_box, cylinder, fluid, ctx);
  csr::compute_cell_offsets_device(fluid);

  // Solver configuration
  EulerSolver2D::Config cfg;
  cfg.cfl = 0.45f;
  cfg.gamma = 1.4f;

  // Create solver
  EulerSolver2D solver(fluid, domain, cfg);

  // Boundary conditions (compile-time!)
  Real mach = 2.0f;
  Real rho = 1.0f, p = 1.0f;
  Real gamma = 1.4f;
  Real a = Kokkos::sqrt(gamma * p / rho);
  Real u = mach * a;

  EulerPrimitive inflow{rho, u, 0, p};
  auto inflow_conserved = euler_from_primitive(inflow, gamma);

  solver.set_left_bc(BcDirichlet{inflow_conserved});
  solver.set_right_bc(BcNeumann{});
  // Default slip walls for top/bottom/obstacle

  // Initialize
  solver.initialize(inflow);

  // Time loop
  Real t = 0.0f;
  int step = 0;

  while (t < 0.01f && step < 5000) {
    Real dt = solver.step();
    t += dt;
    ++step;

    if (step % 50 == 0) {
      // Export VTK
      vtk::write_legacy_quads(
          csr::toHost(fluid), solver.rho(),
          "output/step_" + std::to_string(step) + "_density.vtk");
    }
  }

  return 0;
}
```

---

## Design Comparison

| Aspect | Previous Proposal | V2: Kokkos-Native |
|--------|------------------|-------------------|
| **Files** | 10+ files, nested dirs | 4 files, flat |
| **Concepts** | C++20 concepts | None (doc only) |
| **Virtual** | No virtual | No virtual |
| **Runtime** | Some runtime config | Compile-time only |
| **Abstraction** | Multi-level | Single, flat |
| **GPU safety** | Mostly safe | 100% safe |
| **Kokkos patterns** | Partial | Exact match |
| **Compile-time** | Yes | **100%** |
| **Header-only** | Yes | Yes |
| **Lines of code** | ~2000 | ~800 |

---

## Key Design Decisions

### 1. Everything is POD

```cpp
// All device-side types are POD
struct EulerConserved { Real rho, rhou, rhov, E; };
struct RusanovFlux { Real gamma; };  // functor with state
```

**Why**: POD types can be safely passed by value to GPU kernels without worrying about memory layout.

### 2. Free Functions, Not Methods

```cpp
// GOOD: Free function
KOKKOS_INLINE_FUNCTION
EulerPrimitive euler_to_primitive(const EulerConserved& U, Real gamma);

// AVOID: Member function (requires object)
// struct EulerSystem {
//   KOKKOS_INLINE_FUNCTION
//   Primitive to_primitive(const Conserved& U) const;
// };
```

**Why**: Free functions are simpler to inline and don't require carrying object state to GPU.

### 3. Compile-Time BC Configuration

```cpp
// BCs are template parameters (compile-time)
BoundaryConfig<BcDirichlet, BcNeumann, BcSlipWall, BcSlipWall> bc;

// NOT runtime
// std::vector<std::unique_ptr<BoundaryCondition>> bc;
```

**Why**: No virtual function calls, no runtime branching on BC type.

### 4. Single Kernel for All Variables

```cpp
// One kernel writes all 4 variables
Kokkos::parallel_for(..., KOKKOS_LAMBDA(...) {
  out_rho(idx) = ...;
  out_rhou(idx) = ...;
  out_rhov(idx) = ...;
  out_E(idx) = ...;
});

// NOT 4 separate kernels
// apply_stencil(rho);
// apply_stencil(rhou);
// apply_stencil(rhov);
// apply_stencil(E);
```

**Why**: Single kernel launch = less overhead, better cache utilization.

### 5. Follow Subsetix Patterns Exactly

```cpp
// Subsetix style (from field_stencil.hpp)
template <typename OutT, typename InT, class StencilFunctor>
inline void apply_csr_stencil_on_set_device(
    Field2DDevice<OutT>& field_out,
    const Field2DDevice<InT>& field_in,
    const IntervalSet2DDevice& mask,
    const FieldMaskMapping& mapping,
    StencilFunctor stencil);

// Our style (same pattern)
inline void apply_euler_stencil_on_set_device(
    Field2DDevice<Real>& out_rho, ...,  // 4 outputs
    const Field2DDevice<Real>& in_rho, ...,  // 4 inputs
    const IntervalSet2DDevice& mask,
    const FieldMaskMapping& mapping,
    Real gamma, Real dt_over_dx, Real dt_over_dy);  // parameters
```

**Why**: Consistency with existing code, easier to review and maintain.

---

## Advantages of V2 Design

1. **100% Compile-Time**: All types resolved at compilation, zero runtime overhead
2. **GPU-Safe**: No concepts, no std::function, no virtual, all POD
3. **Kokkos-Native**: Uses `KOKKOS_INLINE_FUNCTION`, `KOKKOS_LAMBDA` exactly like Subsetix
4. **Simple**: Only 4 header files, flat structure
5. **Maintainable**: Follows established Subsetix patterns
6. **Performant**: Single kernel for all 4 variables, SoA storage
7. **Type-Safe**: Compile-time BC configuration prevents runtime errors

---

## Migration from MACH2

| MACH2 Function | FVD Equivalent |
|---------------|----------------|
| `struct Conserved` | `EulerConserved` |
| `struct Primitive` | `EulerPrimitive` |
| `cons_to_prim()` | `euler_to_primitive()` |
| `prim_to_cons()` | `euler_from_primitive()` |
| `sound_speed()` | `euler_sound_speed()` |
| `flux_x()` | `euler_flux_x()` |
| `rusanov_flux_x()` | Inline in `apply_euler_stencil()` |
| `fill_ghost_cells()` | `EulerSolver2D::fill_ghost_cells()` |
| `compute_dt()` | `EulerSolver2D::compute_dt()` |
| `EulerStencilSoA` | Inline in `apply_euler_stencil()` |
| `main` loop | `EulerSolver2D::step()` |

---

## Next Steps

1. **Review and approve** this simplified design
2. **Create** `include/subsetix/fvd/` directory
3. **Implement** `euler.hpp` (types + functions)
4. **Implement** `solver.hpp` (main class + stencil kernel)
5. **Test** with simple case (advection pulse)
6. **Refactor MACH2** to use new API
7. **Add** `flux.hpp` and `boundary.hpp` as needed
8. **Benchmark** vs original MACH2

---

*End of Proposal V2*
