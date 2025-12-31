# FVD High-Level API - Next Steps Design

## Goals

1. **Time-Dependent BCs** - Type and zone can change during simulation
2. **AMR with Coarsening** - Combinable criteria, exclusion zones, custom refinement
3. **High-Order Time Integration** - Compile-time schemes, variable dt

**Design Principles:**
- GPU-first: minimize host/device syncs
- Kokkos-native patterns
- C++20 concepts and templates
- Zero-cost abstraction where possible

---

## 1. Time-Dependent Boundary Conditions

### 1.1 Problem Statement

Current limitations:
- BCs are fixed at construction time
- No time dependence
- Zones are static
- Type cannot change (Dirichlet → Neumann)

Requirements:
- BC *type* can change (Wall → Inlet)
- BC *zone* can change (expanding/contracting regions)
- High performance on GPU (minimal host/device syncs)

### 1.2 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Solver (Host)                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  BCManager (Host-side orchestration)                  │  │
│  │  - add_bc(zone, bc_descriptor)                        │  │
│  │  - update_bc(zone, new_descriptor)                    │  │
│  │  - remove_bc(zone)                                    │  │
│  │  - sync_to_device()  [explicit sync]                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  BCRegistry (Device View)                             │  │
│  │  - Kokkos::View<BCDescriptor*> bc_zones              │  │
│  │  - spatial lookup: (i,j) → BCDescriptor               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼ (device kernel)
┌─────────────────────────────────────────────────────────────┐
│                  BCDescriptor (Device POD)                  │
│  - Type enum (Dirichlet, Neumann, Reflective, TimeDep)     │
│  - Conserved value (for static BCs)                        │
│  - TimeFunction index (for time-dependent BCs)             │
│  - Zone geometry (for zonal BCs)                           │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Time-Dependent BC Interface

```cpp
// fvd/boundary/time_dependent_bc.hpp
#pragma once

#include <Kokkos_Core.hpp>
#include <concepts>
#include "../system/concepts_v2.hpp"

namespace subsetix::fvd::boundary {

// ============================================================================
// TIME-DEPENDENT BC POLICY (Device-Side)
// ============================================================================

/**
 * @brief Policy for time-dependent boundary conditions
 *
 * This is a POD type (no virtual functions) that can be used
 * directly in GPU kernels. All parameters are stored in Kokkos::Views
 * for device access.
 *
 * Design: Template-based policy where the time dependence is
 * encoded as a callable type that is evaluated on device.
 */
template<typename Real>
struct TimeDependentBC {
    // Parameters stored directly (POD)
    Real t0 = Real(0);           // Reference time
    Real frequency = Real(1);    // For periodic BCs
    Real phase = Real(0);        // Phase offset

    // Base values
    Real rho0 = Real(1);
    Real u0 = Real(0);
    Real v0 = Real(0);
    Real p0 = Real(1);

    // Modulation type
    enum Modulation : uint8_t {
        Constant = 0,
        Sinusoidal = 1,
        Linear = 2,
        Exponential = 3
    };
    Modulation rho_mod = Constant;
    Modulation u_mod = Constant;
    Modulation v_mod = Constant;
    Modulation p_mod = Constant;

    KOKKOS_INLINE_FUNCTION
    Real apply_modulation(Real base, Modulation mod, Real t) const {
        Real dt = t - t0;
        switch (mod) {
            case Sinusoidal:
                return base * (Real(1) + Real(0.1) * Kokkos::sin(frequency * dt + phase));
            case Linear:
                return base * (Real(1) + Real(0.1) * dt);
            case Exponential:
                return base * Kokkos::exp(Real(0.1) * dt);
            default:
                return base;
        }
    }

    KOKKOS_INLINE_FUNCTION
    Real rho(Real t) const { return apply_modulation(rho0, rho_mod, t); }
    KOKKOS_INLINE_FUNCTION
    Real u(Real t) const { return apply_modulation(u0, u_mod, t); }
    KOKKOS_INLINE_FUNCTION
    Real v(Real t) const { return apply_modulation(v0, v_mod, t); }
    KOKKOS_INLINE_FUNCTION
    Real p(Real t) const { return apply_modulation(p0, p_mod, t); }
};

// ============================================================================
// ZONAL BC (Spatially varying BCs)
// ============================================================================

/**
 * @brief Zone definition for zonal boundary conditions
 *
 * Zones are defined by geometric predicates. Used for:
 * - Partial inlet (only part of boundary)
 * - Moving boundaries (zone changes with time)
 * - Multi-region BCs (different BCs on different parts)
 */
template<typename Real>
struct Zone {
    using Predicate = enum : uint8_t {
        EntireSide = 0,      // Entire boundary side
        IntervalX = 1,       // x in [x_min, x_max]
        IntervalY = 2,       // y in [y_min, y_max]
        Circle = 3,          // Distance from center < radius
        Rectangle = 4,       // Rectangle in domain coordinates
        Custom = 99          // User-defined predicate
    };

    Predicate predicate = EntireSide;

    // Zone parameters
    Real x_min = Real(0), x_max = Real(1);
    Real y_min = Real(0), y_max = Real(1);
    Real center_x = Real(0), center_y = Real(0);
    Real radius = Real(0);

    KOKKOS_INLINE_FUNCTION
    bool contains(int i, int j, Real dx, Real dy) const {
        Real x = i * dx;
        Real y = j * dy;

        switch (predicate) {
            case EntireSide:
                return true;
            case IntervalX:
                return x >= x_min && x <= x_max;
            case IntervalY:
                return y >= y_min && y <= y_max;
            case Circle: {
                Real dx_center = x - center_x;
                Real dy_center = y - center_y;
                return (dx_center*dx_center + dy_center*dy_center) < radius*radius;
            }
            case Rectangle:
                return x >= x_min && x <= x_max && y >= y_min && y <= y_max;
            default:
                return true;
        }
    }
};

// ============================================================================
// BC DESCRIPTOR (Type Erasure + Time Dependence)
// ============================================================================

/**
 * @brief Unified BC descriptor with type erasure and time dependence
 *
 * This POD struct can represent:
 * 1. Static BCs (Dirichlet, Neumann, Reflective)
 * 2. Time-dependent BCs (via TimeDependentBC policy)
 * 3. Zonal BCs (via Zone specification)
 *
 * GPU-safe: no virtual functions, all data inline.
 */
template<typename System>
    requires FiniteVolumeSystem<System>
struct BcDescriptor {
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;

    // BC type (can be changed at runtime)
    enum Type : uint8_t {
        StaticDirichlet = 0,
        StaticNeumann = 1,
        StaticReflective = 2,
        TimeDependentDirichlet = 3,
        TimeDependentInlet = 4,
        Outflow = 5
    };
    Type type = StaticNeumann;

    // For static BCs
    Conserved static_value;

    // For time-dependent BCs
    TimeDependentBC<Real> time_policy;

    // For zonal BCs
    Zone<Real> zone;

    // Priority (for overlapping zones)
    int priority = 0;

    KOKKOS_INLINE_FUNCTION
    BcDescriptor() = default;

    KOKKOS_INLINE_FUNCTION
    bool matches_zone(int i, int j, Real dx, Real dy) const {
        return zone.contains(i, j, dx, dy);
    }
};

// ============================================================================
// BC REGISTRY (Device-Side Storage)
// ============================================================================

/**
 * @brief Device-side registry of boundary conditions
 *
 * Stores all BC descriptors in GPU-accessible memory.
 * Provides spatial lookup: (side, i, j) → BcDescriptor
 */
template<typename System>
    requires FiniteVolumeSystem<System>
class BcRegistry {
public:
    using Real = typename System::RealType;
    using Descriptor = BcDescriptor<System>;

    // Maximum number of BC descriptors
    static constexpr int max_bcs = 64;

    // BC descriptors
    Kokkos::View<Descriptor[max_bcs]> descriptors;
    Kokkos::View<int> num_descriptors;

    // Spatial lookup table: (side, i) → descriptor index
    // Precomputed on host, synced to device
    Kokkos::View<int*[4]> bc_lookup;  // [4][nx] or [4][ny]

    KOKKOS_FUNCTION
    BcRegistry() = default;

    /**
     * @brief Find BC descriptor for given location
     *
     * Returns the highest priority BC that matches the zone.
     * Returns default (Neumann) if no BC matches.
     */
    KOKKOS_INLINE_FUNCTION
    const Descriptor& find(int side, int i, int j) const {
        // Check lookup table first (fast path)
        int idx = bc_lookup(side, i);
        if (idx >= 0) {
            return descriptors(idx);
        }

        // Fallback: search all descriptors (slow path)
        Descriptor default_bc;
        default_bc.type = BcDescriptor<System>::StaticNeumann;
        return default_bc;
    }
};

// ============================================================================
// BC MANAGER (Host-Side API)
// ============================================================================

/**
 * @brief Host-side manager for boundary conditions
 *
 * Provides a simple API for adding/updating/removing BCs.
 * Handles synchronization to device.
 */
template<typename System>
    requires FiniteVolumeSystem<System>
class BcManager {
public:
    using Real = typename System::RealType;
    using Descriptor = BcDescriptor<System>;

    /**
     * @brief Add a static BC on a side
     */
    void add_static_bc(const std::string& side,
                       typename Descriptor::Type type,
                       const typename System::Primitive& value,
                       Real gamma = System::default_gamma)
    {
        Descriptor desc;
        desc.type = type;
        desc.static_value = System::from_primitive(value, gamma);
        desc.zone.predicate = Zone<Real>::EntireSide;

        add_descriptor(side, std::move(desc));
    }

    /**
     * @brief Add a time-dependent BC
     */
    void add_time_dependent_bc(const std::string& side,
                                const TimeDependentBC<Real>& policy,
                                typename Descriptor::Type type =
                                    Descriptor::TimeDependentDirichlet)
    {
        Descriptor desc;
        desc.type = type;
        desc.time_policy = policy;
        desc.zone.predicate = Zone<Real>::EntireSide;

        add_descriptor(side, std::move(desc));
    }

    /**
     * @brief Add a zonal BC (partial side)
     */
    void add_zonal_bc(const std::string& side,
                      const Zone<Real>& zone,
                      const typename System::Primitive& value,
                      Real gamma = System::default_gamma,
                      int priority = 0)
    {
        Descriptor desc;
        desc.type = Descriptor::StaticDirichlet;
        desc.static_value = System::from_primitive(value, gamma);
        desc.zone = zone;
        desc.priority = priority;

        add_descriptor(side, std::move(desc));
    }

    /**
     * @brief Update an existing BC
     *
     * Changes the BC type and/or values for an existing zone.
     * Requires re-sync to device.
     */
    void update_bc(const std::string& side, int index,
                   typename Descriptor::Type new_type,
                   const typename System::Primitive& new_value,
                   Real gamma = System::default_gamma)
    {
        if (index < descriptors_[side].size()) {
            descriptors_[side][index].type = new_type;
            descriptors_[side][index].static_value =
                System::from_primitive(new_value, gamma);
            needs_sync_ = true;
        }
    }

    /**
     * @brief Remove a BC by index
     */
    void remove_bc(const std::string& side, int index) {
        if (index < descriptors_[side].size()) {
            descriptors_[side].erase(descriptors_[side].begin() + index);
            needs_sync_ = true;
        }
    }

    /**
     * @brief Synchronize changes to device
     *
     * Called automatically by solver before time step.
     * Can be called explicitly for finer control.
     */
    void sync_to_device() {
        if (!needs_sync_) return;

        // Copy descriptors to device views
        // Rebuild lookup table
        build_lookup_table();

        needs_sync_ = false;
    }

    /**
     * @brief Get device-side registry
     */
    const BcRegistry<System>& device_registry() const {
        return device_registry_;
    }

private:
    // Host-side storage
    std::unordered_map<std::string, std::vector<Descriptor>> descriptors_;

    // Device-side registry
    BcRegistry<System> device_registry_;

    // Dirty flag
    bool needs_sync_ = false;

    void add_descriptor(const std::string& side, Descriptor&& desc) {
        descriptors_[side].push_back(std::move(desc));
        needs_sync_ = true;
    }

    void build_lookup_table();
};

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

/**
 * @brief Create a sinusoidal inlet BC
 */
template<typename System>
TimeDependentBC<typename System::RealType>
sinusoidal_inlet(typename System::RealType rho0,
                 typename System::RealType u0,
                 typename System::RealType frequency)
{
    TimeDependentBC<typename System::RealType> bc;
    bc.rho0 = rho0;
    bc.u0 = u0;
    bc.frequency = frequency;
    bc.rho_mod = TimeDependentBC<typename System::RealType>::Sinusoidal;
    return bc;
}

} // namespace subsetix::fvd::boundary
```

### 1.4 Usage Examples

```cpp
// Example 1: Static BC (current behavior)
auto& solver = ...;
solver.add_bc("left", BcType::Dirichlet, Primitive{1.0, 100.0, 0.0, 100000.0});

// Example 2: Time-dependent BC (sinusoidal inflow)
auto sinusoidal = sinusoidal_inlet<Euler2D<float>>(1.0, 100.0, 2.0 * 3.14159);
solver.add_time_dependent_bc("left", sinusoidal);

// Example 3: Zonal BC (partial inlet)
Zone<float> zone;
zone.predicate = Zone<float>::IntervalX;
zone.x_min = 0.2;
zone.x_max = 0.4;
solver.add_zonal_bc("bottom", zone, Primitive{1.0, 50.0, 0.0, 100000.0});

// Example 4: Update BC type during simulation
solver.boundary_manager().update_bc("left", 0, BcType::Neumann, Primitive{});
solver.boundary_manager().sync_to_device();

// Example 5: Combining multiple BCs on same side
solver.add_zonal_bc("bottom", zone1, value1, /*priority=*/1);
solver.add_zonal_bc("bottom", zone2, value2, /*priority=*/2);
// Higher priority takes precedence in overlap region
```

### 1.5 Performance Considerations

1. **Minimize syncs**: BC changes are batched on host, synced once per timestep
2. **Spatial lookup**: Precomputed lookup table for O(1) access on device
3. **POD types**: All BC data is POD, no virtual functions
4. **Inline evaluation**: Time dependence computed directly in kernel

---

## 2. AMR with Coarsening and Combinable Criteria

### 2.1 Problem Statement

Current limitations:
- No coarsening (refinement only)
- Single hardcoded criterion (gradient)
- No discontinuity detection
- No exclusion zones

Requirements:
- Coarsening support
- Combinable criteria (AND/OR logic)
- Discontinuity detection (shock sensors)
- Exclusion zones (keep certain areas fine)
- Customizable remesh frequency

### 2.2 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Solver (Host)                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  RefinementManager                                    │  │
│  │  - add_criterion(...)                                │  │
│  │  - set_remesh_frequency(...)                         │  │
│  │  - add_exclusion_zone(...)                           │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  RefinementCriteria (Device POD)                     │  │
│  │  - Criterion[] criteria  [array of criteria]         │  │
│  │  - LogicOperator op  [AND/OR]                        │  │
│  │  - Zone[] exclusions                                 │  │
│  │  - int remesh_interval                               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼ (device kernel)
┌─────────────────────────────────────────────────────────────┐
│                  evaluate_refinement(i, j)                  │
│  for each criterion:                                         │
│      val = criterion.evaluate(U[i,j], neighbors)            │
│      if op == AND and not val: return COARSEN               │
│      if op == OR and val: return REFINE                     │
│  check exclusions: if in exclusion: return KEEP             │
│  return default                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Refinement Criteria Interface

```cpp
// fvd/amr/refinement_criteria.hpp
#pragma once

#include <Kokkos_Core.hpp>
#include <concepts>
#include "../system/concepts_v2.hpp"

namespace subsetix::fvd::amr {

// ============================================================================
// RESULT ENUM
// ============================================================================

/**
 * @brief Result of refinement evaluation
 */
enum class RefinementAction : int {
    Coarsen = -1,   // Remove this cell (merge with parent)
    Keep = 0,       // Keep current level
    Refine = 1      // Split this cell (add children)
};

// ============================================================================
// BASE CRITERION CONCEPT
// ============================================================================

/**
 * @brief Concept for refinement criteria
 *
 * A criterion must be:
 * - Default constructible (POD-friendly)
 * - Callable with signature: RefinementAction(Conserved, neighbors)
 * - GPU-compatible (KOKKOS_INLINE_FUNCTION)
 */
template<typename T, typename System>
concept RefinementCriterion =
    std::is_default_constructible_v<T> &&
    requires(const T& crit,
             const typename System::Conserved& U,
             const typename System::Primitive& q,
             typename System::RealType dx)
    {
        { crit.evaluate(U, q, dx) } -> std::convertible_to<RefinementAction>;
    };

// ============================================================================
// GRADIENT CRITERION
// ============================================================================

/**
 * @brief Refine based on solution gradient
 *
 * Refines where: |grad(U)| > threshold
 */
template<typename System>
    requires FiniteVolumeSystem<System>
class GradientCriterion {
public:
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    Real threshold = Real(0.1);
    bool use_rho = true;
    bool use_p = true;
    bool use_u = false;

    KOKKOS_INLINE_FUNCTION
    GradientCriterion() = default;

    KOKKOS_INLINE_FUNCTION
    RefinementAction evaluate(const Conserved& U,
                               const Primitive& q,
                               Real /*dx*/) const
    {
        // Placeholder: compute gradient from neighbors
        // In production: would use reconstructed gradients
        Real grad_rho = /* ... */ Real(0);

        if (use_rho && grad_rho > threshold) {
            return RefinementAction::Refine;
        }
        return RefinementAction::Keep;
    }
};

// ============================================================================
// SHOCK SENSOR CRITERION
// ============================================================================

/**
 * @brief Refine based on shock sensor
 *
 * Uses Ducros or Jameson shock sensor to detect discontinuities.
 */
template<typename System>
    requires FiniteVolumeSystem<System>
class ShockSensorCriterion {
public:
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    enum SensorType : uint8_t {
        Ducros = 0,      // Divergence of velocity vs vorticity
        Jameson = 1,     // Pressure gradient
        Persson = 2      // Spectral decay (for high-order)
    };

    SensorType sensor_type = Ducros;
    Real threshold = Real(0.8);

    KOKKOS_INLINE_FUNCTION
    ShockSensorCriterion() = default;

    KOKKOS_INLINE_FUNCTION
    RefinementAction evaluate(const Conserved& U,
                               const Primitive& q,
                               Real dx) const
    {
        Real sensor = compute_sensor(q, dx);

        if (sensor > threshold) {
            return RefinementAction::Refine;
        }
        return RefinementAction::Keep;
    }

private:
    KOKKOS_INLINE_FUNCTION
    Real compute_sensor(const Primitive& q, Real dx) const {
        switch (sensor_type) {
            case Ducros: {
                // div(V) / (|div(V)| + |curl(V)| + eps)
                Real div_v = /* ... */ Real(0);
                Real curl_v = /* ... */ Real(0);
                Real eps = Real(1e-10);
                return div_v / (Kokkos::abs(div_v) + Kokkos::abs(curl_v) + eps);
            }
            case Jameson: {
                // |grad(p)| * dx / p
                Real grad_p = /* ... */ Real(0);
                return grad_p * dx / (q.p + Real(1e-10));
            }
            default:
                return Real(0);
        }
    }
};

// ============================================================================
// VORTICITY CRITERION
// ============================================================================

/**
 * @brief Refine based on vorticity magnitude
 *
 * Useful for capturing shear layers and vortices.
 */
template<typename System>
    requires FiniteVolumeSystem<System>
class VorticityCriterion {
public:
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    Real threshold = Real(1.0);

    KOKKOS_INLINE_FUNCTION
    VorticityCriterion() = default;

    KOKKOS_INLINE_FUNCTION
    RefinementAction evaluate(const Conserved& U,
                               const Primitive& q,
                               Real dx) const
    {
        Real vorticity = /* dv/dx - du/dy */ Real(0);

        if (Kokkos::abs(vorticity) > threshold) {
            return RefinementAction::Refine;
        }
        return RefinementAction::Keep;
    }
};

// ============================================================================
// VALUE RANGE CRITERION
// ============================================================================

/**
 * @brief Refine based on variable value range
 *
 * Refines where variable is in specified range (e.g., tracking a front).
 */
template<typename System>
    requires FiniteVolumeSystem<System>
class ValueRangeCriterion {
public:
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    enum Variable : uint8_t {
        Density = 0,
        Pressure = 1,
        VelocityX = 2,
        VelocityY = 3,
        Mach = 4
    };

    Variable variable = Density;
    Real min_val = Real(0);
    Real max_val = Real(1);
    bool invert = false;  // If true, refine OUTSIDE range

    KOKKOS_INLINE_FUNCTION
    ValueRangeCriterion() = default;

    KOKKOS_INLINE_FUNCTION
    RefinementAction evaluate(const Conserved& U,
                               const Primitive& q,
                               Real /*dx*/) const
    {
        Real val = get_value(q);

        bool in_range = (val >= min_val && val <= max_val);
        bool should_refine = invert ? !in_range : in_range;

        return should_refine ? RefinementAction::Refine : RefinementAction::Keep;
    }

private:
    KOKKOS_INLINE_FUNCTION
    Real get_value(const Primitive& q) const {
        switch (variable) {
            case Density: return q.rho;
            case Pressure: return q.p;
            case VelocityX: return q.u;
            case VelocityY: return q.v;
            case Mach: return q.u / /* sound speed */ Real(1);
            default: return q.rho;
        }
    }
};

// ============================================================================
// COMBINABLE CRITERIA
// ============================================================================

/**
 * @brief Combines multiple criteria with logic operators
 *
 * Supports:
 * - AND: refine only if ALL criteria agree
 * - OR: refine if ANY criterion agrees
 * - Custom: user-defined logic
 */
template<typename System, int MaxCriteria = 8>
    requires FiniteVolumeSystem<System>
class CompositeCriterion {
public:
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    enum LogicOp : uint8_t {
        And = 0,   // All criteria must return Refine
        Or = 1,    // Any criterion returns Refine
        Vote = 2   // Majority vote
    };

    LogicOp logic_op = Or;

    // Array of criterion types (stored as indices into type registry)
    int16_t criterion_indices[MaxCriteria] = {-1, -1, -1, -1, -1, -1, -1, -1};
    int8_t num_criteria = 0;

    KOKKOS_FUNCTION
    CompositeCriterion() = default;

    /**
     * @brief Evaluate composite criterion
     *
     * Device-side evaluation loops over all criteria
     * and combines results using the logic operator.
     */
    KOKKOS_INLINE_FUNCTION
    RefinementAction evaluate(const Conserved& U,
                               const Primitive& q,
                               Real dx) const
    {
        int refine_count = 0;
        int coarsen_count = 0;

        for (int i = 0; i < num_criteria; ++i) {
            if (criterion_indices[i] < 0) break;

            // Get criterion from global registry
            auto& crit = get_criterion(criterion_indices[i]);
            auto action = crit.evaluate(U, q, dx);

            if (action == RefinementAction::Refine) refine_count++;
            if (action == RefinementAction::Coarsen) coarsen_count++;
        }

        // Combine based on logic operator
        switch (logic_op) {
            case And:
                return (refine_count == num_criteria)
                    ? RefinementAction::Refine : RefinementAction::Keep;
            case Or:
                return (refine_count > 0)
                    ? RefinementAction::Refine : RefinementAction::Keep;
            case Vote:
                return (refine_count > num_criteria / 2)
                    ? RefinementAction::Refine : RefinementAction::Keep;
            default:
                return RefinementAction::Keep;
        }
    }

private:
    // Get criterion from registry (implementation-defined)
    static auto& get_criterion(int index);
};

// ============================================================================
// EXCLUSION ZONES
// ============================================================================

/**
 * @brief Zones where refinement is disabled
 *
 * Cells in these zones will never be coarsened below min_level.
 */
template<typename Real>
struct ExclusionZone {
    // Zone definition (same as BC zones)
    enum Predicate : uint8_t {
        EntireDomain = 0,
        Rectangle = 1,
        Circle = 2
    };

    Predicate predicate = Rectangle;
    Real x_min, x_max, y_min, y_max;
    Real center_x, center_y, radius;

    int min_level = 0;  // Minimum level to maintain in this zone

    KOKKOS_INLINE_FUNCTION
    bool contains(int i, int j, Real dx, Real dy) const {
        Real x = i * dx;
        Real y = j * dy;

        switch (predicate) {
            case Rectangle:
                return x >= x_min && x <= x_max && y >= y_min && y <= y_max;
            case Circle: {
                Real dx_c = x - center_x;
                Real dy_c = y - center_y;
                return (dx_c*dx_c + dy_c*dy_c) < radius*radius;
            }
            default:
                return false;
        }
    }
};

// ============================================================================
// REFINEMENT CONFIG (Complete Configuration)
// ============================================================================

/**
 * @brief Complete refinement configuration
 *
 * Combines criteria, exclusions, and parameters into a single POD struct.
 */
template<typename System>
    requires FiniteVolumeSystem<System>
struct RefinementConfig {
    using Real = typename System::RealType;

    // Criterion to use
    CompositeCriterion<System> criterion;

    // Exclusion zones
    static constexpr int max_exclusions = 16;
    ExclusionZone<Real> exclusions[max_exclusions];
    int8_t num_exclusions = 0;

    // Level limits
    int8_t min_level = 0;
    int8_t max_level = 5;

    // Remesh frequency
    int remesh_interval = 100;

    // Coarsening enabled
    bool enable_coarsening = true;

    KOKKOS_FUNCTION
    RefinementConfig() = default;

    /**
     * @brief Check if cell is in exclusion zone
     *
     * Returns minimum level to maintain, or -1 if not in exclusion.
     */
    KOKKOS_INLINE_FUNCTION
    int check_exclusions(int i, int j, Real dx, Real dy) const {
        for (int ie = 0; ie < num_exclusions; ++ie) {
            if (exclusions[ie].contains(i, j, dx, dy)) {
                return exclusions[ie].min_level;
            }
        }
        return -1;
    }
};

// ============================================================================
// MANAGER (Host-Side API)
// ============================================================================

/**
 * @brief Host-side manager for refinement configuration
 *
 * Provides a convenient API for building the refinement config.
 */
template<typename System>
    requires FiniteVolumeSystem<System>
class RefinementManager {
public:
    using Real = typename System::RealType;
    using Config = RefinementConfig<System>;

    Config config;

    /**
     * @brief Add gradient criterion
     */
    void add_gradient_criterion(Real threshold) {
        GradientCriterion<System> crit;
        crit.threshold = threshold;
        add_criterion(std::move(crit));
    }

    /**
     * @brief Add shock sensor criterion
     */
    void add_shock_sensor_criterion(
        typename ShockSensorCriterion<System>::SensorType type,
        Real threshold)
    {
        ShockSensorCriterion<System> crit;
        crit.sensor_type = type;
        crit.threshold = threshold;
        add_criterion(std::move(crit));
    }

    /**
     * @brief Add value range criterion
     */
    void add_value_range_criterion(
        typename ValueRangeCriterion<System>::Variable var,
        Real min_val, Real max_val)
    {
        ValueRangeCriterion<System> crit;
        crit.variable = var;
        crit.min_val = min_val;
        crit.max_val = max_val;
        add_criterion(std::move(crit));
    }

    /**
     * @brief Set logic operator for combining criteria
     */
    void set_logic_op(typename CompositeCriterion<System>::LogicOp op) {
        config.criterion.logic_op = op;
    }

    /**
     * @brief Add exclusion zone
     */
    void add_exclusion_zone(const ExclusionZone<Real>& zone) {
        if (config.num_exclusions < Config::max_exclusions) {
            config.exclusions[config.num_exclusions++] = zone;
        }
    }

    /**
     * @brief Set level limits
     */
    void set_level_limits(int min_level, int max_level) {
        config.min_level = min_level;
        config.max_level = max_level;
    }

    /**
     * @brief Set remesh frequency
     */
    void set_remesh_frequency(int interval) {
        config.remesh_interval = interval;
    }

    /**
     * @brief Enable/disable coarsening
     */
    void set_coarsening(bool enable) {
        config.enable_coarsening = enable;
    }

    /**
     * @brief Build final config (copies to device)
     */
    Config build() const {
        return config;
    }

private:
    void add_criterion(auto&& crit) {
        // Add to criterion registry and update indices
        // Implementation stores criterion and gets index
        int idx = register_criterion(std::forward<decltype(crit)>(crit));

        if (config.criterion.num_criteria < CompositeCriterion<System>::MaxCriteria) {
            config.criterion.criterion_indices[config.criterion.num_criteria++] = idx;
        }
    }

    int register_criterion(auto&& crit);
};

} // namespace subsetix::fvd::amr
```

### 2.4 Usage Examples

```cpp
// Example 1: Simple gradient-based refinement
auto& mgr = solver.refinement_manager();
mgr.add_gradient_criterion(0.1);
mgr.set_level_limits(0, 4);
mgr.set_remesh_frequency(100);
solver.set_refinement(mgr.build());

// Example 2: Combine shock sensor + vorticity (OR logic)
mgr.add_shock_sensor_criterion(ShockSensorCriterion<System>::Ducros, 0.8);
mgr.add_vorticity_criterion(1.0);
mgr.set_logic_op(CompositeCriterion<System>::Or);

// Example 3: Track a density front
mgr.add_value_range_criterion(
    ValueRangeCriterion<System>::Density,
    0.5, 1.5  // Refine where rho is in [0.5, 1.5]
);

// Example 4: Exclude near-wall region from coarsening
ExclusionZone<float> wall_zone;
wall_zone.predicate = ExclusionZone<float>::Rectangle;
wall_zone.y_min = 0.0;
wall_zone.y_max = 0.1;  // Bottom 10% of domain
wall_zone.min_level = 2;  // Always at least level 2
mgr.add_exclusion_zone(wall_zone);

// Example 5: Custom criterion (user-defined)
struct MyCustomCriterion {
    float threshold = 1.0f;

    KOKKOS_INLINE_FUNCTION
    RefinementAction evaluate(const Conserved& U, const Primitive& q, float dx) const {
        // Custom logic here
        return (q.p > threshold) ? RefinementAction::Refine : RefinementAction::Keep;
    }
};
mgr.add_custom_criterion(MyCustomCriterion{1.0f});
```

### 2.5 Coarsening Algorithm

```cpp
// Device-side coarsening logic
KOKKOS_FUNCTION
void evaluate_coarsening(int i, int j, const RefinementConfig<System>& cfg) {
    if (!cfg.enable_coarsening) return;

    // Get sibling cells (4 children that form a parent)
    auto children = get_siblings(i, j);

    // Check if all siblings agree on coarsening
    bool all_agree = true;
    for (auto& child : children) {
        auto action = cfg.criterion.evaluate(child.U, child.q, child.dx);
        if (action != RefinementAction::Coarsen) {
            all_agree = false;
            break;
        }
    }

    if (all_agree) {
        // Check level limits
        int current_level = get_level(i, j);
        if (current_level > cfg.min_level) {
            // Check exclusions
            int min_level = cfg.check_exclusions(i, j, dx, dy);
            if (min_level < 0 || current_level > min_level) {
                mark_for_coarsening(i, j);
            }
        }
    }
}
```

---

## 3. High-Order Time Integration

### 3.1 Problem Statement

Current limitation:
- Only Forward Euler (order 1, single stage)

Requirements:
- RK2, RK3, RK4 schemes
- Compile-time selection (template parameter)
- Variable dt with customizable criteria
- Observer integration

### 3.2 Architecture

```cpp
// Time integrator as policy
template<typename System>
class RK3Integrator {
public:
    static constexpr int order = 3;
    static constexpr int stages = 3;

    // Butcher tableau coefficients
    static constexpr Real a[stages][stages] = {
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0},
        {-1.0, 2.0, 0.0}
    };
    static constexpr Real b[stages] = {1.0/6.0, 2.0/3.0, 1.0/6.0};
    static constexpr Real c[stages] = {0.0, 0.5, 1.0};
};
```

### 3.3 Time Integration Interface

```cpp
// fvd/time/time_integrators.hpp
#pragma once

#include <Kokkos_Core.hpp>
#include <concepts>
#include "../system/concepts_v2.hpp"

namespace subsetix::fvd::time {

// ============================================================================
// TIME INTEGRATOR CONCEPT
// ============================================================================

/**
 * @brief Concept for time integrators
 *
 * A time integrator must provide:
 * - order: Order of accuracy
 * - stages: Number of RHS evaluations per step
 * - a, b, c: Butcher tableau coefficients
 */
template<typename T>
concept TimeIntegrator = requires {
    { T::order } -> std::convertible_to<int>;
    { T::stages } -> std::convertible_to<int>;
    typename T::Real;
    { T::a } -> std::convertible_to<const typename T::Real (*)[T::stages]>;
    { T::b } -> std::convertible_to<const typename T::Real (*)>;
    { T::c } -> std::convertible_to<const typename T::Real (*)>;
};

// ============================================================================
// FORWARD EULER
// ============================================================================

/**
 * @brief Forward Euler (1st order, 1 stage)
 */
template<typename Real = float>
struct ForwardEuler {
    static constexpr int order = 1;
    static constexpr int stages = 1;
    using RealType = Real;

    static constexpr Real a[1][1] = {{0.0}};
    static constexpr Real b[1] = {1.0};
    static constexpr Real c[1] = {0.0};

    static constexpr const char* name = "Forward Euler";
};

// ============================================================================
:: HEUN'S METHOD (RK2)
// ============================================================================

/**
 * @brief Heun's method (2nd order, 2 stages)
 *
 * Also known as improved Euler or explicit trapezoidal method.
 */
template<typename Real = float>
struct Heun2 {
    static constexpr int order = 2;
    static constexpr int stages = 2;
    using RealType = Real;

    static constexpr Real a[2][2] = {
        {0.0, 0.0},
        {1.0, 0.0}
    };
    static constexpr Real b[2] = {0.5, 0.5};
    static constexpr Real c[2] = {0.0, 1.0};

    static constexpr const char* name = "Heun (RK2)";
};

// ============================================================================
// KUTTA'S THIRD ORDER (RK3)
// ============================================================================

/**
 * @brief Kutta's third-order method (RK3)
 *
 * Classic 3-stage, 3rd-order Runge-Kutta method.
 */
template<typename Real = float>
struct Kutta3 {
    static constexpr int order = 3;
    static constexpr int stages = 3;
    using RealType = Real;

    static constexpr Real a[3][3] = {
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0},
        {-1.0, 2.0, 0.0}
    };
    static constexpr Real b[3] = {1.0/6.0, 2.0/3.0, 1.0/6.0};
    static constexpr Real c[3] = {0.0, 0.0, 1.0};

    static constexpr const char* name = "Kutta (RK3)";
};

// ============================================================================
// CLASSIC RK4
// ============================================================================

/**
 * @brief Classic 4th-order Runge-Kutta (RK4)
 */
template<typename Real = float>
struct ClassicRK4 {
    static constexpr int order = 4;
    static constexpr int stages = 4;
    using RealType = Real;

    static constexpr Real a[4][4] = {
        {0.0, 0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0, 0.0},
        {0.0, 0.5, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0}
    };
    static constexpr Real b[4] = {1.0/6.0, 2.0/6.0, 2.0/6.0, 1.0/6.0};
    static constexpr Real c[4] = {0.0, 0.5, 0.5, 1.0};

    static constexpr const char* name = "Classic RK4";
};

// ============================================================================
// LOW-STORAGE RK3 (Williamson)
// ============================================================================

/**
 * @brief Low-storage 3-stage Runge-Kutta (Williamson)
 *
 * Optimized for memory: only 2 solution storage arrays needed.
 * Ideal for GPU where memory is limited.
 */
template<typename Real = float>
struct LowStorageRK3 {
    static constexpr int order = 3;
    static constexpr int stages = 3;
    using RealType = Real;

    // Low-storage formulation
    // Only need alpha and beta coefficients
    static constexpr Real alpha[3] = {0.0, -5.0/9.0, -153.0/128.0};
    static constexpr Real beta[3] = {1.0/3.0, 15.0/16.0, 8.0/15.0};

    // Butcher tableau equivalent (for compatibility)
    static constexpr Real a[3][3] = {
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0},
        {0.0, 0.75, 0.0}
    };
    static constexpr Real b[3] = {1.0/6.0, 2.0/3.0, 1.0/6.0};
    static constexpr Real c[3] = {0.0, 0.5, 0.5};

    static constexpr bool low_storage = true;
    static constexpr const char* name = "Low-Storage RK3";
};

// ============================================================================
// VARIABLE TIME STEP CONTROLLER
// ============================================================================

/**
 * @brief Controls adaptive time stepping
 *
 * Adjusts dt based on:
 * - CFL condition
 * - Error estimates (for embedded schemes)
 * - Custom user criteria
 */
template<typename Real = float>
class TimeStepController {
public:
    struct Config {
        // CFL-based control
        Real cfl_target = Real(0.8);
        Real cfl_max = Real(1.0);
        Real cfl_min = Real(0.1);

        // Adaptive limits
        Real dt_max = Real(1e-2);
        Real dt_min = Real(1e-6);
        Real growth_factor = Real(1.2);   // Max dt increase per step
        Real shrink_factor = Real(0.8);   // Max dt decrease per step

        // Custom criterion
        bool use_custom_criterion = false;
        // Custom criterion: function pointer or lambda (host-side only)
    };

    KOKKOS_INLINE_FUNCTION
    TimeStepController() = default;

    /**
     * @brief Compute new dt based on current CFL
     */
    KOKKOS_INLINE_FUNCTION
    Real compute_dt(Real current_dt, Real current_cfl, const Config& cfg) const {
        Real new_dt = current_dt;

        if (current_cfl > cfg.cfl_max) {
            // Too aggressive: shrink dt
            new_dt = current_dt * cfg.shrink_factor;
        } else if (current_cfl < cfg.cfl_target * Real(0.5)) {
            // Too conservative: grow dt
            new_dt = Kokkos::min(current_dt * cfg.growth_factor, cfg.dt_max);
        }

        // Enforce limits
        new_dt = Kokkos::max(new_dt, cfg.dt_min);
        new_dt = Kokkos::min(new_dt, cfg.dt_max);

        return new_dt;
    }

    /**
     * @brief Check if dt should be changed
     */
    KOKKOS_INLINE_FUNCTION
    bool should_adapt(Real current_cfl, const Config& cfg) const {
        return (current_cfl > cfg.cfl_max) ||
               (current_cfl < cfg.cfl_target * Real(0.5));
    }
};

// ============================================================================
// TIME INTEGRATOR STATE
// ============================================================================

/**
 * @brief Stores state during multi-stage integration
 *
 * For GPU efficiency, all stage data is stored contiguously.
 */
template<typename System>
    requires FiniteVolumeSystem<System>
class IntegratorState {
public:
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;

    // RHS for each stage
    Kokkos::View<Conserved***> stage_rhs;  // [stages][ny][nx][n_vars]

    // Working solution for each stage
    Kokkos::View<Conserved**> stage_solution;  // [ny][nx]

    // Current stage
    int current_stage = 0;

    // Current time within step
    Real stage_time = Real(0);

    KOKKOS_FUNCTION
    IntegratorState() = default;

    /**
     * @brief Initialize for a given integrator
     */
    void initialize(int nx, int ny, int stages) {
        stage_rhs = Kokkos::View<Conserved***>(
            "stage_rhs", stages, ny, nx
        );
        stage_solution = Kokkos::View<Conserved**>(
            "stage_solution", ny, nx
        );
    }
};

// ============================================================================
// RUNGE-KUTTA INTEGRATOR IMPLEMENTATION
// ============================================================================

/**
 * @brief Generic Runge-Kutta integrator
 *
 * Works with any TimeIntegrator policy.
 */
template<
    typename System,
    TimeIntegrator Integrator
>
class RungeKuttaIntegrator {
public:
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using State = IntegratorState<System>;

    static constexpr int stages = Integrator::stages;
    static constexpr int order = Integrator::order;

    /**
     * @brief Perform one time step
     *
     * Template-based: loops are unrolled at compile time for performance.
     */
    template<typename RHSFunc>
    KOKKOS_FUNCTION
    static void step(
        const Kokkos::View<Conserved**>& U,
        Real dt,
        Real t,
        RHSFunc&& rhs,
        State& state)
    {
        // Stage 1: Compute initial RHS
        state.current_stage = 0;
        state.stage_time = t;
        rhs(U, state.stage_rhs, 0);  // k1 = f(t, U)

        // Stages 2 to N: Compute intermediate solutions
        for (int s = 1; s < stages; ++s) {
            state.current_stage = s;

            // U_stage = U + dt * sum(a[s][i] * k_i)
            compute_stage_solution(U, dt, state, s);

            state.stage_time = t + Integrator::c[s] * dt;
            rhs(state.stage_solution, state.stage_rhs, s);  // k_s = f(t + c[s]*dt, U_stage)
        }

        // Final solution: U_new = U + dt * sum(b[i] * k_i)
        combine_stages(U, dt, state);
    }

private:
    KOKKOS_FUNCTION
    static void compute_stage_solution(
        const Kokkos::View<Conserved**>& U,
        Real dt,
        State& state,
        int stage)
    {
        const int nx = U.extent(1);
        const int ny = U.extent(0);

        auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx});

        Kokkos::parallel_for("compute_stage_solution", policy,
            KOKKOS_LAMBDA(int j, int i) {
                Conserved sum{0, 0, 0, 0};

                // Sum previous RHS contributions
                for (int prev = 0; prev < stage; ++prev) {
                    Real coeff = Integrator::a[stage][prev];
                    sum.rho += coeff * state.stage_rhs(prev, j, i).rho;
                    sum.rhou += coeff * state.stage_rhs(prev, j, i).rhou;
                    sum.rhov += coeff * state.stage_rhs(prev, j, i).rhov;
                    sum.E += coeff * state.stage_rhs(prev, j, i).E;
                }

                state.stage_solution(j, i).rho = U(j, i).rho + dt * sum.rho;
                state.stage_solution(j, i).rhou = U(j, i).rhou + dt * sum.rhou;
                state.stage_solution(j, i).rhov = U(j, i).rhov + dt * sum.rhov;
                state.stage_solution(j, i).E = U(j, i).E + dt * sum.E;
            }
        );
    }

    KOKKOS_FUNCTION
    static void combine_stages(
        Kokkos::View<Conserved**>& U,
        Real dt,
        State& state)
    {
        const int nx = U.extent(1);
        const int ny = U.extent(0);

        auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx});

        Kokkos::parallel_for("combine_stages", policy,
            KOKKOS_LAMBDA(int j, int i) {
                Conserved sum{0, 0, 0, 0};

                for (int s = 0; s < stages; ++s) {
                    Real coeff = Integrator::b[s];
                    sum.rho += coeff * state.stage_rhs(s, j, i).rho;
                    sum.rhou += coeff * state.stage_rhs(s, j, i).rhou;
                    sum.rhov += coeff * state.stage_rhs(s, j, i).rhov;
                    sum.E += coeff * state.stage_rhs(s, j, i).E;
                }

                U(j, i).rho += dt * sum.rho;
                U(j, i).rhou += dt * sum.rhou;
                U(j, i).rhov += dt * sum.rhov;
                U(j, i).E += dt * sum.E;
            }
        );
    }
};

// ============================================================================
// LOW-STORAGE IMPLEMENTATION (Optimized for GPU)
// ============================================================================

/**
 * @brief Low-storage RK implementation
 *
 * Only needs 2 solution arrays instead of (stages + 1).
 * Formula: U_new = U_old + dt * sum(beta[s] * RHS_s)
 *          with intermediate updates
 */
template<typename System, typename Integrator>
class LowStorageIntegrator {
public:
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;

    template<typename RHSFunc>
    KOKKOS_FUNCTION
    static void step(
        Kokkos::View<Conserved**> U,
        Kokkos::View<Conserved**> U_work,
        Real dt,
        Real t,
        RHSFunc&& rhs)
    {
        for (int s = 0; s < Integrator::stages; ++s) {
            Real t_stage = t + Integrator::c[s] * dt;

            // Compute RHS at current stage
            rhs(U, U_work, t_stage);

            // Update solution: U = U + dt * beta[s] * RHS
            const int nx = U.extent(1);
            const int ny = U.extent(0);

            auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx});

            Kokkos::parallel_for("low_storage_update", policy,
                KOKKOS_LAMBDA(int j, int i) {
                    Real beta = Integrator::beta[s];
                    U(j, i).rho += dt * beta * U_work(j, i).rho;
                    U(j, i).rhou += dt * beta * U_work(j, i).rhou;
                    U(j, i).rhov += dt * beta * U_work(j, i).rhov;
                    U(j, i).E += dt * beta * U_work(j, i).E;
                }
            );
        }
    }
};

} // namespace subsetix::fvd::time
```

### 3.4 Integration with Solver

```cpp
// Updated solver signature with time integrator
template<
    FiniteVolumeSystem System,
    typename Reconstruction = reconstruction::NoReconstruction,
    template<typename> class FluxScheme = flux::RusanovFlux,
    typename TimeIntegrator = time::ForwardEuler<typename System::RealType>
>
class AdaptiveSolver {
public:
    using Real = typename System::RealType;

    // Time stepping
    void step(Real dt) {
        if constexpr (TimeIntegrator::stages == 1) {
            // Forward Euler: single stage
            single_stage_step(dt);
        } else {
            // Multi-stage Runge-Kutta
            runge_kutta_step(dt);
        }
    }

    /**
     * @brief Step with variable dt
     */
    void step_adaptive() {
        Real current_cfl = compute_max_cfl();

        if (dt_controller_.should_adapt(current_cfl, dt_config_)) {
            dt_ = dt_controller_.compute_dt(dt_, current_cfl, dt_config_);
        }

        step(dt_);

        // Notify observers of sub-stages if RK
        if constexpr (TimeIntegrator::stages > 1) {
            notify_observers_substep();
        }
    }

private:
    Real dt_ = Real(1e-4);
    time::TimeStepController<Real>::Config dt_config_;
    time::IntegratorState<System> integrator_state_;

    void runge_kutta_step(Real dt) {
        auto rhs = [this](auto& U, auto& rhs_out, Real t) {
            this->compute_rhs(U, rhs_out, t);
        };

        time::RungeKuttaIntegrator<System, TimeIntegrator>::step(
            U_, dt, t_, rhs, integrator_state_
        );

        t_ += dt;
    }
};
```

### 3.5 Solver Aliases with Time Integrators

```cpp
// fvd/solver/solver_aliases_with_time.hpp
namespace subsetix::fvd::solver {

using Real = float;

// ============================================================================
// EULER SYSTEM SOLVERS
// ============================================================================

// Forward Euler (default, fastest)
using EulerSolver = AdaptiveSolver<Euler2D<Real>>;

// RK2 (better accuracy)
using EulerSolverRK2 = AdaptiveSolver<
    Euler2D<Real>,
    reconstruction::NoReconstruction,
    flux::RusanovFlux,
    time::Heun2<Real>
>;

// RK3 (good balance)
using EulerSolverRK3 = AdaptiveSolver<
    Euler2D<Real>,
    reconstruction::NoReconstruction,
    flux::RusanovFlux,
    time::Kutta3<Real>
>;

// Low-storage RK3 (GPU-optimized)
using EulerSolverRK3LS = AdaptiveSolver<
    Euler2D<Real>,
    reconstruction::NoReconstruction,
    flux::RusanovFlux,
    time::LowStorageRK3<Real>
>;

// RK4 (most accurate)
using EulerSolverRK4 = AdaptiveSolver<
    Euler2D<Real>,
    reconstruction::NoReconstruction,
    flux::RusanovFlux,
    time::ClassicRK4<Real>
>;

// ============================================================================
// HIGH-ORDER SOLVERS (Reconstruction + High-Order Time)
// ============================================================================

// MUSCL + RK3
using EulerSolverMUSCL_RK3 = AdaptiveSolver<
    Euler2D<Real>,
    reconstruction::MUSCLReconstruction<Real>,
    flux::HLLCFlux,
    time::Kutta3<Real>
>;

// MUSCL + Low-Storage RK3 (GPU-optimized high-order)
using EulerSolverMUSCL_RK3LS = AdaptiveSolver<
    Euler2D<Real>,
    reconstruction::MUSCLReconstruction<Real>,
    flux::HLLCFlux,
    time::LowStorageRK3<Real>
>;

} // namespace subsetix::fvd::solver
```

### 3.6 Usage Examples

```cpp
// Example 1: Use RK3 solver (compile-time selection)
using Solver = EulerSolverRK3;
auto solver = Solver::builder(nx, ny)
    .with_domain(0.0, 2.0, 0.0, 0.8)
    .with_initial_condition(mach2_cylinder)
    .build();

// Example 2: Switch to RK4
using SolverRK4 = EulerSolverRK4;
auto solver = SolverRK4::builder(nx, ny)...;

// Example 3: Enable adaptive time stepping
solver.enable_adaptive_timestepping({
    .cfl_target = 0.8f,
    .dt_max = 1e-2f,
    .growth_factor = 1.2f
});

// Example 4: Low-storage RK3 for GPU efficiency
using SolverRK3LS = EulerSolverRK3LS;
auto solver = SolverRK3LS::builder(nx, ny)...;

// Example 5: Custom time step criterion
solver.set_timestep_criterion([](const auto& U, const auto& q) {
    // Custom logic: adapt dt based on local conditions
    Real max_wave_speed = ...;
    return 0.8f * dx / max_wave_speed;
});
```

### 3.7 Observer Integration

```cpp
// Observer events for multi-stage integration
enum class SolverEvent {
    StepStart,
    StepEnd,
    SubStepStart,      // NEW: for RK stages
    SubStepEnd,        // NEW: for RK stages
    Remesh,
    Checkpoint,
    Error
};

// In solver:
template<typename TimeIntegrator>
void AdaptiveSolver<...>::runge_kutta_step(Real dt) {
    for (int s = 0; s < TimeIntegrator::stages; ++s) {
        observers_.notify(SolverEvent::SubStepStart, {
            .time = t_ + TimeIntegrator::c[s] * dt,
            .dt = dt,
            .stage = s
        });

        // Compute stage...

        observers_.notify(SolverEvent::SubStepEnd, {...});
    }
}
```

---

## 4. Integration: All Three Features Together

### 4.1 Complete Example

```cpp
#include <subsetix/fvd/solver/solver_aliases_with_time.hpp>
#include <subsetix/fvd/boundary/time_dependent_bc.hpp>
#include <subsetix/fvd/amr/refinement_criteria.hpp>

using namespace subsetix::fvd;
using namespace subsetix::fvd::solver;
using namespace subsetix::fvd::boundary;
using namespace subsetix::fvd::amr;

// Configure solver with:
// - RK3 time integration
// - Time-dependent BCs
// - AMR with coarsening
using Solver = EulerSolverMUSCL_RK3LS;

int main() {
    const int nx = 400, ny = 160;

    auto solver = Solver::builder(nx, ny)
        .with_domain(0.0, 2.0, 0.0, 0.8)
        .with_initial_condition(mach2_cylinder)
        .build();

    // ===== Configure time-dependent BCs =====
    auto sinusoidal_inlet = sinusoidal_inlet<Euler2D<float>>(
        1.0,   // rho0
        100.0, // u0
        2.0 * 3.14159  // frequency (1 Hz)
    );

    solver.boundary_manager().add_time_dependent_bc(
        "left",
        sinusoidal_inlet,
        BcDescriptor<Euler2D<float>>::TimeDependentInlet
    );

    // Partial inlet on bottom (zonal BC)
    Zone<float> inlet_zone;
    inlet_zone.predicate = Zone<float>::IntervalX;
    inlet_zone.x_min = 0.2;
    inlet_zone.x_max = 0.4;
    solver.boundary_manager().add_zonal_bc(
        "bottom",
        inlet_zone,
        Primitive{1.0, 50.0, 0.0, 100000.0}
    );

    solver.boundary_manager().sync_to_device();

    // ===== Configure AMR with coarsening =====
    auto& refinement_mgr = solver.refinement_manager();

    // Add shock sensor criterion
    refinement_mgr.add_shock_sensor_criterion(
        ShockSensorCriterion<Euler2D<float>>::Ducros,
        0.8
    );

    // Add vorticity criterion (OR logic)
    refinement_mgr.add_vorticity_criterion(1.0);
    refinement_mgr.set_logic_op(CompositeCriterion<Euler2D<float>>::Or);

    // Exclude near-wall from coarsening
    ExclusionZone<float> wall_zone;
    wall_zone.predicate = ExclusionZone<float>::Rectangle;
    wall_zone.y_min = 0.0;
    wall_zone.y_max = 0.1;
    wall_zone.min_level = 2;
    refinement_mgr.add_exclusion_zone(wall_zone);

    // Set limits and frequency
    refinement_mgr.set_level_limits(0, 5);
    refinement_mgr.set_remesh_frequency(100);
    refinement_mgr.set_coarsening(true);

    solver.set_refinement(refinement_mgr.build());

    // ===== Configure adaptive time stepping =====
    solver.enable_adaptive_timestepping({
        .cfl_target = 0.8f,
        .cfl_max = 1.0f,
        .dt_max = 1e-2f,
        .growth_factor = 1.2f
    });

    // ===== Add observers =====
    solver.observers().on_progress([](const auto& state) {
        std::cout << "Step " << state.step
                  << " t=" << state.time
                  << " dt=" << state.dt
                  << " stages=" << Solver::TimeIntegrator::stages
                  << std::endl;
    });

    // ===== Main loop =====
    Real t_end = 1.0;
    while (solver.time() < t_end) {
        solver.step_adaptive();

        // BC changes can be made between steps
        if (solver.time() > 0.5) {
            // Change BC type after t=0.5
            solver.boundary_manager().update_bc(
                "left", 0,
                BcDescriptor<Euler2D<float>>::StaticNeumann,
                Primitive{}
            );
            solver.boundary_manager().sync_to_device();
        }
    }

    return 0;
}
```

---

## 5. Summary and Implementation Plan

### 5.1 New Files to Create

| File | Purpose |
|------|---------|
| `include/subsetix/fvd/boundary/time_dependent_bc.hpp` | Time-dependent BC system |
| `include/subsetix/fvd/amr/refinement_criteria.hpp` | AMR criteria with coarsening |
| `include/subsetix/fvd/time/time_integrators.hpp` | RK2/RK3/RK4 integrators |
| `include/subsetix/fvd/solver/solver_aliases_with_time.hpp` | Updated solver aliases |

### 5.2 Files to Modify

| File | Changes |
|------|---------|
| `adaptive_solver.hpp` | Add TimeIntegrator template parameter, integrate BCManager and RefinementManager |
| `boundary_generic.hpp` | Add zone support, time-dependent BC types |
| `observer.hpp` | Add SubStepStart/SubStepEnd events |

### 5.3 Implementation Order

1. **Phase 1**: Time integrators (isolated, low risk)
2. **Phase 2**: AMR criteria with coarsening
3. **Phase 3**: Time-dependent BCs
4. **Phase 4**: Integration into AdaptiveSolver
5. **Phase 5**: Tests and examples

### 5.4 GPU Performance Checklist

- [ ] All BC data is POD (no virtual functions on device)
- [ ] Time-dependent BC parameters stored in Kokkos::Views
- [ ] Minimize host/device syncs (batch BC updates)
- [ ] RK stages use device-side parallel_for
- [ ] Low-storage variant available for memory-constrained GPUs
- [ ] AMR criteria evaluated entirely on device

### 5.5 Open Questions for User Confirmation

1. **Observer granularity**: Should observers see each RK sub-stage or only complete steps?

2. **BC zone representation**: Are rectangles/circles sufficient or need arbitrary polygons?

3. **AMR criterion storage**: Static array (fixed max) or dynamic allocation?

4. **Time integrator selection**: Keep compile-time only or add runtime option too?

5. **Low-storage RK**: Worth implementing or standard RK sufficient?

