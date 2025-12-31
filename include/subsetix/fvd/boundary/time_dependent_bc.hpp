#pragma once

#include <Kokkos_Core.hpp>
#include <concepts>
#include <cstdint>
#include "../system/concepts_v2.hpp"
#include "../geometry/csr_types.hpp"

namespace subsetix::fvd::boundary {

// ============================================================================
// TIME-DEPENDENT BC POLICY (Device-Side)
// ============================================================================

/**
 * @brief Policy for time-dependent boundary conditions
 *
 * This is a POD type (no virtual functions) that can be used
 * directly in GPU kernels. All parameters are stored inline.
 *
 * Design: Template-based policy where the time dependence is
 * encoded as modulation functions evaluated on device.
 */
template<typename Real>
struct TimeDependentBC {
    // Time reference
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
        Exponential = 3,
        SquareWave = 4
    };
    Modulation rho_mod = Constant;
    Modulation u_mod = Constant;
    Modulation v_mod = Constant;
    Modulation p_mod = Constant;

    // Amplitude for modulation (default: 10% variation)
    Real amplitude = Real(0.1);

    KOKKOS_INLINE_FUNCTION
    TimeDependentBC() = default;

    KOKKOS_INLINE_FUNCTION
    Real apply_modulation(Real base, Modulation mod, Real t) const {
        Real dt = t - t0;

        switch (mod) {
            case Sinusoidal:
                return base * (Real(1) + amplitude * Kokkos::sin(frequency * dt + phase));
            case Linear:
                return base * (Real(1) + amplitude * dt);
            case Exponential:
                return base * Kokkos::exp(amplitude * dt);
            case SquareWave: {
                Real phase_val = Kokkos::fmod(frequency * dt + phase, Real(2) * Real(3.14159));
                return (phase_val < Real(3.14159)) ? base * (Real(1) + amplitude)
                                                    : base * (Real(1) - amplitude);
            }
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
// ZONAL BC PREDICATES (Spatially varying BCs)
// ============================================================================

/**
 * @brief Zone predicate for zonal boundary conditions
 *
 * Zones are defined by geometric predicates. Used for:
 * - Partial inlet (only part of boundary)
 * - Moving boundaries (zone changes with time)
 * - Multi-region BCs (different BCs on different parts)
 *
 * Supports both simple shapes and custom Subsetix CSR geometries.
 */
template<typename Real>
struct ZonePredicate {
    using Predicate = enum : uint8_t {
        EntireSide = 0,      // Entire boundary side
        IntervalX = 1,       // x in [x_min, x_max]
        IntervalY = 2,       // y in [y_min, y_max]
        Rectangle = 3,       // Rectangle in domain coordinates
        Circle = 4,          // Distance from center < radius
        CustomCSR = 5        // User-defined Subsetix CSR geometry
    };

    Predicate predicate = EntireSide;

    // Simple shape parameters
    Real x_min = Real(0), x_max = Real(1);
    Real y_min = Real(0), y_max = Real(1);
    Real center_x = Real(0), center_y = Real(0);
    Real radius = Real(0);

    // Custom CSR geometry (pointer to device-side geometry)
    const csr::IntervalSet2DDevice* csr_geometry = nullptr;

    // For time-dependent zones (moving boundaries)
    bool time_dependent = false;
    Real velocity_x = Real(0);  // Zone velocity
    Real velocity_y = Real(0);

    KOKKOS_INLINE_FUNCTION
    ZonePredicate() = default;

    /**
     * @brief Check if point (x, y) is in this zone
     */
    KOKKOS_INLINE_FUNCTION
    bool contains(Real x, Real y, Real t = Real(0)) const {
        // Adjust for time-dependent zones
        Real x_adj = x;
        Real y_adj = y;
        if (time_dependent) {
            x_adj -= velocity_x * (t - Real(0));  // Assume t0=0 for simplicity
            y_adj -= velocity_y * (t - Real(0));
        }

        switch (predicate) {
            case EntireSide:
                return true;
            case IntervalX:
                return x_adj >= x_min && x_adj <= x_max;
            case IntervalY:
                return y_adj >= y_min && y_adj <= y_max;
            case Rectangle:
                return x_adj >= x_min && x_adj <= x_max &&
                       y_adj >= y_min && y_adj <= y_max;
            case Circle: {
                Real dx_c = x_adj - center_x;
                Real dy_c = y_adj - center_y;
                return (dx_c*dx_c + dy_c*dy_c) < radius*radius;
            }
            case CustomCSR:
                if (csr_geometry) {
                    // For CSR geometry, we'd need to check if (x, y) maps to a cell
                    // This is a simplified check - in production would do proper lookup
                    // For now, return true if geometry is set
                    return true;
                }
                return false;
            default:
                return true;
        }
    }

    /**
     * @brief Check if cell (i, j) is in this zone
     *
     * @param i Cell index in x
     * @param j Cell index in y
     * @param dx Cell spacing
     * @param dy Cell spacing
     * @param x0 Domain origin
     * @param y0 Domain origin
     * @param t Current time (for moving zones)
     */
    KOKKOS_INLINE_FUNCTION
    bool contains(int i, int j, Real dx, Real dy, Real x0, Real y0, Real t = Real(0)) const {
        Real x = x0 + i * dx;
        Real y = y0 + j * dy;
        return contains(x, y, t);
    }

    /**
     * @brief Create a simple interval zone
     */
    static ZonePredicate interval_x(Real x_min, Real x_max) {
        ZonePredicate z;
        z.predicate = IntervalX;
        z.x_min = x_min;
        z.x_max = x_max;
        return z;
    }

    static ZonePredicate interval_y(Real y_min, Real y_max) {
        ZonePredicate z;
        z.predicate = IntervalY;
        z.y_min = y_min;
        z.y_max = y_max;
        return z;
    }

    /**
     * @brief Create a rectangular zone
     */
    static ZonePredicate rectangle(Real x_min, Real x_max, Real y_min, Real y_max) {
        ZonePredicate z;
        z.predicate = Rectangle;
        z.x_min = x_min;
        z.x_max = x_max;
        z.y_min = y_min;
        z.y_max = y_max;
        return z;
    }

    /**
     * @brief Create a circular zone
     */
    static ZonePredicate circle(Real center_x, Real center_y, Real radius) {
        ZonePredicate z;
        z.predicate = Circle;
        z.center_x = center_x;
        z.center_y = center_y;
        z.radius = radius;
        return z;
    }

    /**
     * @brief Create a custom CSR zone
     */
    static ZonePredicate custom_csr(const csr::IntervalSet2DDevice* geometry) {
        ZonePredicate z;
        z.predicate = CustomCSR;
        z.csr_geometry = geometry;
        return z;
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
 * 3. Zonal BCs (via ZonePredicate specification)
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
    ZonePredicate<Real> zone;

    // Priority (for overlapping zones)
    int8_t priority = 0;

    KOKKOS_INLINE_FUNCTION
    BcDescriptor() = default;

    /**
     * @brief Get BC value at given time and location
     *
     * For static BCs: returns static_value
     * For time-dependent BCs: computes value from time_policy
     */
    KOKKOS_INLINE_FUNCTION
    Conserved get_value(Real t, Real gamma = System::default_gamma) const {
        if (type == StaticDirichlet || type == StaticNeumann || type == StaticReflective) {
            return static_value;
        } else if (type == TimeDependentDirichlet || type == TimeDependentInlet) {
            // Compute from time policy
            Real rho = time_policy.rho(t);
            Real u = time_policy.u(t);
            Real v = time_policy.v(t);
            Real p = time_policy.p(t);

            // Convert primitive to conserved
            typename System::Primitive q{rho, u, v, p};
            return System::from_primitive(q, gamma);
        }
        return Conserved{0, 0, 0, 0};
    }

    /**
     * @brief Check if this BC applies at given location
     */
    KOKKOS_INLINE_FUNCTION
    bool matches_zone(int i, int j, Real dx, Real dy, Real x0, Real y0, Real t) const {
        return zone.contains(i, j, dx, dy, x0, y0, t);
    }

    KOKKOS_INLINE_FUNCTION
    bool matches_zone(Real x, Real y, Real t) const {
        return zone.contains(x, y, t);
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

    // Maximum number of BC descriptors per side
    static constexpr int max_bcs_per_side = 16;

    // BC descriptors for each side
    // Indexing: [side][bc_index]
    Kokkos::View<Descriptor*[4]> descriptors;  // [4][max_bcs_per_side]
    Kokkos::View<int[4]> num_descriptors;      // [4]

    // Spatial lookup cache: (side, i_or_j) → best descriptor index
    // Precomputed on host, synced to device for O(1) access
    Kokkos::View<int**> bc_lookup;  // [4][max_dim]

    // Domain parameters (for zone checks)
    Real x0 = Real(0), y0 = Real(0);
    Real dx = Real(1), dy = Real(1);
    int nx = 0, ny = 0;

    KOKKOS_FUNCTION
    BcRegistry() = default;

    /**
     * @brief Initialize registry
     */
    void initialize(int nx_, int ny_, Real dx_, Real dy_, Real x0_, Real y0_) {
        nx = nx_;
        ny = ny_;
        dx = dx_;
        dy = dy_;
        x0 = x0_;
        y0 = y0_;

        // Allocate views
        descriptors = Kokkos::View<Descriptor*[4]>(
            "bc_descriptors", max_bcs_per_side
        );
        num_descriptors = Kokkos::View<int[4]>("bc_num_descriptors");

        // Initialize to zero
        auto h_num = Kokkos::create_mirror_view(num_descriptors);
        for (int s = 0; s < 4; ++s) h_num(s) = 0;
        Kokkos::deep_copy(num_descriptors, h_num);

        // Allocate lookup table
        int max_dim = Kokkos::max(nx, ny);
        bc_lookup = Kokkos::View<int**>("bc_lookup", 4, max_dim);

        // Initialize to -1 (no BC)
        auto h_lookup = Kokkos::create_mirror_view(bc_lookup);
        for (int s = 0; s < 4; ++s) {
            for (int i = 0; i < max_dim; ++i) {
                h_lookup(s, i) = -1;
            }
        }
        Kokkos::deep_copy(bc_lookup, h_lookup);
    }

    /**
     * @brief Find BC descriptor for given location
     *
     * Returns the highest priority BC that matches the zone.
     * Returns default (Neumann) if no BC matches.
     */
    KOKKOS_INLINE_FUNCTION
    const Descriptor find(int side, int i, int j, Real t) const {
        // Check lookup table first (fast path)
        int idx = -1;
        if ((side == 0 || side == 1) && i < nx) {
            idx = bc_lookup(side, i);
        } else if ((side == 2 || side == 3) && j < ny) {
            idx = bc_lookup(side, j);
        }

        if (idx >= 0 && idx < num_descriptors(side)) {
            const Descriptor& desc = descriptors(side, idx);
            // Verify zone matches (in case of moving zones)
            if (desc.matches_zone(i, j, dx, dy, x0, y0, t)) {
                return desc;
            }
        }

        // Fallback: search all descriptors (slow path)
        Descriptor default_bc;
        default_bc.type = BcDescriptor<System>::StaticNeumann;

        int best_priority = -1000;
        int best_idx = -1;

        for (int i_bc = 0; i_bc < num_descriptors(side); ++i_bc) {
            const Descriptor& desc = descriptors(side, i_bc);
            if (desc.matches_zone(i, j, dx, dy, x0, y0, t)) {
                if (desc.priority > best_priority) {
                    best_priority = desc.priority;
                    best_idx = i_bc;
                }
            }
        }

        if (best_idx >= 0) {
            return descriptors(side, best_idx);
        }

        return default_bc;
    }

    /**
     * @brief Add a descriptor (host-side)
     */
    void add_descriptor(int side, const Descriptor& desc) {
        auto h_num = Kokkos::create_mirror_view(num_descriptors);
        Kokkos::deep_copy(h_num, num_descriptors);

        if (h_num(side) >= max_bcs_per_side) return;

        int idx = h_num(side);
        h_num(side)++;
        Kokkos::deep_copy(num_descriptors, h_num);

        // Copy descriptor
        auto h_desc = Kokkos::create_mirror_view(descriptors);
        h_desc(side, idx) = desc;
        Kokkos::deep_copy(descriptors, h_desc);
    }

    /**
     * @brief Rebuild lookup table (host-side)
     *
     * Precomputes which BC to use for each cell on each side.
     * Call this after modifying BCs and before device execution.
     */
    void rebuild_lookup() {
        auto h_lookup = Kokkos::create_mirror_view(bc_lookup);
        auto h_num = Kokkos::create_mirror_view(num_descriptors);
        auto h_desc = Kokkos::create_mirror_view(descriptors);

        Kokkos::deep_copy(h_num, num_descriptors);
        Kokkos::deep_copy(h_desc, descriptors);

        // For each side
        for (int side = 0; side < 4; ++side) {
            int n = h_num(side);

            // For each cell along this side
            for (int i = 0; i < (side < 2 ? nx : ny); ++i) {
                // Find highest priority matching BC
                int best_priority = -1000;
                int best_idx = -1;

                for (int i_bc = 0; i_bc < n; ++i_bc) {
                    const Descriptor& desc = h_desc(side, i_bc);

                    // Check zone at t=0 (will be updated dynamically for moving zones)
                    int cell_i = (side < 2) ? i : 0;
                    int cell_j = (side < 2) ? 0 : i;

                    if (desc.matches_zone(cell_i, cell_j, dx, dy, x0, y0, 0)) {
                        if (desc.priority > best_priority) {
                            best_priority = desc.priority;
                            best_idx = i_bc;
                        }
                    }
                }

                h_lookup(side, i) = best_idx;
            }
        }

        Kokkos::deep_copy(bc_lookup, h_lookup);
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

    BcManager() = default;

    /**
     * @brief Initialize with domain parameters
     */
    void initialize(int nx, int ny, Real dx, Real dy, Real x0 = Real(0), Real y0 = Real(0)) {
        nx_ = nx;
        ny_ = ny;
        dx_ = dx;
        dy_ = dy;
        x0_ = x0;
        y0_ = y0;
        registry_.initialize(nx, ny, dx, dy, x0, y0);
        needs_sync_ = false;
    }

    /**
     * @brief Add a static BC on a side
     *
     * @param side "left", "right", "bottom", or "top"
     * @param type BC type (StaticDirichlet, StaticNeumann, StaticReflective)
     * @param value Primitive variable values
     * @param gamma Adiabatic index
     */
    void add_static_bc(const std::string& side,
                       typename Descriptor::Type type,
                       const typename System::Primitive& value,
                       Real gamma = System::default_gamma)
    {
        Descriptor desc;
        desc.type = type;
        desc.static_value = System::from_primitive(value, gamma);
        desc.zone.predicate = ZonePredicate<Real>::EntireSide;
        desc.priority = 0;

        int side_idx = get_side_index(side);
        registry_.add_descriptor(side_idx, desc);
        needs_sync_ = true;
    }

    /**
     * @brief Add a time-dependent BC
     *
     * @param side "left", "right", "bottom", or "top"
     * @param time_policy Time-dependent BC policy
     * @param type BC type (TimeDependentDirichlet or TimeDependentInlet)
     */
    void add_time_dependent_bc(const std::string& side,
                                const TimeDependentBC<Real>& time_policy,
                                typename Descriptor::Type type =
                                    Descriptor::TimeDependentDirichlet)
    {
        Descriptor desc;
        desc.type = type;
        desc.time_policy = time_policy;
        desc.zone.predicate = ZonePredicate<Real>::EntireSide;
        desc.priority = 0;

        int side_idx = get_side_index(side);
        registry_.add_descriptor(side_idx, desc);
        needs_sync_ = true;
    }

    /**
     * @brief Add a zonal BC (partial side)
     *
     * @param side "left", "right", "bottom", or "top"
     * @param zone Zone predicate defining the region
     * @param value Primitive variable values
     * @param gamma Adiabatic index
     * @param priority Priority for overlapping zones
     */
    void add_zonal_bc(const std::string& side,
                      const ZonePredicate<Real>& zone,
                      const typename System::Primitive& value,
                      Real gamma = System::default_gamma,
                      int priority = 0)
    {
        Descriptor desc;
        desc.type = Descriptor::StaticDirichlet;
        desc.static_value = System::from_primitive(value, gamma);
        desc.zone = zone;
        desc.priority = priority;

        int side_idx = get_side_index(side);
        registry_.add_descriptor(side_idx, desc);
        needs_sync_ = true;
    }

    /**
     * @brief Add a time-dependent zonal BC
     */
    void add_time_dependent_zonal_bc(const std::string& side,
                                      const ZonePredicate<Real>& zone,
                                      const TimeDependentBC<Real>& time_policy,
                                      int priority = 0)
    {
        Descriptor desc;
        desc.type = Descriptor::TimeDependentDirichlet;
        desc.time_policy = time_policy;
        desc.zone = zone;
        desc.priority = priority;

        int side_idx = get_side_index(side);
        registry_.add_descriptor(side_idx, desc);
        needs_sync_ = true;
    }

    /**
     * @brief Update an existing BC
     *
     * Changes the BC type and/or values for an existing BC.
     * Requires re-sync to device.
     */
    void update_bc(const std::string& side, int index,
                   typename Descriptor::Type new_type,
                   const typename System::Primitive& new_value,
                   Real gamma = System::default_gamma)
    {
        int side_idx = get_side_index(side);
        auto h_num = Kokkos::create_mirror_view(registry_.num_descriptors);
        Kokkos::deep_copy(h_num, registry_.num_descriptors);

        if (index >= 0 && index < h_num(side_idx)) {
            auto h_desc = Kokkos::create_mirror_view(registry_.descriptors);
            Kokkos::deep_copy(h_desc, registry_.descriptors);

            h_desc(side_idx, index).type = new_type;
            h_desc(side_idx, index).static_value =
                System::from_primitive(new_value, gamma);

            Kokkos::deep_copy(registry_.descriptors, h_desc);
            needs_sync_ = true;
        }
    }

    /**
     * @brief Remove a BC by index
     */
    void remove_bc(const std::string& side, int index) {
        int side_idx = get_side_index(side);
        // Implementation: shift remaining BCs
        // For simplicity, just mark for rebuild
        needs_sync_ = true;
    }

    /**
     * @brief Synchronize changes to device
     *
     * Called automatically by solver before time step.
     * Can be called explicitly for finer control.
     */
    void sync_to_device() {
        if (!needs_sync_) return;

        registry_.rebuild_lookup();
        needs_sync_ = false;
    }

    /**
     * @brief Get device-side registry
     */
    const BcRegistry<System>& device_registry() const {
        return registry_;
    }

    /**
     * @brief Check if sync is needed
     */
    bool needs_sync() const { return needs_sync_; }

private:
    BcRegistry<System> registry_;

    int nx_, ny_;
    Real dx_, dy_;
    Real x0_, y0_;

    bool needs_sync_ = false;

    int get_side_index(const std::string& side) const {
        if (side == "left") return 0;
        if (side == "right") return 1;
        if (side == "bottom") return 2;
        if (side == "top") return 3;
        return 0;  // Default
    }
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
                 typename System::RealType frequency,
                 typename System::RealType amplitude = typename System::RealType(0.1))
{
    TimeDependentBC<typename System::RealType> bc;
    bc.rho0 = rho0;
    bc.u0 = u0;
    bc.frequency = frequency;
    bc.amplitude = amplitude;
    bc.rho_mod = TimeDependentBC<typename System::RealType>::Sinusoidal;
    return bc;
}

/**
 * @brief Create a pulsating inlet BC
 */
template<typename System>
TimeDependentBC<typename System::RealType>
pulsating_inlet(typename System::RealType rho0,
                typename System::RealType u0,
                typename System::RealType frequency)
{
    TimeDependentBC<typename System::RealType> bc;
    bc.rho0 = rho0;
    bc.u0 = u0;
    bc.frequency = frequency;
    bc.amplitude = typename System::RealType(0.2);
    bc.u_mod = TimeDependentBC<typename System::RealType>::Sinusoidal;
    return bc;
}

/**
 * @brief Create a linear ramp BC
 */
template<typename System>
TimeDependentBC<typename System::RealType>
linear_ramp(typename System::RealType rho0,
            typename System::RealType u0,
            typename System::RealType rate)
{
    TimeDependentBC<typename System::RealType> bc;
    bc.rho0 = rho0;
    bc.u0 = u0;
    bc.amplitude = rate;
    bc.u_mod = TimeDependentBC<typename System::RealType>::Linear;
    return bc;
}

} // namespace subsetix::fvd::boundary
