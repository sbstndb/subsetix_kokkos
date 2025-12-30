#pragma once

#include <Kokkos_Core.hpp>
#include "../system/concepts.hpp"

namespace subsetix::fvd {

// ============================================================================
// GENERIC BOUNDARY CONDITIONS (TYPE ERASURE)
// ============================================================================

/**
 * @brief Generic boundary condition for ANY System
 *
 * P0-2 FIX: This uses type erasure via enum dispatch to work for
 * any System (Euler2D, Advection2D, etc.) without virtual functions.
 *
 * The BC type is stored as an enum and dispatched at runtime.
 * This is GPU-safe (POD type, no virtual functions).
 */
template<typename System>
struct AnyBc {
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    /// BC types
    enum Type : int {
        Dirichlet = 0,   // Fixed value
        Neumann = 1,     // Zero gradient (copy from interior)
        Reflective = 2,   // Reflective wall (for Euler2D)
        Custom = 3        // User-defined value
    };

    Type type = Neumann;
    Conserved value;

    KOKKOS_INLINE_FUNCTION
    AnyBc() = default;

    KOKKOS_INLINE_FUNCTION
    AnyBc(Type t, const Conserved& v) : type(t), value(v) {}

    /// Apply BC to ghost cell
    KOKKOS_INLINE_FUNCTION
    void apply(Conserved& U_ghost,
               const Conserved& U_interior,
               Real gamma) const
    {
        switch (type) {
            case Dirichlet:
                U_ghost = value;
                break;

            case Neumann:
                U_ghost = U_interior;
                break;

            case Reflective: {
                // Reflect normal velocity
                // For now, just copy (full implementation in production)
                U_ghost = U_interior;

                // TODO: In production, would reflect velocity
                // based on which side (left/right -> u, top/bottom -> v)
                break;
            }

            case Custom:
                U_ghost = value;
                break;

            default:
                U_ghost = U_interior;
                break;
        }
    }

    /// Helper: Create Dirichlet BC from primitive variables
    static AnyBc dirichlet_from_primitive(const Primitive& q, Real gamma) {
        AnyBc bc;
        bc.type = Dirichlet;
        bc.value = System::from_primitive(q, gamma);
        return bc;
    }
};

// ============================================================================
// BOUNDARY CONFIG (4 Sides)
// ============================================================================

/**
 * @brief Complete boundary configuration for all 4 sides
 *
 * P0-2 FIX: Replaces hardcoded BCs with runtime-configurable
 * generic BCs that work for ANY System.
 */
template<typename System>
struct BoundaryConfig {
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    AnyBc<System> left;
    AnyBc<System> right;
    AnyBc<System> bottom;
    AnyBc<System> top;

    KOKKOS_INLINE_FUNCTION
    BoundaryConfig() = default;
};

// ============================================================================
// BOUNDARY CONFIG BUILDER
// ============================================================================

/**
 * @brief Builder for common boundary configurations
 *
 * P0-2 FIX: Provides convenient factory methods for standard BC setups.
 * Works for ANY System (Euler2D, Advection2D, etc.)
 */
template<typename System>
class BoundaryConfigBuilder {
public:
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    /// Inflow from left, outflow elsewhere (Neumann)
    static BoundaryConfig<System> inflow_outflow(const Primitive& inflow, Real gamma) {
        BoundaryConfig<System> cfg;

        // Left: Dirichlet with inflow state
        cfg.left = AnyBc<System>::dirichlet_from_primitive(inflow, gamma);

        // Right, top, bottom: Neumann (zero gradient)
        cfg.right = AnyBc<System>{AnyBc<System>::Neumann, Conserved{}};
        cfg.bottom = AnyBc<System>{AnyBc<System>::Neumann, Conserved{}};
        cfg.top = AnyBc<System>{AnyBc<System>::Neumann, Conserved{}};

        return cfg;
    }

    /// All sides: Dirichlet with fixed state
    static BoundaryConfig<System> dirichlet_all(const Primitive& q, Real gamma) {
        BoundaryConfig<System> cfg;
        auto bc_value = AnyBc<System>::dirichlet_from_primitive(q, gamma);

        cfg.left = bc_value;
        cfg.right = bc_value;
        cfg.bottom = bc_value;
        cfg.top = bc_value;

        return cfg;
    }

    /// All sides: Neumann (zero gradient, outflow)
    static BoundaryConfig<System> neumann_all() {
        BoundaryConfig<System> cfg;
        AnyBc<System> neumann_bc{AnyBc<System>::Neumann, Conserved{}};

        cfg.left = neumann_bc;
        cfg.right = neumann_bc;
        cfg.bottom = neumann_bc;
        cfg.top = neumann_bc;

        return cfg;
    }

    /// Custom configuration (specify each side)
    static BoundaryConfig<System> custom(
        const Primitive& q_left,
        const Primitive& q_right,
        const Primitive& q_bottom,
        const Primitive& q_top,
        Real gamma)
    {
        BoundaryConfig<System> cfg;

        cfg.left = AnyBc<System>::dirichlet_from_primitive(q_left, gamma);
        cfg.right = AnyBc<System>::dirichlet_from_primitive(q_right, gamma);
        cfg.bottom = AnyBc<System>::dirichlet_from_primitive(q_bottom, gamma);
        cfg.top = AnyBc<System>::dirichlet_from_primitive(q_top, gamma);

        return cfg;
    }
};

} // namespace subsetix::fvd
