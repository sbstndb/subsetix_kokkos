#pragma once

#include <Kokkos_Core.hpp>
#include <cmath>
#include "concepts.hpp"

namespace subsetix::fvd {

// ============================================================================
// ADVECTION2D SYSTEM - 2D Linear Advection Equation
// ============================================================================

/**
 * @brief 2D Linear Advection Equation
 *
 * Equation: ∂u/∂t + v_x ∂u/∂x + v_y ∂u/∂y = 0
 *
 * Conserved variables: U = (value)
 * - value: scalar quantity being advected
 *
 * Primitive variables: Q = (value)
 * - value: same as conserved (no conversion needed)
 *
 * This is a simple scalar PDE used to demonstrate system genericity.
 * Phase 6: Extensibility Validation - proves solver works for ANY System.
 *
 * Runtime Parameters:
 * - vx: x-velocity component (constant)
 * - vy: y-velocity component (constant)
 */
template<typename Real = float>
class Advection2D {
public:
    // ========================================================================
    // 1. Type Definitions
    // ========================================================================

    using RealType = Real;

    /// Conserved variables U = (value)
    struct Conserved {
        Real value = Real(0);

        KOKKOS_INLINE_FUNCTION
        Conserved() = default;

        KOKKOS_INLINE_FUNCTION
        Conserved(Real v_) : value(v_) {}

        KOKKOS_INLINE_FUNCTION
        Conserved& operator+=(const Conserved& other) {
            value += other.value;
            return *this;
        }

        KOKKOS_INLINE_FUNCTION
        Conserved& operator*=(Real s) {
            value *= s;
            return *this;
        }

        KOKKOS_INLINE_FUNCTION
        Conserved operator*(Real s) const {
            return Conserved{value * s};
        }

        KOKKOS_INLINE_FUNCTION
        Conserved operator+(const Conserved& other) const {
            return Conserved{value + other.value};
        }

        KOKKOS_INLINE_FUNCTION
        Conserved operator-(const Conserved& other) const {
            return Conserved{value - other.value};
        }
    };

    /// Primitive variables Q = (value)
    /// For advection, primitive = conserved (no conversion needed)
    struct Primitive {
        Real value = Real(0);

        KOKKOS_INLINE_FUNCTION
        Primitive() = default;

        KOKKOS_INLINE_FUNCTION
        Primitive(Real v_) : value(v_) {}
    };

    /// Views wrapper for device access
    struct Views {
        // Field references
        const Real* value = nullptr;

        // Geometry reference
        const void* geometry_ref = nullptr;

        KOKKOS_INLINE_FUNCTION
        Conserved gather(std::size_t idx) const {
            return Conserved{value[idx]};
        }

        KOKKOS_INLINE_FUNCTION
        void scatter(std::size_t idx, const Conserved& U) {
            const_cast<Real*>(value)[idx] = U.value;
        }
    };

    // ========================================================================
    // 2. Runtime Parameters (Instance Data)
    // ========================================================================

    Real vx = Real(1);  // x-velocity component
    Real vy = Real(0);  // y-velocity component

    KOKKOS_INLINE_FUNCTION
    Advection2D() = default;

    KOKKOS_INLINE_FUNCTION
    Advection2D(Real vx_, Real vy_) : vx(vx_), vy(vy_) {}

    // ========================================================================
    // 3. Static Constants
    // ========================================================================

    static constexpr Real default_gamma = Real(1);  // Not used for advection

    // ========================================================================
    // 4. Static Functions (State conversions)
    // ========================================================================

    /// Convert conserved to primitive (identity for advection)
    KOKKOS_INLINE_FUNCTION
    static Primitive to_primitive(const Conserved& U,
                                   Real /*gamma*/ = default_gamma) {
        return Primitive{U.value};
    }

    /// Convert primitive to conserved (identity for advection)
    KOKKOS_INLINE_FUNCTION
    static Conserved from_primitive(const Primitive& q,
                                     Real /*gamma*/ = default_gamma) {
        return Conserved{q.value};
    }

    /// Compute "sound speed" (wave speed for advection = |velocity|)
    /// Note: This is an instance method that uses runtime parameters
    KOKKOS_INLINE_FUNCTION
    Real wave_speed() const {
        return Kokkos::sqrt(vx * vx + vy * vy);
    }

    /// Static sound speed (placeholder for interface compatibility)
    KOKKOS_INLINE_FUNCTION
    static Real sound_speed(const Primitive& /*q*/,
                            Real /*gamma*/ = default_gamma) {
        return Real(1);  // Default max wave speed
    }

    // ========================================================================
    // 5. Physical Fluxes
    // ========================================================================

    /// Physical flux in x-direction: F(U) = vx * value
    /// Static version (required by FiniteVolumeSystem concept)
    /// Uses default velocity vx=1
    KOKKOS_INLINE_FUNCTION
    static Conserved flux_phys_x(const Conserved& U, const Primitive& /*q*/) {
        return Conserved{U.value};  // Default vx = 1
    }

    /// Physical flux in y-direction: G(U) = vy * value
    /// Static version (required by FiniteVolumeSystem concept)
    /// Uses default velocity vy=0
    KOKKOS_INLINE_FUNCTION
    static Conserved flux_phys_y(const Conserved& U, const Primitive& /*q*/) {
        return Conserved{Real(0)};  // Default vy = 0
    }

    /// Instance flux methods with runtime velocities
    /// These use the actual vx, vy parameters
    KOKKOS_INLINE_FUNCTION
    Conserved flux_phys_x_runtime(const Conserved& U, const Primitive& /*q*/) const {
        return Conserved{vx * U.value};
    }

    KOKKOS_INLINE_FUNCTION
    Conserved flux_phys_y_runtime(const Conserved& U, const Primitive& /*q*/) const {
        return Conserved{vy * U.value};
    }

    // ========================================================================
    // 6. System Traits Integration
    // ========================================================================

    static constexpr int num_vars = 1;
    static constexpr const char* name = "Advection2D";
};

// ============================================================================
// SYSTEM TRAITS SPECIALIZATION FOR ADVECTION2D
// ============================================================================

template<typename Real>
struct system_traits<Advection2D<Real>> {
    static constexpr int n_conserved = 1;
    static constexpr const char* const names[1] = {"value"};
    static constexpr int num_vars = Advection2D<Real>::num_vars;
    static constexpr const char* name = Advection2D<Real>::name;

    /// Generic field access for Views
    KOKKOS_INLINE_FUNCTION
    static Real get_field(const typename Advection2D<Real>::Views& views,
                         int /*field_idx*/,
                         std::size_t idx) {
        return views.value[idx];
    }

    KOKKOS_INLINE_FUNCTION
    static void set_field(const typename Advection2D<Real>::Views& views,
                         int /*field_idx*/,
                         std::size_t idx,
                         Real val) {
        const_cast<Real*>(views.value)[idx] = val;
    }

    /// Generic iteration over fields
    template<typename Func>
    KOKKOS_INLINE_FUNCTION
    static void for_each_field(const typename Advection2D<Real>::Conserved& U,
                               Func&& func) {
        func(0, U.value);
    }

    template<typename Func>
    KOKKOS_INLINE_FUNCTION
    static void for_each_field(typename Advection2D<Real>::Conserved& U,
                               Func&& func) {
        func(0, U.value);
    }

    /// Get primitive field by index (0-based) - for refinement criteria compatibility
    /// For Advection2D, index 0 returns the scalar value
    KOKKOS_INLINE_FUNCTION
    static Real get_primitive_field(const typename Advection2D<Real>::Primitive& q,
                                    int field_idx) {
        return q.value;  // All fields map to the scalar value for advection
    }
};

// ============================================================================
// MARK ADVECTION2D AS A VALID SYSTEM
// ============================================================================

template<typename Real>
struct IsSystem<Advection2D<Real>> {
    static constexpr bool value = true;
};

} // namespace subsetix::fvd
