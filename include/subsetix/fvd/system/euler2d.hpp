#pragma once

#include <Kokkos_Core.hpp>
#include <cmath>
#include "concepts.hpp"

namespace subsetix::fvd {

// ============================================================================
// EULER2D SYSTEM - 2D Compressible Euler Equations
// ============================================================================

/**
 * @brief 2D Compressible Euler Equations
 *
 * Conserved variables: U = (rho, rhou, rhov, E)
 * - rho: density
 * - rhou: x-momentum
 * - rhov: y-momentum
 * - E: total energy per unit volume
 *
 * Primitive variables: Q = (rho, u, v, p)
 * - rho: density
 * - u: x-velocity
 * - v: y-velocity
 * - p: pressure
 *
 * Equation of state: Ideal gas
 *   p = (gamma - 1) * (E - 0.5 * rho * (u^2 + v^2))
 */
template<typename Real = float>
class Euler2D {
public:
    // ========================================================================
    // 1. Type Definitions
    // ========================================================================

    using RealType = Real;

    /// Conserved variables U = (rho, rhou, rhov, E)
    struct Conserved {
        Real rho = Real(0);
        Real rhou = Real(0);
        Real rhov = Real(0);
        Real E = Real(0);

        KOKKOS_INLINE_FUNCTION
        Conserved() = default;

        KOKKOS_INLINE_FUNCTION
        Conserved(Real r_, Real rhox_, Real rhoy_, Real E_)
            : rho(r_), rhou(rhox_), rhov(rhoy_), E(E_) {}
    };

    /// Primitive variables Q = (rho, u, v, p)
    struct Primitive {
        Real rho = Real(0);
        Real u = Real(0);
        Real v = Real(0);
        Real p = Real(0);

        KOKKOS_INLINE_FUNCTION
        Primitive() = default;

        KOKKOS_INLINE_FUNCTION
        Primitive(Real r_, Real u_, Real v_, Real p_)
            : rho(r_), u(u_), v(v_), p(p_) {}
    };

    /// Views wrapper for device access
    struct Views {
        // Field references (these would be Kokkos::View in real implementation)
        const Real* rho = nullptr;
        const Real* rhou = nullptr;
        const Real* rhov = nullptr;
        const Real* E = nullptr;

        // Geometry reference (placeholder)
        const void* geometry_ref = nullptr;
    };

    // ========================================================================
    // 2. Static Constants
    // ========================================================================

    static constexpr Real default_gamma = Real(1.4);  // Air at standard conditions

    // ========================================================================
    // 3. Static Functions (State conversions)
    // ========================================================================

    /// Convert conserved to primitive variables
    /// IMPROVEMENT: gamma has default value, so caller can omit it
    KOKKOS_INLINE_FUNCTION
    static Primitive to_primitive(const Conserved& U,
                                   Real gamma = default_gamma) {
        constexpr Real eps = Real(1e-12);
        Real inv_rho = Real(1) / (U.rho + eps);
        Real u = U.rhou * inv_rho;
        Real v = U.rhov * inv_rho;
        Real kinetic = Real(0.5) * U.rho * (u * u + v * v);
        Real p = (gamma - Real(1)) * (U.E - kinetic);
        p = (p > eps) ? p : eps;  // Clamp to avoid negative pressure
        return Primitive{U.rho, u, v, p};
    }

    /// Convert primitive to conserved variables
    /// IMPROVEMENT: gamma has default value, so caller can omit it
    KOKKOS_INLINE_FUNCTION
    static Conserved from_primitive(const Primitive& q,
                                     Real gamma = default_gamma) {
        Real kinetic = Real(0.5) * q.rho * (q.u * q.u + q.v * q.v);
        return Conserved{
            q.rho,
            q.rho * q.u,
            q.rho * q.v,
            q.p / (gamma - Real(1)) + kinetic
        };
    }

    /// Compute sound speed
    /// IMPROVEMENT: gamma has default value, so caller can omit it
    KOKKOS_INLINE_FUNCTION
    static Real sound_speed(const Primitive& q,
                            Real gamma = default_gamma) {
        constexpr Real eps = Real(1e-12);
        return Kokkos::sqrt(gamma * q.p / (q.rho + eps));
    }

    // ========================================================================
    // 4. Physical Fluxes
    // ========================================================================

    /// Physical flux in x-direction: F(U) = (rhou, rhou*u + p, rhou*v, (E+p)*u)
    KOKKOS_INLINE_FUNCTION
    static Conserved flux_phys_x(const Conserved& U, const Primitive& q) {
        return Conserved{
            U.rhou,
            U.rho * q.u * q.u + q.p,
            U.rho * q.u * q.v,
            (U.E + q.p) * q.u
        };
    }

    /// Physical flux in y-direction: G(U) = (rhov, rhov*u, rhov*v + p, (E+p)*v)
    KOKKOS_INLINE_FUNCTION
    static Conserved flux_phys_y(const Conserved& U, const Primitive& q) {
        return Conserved{
            U.rhov,
            U.rho * q.u * q.v,
            U.rho * q.v * q.v + q.p,
            (U.E + q.p) * q.v
        };
    }

    // ========================================================================
    // 5. Pressure computation (utility)
    // ========================================================================

    /// Compute pressure from conserved variables
    /// IMPROVEMENT: gamma has default value, so caller can omit it
    KOKKOS_INLINE_FUNCTION
    static Real pressure(const Conserved& U,
                         Real gamma = default_gamma) {
        auto q = to_primitive(U, gamma);
        return q.p;
    }
};

// ============================================================================
// SYSTEM TRAITS SPECIALIZATION FOR EULER2D
// ============================================================================

template<typename Real>
struct system_traits<Euler2D<Real>> {
    static constexpr int n_conserved = 4;

    static constexpr const char* const names[n_conserved] = {
        "rho", "rhou", "rhov", "E"
    };
};

// ============================================================================
// MARK EULER2D AS A VALID SYSTEM
// ============================================================================

template<typename Real>
struct IsSystem<Euler2D<Real>> {
    static constexpr bool value = true;
};

} // namespace subsetix::fvd
