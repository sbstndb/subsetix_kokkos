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

        // Phase 6: Generic operators for multi-system support
        KOKKOS_INLINE_FUNCTION
        Conserved& operator+=(const Conserved& other) {
            rho  += other.rho;
            rhou += other.rhou;
            rhov += other.rhov;
            E    += other.E;
            return *this;
        }

        KOKKOS_INLINE_FUNCTION
        Conserved& operator*=(Real s) {
            rho  *= s;
            rhou *= s;
            rhov *= s;
            E    *= s;
            return *this;
        }

        KOKKOS_INLINE_FUNCTION
        Conserved operator+(const Conserved& other) const {
            return Conserved{rho + other.rho, rhou + other.rhou,
                            rhov + other.rhov, E + other.E};
        }

        KOKKOS_INLINE_FUNCTION
        Conserved operator*(Real s) const {
            return Conserved{rho * s, rhou * s, rhov * s, E * s};
        }

        KOKKOS_INLINE_FUNCTION
        Conserved operator-(const Conserved& other) const {
            return Conserved{rho - other.rho, rhou - other.rhou,
                            rhov - other.rhov, E - other.E};
        }
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
    ///
    /// WARNING: This function clips pressure to eps (1e-12) to avoid
    /// negative pressures. This creates INCONSISTENCY in round-trip:
    ///   U -> to_primitive -> from_primitive -> U'
    /// where U'.E ≠ U.E if original pressure was negative.
    ///
    /// For MUSCL reconstruction, this means:
    /// 1. Stencil cells with slightly negative pressure get clipped
    /// 2. Reconstruction operates on modified states
    /// 3. Converted-back conserved variables have wrong energy
    ///
    /// IMPROVEMENT: gamma has default value, so caller can omit it
    KOKKOS_INLINE_FUNCTION
    static Primitive to_primitive(const Conserved& U,
                                   Real gamma = default_gamma) {
        constexpr Real eps = Real(1e-12);
        Real inv_rho = Real(1) / (U.rho + eps);
        Real u = U.rhou * inv_rho;
        Real v = U.rhov * inv_rho;
        Real kinetic = Real(0.5) * (u * u + v * v);
        Real p = (gamma - Real(1)) * (U.E - U.rho * kinetic);
        p = Kokkos::fmax(p, eps);  // FIX: Use Kokkos::fmax for clarity and consistency
        return Primitive{U.rho, u, v, p};
    }

    /// Convert primitive to conserved variables
    ///
    /// NOTE: This function does NOT clip pressure. If you need perfect
    /// round-trip consistency with to_primitive, handle negative pressures
    /// explicitly before calling this function.
    ///
    /// IMPROVEMENT: gamma has default value, so caller can omit it
    KOKKOS_INLINE_FUNCTION
    static Conserved from_primitive(const Primitive& q,
                                     Real gamma = default_gamma) {
        constexpr Real eps = Real(1e-12);
        Real kinetic = Real(0.5) * q.rho * (q.u * q.u + q.v * q.v);
        // FIX: Match the clipping behavior from to_primitive for consistency
        Real p_safe = Kokkos::fmax(q.p, eps);
        return Conserved{
            q.rho,
            q.rho * q.u,
            q.rho * q.v,
            p_safe / (gamma - Real(1)) + kinetic
        };
    }

    /// Validate round-trip consistency
    /// Returns true if to_primitive(from_primitive(q)) == q (within tolerance)
    KOKKOS_INLINE_FUNCTION
    static bool validate_consistency(const Primitive& q,
                                      Real gamma = default_gamma,
                                      Real tol = Real(1e-10)) {
        constexpr Real eps = Real(1e-12);

        // Round-trip test
        Conserved U = from_primitive(q, gamma);
        Primitive q2 = to_primitive(U, gamma);

        // Check each component
        Real rho_err = Kokkos::fabs(q.rho - q2.rho) / (Kokkos::fabs(q.rho) + eps);
        Real u_err = Kokkos::fabs(q.u - q2.u) / (Kokkos::fabs(q.u) + eps);
        Real v_err = Kokkos::fabs(q.v - q2.v) / (Kokkos::fabs(q.v) + eps);
        Real p_err = Kokkos::fabs(q.p - q2.p) / (Kokkos::fabs(q.p) + eps);

        return (rho_err < tol) && (u_err < tol) &&
               (v_err < tol) && (p_err < tol);
    }

    /// Validate physical admissibility of state
    KOKKOS_INLINE_FUNCTION
    static bool is_physically_admissible(const Conserved& U,
                                      Real gamma = default_gamma) {
        constexpr Real eps = Real(1e-12);

        // Check density
        if (U.rho < eps) return false;

        // Check energy
        if (U.E < eps) return false;

        // Check pressure
        Real inv_rho = Real(1) / U.rho;
        Real u = U.rhou * inv_rho;
        Real v = U.rhov * inv_rho;
        Real kinetic = Real(0.5) * (u * u + v * v);
        Real p = (gamma - Real(1)) * (U.E - U.rho * kinetic);

        if (p < eps) return false;

        return true;
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

    // ========================================================================
    // 6. System metadata
    // ========================================================================

    static constexpr int n_conserved = 4;  // Number of conserved variables
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

    /// Get primitive field by index (0-based) - for refinement criteria compatibility
    /// Maps: 0->rho, 1->u, 2->v, 3->p
    KOKKOS_INLINE_FUNCTION
    static Real get_primitive_field(const typename Euler2D<Real>::Primitive& q,
                                    int field_idx) {
        switch (field_idx) {
            case 0: return q.rho;
            case 1: return q.u;
            case 2: return q.v;
            case 3: return q.p;
            default: return Real(0);
        }
    }

    /// Get conserved field by index (0-based) - for refinement criteria compatibility
    /// Maps: 0->rho, 1->rhou, 2->rhov, 3->E
    KOKKOS_INLINE_FUNCTION
    static Real get_conserved_field(const typename Euler2D<Real>::Conserved& U,
                                     int field_idx) {
        switch (field_idx) {
            case 0: return U.rho;
            case 1: return U.rhou;
            case 2: return U.rhov;
            case 3: return U.E;
            default: return Real(0);
        }
    }
};

// ============================================================================
// MARK EULER2D AS A VALID SYSTEM
// ============================================================================

template<typename Real>
struct IsSystem<Euler2D<Real>> {
    static constexpr bool value = true;
};

} // namespace subsetix::fvd
