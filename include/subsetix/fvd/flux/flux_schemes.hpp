#pragma once

#include <Kokkos_Core.hpp>
#include "../system/concepts.hpp"

namespace subsetix::fvd::flux {

// ============================================================================
// NUMERICAL FLUX CONCEPT
// ============================================================================

/**
 * @brief Numerical flux concept
 *
 * A numerical flux scheme must provide:
 * - System system_instance (for runtime parameters)
 * - Real gamma (or other parameters)
 * - flux_x(UL, UR, qL, qR) -> numerical flux in x-direction
 * - flux_y(UL, UR, qL, qR) -> numerical flux in y-direction
 *
 * P0-4 FIX: System instance is stored to support Systems with runtime
 * parameters (e.g., Advection2D with vx, vy)
 */

// ============================================================================
// RUSANOV FLUX (Local Lax-Friedrichs)
// ============================================================================

/**
 * @brief Rusanov flux - simplest upwind-type flux
 *
 * F = 0.5 * (FL + FR) - 0.5 * smax * (UR - UL)
 *
 * where smax = max(|uL| + aL, |uR| + aR)
 *
 * Pros: Very robust, simple, TVD
 * Cons: Very dissipative (smears shocks)
 */
template<typename System>
struct RusanovFlux {
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    // P0-4 FIX: Store System instance for runtime parameters
    System system_instance;
    Real gamma = System::default_gamma;

    // Default constructor
    RusanovFlux() = default;

    // Constructor with gamma
    KOKKOS_INLINE_FUNCTION
    RusanovFlux(Real g) : gamma(g) {}

    // P0-4 FIX: Constructor with System instance
    KOKKOS_INLINE_FUNCTION
    RusanovFlux(Real g, const System& sys)
        : system_instance(sys), gamma(g) {}

    /// Numerical flux in x-direction
    KOKKOS_INLINE_FUNCTION
    Conserved flux_x(
        const Conserved& UL,
        const Conserved& UR,
        const Primitive& qL,
        const Primitive& qR) const
    {
        Real aL = System::sound_speed(qL, gamma);
        Real aR = System::sound_speed(qR, gamma);
        Real smax = Kokkos::fmax(Kokkos::fabs(qL.u) + aL, Kokkos::fabs(qR.u) + aR);

        auto FL = system_instance.flux_phys_x(UL, qL);
        auto FR = system_instance.flux_phys_x(UR, qR);

        Conserved F;
        Real half = Real(0.5);
        F.rho  = half * (FL.rho + FR.rho)  - half * smax * (UR.rho - UL.rho);
        F.rhou = half * (FL.rhou + FR.rhou) - half * smax * (UR.rhou - UL.rhou);
        F.rhov = half * (FL.rhov + FR.rhov) - half * smax * (UR.rhov - UL.rhov);
        F.E    = half * (FL.E + FR.E)      - half * smax * (UR.E - UL.E);

        return F;
    }

    /// Numerical flux in y-direction
    KOKKOS_INLINE_FUNCTION
    Conserved flux_y(
        const Conserved& UL,
        const Conserved& UR,
        const Primitive& qL,
        const Primitive& qR) const
    {
        Real aL = System::sound_speed(qL, gamma);
        Real aR = System::sound_speed(qR, gamma);
        Real smax = Kokkos::fmax(Kokkos::fabs(qL.v) + aL, Kokkos::fabs(qR.v) + aR);

        auto FL = system_instance.flux_phys_y(UL, qL);
        auto FR = system_instance.flux_phys_y(UR, qR);

        Conserved F;
        Real half = Real(0.5);
        F.rho  = half * (FL.rho + FR.rho)  - half * smax * (UR.rho - UL.rho);
        F.rhou = half * (FL.rhou + FR.rhou) - half * smax * (UR.rhou - UL.rhou);
        F.rhov = half * (FL.rhov + FR.rhov) - half * smax * (UR.rhov - UL.rhov);
        F.E    = half * (FL.E + FR.E)      - half * smax * (UR.E - UL.E);

        return F;
    }
};

// ============================================================================
// HLLC FLUX (Harten-Lax-van Leer-Contact)
// ============================================================================

/**
 * @brief HLLC flux - captures contact discontinuities
 *
 * Better than Rusanov for shocks, still relatively simple.
 *
 * NOTE: Full implementation in production, this is a stub.
 */
template<typename System>
struct HLLCFlux {
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    System system_instance;
    Real gamma = System::default_gamma;

    HLLCFlux() = default;

    KOKKOS_INLINE_FUNCTION
    HLLCFlux(Real g) : gamma(g) {}

    KOKKOS_INLINE_FUNCTION
    HLLCFlux(Real g, const System& sys)
        : system_instance(sys), gamma(g) {}

    // Stub: falls back to Rusanov for now
    KOKKOS_INLINE_FUNCTION
    Conserved flux_x(
        const Conserved& UL,
        const Conserved& UR,
        const Primitive& qL,
        const Primitive& qR) const
    {
        RusanovFlux<System> rusanov(gamma, system_instance);
        return rusanov.flux_x(UL, UR, qL, qR);
    }

    KOKKOS_INLINE_FUNCTION
    Conserved flux_y(
        const Conserved& UL,
        const Conserved& UR,
        const Primitive& qL,
        const Primitive& qR) const
    {
        RusanovFlux<System> rusanov(gamma, system_instance);
        return rusanov.flux_y(UL, UR, qL, qR);
    }
};

// ============================================================================
// ROE FLUX
// ============================================================================

/**
 * @brief Roe flux - approximate Riemann solver
 *
 * High accuracy, more expensive.
 *
 * NOTE: Full implementation in production, this is a stub.
 */
template<typename System>
struct RoeFlux {
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;
    using Primitive = typename System::Primitive;

    System system_instance;
    Real gamma = System::default_gamma;

    RoeFlux() = default;

    KOKKOS_INLINE_FUNCTION
    RoeFlux(Real g) : gamma(g) {}

    KOKKOS_INLINE_FUNCTION
    RoeFlux(Real g, const System& sys)
        : system_instance(sys), gamma(g) {}

    // Stub: falls back to Rusanov for now
    KOKKOS_INLINE_FUNCTION
    Conserved flux_x(
        const Conserved& UL,
        const Conserved& UR,
        const Primitive& qL,
        const Primitive& qR) const
    {
        RusanovFlux<System> rusanov(gamma, system_instance);
        return rusanov.flux_x(UL, UR, qL, qR);
    }

    KOKKOS_INLINE_FUNCTION
    Conserved flux_y(
        const Conserved& UL,
        const Conserved& UR,
        const Primitive& qL,
        const Primitive& qR) const
    {
        RusanovFlux<System> rusanov(gamma, system_instance);
        return rusanov.flux_y(UL, UR, qL, qR);
    }
};

} // namespace subsetix::fvd::flux
