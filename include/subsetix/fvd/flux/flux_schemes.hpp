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
 * HLLC improves upon HLL by restoring the contact wave.
 *
 * Algorithm:
 * 1. Estimate wave speeds S_L, S_R using Davis (or Einfeldt) estimates
 * 2. Compute contact wave speed S_M
 * 3. Compute star states U*_L, U*_R
 * 4. Select flux based on region
 *
 * References:
 * - Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics"
 * - Batten et al., "On the choice of wave speeds for the HLLC Riemann solver"
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

    /**
     * @brief Compute HLLC flux in x-direction
     *
     * The HLLC flux is:
     * - F_L if 0 <= S_L
     * - F*_L if S_L <= 0 <= S_M
     * - F*_R if S_M <= 0 <= S_R
     * - F_R if 0 >= S_R
     */
    KOKKOS_INLINE_FUNCTION
    Conserved flux_x(
        const Conserved& UL,
        const Conserved& UR,
        const Primitive& qL,
        const Primitive& qR) const
    {
        // Physical fluxes
        auto FL = system_instance.flux_phys_x(UL, qL);
        auto FR = system_instance.flux_phys_x(UR, qR);

        // Sound speeds
        Real aL = System::sound_speed(qL, gamma);
        Real aR = System::sound_speed(qR, gamma);

        // Wave speed estimates (Davis - simpler than Einfeldt)
        // S_L = min(uL - aL, uR - aR)
        // S_R = max(uL + aL, uR + aR)
        Real SL = Kokkos::fmin(qL.u - aL, qR.u - aR);
        Real SR = Kokkos::fmax(qL.u + aL, qR.u + aR);

        // Pressure estimate (Davis)
        Real pL = qL.p;
        Real pR = qR.p;
        Real p_star = Real(0.5) * (pL + pR) -
                      Real(0.5) * (qR.u - qL.u) * Real(0.5) * (UL.rho + UR.rho) * (SR - SL);

        // Contact wave speed (S_M)
        // S_M = (pR - pL + rhoL*uL*(SL - uL) - rhoR*uR*(SR - uR)) /
        //       (rhoL*(SL - uL) - rhoR*(SR - uR))
        Real denom = UL.rho * (SL - qL.u) - UR.rho * (SR - qR.u);
        Real SM;
        if (Kokkos::fabs(denom) > Real(1e-10)) {
            SM = (pR - pL + UL.rho * qL.u * (SL - qL.u) - UR.rho * qR.u * (SR - qR.u)) / denom;
        } else {
            // Fallback to average velocity
            SM = Real(0.5) * (qL.u + qR.u);
        }

        // Compute star states
        // U*_L = UL * (SL - uL) / (SL - SM) + [0, (p* - pL), 0, (p* - pL)*uL] / (SL - SM)
        // U*_R = UR * (SR - uR) / (SR - SM) + [0, (p* - pR), 0, (p* - pR)*uR] / (SR - SM)

        Conserved F;  // Result flux

        if (SL >= Real(0)) {
            // Region 1: flux is FL
            F = FL;
        } else if (SM >= Real(0)) {
            // Region 2: flux is F*_L = FL + SL*(U*_L - UL)
            Real denom_L = SL - SM;
            if (Kokkos::fabs(denom_L) > Real(1e-10)) {
                Real factor_L = (SL - qL.u) / denom_L;
                Real p_star_minus_pL = p_star - pL;

                // U*_L - UL
                Real delta_rho = UL.rho * (factor_L - Real(1));
                Real delta_rhou = UL.rho * (SM - qL.u) + p_star_minus_pL;
                Real delta_rhov = UL.rhov * (factor_L - Real(1));
                Real delta_E = UL.E * (factor_L - Real(1)) + (p_star_minus_pL * SM - pL * qL.u) / denom_L;

                F.rho  = FL.rho  + SL * delta_rho;
                F.rhou = FL.rhou + SL * delta_rhou;
                F.rhov = FL.rhov + SL * delta_rhov;
                F.E    = FL.E    + SL * delta_E;
            } else {
                F = FL;
            }
        } else if (SR >= Real(0)) {
            // Region 3: flux is F*_R = FR + SR*(U*_R - UR)
            Real denom_R = SR - SM;
            if (Kokkos::fabs(denom_R) > Real(1e-10)) {
                Real factor_R = (SR - qR.u) / denom_R;
                Real p_star_minus_pR = p_star - pR;

                // U*_R - UR
                Real delta_rho = UR.rho * (factor_R - Real(1));
                Real delta_rhou = UR.rho * (SM - qR.u) + p_star_minus_pR;
                Real delta_rhov = UR.rhov * (factor_R - Real(1));
                Real delta_E = UR.E * (factor_R - Real(1)) + (p_star_minus_pR * SM - pR * qR.u) / denom_R;

                F.rho  = FR.rho  + SR * delta_rho;
                F.rhou = FR.rhou + SR * delta_rhou;
                F.rhov = FR.rhov + SR * delta_rhov;
                F.E    = FR.E    + SR * delta_E;
            } else {
                F = FR;
            }
        } else {
            // Region 4: flux is FR
            F = FR;
        }

        return F;
    }

    KOKKOS_INLINE_FUNCTION
    Conserved flux_y(
        const Conserved& UL,
        const Conserved& UR,
        const Primitive& qL,
        const Primitive& qR) const
    {
        // For y-direction, we use v as the normal velocity
        // The algorithm is the same, just replace u with v

        // Physical fluxes
        auto FL = system_instance.flux_phys_y(UL, qL);
        auto FR = system_instance.flux_phys_y(UR, qR);

        // Sound speeds
        Real aL = System::sound_speed(qL, gamma);
        Real aR = System::sound_speed(qR, gamma);

        // Wave speed estimates (use v for y-direction)
        Real SL = Kokkos::fmin(qL.v - aL, qR.v - aR);
        Real SR = Kokkos::fmax(qL.v + aL, qR.v + aR);

        // Pressure estimate
        Real pL = qL.p;
        Real pR = qR.p;
        Real p_star = Real(0.5) * (pL + pR) -
                      Real(0.5) * (qR.v - qL.v) * Real(0.5) * (UL.rho + UR.rho) * (SR - SL);

        // Contact wave speed
        Real denom = UL.rho * (SL - qL.v) - UR.rho * (SR - qR.v);
        Real SM;
        if (Kokkos::fabs(denom) > Real(1e-10)) {
            SM = (pR - pL + UL.rho * qL.v * (SL - qL.v) - UR.rho * qR.v * (SR - qR.v)) / denom;
        } else {
            SM = Real(0.5) * (qL.v + qR.v);
        }

        Conserved F;

        if (SL >= Real(0)) {
            F = FL;
        } else if (SM >= Real(0)) {
            Real denom_L = SL - SM;
            if (Kokkos::fabs(denom_L) > Real(1e-10)) {
                Real factor_L = (SL - qL.v) / denom_L;
                Real p_star_minus_pL = p_star - pL;

                Real delta_rho = UL.rho * (factor_L - Real(1));
                Real delta_rhou = UL.rhou * (factor_L - Real(1));
                Real delta_rhov = UL.rho * (SM - qL.v) + p_star_minus_pL;
                Real delta_E = UL.E * (factor_L - Real(1)) + (p_star_minus_pL * SM - pL * qL.v) / denom_L;

                F.rho  = FL.rho  + SL * delta_rho;
                F.rhou = FL.rhou + SL * delta_rhou;
                F.rhov = FL.rhov + SL * delta_rhov;
                F.E    = FL.E    + SL * delta_E;
            } else {
                F = FL;
            }
        } else if (SR >= Real(0)) {
            Real denom_R = SR - SM;
            if (Kokkos::fabs(denom_R) > Real(1e-10)) {
                Real factor_R = (SR - qR.v) / denom_R;
                Real p_star_minus_pR = p_star - pR;

                Real delta_rho = UR.rho * (factor_R - Real(1));
                Real delta_rhou = UR.rhou * (factor_R - Real(1));
                Real delta_rhov = UR.rho * (SM - qR.v) + p_star_minus_pR;
                Real delta_E = UR.E * (factor_R - Real(1)) + (p_star_minus_pR * SM - pR * qR.v) / denom_R;

                F.rho  = FR.rho  + SR * delta_rho;
                F.rhou = FR.rhou + SR * delta_rhou;
                F.rhov = FR.rhov + SR * delta_rhov;
                F.E    = FR.E    + SR * delta_E;
            } else {
                F = FR;
            }
        } else {
            F = FR;
        }

        return F;
    }
};

// ============================================================================
// ROE FLUX
// ============================================================================

/**
 * @brief Roe flux - approximate Riemann solver
 *
 * Roe's solver linearizes the Riemann problem about a Roe-averaged state.
 * Provides high accuracy for smooth flows but requires entropy fix for shocks.
 *
 * Algorithm:
 * 1. Compute Roe-averaged state
 * 2. Compute wave strengths (alpha_i)
 * 3. Compute absolute eigenvalues with entropy fix
 * 4. F_roe = 0.5*(FL + FR) - 0.5*sum(|lambda_i|*alpha_i*r_i)
 *
 * References:
 * - Roe, "Approximate Riemann solvers, parameter vectors, and difference schemes"
 * - Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics"
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

    /**
     * @brief Compute Roe-averaged quantities
     *
     * Roe averages are based on sqrt(rho) weighting:
     * z = sqrt(rho)
     * u_roe = (zL*uL + zR*uR) / (zL + zR)
     * h_roe = (zL*hL + zR*hR) / (zL + zR)
     */
    KOKKOS_INLINE_FUNCTION
    void roe_averages(
        const Primitive& qL,
        const Primitive& qR,
        Real& u_roe,
        Real& v_roe,
        Real& h_roe,
        Real& a_roe) const
    {
        // Enthalpy: h = (E + p) / rho = gamma/(gamma-1) * p/rho + 0.5*(u^2 + v^2)
        Real gm1 = gamma - Real(1);
        Real gp1 = gamma + Real(1);
        Real hL = (gp1 / gm1) * (qL.p / qL.rho) + Real(0.5) * (qL.u * qL.u + qL.v * qL.v);
        Real hR = (gp1 / gm1) * (qR.p / qR.rho) + Real(0.5) * (qR.u * qR.u + qR.v * qR.v);

        // Square root density weighting
        Real zL = Kokkos::sqrt(qL.rho);
        Real zR = Kokkos::sqrt(qR.rho);
        Real inv_z_sum = Real(1) / (zL + zR + Real(1e-10));

        u_roe = (zL * qL.u + zR * qR.u) * inv_z_sum;
        v_roe = (zL * qL.v + zR * qR.v) * inv_z_sum;
        h_roe = (zL * hL + zR * hR) * inv_z_sum;

        // Speed of sound at Roe state: a^2 = (gamma-1)*(h - 0.5*(u^2+v^2))
        Real q2 = u_roe * u_roe + v_roe * v_roe;
        a_roe = Kokkos::sqrt(gm1 * (h_roe - Real(0.5) * q2));
        a_roe = Kokkos::fmax(a_roe, Real(1e-6));  // Prevent div by zero
    }

    /**
     * @brief Entropy fix for Roe scheme
     *
     * Prevents expansion shocks by replacing |lambda| with a smoothed version
     * when eigenvalues change sign.
     */
    KOKKOS_INLINE_FUNCTION
    Real entropy_fix(Real lambda, Real a) const {
        const Real epsilon = Real(0.1) * a;  // 10% threshold

        if (Kokkos::fabs(lambda) < epsilon) {
            // Smoothed version: (lambda^2 + epsilon^2) / (2*epsilon)
            return (lambda * lambda + epsilon * epsilon) / (Real(2) * epsilon);
        }
        return Kokkos::fabs(lambda);
    }

    /**
     * @brief Compute Roe flux in x-direction
     *
     * F_roe = 0.5*(FL + FR) - 0.5*R*|Lambda|*alpha
     * where R is right eigenvector matrix, |Lambda| is abs eigenvalues,
     * alpha is wave strength vector
     */
    KOKKOS_INLINE_FUNCTION
    Conserved flux_x(
        const Conserved& UL,
        const Conserved& UR,
        const Primitive& qL,
        const Primitive& qR) const
    {
        // Physical fluxes
        auto FL = system_instance.flux_phys_x(UL, qL);
        auto FR = system_instance.flux_phys_x(UR, qR);

        // Roe-averaged state
        Real u_roe, v_roe, h_roe, a_roe;
        roe_averages(qL, qR, u_roe, v_roe, h_roe, a_roe);

        // Difference in conserved variables
        Real d_rho = UR.rho - UL.rho;
        Real d_rhou = UR.rhou - UL.rhou;
        Real d_rhov = UR.rhov - UL.rhov;
        Real d_E = UR.E - UL.E;

        // Wave strengths (alpha) for 2D Euler in x-direction
        // alpha[3] (entropy wave) is computed first
        Real gm1 = gamma - Real(1);
        Real q2 = u_roe * u_roe + v_roe * v_roe;
        Real a2 = a_roe * a_roe;

        // Wave strength 3: entropy wave
        Real alpha3 = gm1 * ((h_roe - u_roe * u_roe) * d_rho +
                            u_roe * d_rhou +
                            v_roe * d_rhov -
                            d_E) / (a2);

        // Wave strength 2: shear wave
        Real alpha2 = (qL.rho * qR.v - qR.rho * qL.v +
                      (qL.rho - qR.rho) * v_roe) /
                     (qL.rho + qR.rho + Real(1e-10));

        // Wave strengths 1 and 4: acoustic waves
        Real alpha4 = Real(0.5) / (a_roe * a_roe) * (
            (d_rho * (q2 + gm1 * u_roe * u_roe) +
             d_rhou * ((Real(1) - gm1) * u_roe - a_roe) +
             d_rhov * ((Real(1) - gm1) * v_roe) -
             gm1 * d_E) -
            a_roe * (qL.rho + qR.rho) * alpha3
        );

        Real alpha1 = d_rho - alpha3 - alpha4;

        // Absolute eigenvalues with entropy fix
        Real lambda1 = Kokkos::fabs(u_roe - a_roe);
        Real lambda2 = Kokkos::fabs(u_roe);
        Real lambda3 = Kokkos::fabs(u_roe);
        Real lambda4 = Kokkos::fabs(u_roe + a_roe);

        // Apply entropy fix
        lambda1 = entropy_fix(u_roe - a_roe, a_roe);
        lambda4 = entropy_fix(u_roe + a_roe, a_roe);

        // Compute dissipation term: R * |Lambda| * alpha
        // For 2D Euler x-flux, right eigenvectors are:
        // r1 = [1, u-a, v, H-u*a]^T
        // r2 = [0, 0, 1, v]^T
        // r3 = [1, u, v, 0.5*(u^2+v^2)]^T
        // r4 = [1, u+a, v, H+u*a]^T

        Real H = h_roe + a2 / gm1;  // Total enthalpy

        // Dissipative flux components
        Real diss_rho = lambda1 * alpha1 + lambda2 * Real(0) + lambda3 * alpha3 + lambda4 * alpha4;
        Real diss_rhou = lambda1 * alpha1 * (u_roe - a_roe) +
                        lambda2 * Real(0) +
                        lambda3 * alpha3 * u_roe +
                        lambda4 * alpha4 * (u_roe + a_roe);
        Real diss_rhov = lambda1 * alpha1 * v_roe +
                        lambda2 * alpha2 +
                        lambda3 * alpha3 * v_roe +
                        lambda4 * alpha4 * v_roe;
        Real diss_E = lambda1 * alpha1 * (H - u_roe * a_roe) +
                     lambda2 * alpha2 * v_roe +
                     lambda3 * alpha3 * (Real(0.5) * q2) +
                     lambda4 * alpha4 * (H + u_roe * a_roe);

        // Final Roe flux
        Conserved F;
        Real half = Real(0.5);
        F.rho  = half * (FL.rho  + FR.rho ) - half * diss_rho;
        F.rhou = half * (FL.rhou + FR.rhou) - half * diss_rhou;
        F.rhov = half * (FL.rhov + FR.rhov) - half * diss_rhov;
        F.E    = half * (FL.E    + FR.E   ) - half * diss_E;

        return F;
    }

    /**
     * @brief Compute Roe flux in y-direction
     *
     * Similar to x-direction but with v as normal velocity
     */
    KOKKOS_INLINE_FUNCTION
    Conserved flux_y(
        const Conserved& UL,
        const Conserved& UR,
        const Primitive& qL,
        const Primitive& qR) const
    {
        // Physical fluxes
        auto FL = system_instance.flux_phys_y(UL, qL);
        auto FR = system_instance.flux_phys_y(UR, qR);

        // Roe-averaged state
        Real u_roe, v_roe, h_roe, a_roe;
        roe_averages(qL, qR, u_roe, v_roe, h_roe, a_roe);

        // Difference in conserved variables
        Real d_rho = UR.rho - UL.rho;
        Real d_rhou = UR.rhou - UL.rhou;
        Real d_rhov = UR.rhov - UL.rhov;
        Real d_E = UR.E - UL.E;

        // For y-direction, v is the normal velocity
        Real gm1 = gamma - Real(1);
        Real q2 = u_roe * u_roe + v_roe * v_roe;
        Real a2 = a_roe * a_roe;

        // Wave strength 3: entropy wave
        Real alpha3 = gm1 * ((h_roe - v_roe * v_roe) * d_rho +
                            v_roe * d_rhov +
                            u_roe * d_rhou -
                            d_E) / (a2);

        // Wave strength 2: shear wave (u is now transverse)
        Real alpha2 = (qL.rho * qR.u - qR.rho * qL.u +
                      (qL.rho - qR.rho) * u_roe) /
                     (qL.rho + qR.rho + Real(1e-10));

        // Wave strengths 1 and 4: acoustic waves
        Real alpha4 = Real(0.5) / (a_roe * a_roe) * (
            (d_rho * (q2 + gm1 * v_roe * v_roe) +
             d_rhov * ((Real(1) - gm1) * v_roe - a_roe) +
             d_rhou * ((Real(1) - gm1) * u_roe) -
             gm1 * d_E) -
            a_roe * (qL.rho + qR.rho) * alpha3
        );

        Real alpha1 = d_rho - alpha3 - alpha4;

        // Absolute eigenvalues (using v for y-direction)
        Real lambda1 = Kokkos::fabs(v_roe - a_roe);
        Real lambda2 = Kokkos::fabs(v_roe);
        Real lambda3 = Kokkos::fabs(v_roe);
        Real lambda4 = Kokkos::fabs(v_roe + a_roe);

        // Apply entropy fix
        lambda1 = entropy_fix(v_roe - a_roe, a_roe);
        lambda4 = entropy_fix(v_roe + a_roe, a_roe);

        // Dissipation term for y-direction
        // For y-flux, eigenvectors are similar but with u and v swapped appropriately
        Real H = h_roe + a2 / gm1;

        Real diss_rho = lambda1 * alpha1 + lambda2 * Real(0) + lambda3 * alpha3 + lambda4 * alpha4;
        Real diss_rhou = lambda1 * alpha1 * u_roe +
                        lambda2 * alpha2 +
                        lambda3 * alpha3 * u_roe +
                        lambda4 * alpha4 * u_roe;
        Real diss_rhov = lambda1 * alpha1 * (v_roe - a_roe) +
                        lambda2 * Real(0) +
                        lambda3 * alpha3 * v_roe +
                        lambda4 * alpha4 * (v_roe + a_roe);
        Real diss_E = lambda1 * alpha1 * (H - v_roe * a_roe) +
                     lambda2 * alpha2 * u_roe +
                     lambda3 * alpha3 * (Real(0.5) * q2) +
                     lambda4 * alpha4 * (H + v_roe * a_roe);

        // Final Roe flux
        Conserved F;
        Real half = Real(0.5);
        F.rho  = half * (FL.rho  + FR.rho ) - half * diss_rho;
        F.rhou = half * (FL.rhou + FR.rhou) - half * diss_rhou;
        F.rhov = half * (FL.rhov + FR.rhov) - half * diss_rhov;
        F.E    = half * (FL.E    + FR.E   ) - half * diss_E;

        return F;
    }
};

} // namespace subsetix::fvd::flux
