#pragma once

#include <Kokkos_Core.hpp>
#include "../system/concepts.hpp"

namespace subsetix::fvd::reconstruction {

// ============================================================================
// RECONSTRUCTION CONCEPT
// ============================================================================

/**
 * @brief Reconstruction concept for high-order schemes
 *
 * A reconstruction scheme must provide:
 * - Operator() that takes cell-centered values and returns
 *   left/right interface values for MUSCL reconstruction
 *
 * For 1st order (no reconstruction), returns cell values directly.
 */

// ============================================================================
// NO RECONSTRUCTION (1st Order)
// ============================================================================

/**
 * @brief No reconstruction - 1st order upwind
 *
 * Returns cell values directly (piecewise constant).
 */
struct NoReconstruction {
    // Empty policy - no state needed
    NoReconstruction() = default;

    // 1st order: just return cell value (no reconstruction done here)
    // The actual scheme is handled in the flux computation
};

// ============================================================================
// MUSCL RECONSTRUCTION (2nd Order)
// ============================================================================

/**
 * @brief MUSCL reconstruction with slope limiters
 *
 * MUSCL (Monotonic Upstream-Centered Scheme for Conservation Laws)
 * achieves 2nd order spatial accuracy by reconstructing piecewise-linear
 * states within each cell.
 *
 * Algorithm:
 * 1. Compute slopes: delta_i = limiter(UL[i] - UL[i-1], UL[i+1] - UL[i])
 * 2. Reconstruct interface values:
 *    - U_L at i+1/2: U[i] + 0.5 * delta_i
 *    - U_R at i+1/2: U[i+1] - 0.5 * delta_{i+1}
 *
 * The limiter function ensures TVD (Total Variation Diminishing) property
 * to prevent oscillations near discontinuities.
 *
 * References:
 * - van Leer, "Towards the ultimate conservative difference scheme V"
 * - Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics"
 */
template<template<typename> class Limiter>
struct MUSCL_Reconstruction {
    // MUSCL reconstruction is stateless (limiter is compile-time policy)
    MUSCL_Reconstruction() = default;

    /**
     * @brief Reconstruct left interface value for cell i
     *
     * For interface at i+1/2 (right side of cell i):
     * U_L = U[i] + 0.5 * limiter(delta_L, delta_R)
     *
     * where delta_L = U[i] - U[i-1] (left slope)
     *       delta_R = U[i+1] - U[i] (right slope)
     *
     * @param U_center Cell value at i
     * @param U_left Cell value at i-1
     * @param U_right Cell value at i+1
     * @return Reconstructed left state at i+1/2
     */
    template<typename Real>
    KOKKOS_INLINE_FUNCTION
    static Real reconstruct_left(Real U_center, Real U_left, Real U_right) {
        Real delta_L = U_center - U_left;
        Real delta_R = U_right - U_center;
        Real limited_slope = Limiter<Real>::limit(delta_L, delta_R);
        return U_center + Real(0.5) * limited_slope;
    }

    /**
     * @brief Reconstruct right interface value for cell i+1
     *
     * For interface at i+1/2 (left side of cell i+1):
     * U_R = U[i+1] - 0.5 * limiter(delta_L, delta_R)
     *
     * @param U_center Cell value at i+1
     * @param U_left Cell value at i (previous cell)
     * @param U_right Cell value at i+2
     * @return Reconstructed right state at i+1/2
     */
    template<typename Real>
    KOKKOS_INLINE_FUNCTION
    static Real reconstruct_right(Real U_center, Real U_left, Real U_right) {
        Real delta_L = U_center - U_left;
        Real delta_R = U_right - U_center;
        Real limited_slope = Limiter<Real>::limit(delta_L, delta_R);
        return U_center - Real(0.5) * limited_slope;
    }

    /**
     * @brief Reconstruct primitive variables at interface
     *
     * This is the main interface used by finite volume schemes.
     * For each primitive variable, reconstruct left and right states
     * at the interface using MUSCL with limiters.
     *
     * @param UL_left Primitive at i-1 (left of left cell)
     * @param UL_center Primitive at i (left cell)
     * @param UR_center Primitive at i+1 (right cell)
     * @param UR_right Primitive at i+2 (right of right cell)
     * @param qL_reconstructed Output: left state at interface
     * @param qR_reconstructed Output: right state at interface
     *
     * Usage in FV scheme:
     *   qL = MUSCL::reconstruct_left(q[i-1], q[i], q[i+1])
     *   qR = MUSCL::reconstruct_right(q[i], q[i+1], q[i+2])
     *   flux = compute_flux(qL, qR)
     */
    template<typename Primitive>
    KOKKOS_INLINE_FUNCTION
    static void reconstruct_interface(
        const Primitive& UL_left,
        const Primitive& UL_center,
        const Primitive& UR_center,
        const Primitive& UR_right,
        Primitive& qL_reconstructed,
        Primitive& qR_reconstructed)
    {
        using Real = decltype(Primitive::rho);

        // Reconstruct left state at interface (from UL_center)
        qL_reconstructed.rho = reconstruct_left(UL_center.rho, UL_left.rho, UR_center.rho);
        qL_reconstructed.u   = reconstruct_left(UL_center.u, UL_left.u, UR_center.u);
        qL_reconstructed.v   = reconstruct_left(UL_center.v, UL_left.v, UR_center.v);
        qL_reconstructed.p   = reconstruct_left(UL_center.p, UL_left.p, UR_center.p);

        // Reconstruct right state at interface (from UR_center)
        qR_reconstructed.rho = reconstruct_right(UR_center.rho, UL_center.rho, UR_right.rho);
        qR_reconstructed.u   = reconstruct_right(UR_center.u, UL_center.u, UR_right.u);
        qR_reconstructed.v   = reconstruct_right(UR_center.v, UL_center.v, UR_right.v);
        qR_reconstructed.p   = reconstruct_right(UR_center.p, UL_center.p, UR_right.p);
    }

    /**
     * @brief Characteristic-wise reconstruction (more accurate but expensive)
     *
     * Reconstructs in characteristic variables instead of primitive variables.
     * This provides better shock resolution but requires eigendecomposition.
     *
     * NOTE: This is an advanced feature - for most cases, primitive variable
     * reconstruction (reconstruct_interface) is sufficient and recommended.
     */
    template<typename System, typename Primitive>
    KOKKOS_INLINE_FUNCTION
    static void reconstruct_characteristic(
        const Primitive& UL_left,
        const Primitive& UL_center,
        const Primitive& UR_center,
        const Primitive& UR_right,
        Primitive& qL_reconstructed,
        Primitive& qR_reconstructed,
        typename System::RealType gamma)
    {
        // For characteristic-wise reconstruction, we would:
        // 1. Convert primitives to characteristic variables
        // 2. Apply limiter in characteristic space
        // 3. Convert back to primitive variables
        //
        // This requires the left and right eigenvectors of the flux Jacobian.
        // For simplicity, we fall back to primitive reconstruction here.
        //
        // TODO: Implement full characteristic-wise reconstruction for
        //       improved accuracy on strong shocks.

        reconstruct_interface(UL_left, UL_center, UR_center, UR_right,
                             qL_reconstructed, qR_reconstructed);
    }
};

// ============================================================================
// SLOPE LIMITERS
// ============================================================================

/**
 * @brief Minmod limiter - most dissipative but TVD
 */
template<typename Real>
struct MinmodLimiter {
    KOKKOS_INLINE_FUNCTION
    static Real limit(Real a, Real b) {
        // minmod(a,b) = 0.5 * (sign(a) + sign(b)) * min(|a|, |b|)
        // Returns 0 if a and b have different signs
        Real eps = Real(1e-12);
        if (Kokkos::fabs(a) < eps) return Real(0);
        if (Kokkos::fabs(b) < eps) return Real(0);

        Real sa = (a > Real(0)) ? Real(1) : Real(-1);
        Real sb = (b > Real(0)) ? Real(1) : Real(-1);

        if (sa != sb) return Real(0);  // Different signs

        return (sa + sb) * Real(0.5) * Kokkos::min(Kokkos::fabs(a), Kokkos::fabs(b));
    }
};

/**
 * @brief MC (Monotonized Central) limiter - less dissipative
 */
template<typename Real>
struct MCLimiter {
    KOKKOS_INLINE_FUNCTION
    static Real limit(Real a, Real b) {
        // MC limiter: minmod(2*a, 2*b, (a+b)/2)
        Real eps = Real(1e-12);
        if (Kokkos::fabs(a) < eps) return Real(0);
        if (Kokkos::fabs(b) < eps) return Real(0);

        Real two_a = Real(2) * a;
        Real two_b = Real(2) * b;
        Real ab_sum_2 = (a + b) * Real(0.5);

        // Use minmod for all pairs
        Real r1 = MinmodLimiter<Real>::limit(two_a, two_b);
        Real r2 = MinmodLimiter<Real>::limit(r1, ab_sum_2);

        return r2;
    }
};

/**
 * @brief Superbee limiter - least dissipative
 */
template<typename Real>
struct SuperbeeLimiter {
    KOKKOS_INLINE_FUNCTION
    static Real limit(Real a, Real b) {
        // Superbee limiter implementation
        Real eps = Real(1e-12);
        if (Kokkos::fabs(a) < eps) return Real(0);
        if (Kokkos::fabs(b) < eps) return Real(0);

        Real sa = (a > Real(0)) ? Real(1) : Real(-1);
        Real sb = (b > Real(0)) ? Real(1) : Real(-1);

        if (sa != sb) return Real(0);

        Real abs_a = Kokkos::fabs(a);
        Real abs_b = Kokkos::fabs(b);
        Real min_abs = Kokkos::min(abs_a, abs_b);
        Real max_abs = Kokkos::max(abs_a, abs_b);

        // Superbee formula
        Real r1 = Kokkos::max(min_abs, Real(2) * max_abs);
        Real r2 = Kokkos::max(Real(2) * min_abs, max_abs);

        return sa * Kokkos::min(r1, r2);
    }
};

/**
 * @brief Van Leer limiter - smooth, symmetric
 */
template<typename Real>
struct VanLeerLimiter {
    KOKKOS_INLINE_FUNCTION
    static Real limit(Real a, Real b) {
        Real eps = Real(1e-12);
        if (Kokkos::fabs(a) < eps) return Real(0);
        if (Kokkos::fabs(b) < eps) return Real(0);

        Real abs_a = Kokkos::fabs(a);
        Real abs_b = Kokkos::fabs(b);

        // Van Leer formula: (a*b + |a*b|) / (a + b + eps)
        Real numerator = a * b + Kokkos::fabs(a * b);
        Real denominator = a + b + eps;

        return numerator / denominator;
    }
};

} // namespace subsetix::fvd::reconstruction
