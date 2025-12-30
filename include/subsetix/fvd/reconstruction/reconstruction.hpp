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
 * Reconstructs interface values using limited slopes.
 *
 * NOTE: Full implementation in production, this is a stub.
 */
template<template<typename> class Limiter>
struct MUSCL_Reconstruction {
    // Empty policy for now
    MUSCL_Reconstruction() = default;
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
