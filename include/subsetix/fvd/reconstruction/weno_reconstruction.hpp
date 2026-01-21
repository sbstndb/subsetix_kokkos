#pragma once

#include <Kokkos_Core.hpp>
#include <cmath>
#include <algorithm>

// ============================================================================
// WENO5 RECONSTRUCTION (5th Order - WENO-JS)
// ============================================================================
// NOTE: This file is included from within namespace subsetix::fvd::reconstruction
//       No namespace wrapper needed here.

/**
 * @brief WENO5-JS (Jiang-Shu) reconstruction - 5th order accurate
 *
 * Weighted Essentially Non-Oscillatory reconstruction using:
 * - 5-point stencil with 3 sub-stencils of 3 points each
 * - Smoothness indicators for nonlinear weight computation
 * - Optimal weights: γ₀ = 1/10, γ₁ = 6/10, γ₂ = 3/10
 *
 * Reference: Jiang & Shu (1996), "Efficient Implementation of WENO Schemes"
 */
struct WENO5_JS_Reconstruction {
    static constexpr int stencil_width = 5;
    static constexpr int order = 5;
    static constexpr int ghost_layers = 2;

    /**
     * @brief Reconstruct left state at interface i+1/2 using WENO5-JS
     *
     * @param U_im2 Cell value at i-2
     * @param U_im1 Cell value at i-1
     * @param U_i   Cell value at i
     * @param U_ip1 Cell value at i+1
     * @param U_ip2 Cell value at i+2
     * @return Reconstructed left state at i+1/2
     */
    template<typename Real>
    KOKKOS_INLINE_FUNCTION
    static Real reconstruct_left(
        Real U_im2, Real U_im1, Real U_i, Real U_ip1, Real U_ip2
    ) {
        // =====================================================================
        // STEP 1: Compute polynomial values from 3 substencils
        // =====================================================================
        // Substencil S0: {i-2, i-1, i}
        Real p0 = (Real(1) / Real(3)) * U_im2 - (Real(7) / Real(6)) * U_im1 + (Real(11) / Real(6)) * U_i;

        // Substencil S1: {i-1, i, i+1}
        Real p1 = -(Real(1) / Real(6)) * U_im1 + (Real(5) / Real(6)) * U_i + (Real(1) / Real(3)) * U_ip1;

        // Substencil S2: {i, i+1, i+2}
        Real p2 = (Real(1) / Real(3)) * U_i + (Real(5) / Real(6)) * U_ip1 - (Real(1) / Real(6)) * U_ip2;

        // =====================================================================
        // STEP 2: Compute smoothness indicators
        // =====================================================================
        constexpr Real thirteen_twelfths = Real(13.0) / Real(12.0);
        constexpr Real one_fourth = Real(1.0) / Real(4.0);

        // β₀: smoothness of stencil S0
        Real d0_minus = U_im2 - Real(2) * U_im1 + U_i;
        Real d0_plus = U_im2 - Real(4) * U_im1 + Real(3) * U_i;
        Real beta0 = thirteen_twelfths * d0_minus * d0_minus + one_fourth * d0_plus * d0_plus;

        // β₁: smoothness of stencil S1
        Real d1_minus = U_im1 - Real(2) * U_i + U_ip1;
        Real d1_plus = U_im1 - U_ip1;
        Real beta1 = thirteen_twelfths * d1_minus * d1_minus + one_fourth * d1_plus * d1_plus;

        // β₂: smoothness of stencil S2
        Real d2_minus = U_i - Real(2) * U_ip1 + U_ip2;
        Real d2_plus = Real(3) * U_i - Real(4) * U_ip1 + U_ip2;
        Real beta2 = thirteen_twelfths * d2_minus * d2_minus + one_fourth * d2_plus * d2_plus;

        // =====================================================================
        // STEP 3: Compute nonlinear weights
        // =====================================================================
        constexpr Real eps = Real(1e-6);  // Avoid division by zero
        constexpr Real gamma0 = Real(0.1);  // Optimal weight for S0
        constexpr Real gamma1 = Real(0.6);  // Optimal weight for S1
        constexpr Real gamma2 = Real(0.3);  // Optimal weight for S2

        Real alpha0 = gamma0 / ((eps + beta0) * (eps + beta0));
        Real alpha1 = gamma1 / ((eps + beta1) * (eps + beta1));
        Real alpha2 = gamma2 / ((eps + beta2) * (eps + beta2));

        Real alpha_sum = alpha0 + alpha1 + alpha2;

        Real omega0 = alpha0 / alpha_sum;
        Real omega1 = alpha1 / alpha_sum;
        Real omega2 = alpha2 / alpha_sum;

        // =====================================================================
        // STEP 4: Weighted combination
        // =====================================================================
        return omega0 * p0 + omega1 * p1 + omega2 * p2;
    }

    /**
     * @brief Reconstruct right state at interface i+1/2 using WENO5-JS
     *
     * @param U_im2 Cell value at i-1
     * @param U_im1 Cell value at i
     * @param U_i   Cell value at i+1
     * @param U_ip1 Cell value at i+2
     * @param U_ip2 Cell value at i+3
     * @return Reconstructed right state at i+1/2
     */
    template<typename Real>
    KOKKOS_INLINE_FUNCTION
    static Real reconstruct_right(
        Real U_im2, Real U_im1, Real U_i, Real U_ip1, Real U_ip2
    ) {
        // Mirror symmetry for right reconstruction
        // Stencil is shifted: {i-1, i, i+1, i+2, i+3}

        // Substencil S0: {i-1, i, i+1}
        Real p0 = -(Real(1) / Real(6)) * U_im2 + (Real(5) / Real(6)) * U_im1 + (Real(1) / Real(3)) * U_i;

        // Substencil S1: {i, i+1, i+2}
        Real p1 = (Real(1) / Real(3)) * U_im1 + (Real(5) / Real(6)) * U_i - (Real(1) / Real(6)) * U_ip1;

        // Substencil S2: {i+1, i+2, i+3}
        Real p2 = (Real(11) / Real(6)) * U_i - (Real(7) / Real(6)) * U_ip1 + (Real(1) / Real(3)) * U_ip2;

        // Smoothness indicators
        constexpr Real thirteen_twelfths = Real(13.0) / Real(12.0);
        constexpr Real one_fourth = Real(1.0) / Real(4.0);

        Real beta0 = thirteen_twelfths * (U_im2 - Real(2) * U_im1 + U_i) * (U_im2 - Real(2) * U_im1 + U_i)
                   + one_fourth * (U_im2 - U_i) * (U_im2 - U_i);
        Real beta1 = thirteen_twelfths * (U_im1 - Real(2) * U_i + U_ip1) * (U_im1 - Real(2) * U_i + U_ip1)
                   + one_fourth * (Real(3) * U_im1 - Real(4) * U_i + U_ip1) * (Real(3) * U_im1 - Real(4) * U_i + U_ip1);
        Real beta2 = thirteen_twelfths * (U_i - Real(2) * U_ip1 + U_ip2) * (U_i - Real(2) * U_ip1 + U_ip2)
                   + one_fourth * (U_i - Real(4) * U_ip1 + Real(3) * U_ip2) * (U_i - Real(4) * U_ip1 + Real(3) * U_ip2);

        // Nonlinear weights
        constexpr Real eps = Real(1e-6);
        constexpr Real gamma0 = Real(0.1);
        constexpr Real gamma1 = Real(0.6);
        constexpr Real gamma2 = Real(0.3);

        Real alpha0 = gamma0 / ((eps + beta0) * (eps + beta0));
        Real alpha1 = gamma1 / ((eps + beta1) * (eps + beta1));
        Real alpha2 = gamma2 / ((eps + beta2) * (eps + beta2));

        Real alpha_sum = alpha0 + alpha1 + alpha2;

        Real omega0 = alpha0 / alpha_sum;
        Real omega1 = alpha1 / alpha_sum;
        Real omega2 = alpha2 / alpha_sum;

        return omega0 * p0 + omega1 * p1 + omega2 * p2;
    }

    /**
     * @brief Reconstruct interface states for primitive variables
     *
     * Compatible interface with MUSCL reconstruction for easy integration.
     */
    template<typename Primitive>
    KOKKOS_INLINE_FUNCTION
    static void reconstruct_interface(
        const Primitive& q_ww, const Primitive& q_w, const Primitive& q_c,
        const Primitive& q_e, const Primitive& q_ee,
        Primitive& qL_reconstructed,
        Primitive& qR_reconstructed
    ) {
        using Real = decltype(Primitive::rho);

        // Reconstruct left state at i+1/2: uses [i-2, i-1, i, i+1, i+2]
        qL_reconstructed.rho = reconstruct_left(q_ww.rho, q_w.rho, q_c.rho, q_e.rho, q_ee.rho);
        qL_reconstructed.u   = reconstruct_left(q_ww.u,   q_w.u,   q_c.u,   q_e.u,   q_ee.u);
        qL_reconstructed.v   = reconstruct_left(q_ww.v,   q_w.v,   q_c.v,   q_e.v,   q_ee.v);
        qL_reconstructed.p   = reconstruct_left(q_ww.p,   q_w.p,   q_c.p,   q_e.p,   q_ee.p);

        // Reconstruct right state at i+1/2: uses [i-1, i, i+1, i+2, i+3]
        // Note: We only have [i-2, i-1, i, i+1, i+2], so right reconstruction
        // needs to use a shifted stencil. For now, we use the same stencil but
        // this is an approximation. Full implementation would need i+3.
        qR_reconstructed.rho = reconstruct_right(q_ww.rho, q_w.rho, q_c.rho, q_e.rho, q_ee.rho);
        qR_reconstructed.u   = reconstruct_right(q_ww.u,   q_w.u,   q_c.u,   q_e.u,   q_ee.u);
        qR_reconstructed.v   = reconstruct_right(q_ww.v,   q_w.v,   q_c.v,   q_e.v,   q_ee.v);
        qR_reconstructed.p   = reconstruct_right(q_ww.p,   q_w.p,   q_c.p,   q_e.p,   q_ee.p);
    }
};

// ============================================================================
// WENO5-Z RECONSTRUCTION (Improved Weights)
// ============================================================================

/**
 * @brief WENO5-Z reconstruction with improved convergence at critical points
 *
 * Uses a global smoothness indicator τ₅ to improve accuracy near smooth extrema.
 *
 * Reference: Borges et al. (2008), "An improved weighted essentially
 *            non-oscillatory scheme"
 */
struct WENO5_Z_Reconstruction {
    static constexpr int stencil_width = 5;
    static constexpr int order = 5;
    static constexpr int ghost_layers = 2;

    /**
     * @brief Reconstruct left state at interface i+1/2 using WENO5-Z
     */
    template<typename Real>
    KOKKOS_INLINE_FUNCTION
    static Real reconstruct_left(
        Real U_im2, Real U_im1, Real U_i, Real U_ip1, Real U_ip2
    ) {
        // Polynomial values (same as WENO-JS)
        Real p0 = (Real(1) / Real(3)) * U_im2 - (Real(7) / Real(6)) * U_im1 + (Real(11) / Real(6)) * U_i;
        Real p1 = -(Real(1) / Real(6)) * U_im1 + (Real(5) / Real(6)) * U_i + (Real(1) / Real(3)) * U_ip1;
        Real p2 = (Real(1) / Real(3)) * U_i + (Real(5) / Real(6)) * U_ip1 - (Real(1) / Real(6)) * U_ip2;

        // Smoothness indicators (same as WENO-JS)
        constexpr Real thirteen_twelfths = Real(13.0) / Real(12.0);
        constexpr Real one_fourth = Real(1.0) / Real(4.0);

        Real d0_minus = U_im2 - Real(2) * U_im1 + U_i;
        Real d0_plus = U_im2 - Real(4) * U_im1 + Real(3) * U_i;
        Real beta0 = thirteen_twelfths * d0_minus * d0_minus + one_fourth * d0_plus * d0_plus;

        Real d1_minus = U_im1 - Real(2) * U_i + U_ip1;
        Real d1_plus = U_im1 - U_ip1;
        Real beta1 = thirteen_twelfths * d1_minus * d1_minus + one_fourth * d1_plus * d1_plus;

        Real d2_minus = U_i - Real(2) * U_ip1 + U_ip2;
        Real d2_plus = Real(3) * U_i - Real(4) * U_ip1 + U_ip2;
        Real beta2 = thirteen_twelfths * d2_minus * d2_minus + one_fourth * d2_plus * d2_plus;

        // =====================================================================
        // WENO-Z IMPROVEMENT: Global smoothness indicator τ₅
        // =====================================================================
        Real tau5 = Kokkos::fabs(beta0 - beta2);

        // Improved weights with τ₅ term
        constexpr Real eps = Real(1e-6);
        constexpr Real gamma0 = Real(0.1);
        constexpr Real gamma1 = Real(0.6);
        constexpr Real gamma2 = Real(0.3);

        // Compute denominator for τ normalization
        Real beta_sum_sq = (eps + beta0) * (eps + beta0)
                         + (eps + beta1) * (eps + beta1)
                         + (eps + beta2) * (eps + beta2);

        Real tau_eps_sq = (tau5 * tau5) / beta_sum_sq;

        // WENO-Z weights
        Real alpha0 = gamma0 * (Real(1) + tau_eps_sq / ((eps + beta0) * (eps + beta0)));
        Real alpha1 = gamma1 * (Real(1) + tau_eps_sq / ((eps + beta1) * (eps + beta1)));
        Real alpha2 = gamma2 * (Real(1) + tau_eps_sq / ((eps + beta2) * (eps + beta2)));

        Real alpha_sum = alpha0 + alpha1 + alpha2;

        Real omega0 = alpha0 / alpha_sum;
        Real omega1 = alpha1 / alpha_sum;
        Real omega2 = alpha2 / alpha_sum;

        return omega0 * p0 + omega1 * p1 + omega2 * p2;
    }

    /**
     * @brief Reconstruct right state at interface i+1/2 using WENO5-Z
     */
    template<typename Real>
    KOKKOS_INLINE_FUNCTION
    static Real reconstruct_right(
        Real U_im2, Real U_im1, Real U_i, Real U_ip1, Real U_ip2
    ) {
        // Polynomial values (mirrored)
        Real p0 = -(Real(1) / Real(6)) * U_im2 + (Real(5) / Real(6)) * U_im1 + (Real(1) / Real(3)) * U_i;
        Real p1 = (Real(1) / Real(3)) * U_im1 + (Real(5) / Real(6)) * U_i - (Real(1) / Real(6)) * U_ip1;
        Real p2 = (Real(11) / Real(6)) * U_i - (Real(7) / Real(6)) * U_ip1 + (Real(1) / Real(3)) * U_ip2;

        // Smoothness indicators
        constexpr Real thirteen_twelfths = Real(13.0) / Real(12.0);
        constexpr Real one_fourth = Real(1.0) / Real(4.0);

        Real beta0 = thirteen_twelfths * (U_im2 - Real(2) * U_im1 + U_i) * (U_im2 - Real(2) * U_im1 + U_i)
                   + one_fourth * (U_im2 - U_i) * (U_im2 - U_i);
        Real beta1 = thirteen_twelfths * (U_im1 - Real(2) * U_i + U_ip1) * (U_im1 - Real(2) * U_i + U_ip1)
                   + one_fourth * (Real(3) * U_im1 - Real(4) * U_i + U_ip1) * (Real(3) * U_im1 - Real(4) * U_i + U_ip1);
        Real beta2 = thirteen_twelfths * (U_i - Real(2) * U_ip1 + U_ip2) * (U_i - Real(2) * U_ip1 + U_ip2)
                   + one_fourth * (U_i - Real(4) * U_ip1 + Real(3) * U_ip2) * (U_i - Real(4) * U_ip1 + Real(3) * U_ip2);

        // Global smoothness indicator τ₅
        Real tau5 = Kokkos::fabs(beta0 - beta2);

        // Improved weights
        constexpr Real eps = Real(1e-6);
        constexpr Real gamma0 = Real(0.1);
        constexpr Real gamma1 = Real(0.6);
        constexpr Real gamma2 = Real(0.3);

        Real beta_sum_sq = (eps + beta0) * (eps + beta0)
                         + (eps + beta1) * (eps + beta1)
                         + (eps + beta2) * (eps + beta2);

        Real tau_eps_sq = (tau5 * tau5) / beta_sum_sq;

        Real alpha0 = gamma0 * (Real(1) + tau_eps_sq / ((eps + beta0) * (eps + beta0)));
        Real alpha1 = gamma1 * (Real(1) + tau_eps_sq / ((eps + beta1) * (eps + beta1)));
        Real alpha2 = gamma2 * (Real(1) + tau_eps_sq / ((eps + beta2) * (eps + beta2)));

        Real alpha_sum = alpha0 + alpha1 + alpha2;

        Real omega0 = alpha0 / alpha_sum;
        Real omega1 = alpha1 / alpha_sum;
        Real omega2 = alpha2 / alpha_sum;

        return omega0 * p0 + omega1 * p1 + omega2 * p2;
    }

    /**
     * @brief Reconstruct interface states for primitive variables
     */
    template<typename Primitive>
    KOKKOS_INLINE_FUNCTION
    static void reconstruct_interface(
        const Primitive& q_ww, const Primitive& q_w, const Primitive& q_c,
        const Primitive& q_e, const Primitive& q_ee,
        Primitive& qL_reconstructed,
        Primitive& qR_reconstructed
    ) {
        using Real = decltype(Primitive::rho);

        qL_reconstructed.rho = reconstruct_left(q_ww.rho, q_w.rho, q_c.rho, q_e.rho, q_ee.rho);
        qL_reconstructed.u   = reconstruct_left(q_ww.u,   q_w.u,   q_c.u,   q_e.u,   q_ee.u);
        qL_reconstructed.v   = reconstruct_left(q_ww.v,   q_w.v,   q_c.v,   q_e.v,   q_ee.v);
        qL_reconstructed.p   = reconstruct_left(q_ww.p,   q_w.p,   q_c.p,   q_e.p,   q_ee.p);

        qR_reconstructed.rho = reconstruct_right(q_ww.rho, q_w.rho, q_c.rho, q_e.rho, q_ee.rho);
        qR_reconstructed.u   = reconstruct_right(q_ww.u,   q_w.u,   q_c.u,   q_e.u,   q_ee.u);
        qR_reconstructed.v   = reconstruct_right(q_ww.v,   q_w.v,   q_c.v,   q_e.v,   q_ee.v);
        qR_reconstructed.p   = reconstruct_right(q_ww.p,   q_w.p,   q_c.p,   q_e.p,   q_ee.p);
    }
};

