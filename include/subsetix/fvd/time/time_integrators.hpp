#pragma once

#include <Kokkos_Core.hpp>
#include <concepts>
#include <type_traits>
#include "../system/concepts_v2.hpp"

namespace subsetix::fvd::time {

// ============================================================================
// TIME INTEGRATOR CONCEPT
// ============================================================================

/**
 * @brief Concept for time integrators
 *
 * A time integrator must provide:
 * - order: Order of accuracy
 * - stages: Number of RHS evaluations per step
 * - a, b, c: Butcher tableau coefficients
 */
template<typename T>
concept TimeIntegrator = requires {
    { T::order } -> std::convertible_to<int>;
    { T::stages } -> std::convertible_to<int>;
    typename T::RealType;
};

// ============================================================================
// FORWARD EULER (1st order)
// ============================================================================

/**
 * @brief Forward Euler (1st order, 1 stage)
 *
 * Simplest explicit method: U_{n+1} = U_n + dt * f(t_n, U_n)
 */
template<typename Real = float>
struct ForwardEuler {
    static constexpr int order = 1;
    static constexpr int stages = 1;
    using RealType = Real;

    // Butcher tableau (simplified for 1-stage)
    static constexpr Real a[1][1] = {{0.0}};
    static constexpr Real b[1] = {1.0};
    static constexpr Real c[1] = {0.0};

    static constexpr const char* name = "Forward Euler";
};

// ============================================================================
// HEUN'S METHOD (RK2)
// ============================================================================

/**
 * @brief Heun's method (2nd order, 2 stages)
 *
 * Also known as improved Euler or explicit trapezoidal method.
 * Predictor-corrector: U* = U_n + dt*f(t_n, U_n)
 *                      U_{n+1} = U_n + dt/2*(f(t_n, U_n) + f(t_{n+1}, U*))
 */
template<typename Real = float>
struct Heun2 {
    static constexpr int order = 2;
    static constexpr int stages = 2;
    using RealType = Real;

    // Butcher tableau:
    // 0   |
    // 1   | 1
    // ----|-------------
    //     | 1/2  1/2
    static constexpr Real a[2][2] = {
        {0.0, 0.0},
        {1.0, 0.0}
    };
    static constexpr Real b[2] = {0.5, 0.5};
    static constexpr Real c[2] = {0.0, 1.0};

    static constexpr const char* name = "Heun (RK2)";
};

// ============================================================================
// KUTTA'S THIRD ORDER (RK3)
// ============================================================================

/**
 * @brief Kutta's third-order method (RK3)
 *
 * Classic 3-stage, 3rd-order Runge-Kutta method.
 * Good balance between accuracy and computational cost.
 */
template<typename Real = float>
struct Kutta3 {
    static constexpr int order = 3;
    static constexpr int stages = 3;
    using RealType = Real;

    // Butcher tableau:
    // 0     |
    // 1/2   | 1/2
    // 1     | -1   2
    // ------|-----------------
    //       | 1/6  2/3  1/6
    static constexpr Real a[3][3] = {
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0},
        {-1.0, 2.0, 0.0}
    };
    static constexpr Real b[3] = {Real(1.0/6.0), Real(2.0/3.0), Real(1.0/6.0)};
    static constexpr Real c[3] = {0.0, 0.5, 1.0};

    static constexpr const char* name = "Kutta (RK3)";
};

// ============================================================================
// CLASSIC RK4
// ============================================================================

/**
 * @brief Classic 4th-order Runge-Kutta (RK4)
 *
 * The most widely used RK method.
 * k1 = f(t_n, U_n)
 * k2 = f(t_n + dt/2, U_n + dt*k1/2)
 * k3 = f(t_n + dt/2, U_n + dt*k2/2)
 * k4 = f(t_n + dt, U_n + dt*k3)
 * U_{n+1} = U_n + dt/6*(k1 + 2*k2 + 2*k3 + k4)
 */
template<typename Real = float>
struct ClassicRK4 {
    static constexpr int order = 4;
    static constexpr int stages = 4;
    using RealType = Real;

    // Butcher tableau:
    // 0     |
    // 1/2   | 1/2
    // 1/2   | 0    1/2
    // 1     | 0    0    1
    // ------|------------------------
    //       | 1/6  1/3  1/3  1/6
    static constexpr Real a[4][4] = {
        {0.0, 0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0, 0.0},
        {0.0, 0.5, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0}
    };
    static constexpr Real b[4] = {
        Real(1.0/6.0), Real(1.0/3.0), Real(1.0/3.0), Real(1.0/6.0)
    };
    static constexpr Real c[4] = {0.0, 0.5, 0.5, 1.0};

    static constexpr const char* name = "Classic RK4";
};

// ============================================================================
// RALSTON'S THIRD ORDER (RK3)
// ============================================================================

/**
 * @brief Ralston's third-order method (RK3)
 *
 * Alternative 3rd-order method with smaller error constant than Kutta3.
 */
template<typename Real = float>
struct Ralston3 {
    static constexpr int order = 3;
    static constexpr int stages = 3;
    using RealType = Real;

    // Butcher tableau:
    // 0       |
    // 1/2     | 1/2
    // 3/4     | 0    3/4
    // --------|----------------------
    //         | 2/9  1/3  4/9
    static constexpr Real a[3][3] = {
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0},
        {0.0, 0.75, 0.0}
    };
    static constexpr Real b[3] = {Real(2.0/9.0), Real(1.0/3.0), Real(4.0/9.0)};
    static constexpr Real c[3] = {0.0, 0.5, 0.75};

    static constexpr const char* name = "Ralston (RK3)";
};

// ============================================================================
// SSPRK3 (Strong Stability Preserving RK3)
// ============================================================================

/**
 * @brief SSPRK3 - Third-order Strong Stability Preserving Runge-Kutta
 *
 * Optimized for hyperbolic conservation laws.
 * Commonly used in CFD for its good stability properties.
 */
template<typename Real = float>
struct SSPRK3 {
    static constexpr int order = 3;
    static constexpr int stages = 3;
    using RealType = Real;

    // Butcher tableau (SSP form):
    // 0   |
    // 1   | 1
    // 1/2 | 1/4  1/4
    // ----|-----------------
    //     | 1/6  1/6  2/3
    static constexpr Real a[3][3] = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.25, 0.25, 0.0}
    };
    static constexpr Real b[3] = {Real(1.0/6.0), Real(1.0/6.0), Real(2.0/3.0)};
    static constexpr Real c[3] = {0.0, 1.0, 0.5};

    static constexpr const char* name = "SSPRK3";
};

// ============================================================================
// TIME STEP CONTROLLER
// ============================================================================

/**
 * @brief Controls adaptive time stepping
 *
 * Adjusts dt based on:
 * - CFL condition
 * - Adaptive limits (growth/shrink factors)
 * - Custom user criteria
 */
template<typename Real = float>
class TimeStepController {
public:
    struct Config {
        // CFL-based control
        Real cfl_target = Real(0.8);
        Real cfl_max = Real(1.0);
        Real cfl_min = Real(0.1);

        // Adaptive limits
        Real dt_max = Real(1e-2);
        Real dt_min = Real(1e-6);
        Real growth_factor = Real(1.2);   // Max dt increase per step
        Real shrink_factor = Real(0.8);   // Max dt decrease per step

        // Adjustment frequency
        int adjust_interval = 1;  // Check every N steps
    };

    KOKKOS_INLINE_FUNCTION
    TimeStepController() = default;

    /**
     * @brief Compute new dt based on current CFL
     */
    KOKKOS_INLINE_FUNCTION
    Real compute_dt(Real current_dt, Real current_cfl, const Config& cfg) const {
        Real new_dt = current_dt;

        if (current_cfl > cfg.cfl_max) {
            // Too aggressive: shrink dt
            new_dt = current_dt * cfg.shrink_factor;
        } else if (current_cfl < cfg.cfl_target * Real(0.5)) {
            // Too conservative: grow dt
            new_dt = Kokkos::min(current_dt * cfg.growth_factor, cfg.dt_max);
        }

        // Enforce limits
        new_dt = Kokkos::max(new_dt, cfg.dt_min);
        new_dt = Kokkos::min(new_dt, cfg.dt_max);

        return new_dt;
    }

    /**
     * @brief Check if dt should be changed
     */
    KOKKOS_INLINE_FUNCTION
    bool should_adapt(Real current_cfl, const Config& cfg) const {
        return (current_cfl > cfg.cfl_max) ||
               (current_cfl < cfg.cfl_target * Real(0.5));
    }

    /**
     * @brief Compute dt from CFL condition
     *
     * dt = CFL * dx / max_wave_speed
     */
    KOKKOS_INLINE_FUNCTION
    Real compute_dt_from_cfl(Real cfl, Real dx, Real max_wave_speed,
                             Real dt_min, Real dt_max) const
    {
        Real dt = cfl * dx / (max_wave_speed + Real(1e-10));
        return Kokkos::max(dt_min, Kokkos::min(dt, dt_max));
    }
};

// ============================================================================
// GENERIC RUNGE-KUTTA INTEGRATOR
// ============================================================================

/**
 * @brief Generic Runge-Kutta integrator implementation
 *
 * Works with any TimeIntegrator policy.
 * Device-side implementation using Kokkos parallel loops.
 *
 * @note This is the core RK implementation that handles:
 * - Stage computation
 * - RHS evaluation at each stage
 * - Final solution combination
 */
namespace detail {

/**
 * @brief Compute stage solution for RK methods
 *
 * For stage s, computes:
 * U_stage = U_n + dt * sum_{i=0}^{s-1} a[s][i] * k_i
 *
 * where k_i are the RHS values from previous stages.
 */
template<typename System, typename Integrator>
KOKKOS_FUNCTION
void compute_stage_solution(
    const Kokkos::View<typename System::Conserved**>& U,
    typename System::RealType dt,
    int stage,
    const Kokkos::View<typename System::Conserved***>& stage_rhs,
    const Kokkos::View<typename System::Conserved**>& stage_solution)
{
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;

    const int nx = U.extent(1);
    const int ny = U.extent(0);

    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx});

    Kokkos::parallel_for(
        "rk_compute_stage_" + std::to_string(stage),
        policy,
        KOKKOS_LAMBDA(int j, int i) {
            Conserved sum{0, 0, 0, 0};

            // Sum previous RHS contributions
            for (int prev = 0; prev < stage; ++prev) {
                Real coeff = Integrator::a[stage][prev];
                const Conserved& k = stage_rhs(prev, j, i);
                sum.rho += coeff * k.rho;
                sum.rhou += coeff * k.rhou;
                sum.rhov += coeff * k.rhov;
                sum.E += coeff * k.E;
            }

            stage_solution(j, i).rho = U(j, i).rho + dt * sum.rho;
            stage_solution(j, i).rhou = U(j, i).rhou + dt * sum.rhou;
            stage_solution(j, i).rhov = U(j, i).rhov + dt * sum.rhov;
            stage_solution(j, i).E = U(j, i).E + dt * sum.E;
        }
    );
}

/**
 * @brief Combine stages for final solution
 *
 * U_{n+1} = U_n + dt * sum_{i=0}^{stages-1} b[i] * k_i
 */
template<typename System, typename Integrator>
KOKKOS_FUNCTION
void combine_stages(
    const Kokkos::View<typename System::Conserved**>& U,
    typename System::RealType dt,
    const Kokkos::View<typename System::Conserved***>& stage_rhs)
{
    using Real = typename System::RealType;
    using Conserved = typename System::Conserved;

    const int nx = U.extent(1);
    const int ny = U.extent(0);
    constexpr int stages = Integrator::stages;

    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx});

    Kokkos::parallel_for(
        "rk_combine_stages",
        policy,
        KOKKOS_LAMBDA(int j, int i) {
            Conserved sum{0, 0, 0, 0};

            for (int s = 0; s < stages; ++s) {
                Real coeff = Integrator::b[s];
                const Conserved& k = stage_rhs(s, j, i);
                sum.rho += coeff * k.rho;
                sum.rhou += coeff * k.rhou;
                sum.rhov += coeff * k.rhov;
                sum.E += coeff * k.E;
            }

            U(j, i).rho += dt * sum.rho;
            U(j, i).rhou += dt * sum.rhou;
            U(j, i).rhov += dt * sum.rhov;
            U(j, i).E += dt * sum.E;
        }
    );
}

/**
 * @brief Copy U to stage_solution (for stage 0 of some methods)
 */
template<typename System>
KOKKOS_FUNCTION
void copy_solution(
    const Kokkos::View<typename System::Conserved**>& U,
    const Kokkos::View<typename System::Conserved**>& stage_solution)
{
    const int nx = U.extent(1);
    const int ny = U.extent(0);

    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx});

    Kokkos::parallel_for(
        "copy_solution",
        policy,
        KOKKOS_LAMBDA(int j, int i) {
            stage_solution(j, i) = U(j, i);
        }
    );
}

} // namespace detail

// ============================================================================
// RUNGE-KUTTA STEPPER FUNCTION
// ============================================================================

/**
 * @brief Perform one Runge-Kutta time step
 *
 * This function orchestrates the complete RK step:
 * 1. For each stage s = 0 to stages-1:
 *    a. Compute intermediate solution: U_s = U_n + dt * sum(a[s][i] * k_i)
 *    b. Evaluate RHS: k_s = f(t + c[s]*dt, U_s)
 * 2. Combine stages: U_{n+1} = U_n + dt * sum(b[i] * k_i)
 *
 * @tparam System The PDE system (e.g., Euler2D<float>)
 * @tparam Integrator The time integrator policy (e.g., Kutta3<float>)
 *
 * @param U Solution view (modified in place)
 * @param dt Time step size
 * @param t Current time
 * @param rhs RHS function: void(const View<Conserved**>&, View<Conserved**>&, Real)
 * @param stage_storage Pre-allocated storage for stage RHS
 * @param stage_solution Pre-allocated workspace for intermediate solution
 *
 * @note The RHS function must compute the right-hand side:
 *       dU/dt = -div(F) + S
 *
 * Usage example:
 * @code
 * auto rhs = [&](const auto& U, auto& rhs_out, Real t) {
 *     solver.compute_rhs(U, rhs_out, t);
 * };
 *
 * time::rk_step<System, Kutta3<float>>(U, dt, t, rhs, stage_rhs, stage_work);
 * @endcode
 */
template<
    FiniteVolumeSystem System,
    TimeIntegrator Integrator
>
KOKKOS_FUNCTION
void rk_step(
    const Kokkos::View<typename System::Conserved**>& U,
    typename System::RealType dt,
    typename System::RealType t,
    auto&& rhs,
    const Kokkos::View<typename System::Conserved***>& stage_storage,
    const Kokkos::View<typename System::Conserved**>& stage_solution)
{
    constexpr int stages = Integrator::stages;
    using Real = typename System::RealType;

    // Stage loop
    for (int s = 0; s < stages; ++s) {
        Real stage_time = t + Integrator::c[s] * dt;

        if (s == 0) {
            // For first stage, compute RHS directly from U
            rhs(U, stage_storage, s, stage_time);
        } else {
            // Compute intermediate solution for this stage
            detail::compute_stage_solution<System, Integrator>(
                U, dt, s, stage_storage, stage_solution
            );

            // Evaluate RHS at intermediate solution
            rhs(stage_solution, stage_storage, s, stage_time);
        }
    }

    // Combine all stages for final solution
    detail::combine_stages<System, Integrator>(U, dt, stage_storage);
}

/**
 * @brief Specialization for Forward Euler (single stage)
 *
 * Simpler and faster than the general RK implementation.
 */
template<FiniteVolumeSystem System>
KOKKOS_FUNCTION
void euler_step(
    const Kokkos::View<typename System::Conserved**>& U,
    typename System::RealType dt,
    typename System::RealType t,
    auto&& rhs,
    const Kokkos::View<typename System::Conserved**>& rhs_work)
{
    // Compute RHS
    rhs(U, rhs_work, 0, t);

    // Update: U_new = U_old + dt * RHS
    const int nx = U.extent(1);
    const int ny = U.extent(0);

    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx});

    Kokkos::parallel_for(
        "euler_update",
        policy,
        KOKKOS_LAMBDA(int j, int i) {
            U(j, i).rho += dt * rhs_work(j, i).rho;
            U(j, i).rhou += dt * rhs_work(j, i).rhou;
            U(j, i).rhov += dt * rhs_work(j, i).rhov;
            U(j, i).E += dt * rhs_work(j, i).E;
        }
    );
}

} // namespace subsetix::fvd::time
