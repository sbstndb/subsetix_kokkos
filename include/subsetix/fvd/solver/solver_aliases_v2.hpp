#pragma once

#include <Kokkos_Core.hpp>
#include "../system/euler2d.hpp"
#include "../reconstruction/reconstruction.hpp"
#include "../flux/flux_schemes.hpp"
#include "../time/time_integrators.hpp"
#include "adaptive_solver.hpp"

namespace subsetix::fvd::solver {

// ============================================================================
// FORWARD DECLARATION OF UPDATED SOLVER
// ============================================================================

/**
 * @brief Updated AdaptiveSolver with time integrator support
 *
 * This is the new signature that includes:
 * - System: The PDE system (Euler2D, etc.)
 * - Reconstruction: Spatial reconstruction scheme
 * - FluxScheme: Numerical flux scheme
 * - TimeIntegrator: Time integration scheme (ForwardEuler, RK2, RK3, RK4)
 */
template<
    FiniteVolumeSystem System,
    typename Reconstruction = reconstruction::NoReconstruction,
    template<typename> class FluxScheme = flux::RusanovFlux,
    typename TimeIntegrator = time::ForwardEuler<typename System::RealType>
>
class AdaptiveSolver;

// ============================================================================
// TYPE ALIASES FOR EULER 2D SYSTEM (float)
// ============================================================================

using Real = float;
using Euler = Euler2D<Real>;

// ===========================================================================
// BASIC SOLVERS (Forward Euler)
// ===========================================================================

/**
 * @brief Base Euler solver (1st order space, 1st order time)
 * Fastest but least accurate
 */
using EulerSolver = AdaptiveSolver<
    Euler,
    reconstruction::NoReconstruction,
    flux::RusanovFlux,
    time::ForwardEuler<Real>
>;

// ===========================================================================
// RK2 SOLVERS (2nd order time)
// ===========================================================================

/**
 * @brief Heun's method (2nd order time, 1st order space)
 * Good balance of accuracy and speed
 */
using EulerSolverRK2 = AdaptiveSolver<
    Euler,
    reconstruction::NoReconstruction,
    flux::RusanovFlux,
    time::Heun2<Real>
>;

// ===========================================================================
// RK3 SOLVERS (3rd order time)
// ===========================================================================

/**
 * @brief Kutta's RK3 (3rd order time, 1st order space)
 */
using EulerSolverRK3 = AdaptiveSolver<
    Euler,
    reconstruction::NoReconstruction,
    flux::RusanovFlux,
    time::Kutta3<Real>
>;

// ===========================================================================
// RK4 SOLVERS (4th order time)
// ===========================================================================

/**
 * @brief Classic RK4 (4th order time, 1st order space)
 * Highest accuracy but most expensive
 */
using EulerSolverRK4 = AdaptiveSolver<
    Euler,
    reconstruction::NoReconstruction,
    flux::RusanovFlux,
    time::ClassicRK4<Real>
>;

// ============================================================================
// CONVENIENCE TYPEDEFS
// ============================================================================

// Default solver (good balance)
using DefaultSolver = EulerSolverRK3;

// Fast solver (for testing)
using FastSolver = EulerSolver;

// High-order solver (for smooth flows)
using HighOrderSolver = EulerSolverRK4;

} // namespace subsetix::fvd::solver
