#pragma once

#include <Kokkos_Core.hpp>
#include "../system/euler2d.hpp"
#include "../reconstruction/reconstruction.hpp"
#include "../flux/flux_schemes.hpp"
#include "../time/time_integrators.hpp"
#include "adaptive_solver.hpp"

namespace subsetix::fvd::solver {

// ============================================================================
// BRING ADAPTIVESOLVER INTO SOLVER NAMESPACE
// ============================================================================

/**
 * @brief Import AdaptiveSolver from parent namespace
 *
 * The actual AdaptiveSolver class is defined in subsetix::fvd namespace.
 * We bring it into the solver namespace here for the aliases below.
 */
using subsetix::fvd::AdaptiveSolver;

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
