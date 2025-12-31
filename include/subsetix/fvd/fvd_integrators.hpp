#pragma once

/**
 * @file fvd_integrators.hpp
 *
 * @brief Complete integration header for FVD high-level API
 *
 * This header includes all components for time-dependent BCs,
 * AMR with coarsening, and high-order time integration.
 *
 * Usage:
 *   #include <subsetix/fvd/fvd_integrators.hpp>
 *
 *   using namespace subsetix::fvd;
 *   using namespace subsetix::fvd::solver;
 *
 *   auto solver = EulerSolverMUSCL_RK3::builder(nx, ny)
 *       .with_domain(0.0, 2.0, 0.0, 0.8)
 *       .with_initial_condition(mach2_cylinder)
 *       .build();
 */

// ============================================================================
// SYSTEM AND CONCEPTS
// ============================================================================
#include "system/euler2d.hpp"
#include "system/concepts_v2.hpp"

// ============================================================================
// TIME INTEGRATION
// ============================================================================
#include "time/time_integrators.hpp"

// ============================================================================
// SPATIAL DISCRETIZATION
// ============================================================================
#include "reconstruction/reconstruction.hpp"
#include "flux/flux_schemes.hpp"

// ============================================================================
// BOUNDARY CONDITIONS
// ============================================================================
#include "solver/boundary_generic.hpp"
#include "boundary/time_dependent_bc.hpp"

// ============================================================================
// AMR WITH COARSENING
// ============================================================================
#include "amr/refinement_criteria.hpp"

// ============================================================================
// OBSERVERS
// ============================================================================
#include "solver/observer.hpp"

// ============================================================================
// MAIN SOLVER
// ============================================================================
#include "solver/adaptive_solver.hpp"

// ============================================================================
// SOLVER ALIASES WITH TIME INTEGRATORS
// ============================================================================
#include "solver/solver_aliases_v2.hpp"

namespace subsetix::fvd {

// ============================================================================
// INTEGRATED SOLVER BUILDER
// ============================================================================

/**
 * @brief Convenience namespace for solver construction
 */
namespace builder {

/**
 * @brief Create a solver with specified components
 *
 * Example:
 *   auto solver = builder::create<Euler2D<float>,
 *                                 MUSCLReconstruction<float>,
 *                                 HLLCFlux,
 *                                 Kutta3<float>>(nx, ny);
 */
template<
    typename System,
    typename Reconstruction = reconstruction::NoReconstruction,
    template<typename> class FluxScheme = flux::RusanovFlux,
    typename TimeIntegrator = time::ForwardEuler<typename System::RealType>
>
auto create(int nx, int ny) {
    return solver::AdaptiveSolver<System, Reconstruction, FluxScheme, TimeIntegrator>::builder(nx, ny);
}

} // namespace builder

// ============================================================================
// CONVENIENCE FUNCTIONS FOR COMMON SETUP
// ============================================================================

/**
 * @brief Setup standard time-dependent inlet BC
 *
 * Creates a sinusoidal inlet with given parameters.
 */
template<typename System>
boundary::TimeDependentBC<typename System::RealType>
standard_inlet(typename System::RealType rho0,
              typename System::RealType u0,
              typename System::RealType frequency = typename System::RealType(2 * 3.14159))
{
    return boundary::sinusoidal_inlet<System>(rho0, u0, frequency);
}

/**
 * @brief Setup standard AMR configuration
 *
 * Creates a refinement config with shock sensor and vorticity criteria.
 */
template<typename System>
amr::RefinementManager<System> standard_amr()
{
    using Real = typename System::RealType;
    amr::RefinementManager<System> mgr;

    // Add shock sensor criterion
    mgr.add_shock_sensor_criterion(
        amr::ShockSensorCriterion<System>::Ducros,
        Real(0.8)
    );

    // Add vorticity criterion
    mgr.add_vorticity_criterion(Real(1.0));

    // Use OR logic (refine if either condition is met)
    mgr.set_logic_op(amr::CompositeCriterion<System, 8>::Or);

    // Set level limits
    mgr.set_level_limits(0, 5);

    // Set remesh frequency
    mgr.set_remesh_frequency(100);

    // Enable coarsening
    mgr.set_coarsening(true);

    return mgr;
}

/**
 * @brief Setup adaptive time stepping
 *
 * Returns a standard time step controller config.
 */
template<typename Real = float>
typename time::TimeStepController<Real>::Config standard_adaptive_dt()
{
    typename time::TimeStepController<Real>::Config cfg;
    cfg.cfl_target = Real(0.8);
    cfg.cfl_max = Real(1.0);
    cfg.dt_max = Real(1e-2);
    cfg.dt_min = Real(1e-6);
    cfg.growth_factor = Real(1.2);
    cfg.shrink_factor = Real(0.8);
    return cfg;
}

} // namespace subsetix::fvd
