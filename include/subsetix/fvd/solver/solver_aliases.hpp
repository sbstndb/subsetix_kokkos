#pragma once

#include <Kokkos_Core.hpp>
#include "adaptive_solver.hpp"
#include "../system/euler2d.hpp"
#include "../flux/flux_schemes.hpp"
#include "../reconstruction/reconstruction.hpp"

namespace subsetix::fvd {

// ============================================================================
// SOLVER ALIAS TEMPLATES FOR EULER2D
// ============================================================================

/**
 * @brief Solver alias library for Euler2D
 *
 * GAME CHANGER: 90% reduction in API verbosity!
 *
 * Instead of writing:
 *   AdaptiveSolver<Euler2D<>, MUSCL_Reconstruction<MinmodLimiter>, HLLCFlux>
 *
 * Just write:
 *   EulerSolver2ndHLLC<>
 *
 * All aliases are template on Real type (default float).
 * Use EulerSolver2ndHLLC<double> for double precision.
 */

// ============================================================================
// 1ST ORDER SOLVERS (No Reconstruction)
// ============================================================================

/**
 * @brief 1st order Euler solver with Rusanov flux
 *
 * Simplest, most robust. Good for debugging.
 */
template<typename Real = float>
using EulerSolver1st = AdaptiveSolver<Euler2D<Real>,
                                       reconstruction::NoReconstruction,
                                       flux::RusanovFlux>;

/**
 * @brief 1st order Euler solver with HLLC flux
 *
 * Better shock capturing than Rusanov, still 1st order.
 */
template<typename Real = float>
using EulerSolver1stHLLC = AdaptiveSolver<Euler2D<Real>,
                                          reconstruction::NoReconstruction,
                                          flux::HLLCFlux>;

/**
 * @brief 1st order Euler solver with Roe flux
 *
 * High accuracy, 1st order in space.
 */
template<typename Real = float>
using EulerSolver1stRoe = AdaptiveSolver<Euler2D<Real>,
                                         reconstruction::NoReconstruction,
                                         flux::RoeFlux>;

// ============================================================================
// 2ND ORDER SOLVERS (MUSCL Reconstruction)
// ============================================================================

/**
 * @brief 2nd order Euler solver with Rusanov flux
 *
 * MUSCL reconstruction + Minmod limiter + Rusanov flux.
 * Good balance of accuracy and robustness.
 */
template<typename Real = float,
         template<typename> class Limiter = reconstruction::MinmodLimiter>
using EulerSolver2nd = AdaptiveSolver<Euler2D<Real>,
                                       reconstruction::MUSCL_Reconstruction<Limiter>,
                                       flux::RusanovFlux>;

/**
 * @brief 2nd order Euler solver with HLLC flux
 *
 * **DEFAULT CHOICE FOR PRODUCTION**
 *
 * MUSCL reconstruction + Minmod limiter + HLLC flux.
 * Best choice for shock-capturing applications.
 */
template<typename Real = float,
         template<typename> class Limiter = reconstruction::MinmodLimiter>
using EulerSolver2ndHLLC = AdaptiveSolver<Euler2D<Real>,
                                          reconstruction::MUSCL_Reconstruction<Limiter>,
                                          flux::HLLCFlux>;

/**
 * @brief 2nd order Euler solver with Roe flux
 *
 * MUSCL reconstruction + Minmod limiter + Roe flux.
 * Highest accuracy, more expensive.
 */
template<typename Real = float,
         template<typename> class Limiter = reconstruction::MinmodLimiter>
using EulerSolver2ndRoe = AdaptiveSolver<Euler2D<Real>,
                                         reconstruction::MUSCL_Reconstruction<Limiter>,
                                         flux::RoeFlux>;

// ============================================================================
// DOUBLE PRECISION ALIASES (convenience)
// ============================================================================

/**
 * @brief Double precision 1st order solver
 */
template<typename Real = double>
using EulerSolver1st_d = EulerSolver1st<Real>;

/**
 * @brief Double precision 2nd order HLLC solver
 */
template<typename Real = double>
using EulerSolver2ndHLLC_d = EulerSolver2ndHLLC<Real>;

// ============================================================================
// CUSTOM LIMITER ALIASES
// ============================================================================

/**
 * @brief 2nd order with MC limiter (less dissipative than Minmod)
 */
template<typename Real = float>
using EulerSolver2ndMC = AdaptiveSolver<Euler2D<Real>,
                                        reconstruction::MUSCL_Reconstruction<reconstruction::MCLimiter>,
                                        flux::HLLCFlux>;

/**
 * @brief 2nd order with Superbee limiter (least dissipative)
 */
template<typename Real = float>
using EulerSolver2ndSuperbee = AdaptiveSolver<Euler2D<Real>,
                                               reconstruction::MUSCL_Reconstruction<reconstruction::SuperbeeLimiter>,
                                               flux::HLLCFlux>;

/**
 * @brief 2nd order with Van Leer limiter (smooth, symmetric)
 */
template<typename Real = float>
using EulerSolver2ndVanLeer = AdaptiveSolver<Euler2D<Real>,
                                               reconstruction::MUSCL_Reconstruction<reconstruction::VanLeerLimiter>,
                                               flux::HLLCFlux>;

// ============================================================================
// ALIAS SUMMARY TABLE
// ============================================================================
//
// | Alias             | Order | Flux    | Limiter    | Use Case                    |
// |-------------------|-------|---------|------------|-----------------------------|
// | EulerSolver1st    | 1st   | Rusanov | -          | Debug, robust               |
// | EulerSolver1stHLLC| 1st   | HLLC    | -          | Shocks, 1st order           |
// | EulerSolver2nd    | 2nd   | Rusanov | Minmod     | Default 2nd order           |
// | EulerSolver2ndHLLC| 2nd   | HLLC    | Minmod     | **PRODUCTION DEFAULT**      |
// | EulerSolver2ndRoe | 2nd   | Roe     | Minmod     | High accuracy               |
// | EulerSolver2ndMC  | 2nd   | HLLC    | MC         | Less dissipative 2nd order  |
// | EulerSolver2ndSuperbee| 2nd| HLLC    | Superbee   | Least dissipative           |
// | EulerSolver2ndVanLeer| 2nd| HLLC    | Van Leer   | Smooth, symmetric           |
//
// ============================================================================
//
// USAGE EXAMPLES:
//
// // Default choice (single precision, 2nd order, HLLC, Minmod)
// using MySolver = EulerSolver2ndHLLC<>;
// MySolver solver(fluid, domain, cfg);
//
// // Double precision
// using MySolverD = EulerSolver2ndHLLC<double>;
//
// // Custom limiter
// using MySolverMC = EulerSolver2ndHLLC<float, reconstruction::MCLimiter>;
// using MySolverSB = EulerSolver2ndSuperbee<>;
//
// ============================================================================
//
// COMPARISON: OLD vs NEW API
//
// ----- OLD API (Verbose) -----
// AdaptiveSolver<Euler2D<>,
//                 MUSCL_Reconstruction<MinmodLimiter>,
//                 HLLCFlux> solver(fluid, domain, cfg);
//
// ----- NEW API (Simple) -----
// EulerSolver2ndHLLC<> solver(fluid, domain, cfg);
//
// RESULT: 90% less code!
//
// ============================================================================

} // namespace subsetix::fvd
