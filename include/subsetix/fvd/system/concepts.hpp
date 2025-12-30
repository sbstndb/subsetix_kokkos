#pragma once

#include <Kokkos_Core.hpp>
#include <type_traits>

namespace subsetix::fvd {

// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================

template<typename Real>
class Euler2D;

// ============================================================================
// SYSTEM CONCEPT - What a System Must Provide
// ============================================================================

/**
 * @brief Core System concept for finite volume methods
 *
 * A System must provide:
 * - Real type (float or double)
 * - Conserved variables (U)
 * - Primitive variables (Q)
 * - Views wrapper (for device access)
 * - Conversion functions (to_primitive, from_primitive)
 * - Physical fluxes (flux_phys_x, flux_phys_y)
 * - Sound speed (sound_speed)
 * - Default parameters (default_gamma)
 *
 * This is a documentation concept - not enforced at compile time for now.
 * Future versions will use C++20 concepts.
 */
template<typename T>
struct IsSystem {
    static constexpr bool value = false;
};

// ============================================================================
// TYPE TRAITS FOR SYSTEMS
// ============================================================================

/**
 * @brief System traits for compile-time introspection
 *
 * Provides information about a System:
 * - n_conserved: Number of conserved variables
 * - Field names for debugging
 * - Generic field access (for_each_field, get_field)
 */
template<typename System>
struct system_traits {
    static constexpr int n_conserved = 0;
};

// ============================================================================
// MARKER TYPES FOR POLICY-BASED DESIGN
// ============================================================================

/**
 * @brief Mark reconstruction as "none" (1st order)
 */
struct NoReconstructionTag {};

/**
 * @brief Mark reconstruction as MUSCL (2nd order)
 */
struct MUSCLReconstructionTag {};

} // namespace subsetix::fvd
