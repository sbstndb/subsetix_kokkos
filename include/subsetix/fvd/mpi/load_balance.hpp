// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#pragma once

#include <Kokkos_Core.hpp>
#include "../system/concepts_v2.hpp"
#include "mpi_config.hpp"

namespace subsetix::fvd::mpi {

// ============================================================================
// CELL COUNT LOAD BALANCE
// ============================================================================

/**
 * @brief Load balancing by cell count
 *
 * Simplest strategy: each rank should have ~N_cells_total / N_ranks
 */
struct CellCountLoadBalance {
    template<typename System>
    struct Config {
        float max_imbalance = 1.1f;  // Imbalance tolerance
        int check_interval = 100;    // Check every N steps

        bool validate() const {
            return max_imbalance > 1.0f && check_interval > 0;
        }
    };

    /**
     * @brief Device-friendly function to compute the cost of a cell
     *
     * All cells have the same cost.
     *
     * @param U Conservative variables
     * @param q Primitive variables
     * @param grad_rho_x Density gradient in X
     * @param grad_rho_y Density gradient in Y
     * @param level Refinement level
     * @return Cost of the cell
     */
    template<typename System>
    KOKKOS_FUNCTION static float compute_cell_cost(
        const typename System::Conserved& U,
        const typename System::Primitive& q,
        const typename System::RealType grad_rho_x,
        const typename System::RealType grad_rho_y,
        int level
    ) {
        (void)U; (void)q; (void)grad_rho_x; (void)grad_rho_y; (void)level;
        return 1.0f;  // All cells have the same cost
    }
};

// ============================================================================
// LEVEL WEIGHTED LOAD BALANCE
// ============================================================================

/**
 * @brief Load based on refinement level
 *
 * Fine cells are more expensive (more work per cell)
 */
struct LevelWeightedLoadBalance {
    template<typename System>
    struct Config {
        float level_weight = 2.0f;  // A cell at level l+1 costs level_weight times more
        float max_imbalance = 1.1f;
        int check_interval = 100;

        bool validate() const {
            return level_weight > 1.0f && max_imbalance > 1.0f;
        }
    };

    template<typename System>
    KOKKOS_FUNCTION static float compute_cell_cost(
        const typename System::Conserved& U,
        const typename System::Primitive& q,
        const typename System::RealType grad_rho_x,
        const typename System::RealType grad_rho_y,
        int level
    ) {
        (void)U; (void)q; (void)grad_rho_x; (void)grad_rho_y;
        // Exponential cost with level
        return Kokkos::pow(level_weight, level);
    }

    template<typename System>
    static constexpr float level_weight = 2.0f;
};

// ============================================================================
// PHYSICS WEIGHTED LOAD BALANCE
// ============================================================================

/**
 * @brief Load based on physical activity (gradient, etc.)
 *
 * Active regions (shocks, strong gradients) are more expensive
 */
struct PhysicsWeightedLoadBalance {
    template<typename System>
    struct Config {
        float gradient_weight = 1.0f;
        float shock_weight = 2.0f;
        float base_cost = 1.0f;
        float max_imbalance = 1.1f;
        int check_interval = 100;

        bool validate() const {
            return gradient_weight >= 0.0f && shock_weight >= 1.0f;
        }
    };

    template<typename System>
    KOKKOS_FUNCTION static float compute_cell_cost(
        const typename System::Conserved& U,
        const typename System::Primitive& q,
        const typename System::RealType grad_rho_x,
        const typename System::RealType grad_rho_y,
        int level
    ) {
        float cost = base_cost;

        // Cost based on density gradients
        using Real = typename System::RealType;
        Real grad_mag = Kokkos::sqrt(grad_rho_x * grad_rho_x + grad_rho_y * grad_rho_y);
        cost += gradient_weight * grad_mag;

        // Bonus for shocks (strong compression = negative divergence)
        // To simplify, we use the velocity gradient
        // In a complete implementation, we would compute div(v)
        if (grad_mag > 0.5f) {  // Arbitrary threshold for "strong activity"
            cost *= shock_weight;
        }

        // Bonus for refinement
        cost *= Kokkos::pow(2.0f, level);

        return cost;
    }
};

// ============================================================================
// CUSTOM LOAD BALANCE (User-Defined, Compile-Time)
// ============================================================================

/**
 * @brief Custom load defined by the user (compile-time)
 *
 * The user provides a lambda that becomes a device function via KOKKOS_LAMBDA
 *
 * Exemple d'utilisation:
 * @code
 * auto my_cost_func = KOKKOS_LAMBDA(
 *     const Euler2D<float>::Conserved& U,
 *     const Euler2D<float>::Primitive& q,
 *     float grad_rho_x,
 *     float grad_rho_y,
 *     int level
 * ) -> float {
 *     // Your cost logic here
 *     return some_cost;
 * };
 *
 * solver.with_load_balancing<CustomLoadBalance<System, decltype(my_cost_func)>>({
 *     .cost_func = my_cost_func,
 *     .max_imbalance = 1.15f
 * });
 * @endcode
 */
template<typename System, typename CostFunc>
struct CustomLoadBalance {
    struct Config {
        CostFunc cost_func;
        float max_imbalance = 1.1f;
        int check_interval = 100;

        bool validate() const {
            return max_imbalance > 1.0f && check_interval > 0;
        }
    };

    template<typename... Args>
    KOKKOS_FUNCTION static float compute_cell_cost(Args&&... args) {
        return CostFunc{}(std::forward<Args>(args)...);
    }
};

// ============================================================================
// LOAD BALANCE POLICY TRAITS
// ============================================================================

/**
 * @brief Traits for load balancing policies
 *
 * Allows querying properties of a policy at compile time.
 */
template<typename LoadBalancePolicy, typename System>
struct LoadBalanceTraits {
    using Config = typename LoadBalancePolicy::template Config<System>;

    static constexpr bool has_custom_cost_function = requires {
        { LoadBalancePolicy::template compute_cell_cost<System>(
            std::declval<const typename System::Conserved&>(),
            std::declval<const typename System::Primitive&>(),
            std::declval<const typename System::RealType>(),
            std::declval<const typename System::RealType>(),
            std::declval<int>()
        )} -> std::convertible_to<float>;
    };
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * @brief Create a default load balancing configuration
 *
 * @tparam LoadBalancePolicy Load balancing policy
 * @tparam System System (Euler2D, etc.)
 * @return Config Default configuration
 */
template<template<typename> class LoadBalancePolicy, typename System>
typename LoadBalancePolicy<System>::Config default_load_balance_config() {
    return typename LoadBalancePolicy<System>::Config{};
}

/**
 * @brief Validate a load balancing configuration
 *
 * @tparam LoadBalancePolicy Load balancing policy
 * @tparam System Système
 * @param cfg Configuration to validate
 * @return true If valid
 */
template<template<typename> class LoadBalancePolicy, typename System>
bool validate_load_balance_config(const typename LoadBalancePolicy<System>::Config& cfg) {
    return cfg.validate();
}

} // namespace subsetix::fvd::mpi
