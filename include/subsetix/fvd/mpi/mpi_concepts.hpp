// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#pragma once

#include <type_traits>
#include <concepts>

namespace subsetix::fvd::mpi {

// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================

template<typename T>
struct DecompositionInfo;

enum class Boundary : int;

// ============================================================================
// DECOMPOSITION POLICY CONCEPT
// ============================================================================

/**
 * @brief Concept for domain decomposition policies
 *
 * A decomposition policy must provide:
 * - Config: Configuration structure (POD, GPU-safe)
 * - DecompositionInfo: Returned information structure
 * - init(): Static function to initialize the decomposition
 * - find_neighbors(): Function to find neighbors
 * - is_on_boundary(): Function to check if on a boundary
 */
template<typename T>
concept DecompositionPolicy = requires {
    // Configuration type (must be trivial for GPU)
    typename T::Config;

    // Returned information type
    typename T::DecompositionInfo;

    // Decomposition initialization
    { T::init(std::declval<const typename T::Config&>(), std::declval<MPI_Comm>()) }
        -> std::same_as<typename T::DecompositionInfo>;

    // Find the neighbors of a rank
    { T::find_neighbors(std::declval<const typename T::DecompositionInfo&>()) }
        -> std::convertible_to<std::array<int, 4>>;

    // Check if on a boundary
    { T::is_on_boundary(std::declval<const typename T::DecompositionInfo&>(),
                       std::declval<Boundary>()) }
        -> std::same_as<bool>;
};

// ============================================================================
// LOAD BALANCE POLICY CONCEPT
// ============================================================================

/**
 * @brief Concept for load balancing policies
 *
 * Must provide a device-friendly function to compute the cost of a cell.
 */
template<typename T, typename System>
concept LoadBalancePolicy = requires {
    typename T::template Config<System>;

    // Function to compute the cost of a cell (must be KOKKOS_FUNCTION)
    { T::template compute_cell_cost<System>(
        std::declval<const typename System::Conserved&>(),
        std::declval<const typename System::Primitive&>(),
        std::declval<const typename System::RealType&>(),
        std::declval<const typename System::RealType&>(),
        std::declval<int>()
    )} -> std::convertible_to<float>;
};

// ============================================================================
// COMM POLICY CONCEPT (for communication between ranks)
// ============================================================================

/**
 * @brief Concept for communication policies
 */
template<typename T>
concept CommPolicy = requires {
    // Configuration type
    typename T::Config;

    // Method to exchange halos
    { T::exchange_halos(std::declval<const typename T::Config&>()) }
        -> std::same_as<void>;

    // Method for barrier
    { T::barrier(std::declval<const typename T::Config&>()) }
        -> std::same_as<void>;
};

} // namespace subsetix::fvd::mpi
