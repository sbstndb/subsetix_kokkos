// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#pragma once

#include <cstdint>
#include "mpi_stub.hpp"

namespace subsetix::fvd::mpi {

// ============================================================================
// ENUMS
// ============================================================================

/**
 * @brief Communication mode between ranks
 */
enum class CommMode : int {
    /**
     * Synchronous blocking communications after each step
     * - Simple and safe
     * - May underutilize GPU/CPU during comm
     */
    Synchronous = 0,

    /**
     * Asynchronous non-blocking communications
     * - Computation/communication overlap
     * - Better performance
     */
    Asynchronous = 1,

    /**
     * Communications with GPU-Direct (if available)
     * - Direct GPU to GPU memory
     * - Requires CUDA-aware MPI
     */
    GPUDirect = 2,

    /**
     * Hybrid mode: automatic based on runtime
     */
    Hybrid = 3
};

/**
 * @brief Observer behavior in multi-rank environment
 */
enum class ObserverMode : int {
    /**
     * Only rank 0 displays/writes messages
     * - Other ranks silent
     * - Console/VTK output only from rank 0
     */
    Rank0Only = 0,

    /**
     * All ranks display with prefix
     * - Format: "[Rank 0/4] Step 100: t=0.5"
     * - Useful for debugging
     */
    AllRanks = 1,

    /**
     * Automatic reductions for global statistics
     * - Observers perform MPI_Allreduce
     * - Displays global min/max/average
     */
    Reduced = 2,

    /**
     * Hybrid mode: reduced for stats, rank 0 for output
     */
    Smart = 3
};

/**
 * @brief MPI initialization mode
 */
enum class MPICommMode : int {
    /**
     * Implicit MPI auto-initialization
     */
    Auto = 0,

    /**
     * User manages MPI_Init/MPI_Finalize
     */
    UserManaged = 1,

    /**
     * User provides a custom MPI_Comm
     */
    Custom = 2
};

/**
 * @brief Boundary directions
 */
enum class Boundary : int {
    Left = 0,
    Right = 1,
    Bottom = 2,
    Top = 3
};

/**
 * @brief Load balancing statistics
 */
struct LoadBalanceStats {
    std::size_t total_cells = 0;      // Total number of cells
    std::size_t min_cells = 0;        // Min cells per rank
    std::size_t max_cells = 0;        // Max cells per rank
    std::size_t avg_cells = 0;        // Average cells per rank
    float initial_ratio = 1.0f;       // Initial max/avg ratio
    float final_ratio = 1.0f;         // Final max/avg ratio
    float target_ratio = 1.1f;        // Target ratio
    int num_rebalances = 0;           // Number of redistributions
    double total_comm_time = 0.0;     // Total time in communication
    double comm_ratio = 0.0;          // Communication time / total time ratio
    int most_loaded_rank = 0;         // Most loaded rank
};

/**
 * @brief MPI communication configuration
 */
struct CommConfig {
    CommMode mode = CommMode::Synchronous;
    bool auto_comm = true;
    int halo_width = 1;
    bool use_gpu_direct = false;  // Auto-detect if Hybrid

    // For asynchronous mode
    bool enable_overlap = true;   // Computation/communication overlap

    // Timeouts
    double comm_timeout = 30.0;   // Timeout in seconds
};

/**
 * @brief MPI observer configuration
 */
struct ObserverConfig {
    ObserverMode mode = ObserverMode::Rank0Only;

    // For Reduced mode
    bool enable_allreduce = true;
    int reduce_interval = 1;  // Reduce every N steps

    // For AllRanks mode
    bool show_rank_prefix = true;
    const char* rank_format = "[Rank %d/%d]";
};

/**
 * @brief Global MPI configuration
 */
struct MPIConfig {
    MPICommMode comm_mode = MPICommMode::Auto;
    MPI_Comm custom_comm = MPI_COMM_WORLD;

    CommConfig comm;
    ObserverConfig observer;

    // Load balancing
    bool enable_auto_load_balance = false;
    float load_balance_tolerance = 1.1f;
    int load_balance_interval = 100;

    // Validation
    bool validate_decomposition = true;
    bool check_neighbor_consistency = true;
};

} // namespace subsetix::fvd::mpi
