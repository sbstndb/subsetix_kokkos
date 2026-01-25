// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#pragma once

#include <Kokkos_Core.hpp>
#include <functional>
#include <vector>
#include <memory>
#include <string>
#include "../solver/observer.hpp"
#include "mpi_config.hpp"
#include "mpi_stub.hpp"

namespace subsetix::fvd::mpi {

// ============================================================================
// MPI SOLVER STATE (Extended SolverState with MPI information)
// ============================================================================

/**
 * @brief Solver snapshot with MPI information
 *
 * Extends SolverState<Real> with MPI-specific information.
 */
template<typename Real = float>
struct MPISolverState : public SolverState<Real> {
    // Inherits from SolverState<Real>:
    // - Real time, dt
    // - int step, stage, max_level
    // - std::size_t total_cells, cells_per_level[10]
    // - double wall_time, step_time
    // - Real residual_rho, residual_momentum, residual_energy

    // MPI information
    int rank = 0;                 // My rank
    int nranks = 1;               // Total number of ranks

    // Local statistics
    std::size_t local_cells = 0;  // Number of cells on this rank
    Real local_min_rho = Real(0); // Local minimum of density
    Real local_max_rho = Real(0); // Local maximum of density
    Real local_avg_rho = Real(0); // Local average of density

    // Global statistics (reduced via MPI_Allreduce)
    std::size_t global_cells = 0; // Total number of cells (all ranks)
    Real global_min_rho = Real(0); // Global minimum of density
    Real global_max_rho = Real(0); // Global maximum of density
    Real global_avg_rho = Real(0); // Global average of density

    // Load balancing
    float load_balance_ratio = 1.0f;  // max_cells / avg_cells
    int most_loaded_rank = 0;         // Most loaded rank

    // Communication
    double last_comm_time = 0.0;      // Time spent in MPI comm (seconds)
    double comm_overlap_ratio = 0.0;  // Computation/communication overlap ratio
    std::size_t bytes_sent = 0;       // Bytes sent during last exchange
    std::size_t bytes_received = 0;   // Bytes received during last exchange

    // Voisins
    int num_neighbors = 0;            // Number of neighbors
    std::vector<int> neighbor_ranks;  // List of neighbor ranks
};

// ============================================================================
// MPI-AWARE CALLBACK TYPES
// ============================================================================

/**
 * @brief MPI-aware progress callback
 */
template<typename Real = float>
using MPIProgressCallback = std::function<void(const MPISolverState<Real>&)>;

/**
 * @brief Load balancing callback
 */
template<typename Real = float>
using LoadBalanceCallback = std::function<void(
    const MPISolverState<Real>&,
    float old_ratio,
    float new_ratio
)>;

/**
 * @brief Communication callback
 */
template<typename Real = float>
using CommCallback = std::function<void(
    const MPISolverState<Real>&,
    double comm_time
)>;

// ============================================================================
// MPI OBSERVER MANAGER
// ============================================================================

/**
 * @brief MPI-aware observer manager
 *
 * Extends ObserverManager with MPI-specific functionality.
 */
template<typename Real = float>
class MPIObserverManager : public ObserverManager<Real> {
public:
    using Base = ObserverManager<Real>;

    // ========================================================================
    // CONSTRUCTORS
    // ========================================================================

    MPIObserverManager() = default;
    ~MPIObserverManager() = default;

    // Disable copy, enable move
    MPIObserverManager(const MPIObserverManager&) = delete;
    MPIObserverManager& operator=(const MPIObserverManager&) = delete;
    MPIObserverManager(MPIObserverManager&&) = default;
    MPIObserverManager& operator=(MPIObserverManager&&) = default;

    // ========================================================================
    // CONFIGURATION
    // ========================================================================

    /**
     * @brief Configure the observer mode
     *
     * @param mode Observer mode (Rank0Only, AllRanks, Reduced)
     */
    void set_observer_mode(ObserverMode mode) {
        mode_ = mode;
    }

    /**
     * @brief Return the current observer mode
     */
    ObserverMode observer_mode() const {
        return mode_;
    }

    // ========================================================================
    // REGISTER MPI-AWARE CALLBACKS
    // ========================================================================

    /**
     * @brief Register an MPI-aware progress callback
     *
     * The callback adapts to the configured mode:
     * - Rank0Only: called only on rank 0
     * - AllRanks: called on all ranks
     * - Reduced: called on all ranks with global stats
     *
     * @param callback Function to call after each step
     * @return int Callback ID
     */
    int on_mpi_progress(MPIProgressCallback<Real> callback);

    /**
     * @brief Register a load balancing callback
     *
     * @param callback Function called after each redistribution
     * @return int Callback ID
     */
    int on_load_balance(LoadBalanceCallback<Real> callback);

    /**
     * @brief Register a communication callback
     *
     * @param callback Function called after each halo exchange
     * @return int Callback ID
     */
    int on_communication(CommCallback<Real> callback);

    // ========================================================================
    // NOTIFY MPI EVENTS
    // ========================================================================

    /**
     * @brief Notify all observers of an MPI event
     *
     * @param event Event
     * @param state Solver state with MPI info
     */
    void notify_mpi(SolverEvent event, const MPISolverState<Real>& state);

    /**
     * @brief Notify a load balancing event
     *
     * @param state Solver state
     * @param old_ratio Old load balance ratio
     * @param new_ratio New load balance ratio
     */
    void notify_load_balance(const MPISolverState<Real>& state,
                            float old_ratio, float new_ratio);

    // ========================================================================
    // UTILITY
    // ========================================================================

    /**
     * @brief Check if this rank should display/write
     *
     * @return true If this rank should output
     */
    bool should_output() const;

    /**
     * @brief Return the rank of this manager
     */
    int rank() const { return rank_; }

    /**
     * @brief Return the total number of ranks
     */
    int nranks() const { return nranks_; }

    /**
     * @brief Configure the MPI communicator
     */
    void set_comm(MPI_Comm comm);

private:
    ObserverMode mode_ = ObserverMode::Rank0Only;
    MPI_Comm comm_ = MPI_COMM_WORLD;
    int rank_ = 0;
    int nranks_ = 1;

    // MPI-aware callbacks
    std::vector<MPIProgressCallback<Real>> mpi_progress_callbacks_;
    std::vector<LoadBalanceCallback<Real>> load_balance_callbacks_;
    std::vector<CommCallback<Real>> comm_callbacks_;
};

// ============================================================================
// PREDEFINED MPI OBSERVERS
// ============================================================================

/**
 * @brief Predefined observers for MPI
 */
class MPIObservers {
public:
    /**
     * @brief Create an MPI-aware progress callback
     *
     * S'adapte au mode configuré:
     * - Rank0Only: displays only on rank 0
     * - AllRanks: displays with prefix [Rank X/Y]
     * - Reduced: displays global stats
     *
     * @param print_interval Display every N steps
     * @return MPIProgressCallback<Real>
     */
    template<typename Real = float>
    static MPIProgressCallback<Real> mpi_progress_printer(int print_interval = 1);

    /**
     * @brief Create an MPI-aware CSV logging callback
     *
     * Only rank 0 writes the file.
     *
     * @param filename CSV file name
     * @return Callback for CSV writing
     */
    template<typename Real = float>
    static std::function<void(SolverEvent, const MPISolverState<Real>&)>
    mpi_csv_logger(const std::string& filename);

    /**
     * @brief Create a load balancing report callback
     *
     * @return LoadBalanceCallback<Real>
     */
    template<typename Real = float>
    static LoadBalanceCallback<Real> load_balance_reporter();

    /**
     * @brief Create a communication report callback
     *
     * @return CommCallback<Real>
     */
    template<typename Real = float>
    static CommCallback<Real> comm_reporter();
};

// ============================================================================
// TYPE ALIASES
// ============================================================================

using MPIObserverManagerf = MPIObserverManager<float>;
using MPIObserverManagerd = MPIObserverManager<double>;
using MPISolverStatef = MPISolverState<float>;
using MPISolverStated = MPISolverState<double>;

} // namespace subsetix::fvd::mpi
