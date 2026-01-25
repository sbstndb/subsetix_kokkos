// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#pragma once

/**
 * @file adaptive_solver_mpi.hpp
 *
 * @brief MPI extension for AdaptiveSolver
 *
 * This file extends AdaptiveSolver with MPI functionality.
 * It can be included after adaptive_solver.hpp to add MPI support.
 *
 * Usage:
 *   #include <subsetix/fvd/solver/adaptive_solver.hpp>
 *   #include <subsetix/fvd/solver/adaptive_solver_mpi.hpp>
 *
 *   auto solver = EulerSolverRK3::builder(nx, ny)
 *       .with_mpi<MPIComm::Auto>()
 *       .with_decomposition<mpi::Cartesian2D>({...})
 *       .build();
 */

#include <Kokkos_Core.hpp>
#include "../mpi/fvd_mpi.hpp"
#include "../mpi/mpi_stub.hpp"
#include "adaptive_solver.hpp"

namespace subsetix::fvd::solver {

// ============================================================================
// MPI-ENABLED ADAPTIVE SOLVER (Extension)
// ============================================================================

/**
 * @brief MPI extension for AdaptiveSolver
 *
 * This class extends AdaptiveSolver with MPI functionality
 * via inheritance. It does not modify the base class.
 *
 * MPI methods are added without breaking existing code.
 */
template<
    FiniteVolumeSystem System,
    typename Reconstruction = reconstruction::NoReconstruction,
    template<typename> class FluxScheme = flux::RusanovFlux,
    typename Decomposition = mpi::Cartesian1D
>
class MPIAdaptiveSolver : public AdaptiveSolver<System, Reconstruction, FluxScheme> {
public:
    using Base = AdaptiveSolver<System, Reconstruction, FluxScheme>;
    using Real = typename Base::Real;
    using MPIReal = mpi::MPISolverState<Real>;

    // ========================================================================
    // CONFIGURATION
    // ========================================================================

    /**
     * @brief Extended configuration for MPI
     */
    struct Config : public Base::Config {
        // MPI configuration
        mpi::MPIConfig mpi_config;

        // Decomposition configuration
        typename Decomposition::Config decomp_config;
    };

    // ========================================================================
    // BUILDER (Extended)
    // ========================================================================

    /**
     * @brief Extended builder for MPI solver
     */
    class Builder : public Base::Builder {
    public:
        using BaseBuilder = typename Base::Builder;

        // Constructor
        Builder(int nx, int ny) : BaseBuilder(nx, ny) {
            // Configure MPI by default
            mpi_config_ = mpi::default_mpi_config();
        }

        // ====================================================================
        // MPI Configuration Methods
        // ====================================================================

        /**
         * @brief Specify the domain decomposition type
         *
         * @tparam Decomp The decomposition policy (Cartesian1D, Cartesian2D, Metis, etc.)
         * @param config Decomposition configuration
         *
         * Exemple:
         *   builder.with_decomposition<mpi::Cartesian2D>({
         *       .nx_global = 1000,
         *       .ny_global = 500,
         *       .px = 2,
         *       .py = 2
         *   });
         */
        template<typename Decomp>
        Builder& with_decomposition(typename Decomp::Config config) {
            decomp_config_ = config;
            decomp_type_ = GenericDecompositionInfo::Type::Cartesian1D;

            if constexpr (std::is_same_v<Decomp, mpi::Cartesian1D>) {
                decomp_type_ = GenericDecompositionInfo::Type::Cartesian1D;
            } else if constexpr (std::is_same_v<Decomp, mpi::Cartesian2D>) {
                decomp_type_ = GenericDecompositionInfo::Type::Cartesian2D;
            } else if constexpr (std::is_same_v<Decomp, mpi::SpaceFillingCurve>) {
                decomp_type_ = GenericDecompositionInfo::Type::SpaceFilling;
            } else if constexpr (std::is_same_v<Decomp, mpi::MetisDecomposition>) {
                decomp_type_ = GenericDecompositionInfo::Type::Metis;
            } else if constexpr (std::is_same_v<Decomp, mpi::StaticDecomposition>) {
                decomp_type_ = GenericDecompositionInfo::Type::Static;
            }

            return *this;
        }

        /**
         * @brief Configure the width of halo cells for MPI communications
         *
         * @param halo_width Number of ghost cell layers (default: 1)
         */
        Builder& with_halo_width(int halo_width = 1) {
            mpi_config_.comm.halo_width = halo_width;
            return *this;
        }

        /**
         * @brief Enable/disable automatic communications
         *
         * @param enable If true, halo exchanges are automatic after each step()
         */
        Builder& with_auto_comm(bool enable = true) {
            mpi_config_.comm.auto_comm = enable;
            return *this;
        }

        /**
         * @brief Configure the communication mode
         *
         * @param mode Communication mode
         */
        Builder& with_comm_mode(mpi::CommMode mode) {
            mpi_config_.comm.mode = mode;
            return *this;
        }

        /**
         * @brief Configure observer behavior in multi-rank
         *
         * @param mode Observer mode
         */
        Builder& with_observer_mode(mpi::ObserverMode mode) {
            mpi_config_.observer.mode = mode;
            return *this;
        }

        /**
         * @brief Configure the MPI communicator
         *
         * @param comm_mode MPI initialization mode
         * @param custom_comm Custom communicator (optional)
         */
        Builder& with_mpi_comm(
            mpi::MPICommMode comm_mode = mpi::MPICommMode::Auto,
            MPI_Comm custom_comm = MPI_COMM_WORLD
        ) {
            mpi_config_.comm_mode = comm_mode;
            mpi_config_.custom_comm = custom_comm;
            return *this;
        }

        /**
         * @brief Configure load balancing for AMR
         *
         * @tparam LoadBalancePolicy Load balancing policy
         * @param policy Policy configuration
         */
        template<typename LoadBalancePolicy>
        Builder& with_load_balancing(typename LoadBalancePolicy::template Config<System> policy) {
            // Store the policy (to implement)
            mpi_config_.enable_auto_load_balance = true;
            mpi_config_.load_balance_tolerance = policy.max_imbalance;
            return *this;
        }

        /**
         * @brief Build the MPI solver
         *
         * @return MPIAdaptiveSolver The configured solver
         */
        MPIAdaptiveSolver build() {
            // Build the configuration
            Config config;
            config.mpi_config = mpi_config_;
            config.decomp_config = decomp_config_;

            // Build the solver
            return MPIAdaptiveSolver(config);
        }

    private:
        mpi::MPIConfig mpi_config_;
        typename Decomposition::Config decomp_config_;
        GenericDecompositionInfo::Type decomp_type_ =
            GenericDecompositionInfo::Type::Cartesian1D;
    };

    // ========================================================================
    // MPI QUERY METHODS
    // ========================================================================

    /**
     * @brief Return the MPI information of this solver
     */
    const mpi::TopologyInfo<Real>& mpi_info() const {
        return topology_.info();
    }

    /**
     * @brief Return my rank
     */
    int rank() const {
        return topology_.rank();
    }

    /**
     * @brief Return the total number of ranks
     */
    int nranks() const {
        return topology_.nranks();
    }

    /**
     * @brief Return true if I am rank 0
     */
    bool is_rank0() const {
        return topology_.is_rank0();
    }

    /**
     * @brief Return the list of my neighbors
     */
    const std::vector<int>& neighbors() const {
        return topology_.neighbors();
    }

    /**
     * @brief Return the number of neighbors
     */
    int num_neighbors() const {
        return topology_.num_neighbors();
    }

    /**
     * @brief Check if a rank is my neighbor
     */
    bool is_neighbor(int other_rank) const {
        return topology_.is_neighbor(other_rank);
    }

    // ========================================================================
    // COMMUNICATION CONTROL
    // ========================================================================

    /**
     * @brief Enable/disable automatic communications
     */
    void enable_auto_comm(bool enable = true) {
        comm_manager_.enable_auto_comm(enable);
    }

    /**
     * @brief Manually execute a halo exchange
     */
    void exchange_halos() {
        // To implement with current fields
        // comm_manager_.exchange_halos(topology_, fields, num_fields);
    }

    /**
     * @brief Synchronize all ranks (MPI barrier)
     */
    void barrier() {
        comm_manager_.barrier();
    }

    /**
     * @brief Reduce a scalar value across all ranks
     */
    Real allreduce(Real local_value, MPI_Op op = MPI_SUM) const {
        return comm_manager_.allreduce(local_value, op);
    }

    /**
     * @brief Broadcast a value from rank 0
     */
    void broadcast(Real& value, int root = 0) const {
        comm_manager_.broadcast(value, root);
    }

    // ========================================================================
    // LOAD BALANCING
    // ========================================================================

    /**
     * @brief Enable automatic load balancing for AMR
     */
    void enable_auto_load_balance(bool enable = true) {
        // To implement
    }

    /**
     * @brief Force an immediate load balancing
     */
    void load_balance();

    /**
     * @brief Return statistics about load balancing
     */
    mpi::LoadBalanceStats load_balance_stats() const;

    // ========================================================================
    // GHOST CELLS / HALO
    // ========================================================================

    /**
     * @brief Return the current halo width
     */
    int halo_width() const {
        return comm_manager_.config().halo_width;
    }

    /**
     * @brief Modify the halo width
     */
    void set_halo_width(int width) {
        auto cfg = comm_manager_.config();
        cfg.halo_width = width;
        comm_manager_.set_config(cfg);
    }

    /**
     * @brief Synchronize ghost cells (synonym of exchange_halos)
     */
    void sync_ghosts() {
        exchange_halos();
    }

    // ========================================================================
    // STEP (Extended with automatic halo exchange)
    // ========================================================================

    /**
     * @brief Perform a time step with automatic halo exchange
     */
    void step() {
        // Normal step of the base solver
        Base::step();

        // Automatic halo exchange if enabled
        if (comm_manager_.auto_comm_enabled()) {
            exchange_halos();
        }
    }

    /**
     * @brief Step without communication (for manual usage)
     */
    void step_without_comm() {
        Base::step();
    }

    // ========================================================================
    // OBSERVERS (MPI-aware)
    // ========================================================================

    /**
     * @brief Return the MPI observer manager
     */
    mpi::MPIObserverManager<Real>& mpi_observers() {
        return mpi_observers_;
    }

    /**
     * @brief Return the MPI observer manager (const)
     */
    const mpi::MPIObserverManager<Real>& mpi_observers() const {
        return mpi_observers_;
    }

private:
    // ========================================================================
    // CONSTRUCTORS
    // ========================================================================

    /**
     * @brief Constructor privé (utiliser Builder)
     */
    explicit MPIAdaptiveSolver(const Config& config)
        : Base(config)
        , comm_manager_(config.mpi_config.comm, config.mpi_config.custom_comm)
        , topology_()
        , mpi_observers_()
    {
        // Initialize MPI if necessary
        if (config.mpi_config.comm_mode == mpi::MPICommMode::Auto) {
            mpi::MPIInitializer::initialize();
        }

        // Initialize the decomposition
        init_decomposition(config.decomp_config);

        // Configure MPI observers
        mpi_observers_.set_observer_mode(config.mpi_config.observer.mode);
        mpi_observers_.set_comm(config.mpi_config.custom_comm);
    }

    /**
     * @brief Initialize the decomposition de domaine
     */
    void init_decomposition(const typename Decomposition::Config& config) {
        // Call the decomposition policy
        auto decomp_info = Decomposition::init(config, mpi_config_.custom_comm);

        // Build the topology
        topology_ = mpi::TopologyQuery(decomp_info, mpi_config_.custom_comm);

        // Build the halo info
        halo_info_ = mpi::HaloBuilder::build(topology_, mpi_config_.comm.halo_width);
    }

    // ========================================================================
    // MEMBERS
    // ========================================================================

    mpi::CommManager<Real> comm_manager_;
    mpi::TopologyQuery topology_;
    mpi::HaloInfo halo_info_;
    mpi::MPIObserverManager<Real> mpi_observers_;
    mpi::MPIConfig mpi_config_;
};

// ============================================================================
// CONVENIENCE TYPE ALIASES
// ============================================================================

// MPI solver with Euler2D
template<typename Real = float>
using EulerSolverMPI = MPIAdaptiveSolver<Euler2D<Real>,
                                        reconstruction::NoReconstruction,
                                        flux::RusanovFlux,
                                        mpi::Cartesian2D>;

// MPI RK3 solver
template<typename Real = float>
using EulerSolverRK3_MPI = MPIAdaptiveSolver<Euler2D<Real>,
                                            reconstruction::NoReconstruction,
                                            flux::RusanovFlux,
                                            mpi::Cartesian2D>;

} // namespace subsetix::fvd::solver
