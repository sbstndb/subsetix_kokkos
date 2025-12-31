#pragma once

/**
 * @file adaptive_solver_mpi.hpp
 *
 * @brief Extension MPI pour AdaptiveSolver
 *
 * Ce fichier étend AdaptiveSolver avec les fonctionnalités MPI.
 * Il peut être inclus après adaptive_solver.hpp pour ajouter le support MPI.
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
 * @brief Extension MPI pour AdaptiveSolver
 *
 * Cette classe étend AdaptiveSolver avec les fonctionnalités MPI
 * via l'héritage. Elle ne modifie pas la classe de base.
 *
 * Les méthodes MPI sont ajoutées sans casser le code existant.
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
     * @brief Configuration étendue pour MPI
     */
    struct Config : public Base::Config {
        // Configuration MPI
        mpi::MPIConfig mpi_config;

        // Configuration de décomposition
        typename Decomposition::Config decomp_config;
    };

    // ========================================================================
    // BUILDER (Extended)
    // ========================================================================

    /**
     * @brief Builder étendu pour le solveur MPI
     */
    class Builder : public Base::Builder {
    public:
        using BaseBuilder = typename Base::Builder;

        // Constructeur
        Builder(int nx, int ny) : BaseBuilder(nx, ny) {
            // Configurer MPI par défaut
            mpi_config_ = mpi::default_mpi_config();
        }

        // ====================================================================
        // MPI Configuration Methods
        // ====================================================================

        /**
         * @brief Spécifie le type de décomposition de domaine
         *
         * @tparam Decomp La politique de décomposition (Cartesian1D, Cartesian2D, Metis, etc.)
         * @param config Configuration de la décomposition
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
         * @brief Configure la largeur des halo cells pour les communications MPI
         *
         * @param halo_width Nombre de couches de ghost cells (défaut: 1)
         */
        Builder& with_halo_width(int halo_width = 1) {
            mpi_config_.comm.halo_width = halo_width;
            return *this;
        }

        /**
         * @brief Active/désactive les communications automatiques
         *
         * @param enable Si true, les halo exchanges sont automatiques après chaque step()
         */
        Builder& with_auto_comm(bool enable = true) {
            mpi_config_.comm.auto_comm = enable;
            return *this;
        }

        /**
         * @brief Configure le mode de communication
         *
         * @param mode Mode de communication
         */
        Builder& with_comm_mode(mpi::CommMode mode) {
            mpi_config_.comm.mode = mode;
            return *this;
        }

        /**
         * @brief Configure le comportement des observateurs en multi-rank
         *
         * @param mode Mode des observateurs
         */
        Builder& with_observer_mode(mpi::ObserverMode mode) {
            mpi_config_.observer.mode = mode;
            return *this;
        }

        /**
         * @brief Configure le communicateur MPI
         *
         * @param comm_mode Mode d'initialisation MPI
         * @param custom_comm Communicateur custom (optionnel)
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
         * @brief Configure le load balancing pour AMR
         *
         * @tparam LoadBalancePolicy Politique de load balancing
         * @param policy Configuration de la politique
         */
        template<typename LoadBalancePolicy>
        Builder& with_load_balancing(typename LoadBalancePolicy::template Config<System> policy) {
            // Stocker la politique (à implémenter)
            mpi_config_.enable_auto_load_balance = true;
            mpi_config_.load_balance_tolerance = policy.max_imbalance;
            return *this;
        }

        /**
         * @brief Construit le solveur MPI
         *
         * @return MPIAdaptiveSolver Le solveur configuré
         */
        MPIAdaptiveSolver build() {
            // Construire la configuration
            Config config;
            config.mpi_config = mpi_config_;
            config.decomp_config = decomp_config_;

            // Construire le solveur
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
     * @brief Retourne les informations MPI de ce solveur
     */
    const mpi::TopologyInfo<Real>& mpi_info() const {
        return topology_.info();
    }

    /**
     * @brief Retourne mon rank
     */
    int rank() const {
        return topology_.rank();
    }

    /**
     * @brief Retourne le nombre total de ranks
     */
    int nranks() const {
        return topology_.nranks();
    }

    /**
     * @brief Retourne vrai si je suis le rang 0
     */
    bool is_rank0() const {
        return topology_.is_rank0();
    }

    /**
     * @brief Retourne la liste de mes voisins
     */
    const std::vector<int>& neighbors() const {
        return topology_.neighbors();
    }

    /**
     * @brief Retourne le nombre de voisins
     */
    int num_neighbors() const {
        return topology_.num_neighbors();
    }

    /**
     * @brief Vérifie si un rank est mon voisin
     */
    bool is_neighbor(int other_rank) const {
        return topology_.is_neighbor(other_rank);
    }

    // ========================================================================
    // COMMUNICATION CONTROL
    // ========================================================================

    /**
     * @brief Active/désactive les communications automatiques
     */
    void enable_auto_comm(bool enable = true) {
        comm_manager_.enable_auto_comm(enable);
    }

    /**
     * @brief Exécute manuellement un échange de halos
     */
    void exchange_halos() {
        // À implémenter avec les champs actuels
        // comm_manager_.exchange_halos(topology_, fields, num_fields);
    }

    /**
     * @brief Synchronise tous les ranks (barrière MPI)
     */
    void barrier() {
        comm_manager_.barrier();
    }

    /**
     * @brief Réduit une valeur scalaire sur tous les ranks
     */
    Real allreduce(Real local_value, MPI_Op op = MPI_SUM) const {
        return comm_manager_.allreduce(local_value, op);
    }

    /**
     * @brief Broadcast une valeur depuis le rank 0
     */
    void broadcast(Real& value, int root = 0) const {
        comm_manager_.broadcast(value, root);
    }

    // ========================================================================
    // LOAD BALANCING
    // ========================================================================

    /**
     * @brief Active le load balancing automatique pour AMR
     */
    void enable_auto_load_balance(bool enable = true) {
        // À implémenter
    }

    /**
     * @brief Force un load balancing immédiat
     */
    void load_balance();

    /**
     * @brief Retourne des statistiques sur le load balancing
     */
    mpi::LoadBalanceStats load_balance_stats() const;

    // ========================================================================
    // GHOST CELLS / HALO
    // ========================================================================

    /**
     * @brief Retourne la largeur actuelle du halo
     */
    int halo_width() const {
        return comm_manager_.config().halo_width;
    }

    /**
     * @brief Modifie la largeur du halo
     */
    void set_halo_width(int width) {
        auto cfg = comm_manager_.config();
        cfg.halo_width = width;
        comm_manager_.set_config(cfg);
    }

    /**
     * @brief Synchronise les ghost cells (synonyme de exchange_halos)
     */
    void sync_ghosts() {
        exchange_halos();
    }

    // ========================================================================
    // STEP (Extended with automatic halo exchange)
    // ========================================================================

    /**
     * @brief Effectue un pas de temps avec échange de halos automatique
     */
    void step() {
        // Step normal du solveur de base
        Base::step();

        // Échange automatique des halos si activé
        if (comm_manager_.auto_comm_enabled()) {
            exchange_halos();
        }
    }

    /**
     * @brief Step sans communication (pour usage manuel)
     */
    void step_without_comm() {
        Base::step();
    }

    // ========================================================================
    // OBSERVERS (MPI-aware)
    // ========================================================================

    /**
     * @brief Retourne le gestionnaire d'observateurs MPI
     */
    mpi::MPIObserverManager<Real>& mpi_observers() {
        return mpi_observers_;
    }

    /**
     * @brief Retourne le gestionnaire d'observateurs MPI (const)
     */
    const mpi::MPIObserverManager<Real>& mpi_observers() const {
        return mpi_observers_;
    }

private:
    // ========================================================================
    // CONSTRUCTORS
    // ========================================================================

    /**
     * @brief Constructeur privé (utiliser Builder)
     */
    explicit MPIAdaptiveSolver(const Config& config)
        : Base(config)
        , comm_manager_(config.mpi_config.comm, config.mpi_config.custom_comm)
        , topology_()
        , mpi_observers_()
    {
        // Initialiser MPI si nécessaire
        if (config.mpi_config.comm_mode == mpi::MPICommMode::Auto) {
            mpi::MPIInitializer::initialize();
        }

        // Initialiser la décomposition
        init_decomposition(config.decomp_config);

        // Configurer les observateurs MPI
        mpi_observers_.set_observer_mode(config.mpi_config.observer.mode);
        mpi_observers_.set_comm(config.mpi_config.custom_comm);
    }

    /**
     * @brief Initialiser la décomposition de domaine
     */
    void init_decomposition(const typename Decomposition::Config& config) {
        // Appeler la politique de décomposition
        auto decomp_info = Decomposition::init(config, mpi_config_.custom_comm);

        // Construire la topologie
        topology_ = mpi::TopologyQuery(decomp_info, mpi_config_.custom_comm);

        // Construire les infos de halo
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

// Solveur MPI avec Euler2D
template<typename Real = float>
using EulerSolverMPI = MPIAdaptiveSolver<Euler2D<Real>,
                                        reconstruction::NoReconstruction,
                                        flux::RusanovFlux,
                                        mpi::Cartesian2D>;

// Solveur MPI RK3
template<typename Real = float>
using EulerSolverRK3_MPI = MPIAdaptiveSolver<Euler2D<Real>,
                                            reconstruction::NoReconstruction,
                                            flux::RusanovFlux,
                                            mpi::Cartesian2D>;

} // namespace subsetix::fvd::solver
