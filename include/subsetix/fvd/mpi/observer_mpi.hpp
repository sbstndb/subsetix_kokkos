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
 * @brief Snapshot du solveur avec informations MPI
 *
 * Étend SolverState<Real> avec les informations spécifiques MPI.
 */
template<typename Real = float>
struct MPISolverState : public SolverState<Real> {
    // Hérite de SolverState<Real>:
    // - Real time, dt
    // - int step, stage, max_level
    // - std::size_t total_cells, cells_per_level[10]
    // - double wall_time, step_time
    // - Real residual_rho, residual_momentum, residual_energy

    // Informations MPI
    int rank = 0;                 // Mon rank
    int nranks = 1;               // Nombre total de ranks

    // Statistiques locales
    std::size_t local_cells = 0;  // Nombre de cellules sur ce rank
    Real local_min_rho = Real(0); // Min locale de la densité
    Real local_max_rho = Real(0); // Max locale de la densité
    Real local_avg_rho = Real(0); // Moyenne locale de la densité

    // Statistiques globales (réduites via MPI_Allreduce)
    std::size_t global_cells = 0; // Nombre total de cellules (tous ranks)
    Real global_min_rho = Real(0); // Min global de la densité
    Real global_max_rho = Real(0); // Max global de la densité
    Real global_avg_rho = Real(0); // Moyenne globale de la densité

    // Load balancing
    float load_balance_ratio = 1.0f;  // max_cells / avg_cells
    int most_loaded_rank = 0;         // Rank le plus chargé

    // Communication
    double last_comm_time = 0.0;      // Temps passé en comm MPI (secondes)
    double comm_overlap_ratio = 0.0;  // Ratio de recouvrement calcul/comm
    std::size_t bytes_sent = 0;       // Octets envoyés lors du dernier exchange
    std::size_t bytes_received = 0;   // Octets reçus lors du dernier exchange

    // Voisins
    int num_neighbors = 0;            // Nombre de voisins
    std::vector<int> neighbor_ranks;  // Liste des rangs voisins
};

// ============================================================================
// MPI-AWARE CALLBACK TYPES
// ============================================================================

/**
 * @brief Callback de progression MPI-aware
 */
template<typename Real = float>
using MPIProgressCallback = std::function<void(const MPISolverState<Real>&)>;

/**
 * @brief Callback de load balancing
 */
template<typename Real = float>
using LoadBalanceCallback = std::function<void(
    const MPISolverState<Real>&,
    float old_ratio,
    float new_ratio
)>;

/**
 * @brief Callback de communication
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
 * @brief Gestionnaire d'observateurs MPI-aware
 *
 * Étend ObserverManager avec des fonctionnalités spécifiques MPI.
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
     * @brief Configure le mode des observateurs
     *
     * @param mode Mode des observateurs (Rank0Only, AllRanks, Reduced)
     */
    void set_observer_mode(ObserverMode mode) {
        mode_ = mode;
    }

    /**
     * @brief Retourne le mode actuel des observateurs
     */
    ObserverMode observer_mode() const {
        return mode_;
    }

    // ========================================================================
    // REGISTER MPI-AWARE CALLBACKS
    // ========================================================================

    /**
     * @brief Enregistrer un callback de progression MPI-aware
     *
     * Le callback s'adapte au mode configuré:
     * - Rank0Only: appelé seulement sur rank 0
     * - AllRanks: appelé sur tous les ranks
     * - Reduced: appelé sur tous les ranks avec stats globales
     *
     * @param callback Fonction à appeler après chaque step
     * @return int ID du callback
     */
    int on_mpi_progress(MPIProgressCallback<Real> callback);

    /**
     * @brief Enregistrer un callback de load balancing
     *
     * @param callback Fonction appelée après chaque redistribution
     * @return int ID du callback
     */
    int on_load_balance(LoadBalanceCallback<Real> callback);

    /**
     * @brief Enregistrer un callback de communication
     *
     * @param callback Fonction appelée après chaque échange de halos
     * @return int ID du callback
     */
    int on_communication(CommCallback<Real> callback);

    // ========================================================================
    // NOTIFY MPI EVENTS
    // ========================================================================

    /**
     * @brief Notifier tous les observateurs d'un événement MPI
     *
     * @param event Événement
     * @param state État du solveur avec info MPI
     */
    void notify_mpi(SolverEvent event, const MPISolverState<Real>& state);

    /**
     * @brief Notifier un événement de load balancing
     *
     * @param state État du solveur
     * @param old_ratio Ancien ratio de load balance
     * @param new_ratio Nouveau ratio de load balance
     */
    void notify_load_balance(const MPISolverState<Real>& state,
                            float old_ratio, float new_ratio);

    // ========================================================================
    // UTILITY
    // ========================================================================

    /**
     * @brief Vérifier si ce rank doit afficher/écrire
     *
     * @return true Si ce rank doit output
     */
    bool should_output() const;

    /**
     * @brief Retourne le rank de ce gestionnaire
     */
    int rank() const { return rank_; }

    /**
     * @brief Retourne le nombre total de ranks
     */
    int nranks() const { return nranks_; }

    /**
     * @brief Configure le communicateur MPI
     */
    void set_comm(MPI_Comm comm);

private:
    ObserverMode mode_ = ObserverMode::Rank0Only;
    MPI_Comm comm_ = MPI_COMM_WORLD;
    int rank_ = 0;
    int nranks_ = 1;

    // Callbacks MPI-aware
    std::vector<MPIProgressCallback<Real>> mpi_progress_callbacks_;
    std::vector<LoadBalanceCallback<Real>> load_balance_callbacks_;
    std::vector<CommCallback<Real>> comm_callbacks_;
};

// ============================================================================
// PREDEFINED MPI OBSERVERS
// ============================================================================

/**
 * @brief Observateurs prédéfinis pour MPI
 */
class MPIObservers {
public:
    /**
     * @brief Créer un callback de progression MPI-aware
     *
     * S'adapte au mode configuré:
     * - Rank0Only: affiche seulement sur rank 0
     * - AllRanks: affiche avec préfixe [Rank X/Y]
     * - Reduced: affiche les stats globales
     *
     * @param print_interval Afficher tous les N steps
     * @return MPIProgressCallback<Real>
     */
    template<typename Real = float>
    static MPIProgressCallback<Real> mpi_progress_printer(int print_interval = 1);

    /**
     * @brief Créer un callback de logging CSV MPI-aware
     *
     * Seul rank 0 écrit le fichier.
     *
     * @param filename Nom du fichier CSV
     * @return Callback pour écriture CSV
     */
    template<typename Real = float>
    static std::function<void(SolverEvent, const MPISolverState<Real>&)>
    mpi_csv_logger(const std::string& filename);

    /**
     * @brief Créer un callback de rapport de load balancing
     *
     * @return LoadBalanceCallback<Real>
     */
    template<typename Real = float>
    static LoadBalanceCallback<Real> load_balance_reporter();

    /**
     * @brief Créer un callback de rapport de communication
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
