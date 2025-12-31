#pragma once

#include <Kokkos_Core.hpp>
#include <vector>
#include <array>
#include "mpi_config.hpp"
#include "topology.hpp"
#include "mpi_stub.hpp"

namespace subsetix::fvd::mpi {

// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================

template<typename Real>
struct MPISolverState;

// ============================================================================
// COMMUNICATION STATISTICS
// ============================================================================

/**
 * @brief Statistiques de communication MPI
 */
struct CommStats {
    double total_time = 0.0;         // Temps total passé en communication
    double last_time = 0.0;          // Temps du dernier échange
    std::size_t total_bytes_sent = 0;     // Total octets envoyés
    std::size_t total_bytes_recv = 0;     // Total octets reçus
    int num_exchanges = 0;           // Nombre d'échanges effectués
    double overlap_ratio = 0.0;      // Ratio de recouvrement calcul/comm

    /**
     * @brief Réinitialise les statistiques
     */
    void reset() {
        *this = CommStats{};
    }
};

// ============================================================================
// COMMUNICATION MANAGER
// ============================================================================

/**
 * @brief Gestionnaire des communications MPI
 *
 * Gère les échanges de halos, les réductions, etc.
 */
template<typename Real = float>
class CommManager {
public:
    /**
     * @brief Constructeur
     *
     * @param config Configuration de communication
     * @param comm Communicateur MPI
     */
    explicit CommManager(const CommConfig& config, MPI_Comm comm = MPI_COMM_WORLD);

    /**
     * @brief Destructeur
     */
    ~CommManager() = default;

    // Disable copy, enable move
    CommManager(const CommManager&) = delete;
    CommManager& operator=(const CommManager&) = delete;
    CommManager(CommManager&&) = default;
    CommManager& operator=(CommManager&&) = default;

    // ========================================================================
    // CONFIGURATION
    // ========================================================================

    /**
     * @brief Modifie la configuration
     */
    void set_config(const CommConfig& config);

    /**
     * @brief Retourne la configuration actuelle
     */
    const CommConfig& config() const { return config_; }

    /**
     * @brief Active/désactive les communications automatiques
     */
    void enable_auto_comm(bool enable = true) {
        config_.auto_comm = enable;
    }

    /**
     * @brief Vérifie si les communications automatiques sont actives
     */
    bool auto_comm_enabled() const {
        return config_.auto_comm;
    }

    // ========================================================================
    // HALO EXCHANGE
    // ========================================================================

    /**
     * @brief Échange les halos avec les voisins
     *
     * Cette fonction gère la communication complète:
     * - Pack des données à envoyer
     * - Isend/Irecv (asynchrone si configuré)
     * - Unpack des données reçues
     * - Wait pour complétion
     *
     * @param topology Topologie actuelle
     * @param fields Champs à échanger (tableau de Kokkos::View)
     * @param num_fields Nombre de champs
     * @return double Temps passé en communication (secondes)
     */
    double exchange_halos(
        const TopologyQuery& topology,
        Kokkos::View<Real*, DeviceMemorySpace>* fields,
        int num_fields
    );

    /**
     * @brief Échange les halos (version simple pour un champ)
     */
    double exchange_halos(
        const TopologyQuery& topology,
        Kokkos::View<Real*, DeviceMemorySpace> field
    );

    /**
     * @brief Échange les halos (version synchrone)
     */
    double exchange_halos_sync(
        const TopologyQuery& topology,
        Kokkos::View<Real*, DeviceMemorySpace> field
    );

    /**
     * @brief Échange les halos (version asynchrone)
     *
     * Retourne immédiatement. L'utilisateur doit appeler wait_halos().
     *
     * @return Request pour attendre la complétion
     */
    struct AsyncRequest {
        std::vector<MPI_Request> requests;
        bool active = false;
    };

    AsyncRequest exchange_halos_async(
        const TopologyQuery& topology,
        Kokkos::View<Real*, DeviceMemorySpace> field
    );

    /**
     * @brief Attend la complétion d'un échange asynchrone
     */
    void wait_halos(AsyncRequest& req);

    // ========================================================================
    // COLLECTIVE OPERATIONS
    // ========================================================================

    /**
     * @brief Barrière MPI
     */
    void barrier();

    /**
     * @brief Réduction Allreduce
     *
     * @param local_value Valeur locale
     * @param op Opération (MPI_SUM, MPI_MAX, etc.)
     * @return Real Valeur réduite (disponible sur tous les ranks)
     */
    Real allreduce(Real local_value, MPI_Op op = MPI_SUM);

    /**
     * @brief Broadcast depuis le rang 0
     *
     * @param value Valeur à broadcaster
     * @param root Rang source (défaut: 0)
     */
    void broadcast(Real& value, int root = 0);

    /**
     * @brief Gather sur le rang 0
     *
     * @param local_data Données locales
     * @param global_data Données globales (seulement sur rank 0)
     * @param count Nombre d'éléments
     */
    void gather(const Real* local_data, Real* global_data, int count);

    // ========================================================================
    // GPU-DIRECT MPI
    // ========================================================================

    /**
     * @brief Vérifie si GPU-Direct est disponible
     *
     * @return true Si CUDA-aware MPI est disponible
     */
    bool has_gpu_direct() const {
        return has_gpu_direct_;
    }

    /**
     * @brief Active/désactive GPU-Direct
     *
     * Si désactivé, utilise CPU staging.
     */
    void set_gpu_direct(bool enable) {
        use_gpu_direct_ = enable && has_gpu_direct_;
    }

    // ========================================================================
    // STATISTICS
    // ========================================================================

    /**
     * @brief Retourne les statistiques de communication
     */
    const CommStats& stats() const {
        return stats_;
    }

    /**
     * @brief Réinitialise les statistiques
     */
    void reset_stats() {
        stats_.reset();
    }

    /**
     * @brief Met à jour l'état du solveur avec les stats de comm
     */
    void update_solver_state(MPISolverState<Real>& state) const;

private:
    CommConfig config_;
    MPI_Comm comm_;
    int rank_ = 0;
    int nranks_ = 1;

    // GPU-Direct
    bool has_gpu_direct_ = false;
    bool use_gpu_direct_ = false;

    // Statistiques
    CommStats stats_;

    // Configuration du halo
    HaloInfo halo_info_;

    // Méthodes internes
    bool detect_gpu_direct();
    void setup_gpu_direct_buffers();
};

// ============================================================================
// INLINE FUNCTIONS
// ============================================================================

inline void CommManager<float>::broadcast(float& value, int root) {
    MPI_Bcast(&value, 1, MPI_FLOAT, root, comm_);
}

inline void CommManager<double>::broadcast(double& value, int root) {
    MPI_Bcast(&value, 1, MPI_DOUBLE, root, comm_);
}

} // namespace subsetix::fvd::mpi
