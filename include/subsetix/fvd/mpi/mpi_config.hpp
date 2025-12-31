#pragma once

#include <cstdint>
#include "mpi_stub.hpp"

namespace subsetix::fvd::mpi {

// ============================================================================
// ENUMS
// ============================================================================

/**
 * @brief Mode de communication entre ranks
 */
enum class CommMode : int {
    /**
     * Communications synchrones bloquantes après chaque step
     * - Simple et sûr
     * - Peut sous-utiliser GPU/CPU pendant les comm
     */
    Synchronous = 0,

    /**
     * Communications asynchrones non-bloquantes
     * - Recouvrement calcul/communication
     * - Meilleure performance
     */
    Asynchronous = 1,

    /**
     * Communications avec GPU-Direct (si disponible)
     * - Mémoire GPU vers GPU directe
     * - Requis CUDA-aware MPI
     */
    GPUDirect = 2,

    /**
     * Mode hybride: automatique selon le runtime
     */
    Hybrid = 3
};

/**
 * @brief Comportement des observateurs en environnement multi-rank
 */
enum class ObserverMode : int {
    /**
     * Seul le rang 0 affiche/écrit les messages
     * - Autres ranks silencieux
     * - Sortie console/VTK uniquement depuis rank 0
     */
    Rank0Only = 0,

    /**
     * Tous les ranks affichent avec préfixe
     * - Format: "[Rank 0/4] Step 100: t=0.5"
     * - Utile pour debugging
     */
    AllRanks = 1,

    /**
     * Réductions automatiques pour statistiques globales
     * - Les observers font des MPI_Allreduce
     * - Affiche les min/max/moyenne global
     */
    Reduced = 2,

    /**
     * Mode hybride: réduit pour les stats, rank 0 pour la sortie
     */
    Smart = 3
};

/**
 * @brief Mode d'initialisation MPI
 */
enum class MPICommMode : int {
    /**
     * Auto-initialisation MPI implicite
     */
    Auto = 0,

    /**
     * Utilisateur gère MPI_Init/MPI_Finalize
     */
    UserManaged = 1,

    /**
     * Utilisateur fournit un MPI_Comm custom
     */
    Custom = 2
};

/**
 * @brief Directions de frontière
 */
enum class Boundary : int {
    Left = 0,
    Right = 1,
    Bottom = 2,
    Top = 3
};

/**
 * @brief Statistiques de load balancing
 */
struct LoadBalanceStats {
    std::size_t total_cells = 0;      // Nombre total de cellules
    std::size_t min_cells = 0;        // Min cellules par rank
    std::size_t max_cells = 0;        // Max cellules par rank
    std::size_t avg_cells = 0;        // Moyenne cellules par rank
    float initial_ratio = 1.0f;       // Ratio max/avg initial
    float final_ratio = 1.0f;         // Ratio max/avg final
    float target_ratio = 1.1f;        // Ratio cible
    int num_rebalances = 0;           // Nombre de redistributions
    double total_comm_time = 0.0;     // Temps total en communication
    double comm_ratio = 0.0;          // Ratio temps comm / temps total
    int most_loaded_rank = 0;         // Rank le plus chargé
};

/**
 * @brief Configuration de communication MPI
 */
struct CommConfig {
    CommMode mode = CommMode::Synchronous;
    bool auto_comm = true;
    int halo_width = 1;
    bool use_gpu_direct = false;  // Auto-detect if Hybrid

    // Pour mode asynchrone
    bool enable_overlap = true;   // Recouvrement calcul/comm

    // Timeouts
    double comm_timeout = 30.0;   // Timeout en secondes
};

/**
 * @brief Configuration des observateurs MPI
 */
struct ObserverConfig {
    ObserverMode mode = ObserverMode::Rank0Only;

    // Pour mode Reduced
    bool enable_allreduce = true;
    int reduce_interval = 1;  // Réduire tous les N steps

    // Pour mode AllRanks
    bool show_rank_prefix = true;
    const char* rank_format = "[Rank %d/%d]";
};

/**
 * @brief Configuration MPI globale
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
