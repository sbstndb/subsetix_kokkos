#pragma once

/**
 * @file fvd_mpi.hpp
 *
 * @brief Header d'intégration MPI pour FVD
 *
 * Ce header inclut tous les composants nécessaires pour utiliser
 * l'API MPI de Subsetix FVD.
 *
 * Usage:
 *   #include <subsetix/fvd/mpi/fvd_mpi.hpp>
 *
 *   using namespace subsetix::fvd;
 *   using namespace subsetix::fvd::solver;
 *   using namespace subsetix::fvd::mpi;
 *
 *   auto solver = EulerSolverRK3::builder(1000, 500)
 *       .with_decomposition<mpi::Cartesian2D>({...})
 *       .with_mpi<MPIComm::Auto>()
 *       .build();
 */

// ============================================================================
// CONCEPTS
// ============================================================================
#include "mpi_concepts.hpp"

// ============================================================================
// CONFIGURATION
// ============================================================================
#include "mpi_config.hpp"

// ============================================================================
// DECOMPOSITION
// ============================================================================
#include "decomposition.hpp"

// ============================================================================
// TOPOLOGY
// ============================================================================
#include "topology.hpp"

// ============================================================================
// COMMUNICATION
// ============================================================================
#include "comm_manager.hpp"

// ============================================================================
// OBSERVERS
// ============================================================================
#include "observer_mpi.hpp"

// ============================================================================
// LOAD BALANCING
// ============================================================================
#include "load_balance.hpp"

namespace subsetix::fvd::mpi {

// ============================================================================
// MPI INITIALIZATION
// ============================================================================

/**
 * @brief Initialiseur MPI implicite
 *
 * Si l'utilisateur n'a pas appelé MPI_Init, cette classe
 * l'initialise automatiquement au premier usage.
 */
class MPIInitializer {
public:
    /**
     * @brief Initialise MPI si nécessaire
     */
    static void initialize();

    /**
     * @brief Finalize MPI si nous l'avons initialisé
     */
    static void finalize();

    /**
     * @brief Vérifie si MPI est initialisé
     */
    static bool is_initialized();

    /**
     * @brief Retourne le communicateur par défaut
     */
    static MPI_Comm default_comm();

private:
    static bool we_initialized_;
    static MPI_Comm default_comm_;
};

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

/**
 * @brief Créer une configuration MPI par défaut
 *
 * Configuration pour usage standard:
 * - Détection automatique de MPI
 * - Décomposition cartésienne 2D auto
 * - Communications automatiques
 * - Observateurs Rank0Only
 */
inline MPIConfig default_mpi_config() {
    MPIConfig cfg;
    cfg.comm_mode = MPICommMode::Auto;
    cfg.comm.mode = CommMode::Synchronous;
    cfg.comm.auto_comm = true;
    cfg.comm.halo_width = 1;
    cfg.observer.mode = ObserverMode::Rank0Only;
    cfg.enable_auto_load_balance = false;
    cfg.load_balance_tolerance = 1.1f;
    return cfg;
}

/**
 * @brief Créer une configuration pour communication asynchrone
 */
inline MPIConfig async_mpi_config() {
    auto cfg = default_mpi_config();
    cfg.comm.mode = CommMode::Asynchronous;
    cfg.comm.enable_overlap = true;
    return cfg;
}

/**
 * @brief Créer une configuration pour GPU-Direct
 */
inline MPIConfig gpu_direct_mpi_config() {
    auto cfg = default_mpi_config();
    cfg.comm.mode = CommMode::GPUDirect;
    cfg.comm.use_gpu_direct = true;  // Auto-detect with fallback
    return cfg;
}

/**
 * @brief Créer une configuration pour mode réduit (stats globales)
 */
inline MPIConfig reduced_observer_config() {
    auto cfg = default_mpi_config();
    cfg.observer.mode = ObserverMode::Reduced;
    cfg.observer.enable_allreduce = true;
    return cfg;
}

/**
 * @brief Créer une configuration pour load balancing
 */
template<typename System>
MPIConfig load_balance_config(float max_imbalance = 1.1f) {
    auto cfg = default_mpi_config();
    cfg.enable_auto_load_balance = true;
    cfg.load_balance_tolerance = max_imbalance;
    return cfg;
}

// ============================================================================
// DECOMPOSITION BUILDERS
// ============================================================================

/**
 * @brief Créer une configuration de décomposition cartésienne 1D
 */
inline Cartesian1D::Config cartesian_1d(int nx_global, int ny_global, int padding = 1) {
    return Cartesian1D::Config{nx_global, ny_global, padding};
}

/**
 * @brief Créer une configuration de décomposition cartésienne 2D
 */
inline Cartesian2D::Config cartesian_2d(int nx_global, int ny_global,
                                       int px = -1, int py = -1, int padding = 1) {
    return Cartesian2D::Config{nx_global, ny_global, px, py, padding};
}

/**
 * @brief Créer une configuration de décomposition Metis
 */
inline MetisDecomposition::Config metis_decomposition(
    int nx, int ny, int halo_width = 1, float imbalance = 1.05f
) {
    MetisDecomposition::Config cfg;
    cfg.geometry = MetisDecomposition::GeometryInput{nx, ny, halo_width};
    cfg.imbalance = imbalance;
    cfg.nparts = 0;  // Auto = nranks
    return cfg;
}

} // namespace subsetix::fvd::mpi
