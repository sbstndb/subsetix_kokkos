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
 * @brief Concept pour les politiques de décomposition de domaine
 *
 * Une politique de décomposition doit fournir:
 * - Config: Structure de configuration (POD, GPU-safe)
 * - DecompositionInfo: Structure d'information retournée
 * - init(): Fonction statique pour initialiser la décomposition
 * - find_neighbors(): Fonction pour trouver les voisins
 * - is_on_boundary(): Fonction pour vérifier si sur une frontière
 */
template<typename T>
concept DecompositionPolicy = requires {
    // Type de configuration (doit être trivial pour GPU)
    typename T::Config;

    // Type d'information retourné
    typename T::DecompositionInfo;

    // Initialisation de la décomposition
    { T::init(std::declval<const typename T::Config&>(), std::declval<MPI_Comm>()) }
        -> std::same_as<typename T::DecompositionInfo>;

    // Trouver les voisins d'un rank
    { T::find_neighbors(std::declval<const typename T::DecompositionInfo&>()) }
        -> std::convertible_to<std::array<int, 4>>;

    // Vérifier si sur une frontière
    { T::is_on_boundary(std::declval<const typename T::DecompositionInfo&>(),
                       std::declval<Boundary>()) }
        -> std::same_as<bool>;
};

// ============================================================================
// LOAD BALANCE POLICY CONCEPT
// ============================================================================

/**
 * @brief Concept pour les politiques de load balancing
 *
 * Doit fournir une fonction device-friendly pour calculer le coût d'une cellule.
 */
template<typename T, typename System>
concept LoadBalancePolicy = requires {
    typename T::template Config<System>;

    // Fonction pour calculer le coût d'une cellule (doit être KOKKOS_FUNCTION)
    { T::template compute_cell_cost<System>(
        std::declval<const typename System::Conserved&>(),
        std::declval<const typename System::Primitive&>(),
        std::declval<const typename System::RealType&>(),
        std::declval<const typename System::RealType&>(),
        std::declval<int>()
    )} -> std::convertible_to<float>;
};

// ============================================================================
// COMM POLICY CONCEPT (pour communication entre ranks)
// ============================================================================

/**
 * @brief Concept pour les politiques de communication
 */
template<typename T>
concept CommPolicy = requires {
    // Type de configuration
    typename T::Config;

    // Méthode pour échanger les halos
    { T::exchange_halos(std::declval<const typename T::Config&>()) }
        -> std::same_as<void>;

    // Méthode pour barrier
    { T::barrier(std::declval<const typename T::Config&>()) }
        -> std::same_as<void>;
};

} // namespace subsetix::fvd::mpi
