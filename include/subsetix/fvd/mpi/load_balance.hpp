#pragma once

#include <Kokkos_Core.hpp>
#include "../system/concepts_v2.hpp"
#include "mpi_config.hpp"

namespace subsetix::fvd::mpi {

// ============================================================================
// CELL COUNT LOAD BALANCE
// ============================================================================

/**
 * @brief Load balancing par nombre de cellules
 *
 * Stratégie la plus simple: chaque rank doit avoir ~N_cells_total / N_ranks
 */
struct CellCountLoadBalance {
    template<typename System>
    struct Config {
        float max_imbalance = 1.1f;  // Tolérance de déséquilibre
        int check_interval = 100;    // Vérifier tous les N steps

        bool validate() const {
            return max_imbalance > 1.0f && check_interval > 0;
        }
    };

    /**
     * @brief Fonction device-friendly pour calculer le coût d'une cellule
     *
     * Toutes les cellules ont le même coût.
     *
     * @param U Variables conservatives
     * @param q Variables primitives
     * @param grad_rho_x Gradient de densité en X
     * @param grad_rho_y Gradient de densité en Y
     * @param level Niveau de raffinement
     * @return Coût de la cellule
     */
    template<typename System>
    KOKKOS_FUNCTION static float compute_cell_cost(
        const typename System::Conserved& U,
        const typename System::Primitive& q,
        const typename System::RealType grad_rho_x,
        const typename System::RealType grad_rho_y,
        int level
    ) {
        (void)U; (void)q; (void)grad_rho_x; (void)grad_rho_y; (void)level;
        return 1.0f;  // Toutes les cellules ont le même coût
    }
};

// ============================================================================
// LEVEL WEIGHTED LOAD BALANCE
// ============================================================================

/**
 * @brief Load basé sur le niveau de raffinement
 *
 * Les cellules fines sont plus coûteuses (plus de travail par cellule)
 */
struct LevelWeightedLoadBalance {
    template<typename System>
    struct Config {
        float level_weight = 2.0f;  // Une cellule de niveau l+1 coûte level_weight fois plus
        float max_imbalance = 1.1f;
        int check_interval = 100;

        bool validate() const {
            return level_weight > 1.0f && max_imbalance > 1.0f;
        }
    };

    template<typename System>
    KOKKOS_FUNCTION static float compute_cell_cost(
        const typename System::Conserved& U,
        const typename System::Primitive& q,
        const typename System::RealType grad_rho_x,
        const typename System::RealType grad_rho_y,
        int level
    ) {
        (void)U; (void)q; (void)grad_rho_x; (void)grad_rho_y;
        // Coût exponentiel avec le niveau
        return Kokkos::pow(level_weight, level);
    }

    template<typename System>
    static constexpr float level_weight = 2.0f;
};

// ============================================================================
// PHYSICS WEIGHTED LOAD BALANCE
// ============================================================================

/**
 * @brief Load basé sur l'activité physique (gradient, etc.)
 *
 * Les régions actives (chocs, gradients forts) sont plus coûteuses
 */
struct PhysicsWeightedLoadBalance {
    template<typename System>
    struct Config {
        float gradient_weight = 1.0f;
        float shock_weight = 2.0f;
        float base_cost = 1.0f;
        float max_imbalance = 1.1f;
        int check_interval = 100;

        bool validate() const {
            return gradient_weight >= 0.0f && shock_weight >= 1.0f;
        }
    };

    template<typename System>
    KOKKOS_FUNCTION static float compute_cell_cost(
        const typename System::Conserved& U,
        const typename System::Primitive& q,
        const typename System::RealType grad_rho_x,
        const typename System::RealType grad_rho_y,
        int level
    ) {
        float cost = base_cost;

        // Coût basé sur les gradients de densité
        using Real = typename System::RealType;
        Real grad_mag = Kokkos::sqrt(grad_rho_x * grad_rho_x + grad_rho_y * grad_rho_y);
        cost += gradient_weight * grad_mag;

        // Bonus pour les chocs (compression forte = divergence négative)
        // Pour simplifier, on utilise le gradient de vitesse
        // Dans une implémentation complète, on calculerait div(v)
        if (grad_mag > 0.5f) {  // Seuil arbitraire pour "activité forte"
            cost *= shock_weight;
        }

        // Bonus pour le raffinement
        cost *= Kokkos::pow(2.0f, level);

        return cost;
    }
};

// ============================================================================
// CUSTOM LOAD BALANCE (User-Defined, Compile-Time)
// ============================================================================

/**
 * @brief Load personnalisé par l'utilisateur (compile-time)
 *
 * L'utilisateur fournit un lambda qui devient une fonction device via KOKKOS_LAMBDA
 *
 * Exemple d'utilisation:
 * @code
 * auto my_cost_func = KOKKOS_LAMBDA(
 *     const Euler2D<float>::Conserved& U,
 *     const Euler2D<float>::Primitive& q,
 *     float grad_rho_x,
 *     float grad_rho_y,
 *     int level
 * ) -> float {
 *     // Votre logique de coût ici
 *     return some_cost;
 * };
 *
 * solver.with_load_balancing<CustomLoadBalance<System, decltype(my_cost_func)>>({
 *     .cost_func = my_cost_func,
 *     .max_imbalance = 1.15f
 * });
 * @endcode
 */
template<typename System, typename CostFunc>
struct CustomLoadBalance {
    struct Config {
        CostFunc cost_func;
        float max_imbalance = 1.1f;
        int check_interval = 100;

        bool validate() const {
            return max_imbalance > 1.0f && check_interval > 0;
        }
    };

    template<typename... Args>
    KOKKOS_FUNCTION static float compute_cell_cost(Args&&... args) {
        return CostFunc{}(std::forward<Args>(args)...);
    }
};

// ============================================================================
// LOAD BALANCE POLICY TRAITS
// ============================================================================

/**
 * @brief Traits pour les politiques de load balancing
 *
 * Permet d'interroger les propriétés d'une politique à compile time.
 */
template<typename LoadBalancePolicy, typename System>
struct LoadBalanceTraits {
    using Config = typename LoadBalancePolicy::template Config<System>;

    static constexpr bool has_custom_cost_function = requires {
        { LoadBalancePolicy::template compute_cell_cost<System>(
            std::declval<const typename System::Conserved&>(),
            std::declval<const typename System::Primitive&>(),
            std::declval<const typename System::RealType>(),
            std::declval<const typename System::RealType>(),
            std::declval<int>()
        )} -> std::convertible_to<float>;
    };
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * @brief Créer une configuration de load balancing par défaut
 *
 * @tparam LoadBalancePolicy Politique de load balancing
 * @tparam System Système (Euler2D, etc.)
 * @return Config Configuration par défaut
 */
template<template<typename> class LoadBalancePolicy, typename System>
typename LoadBalancePolicy<System>::Config default_load_balance_config() {
    return typename LoadBalancePolicy<System>::Config{};
}

/**
 * @brief Valider une configuration de load balancing
 *
 * @tparam LoadBalancePolicy Politique de load balancing
 * @tparam System Système
 * @param cfg Configuration à valider
 * @return true Si valide
 */
template<template<typename> class LoadBalancePolicy, typename System>
bool validate_load_balance_config(const typename LoadBalancePolicy<System>::Config& cfg) {
    return cfg.validate();
}

} // namespace subsetix::fvd::mpi
