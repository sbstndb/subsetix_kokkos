#pragma once

#include <Kokkos_Core.hpp>
#include <array>
#include <vector>
#include "decomposition.hpp"
#include "mpi_stub.hpp"

namespace subsetix::fvd::mpi {

// ============================================================================
// TOPOLOGY INFO
// ============================================================================

/**
 * @brief Information sur la topologie MPI
 *
 * Contient les informations sur les voisins, les frontières, etc.
 */
template<typename Real = float>
struct TopologyInfo {
    int rank = 0;                    // Mon rank
    int nranks = 1;                  // Nombre total de ranks

    // Domaine local
    int nx_local = 0;
    int ny_local = 0;
    int x_offset = 0;                // Position dans le domaine global
    int y_offset = 0;

    // Voisins
    std::vector<int> neighbors;      // Liste des rangs voisins
    std::vector<std::vector<int>> boundary_cells;  // Cellules frontière par voisin

    // Type de décomposition
    GenericDecompositionInfo::Type decomp_type = GenericDecompositionInfo::Type::Cartesian1D;

    // Pour grille cartésienne
    struct {
        int grid_x = 0;
        int grid_y = 0;
        int grid_nx = 1;
        int grid_ny = 1;
    } cartesian;

    /**
     * @brief Retourne le nombre de voisins
     */
    int num_neighbors() const {
        return static_cast<int>(neighbors.size());
    }

    /**
     * @brief Vérifie si un rank est mon voisin
     *
     * @param other_rank Rank à vérifier
     * @return true Si c'est un voisin
     */
    bool is_neighbor(int other_rank) const {
        return std::find(neighbors.begin(), neighbors.end(), other_rank) != neighbors.end();
    }

    /**
     * @brief Vérifie si je suis sur une frontière globale
     *
     * @param side Direction à vérifier
     * @return true Si sur la frontière
     */
    bool is_on_boundary(Boundary side) const;

    /**
     * @brief Retourne les cellules frontière avec un voisin
     *
     * @param neighbor_rank Rang du voisin
     * @return const std::vector<int>& Liste des cellules frontières
     */
    const std::vector<int>& boundary_cells_with(int neighbor_rank) const;
};

// ============================================================================
// TOPOLOGY QUERY
// ============================================================================

/**
 * @brief Interface pour interroger la topologie MPI
 *
 * Permet à l'utilisateur de demander: "quels sont mes voisins ?",
 * "suis-je sur une frontière ?", etc.
 */
class TopologyQuery {
public:
    using Real = float;

    /**
     * @brief Constructeur à partir d'une info de décomposition
     *
     * @param decomp_info Information de décomposition
     * @param comm Communicateur MPI
     */
    template<typename DecompositionInfo>
    explicit TopologyQuery(const DecompositionInfo& decomp_info, MPI_Comm comm = MPI_COMM_WORLD);

    /**
     * @brief Constructeur par défaut (single rank)
     */
    TopologyQuery();

    // ========================================================================
    // BASIC QUERIES
    // ========================================================================

    /**
     * @brief Retourne mon rank
     */
    int rank() const { return info_.rank; }

    /**
     * @brief Retourne le nombre total de ranks
     */
    int nranks() const { return info_.nranks; }

    /**
     * @brief Retourne vrai si je suis le rang 0
     */
    bool is_rank0() const { return info_.rank == 0; }

    // ========================================================================
    // NEIGHBOR QUERIES
    // ========================================================================

    /**
     * @brief Retourne la liste de mes voisins
     *
     * @return const std::vector<int>& Vecteur des rangs voisins
     */
    const std::vector<int>& neighbors() const {
        return info_.neighbors;
    }

    /**
     * @brief Retourne le nombre de voisins
     */
    int num_neighbors() const {
        return info_.num_neighbors();
    }

    /**
     * @brief Vérifie si un rank est mon voisin
     *
     * @param other_rank Rank à vérifier
     * @return true Si c'est un voisin
     */
    bool is_neighbor(int other_rank) const {
        return info_.is_neighbor(other_rank);
    }

    /**
     * @brief Retourne les cellules frontière avec un voisin
     *
     * @param neighbor_rank Rang du voisin
     * @return const std::vector<int>& Liste des cellules
     */
    const std::vector<int>& boundary_cells_with(int neighbor_rank) const {
        return info_.boundary_cells_with(neighbor_rank);
    }

    // ========================================================================
    // DOMAIN QUERIES
    // ========================================================================

    /**
     * @brief Retourne la taille de mon domaine local
     *
     * @return std::array<int, 2> {nx_local, ny_local}
     */
    std::array<int, 2> local_size() const {
        return {info_.nx_local, info_.ny_local};
    }

    /**
     * @brief Retourne l'offset de mon domaine dans le domaine global
     *
     * @return std::array<int, 2> {x_offset, y_offset}
     */
    std::array<int, 2> local_offset() const {
        return {info_.x_offset, info_.y_offset};
    }

    /**
     * @brief Vérifie si je suis sur une frontière globale
     *
     * @param side Direction à vérifier
     * @return true Si sur la frontière
     */
    bool is_on_boundary(Boundary side) const {
        return info_.is_on_boundary(side);
    }

    /**
     * @brief Retourne les directions où je suis sur une frontière
     *
     * @return std::vector<Boundary> Liste des directions
     */
    std::vector<Boundary> boundaries() const;

    // ========================================================================
    // CARTESIAN GRID QUERIES (si applicable)
    // ========================================================================

    /**
     * @brief Retourne ma position dans la grille cartésienne
     *
     * @return std::array<int, 2> {grid_x, grid_y}
     */
    std::array<int, 2> grid_position() const {
        return {info_.cartesian.grid_x, info_.cartesian.grid_y};
    }

    /**
     * @brief Retourne la taille de la grille cartésienne
     *
     * @return std::array<int, 2> {grid_nx, grid_ny}
     */
    std::array<int, 2> grid_size() const {
        return {info_.cartesian.grid_nx, info_.cartesian.grid_ny};
    }

    /**
     * @brief Retourne le voisin dans une direction (grille cartésienne)
     *
     * @param side Direction
     * @return int Rank du voisin (-1 si pas de voisin)
     */
    int cartesian_neighbor(Boundary side) const;

    // ========================================================================
    // UTILITY
    // ========================================================================

    /**
     * @brief Affiche les informations de topologie
     */
    void print() const;

    /**
     * @brief Retourne les informations de topologie
     */
    const TopologyInfo<Real>& info() const {
        return info_;
    }

private:
    TopologyInfo<Real> info_;
    MPI_Comm comm_ = MPI_COMM_WORLD;
};

// ============================================================================
// HALO INFO
// ============================================================================

/**
 * @brief Information sur les halo cells
 */
struct HaloInfo {
    int width = 1;                       // Largeur du halo
    std::vector<int> send_ranks;         // Rangs auxquels envoyer
    std::vector<int> recv_ranks;         // Rangs desquels recevoir
    std::vector<std::vector<int>> send_cells;   // Cellules à envoyer par rang
    std::vector<std::vector<int>> recv_cells;   // Cellules à recevoir par rang

    /**
     * @brief Retourne le nombre total de cellules à envoyer
     */
    std::size_t total_send_cells() const {
        std::size_t total = 0;
        for (const auto& cells : send_cells) {
            total += cells.size();
        }
        return total;
    }

    /**
     * @brief Retourne le nombre total de cellules à recevoir
     */
    std::size_t total_recv_cells() const {
        std::size_t total = 0;
        for (const auto& cells : recv_cells) {
            total += cells.size();
        }
        return total;
    }
};

// ============================================================================
// HALO BUILDER
// ============================================================================

/**
 * @brief Construit les informations de halo à partir de la topologie
 */
class HaloBuilder {
public:
    /**
     * @brief Construit les infos de halo pour une décomposition
     *
     * @param topology Topologie
     * @param halo_width Largeur du halo
     * @return HaloInfo Informations de halo
     */
    static HaloInfo build(const TopologyQuery& topology, int halo_width = 1);

    /**
     * @brief Construit les infos de halo pour grille cartésienne
     *
     * @param nx_local Taille locale en X
     * @param ny_local Taille locale en Y
     * @param neighbors Voisins {left, right, bottom, top}
     * @param halo_width Largeur du halo
     * @return HaloInfo Informations de halo
     */
    static HaloInfo build_cartesian(
        int nx_local, int ny_local,
        const std::array<int, 4>& neighbors,
        int halo_width = 1
    );
};

} // namespace subsetix::fvd::mpi
