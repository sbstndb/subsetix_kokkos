#pragma once

#include <Kokkos_Core.hpp>
#include <array>
#include <optional>
#include <vector>
#include "../system/concepts_v2.hpp"
#include "mpi_config.hpp"
#include "mpi_stub.hpp"

namespace subsetix::fvd::mpi {

// ============================================================================
// CARTESIAN 1D DECOMPOSITION
// ============================================================================

/**
 * @brief Décomposition selon l'axe X (bandes verticales)
 *
 * Exemple avec 4 ranks:
 *   Rank 0 | Rank 1 | Rank 2 | Rank 3
 *
 * Voisins: left (rank-1), right (rank+1), sauf aux frontières
 */
struct Cartesian1D {
    struct Config {
        int nx_global = 0;           // Taille globale du domaine
        int ny_global = 0;
        int padding = 1;             // Padding pour ghosts cells

        // Validation
        bool validate() const {
            return nx_global > 0 && ny_global > 0 && padding >= 0;
        }
    };

    struct DecompositionInfo {
        int rank = 0;                // Mon rank
        int nranks = 1;              // Nombre total de ranks
        int nx_local = 0;            // Taille locale de mon domaine
        int ny_local = 0;
        int x_offset = 0;            // Position de mon domaine dans le global
        int y_offset = 0;
        int left_neighbor = -1;      // Rank de mon voisin gauche (-1 si bord)
        int right_neighbor = -1;     // Rank de mon voisin droite (-1 si bord)
    };

    /**
     * @brief Initialisation de la décomposition
     *
     * @param cfg Configuration de la décomposition
     * @param comm Communicateur MPI
     * @return DecompositionInfo Information sur la décomposition
     */
    static DecompositionInfo init(const Config& cfg, MPI_Comm comm);

    /**
     * @brief Trouver les voisins
     *
     * @param info Information de décomposition
     * @return std::array<int, 4> {left, right, bottom, top} (-1 si pas de voisin)
     */
    static std::array<int, 4> find_neighbors(const DecompositionInfo& info);

    /**
     * @brief Vérifier si un rank est sur la frontière globale
     *
     * @param info Information de décomposition
     * @param side Direction à vérifier
     * @return true Si sur la frontière
     */
    static bool is_on_boundary(const DecompositionInfo& info, Boundary side);
};

// ============================================================================
// CARTESIAN 2D DECOMPOSITION
// ============================================================================

/**
 * @brief Décomposition en grille cartésienne 2D
 *
 * Exemple avec 4 ranks (grille 2x2):
 *   Rank 0 | Rank 1
 *   -------+-------
 *   Rank 2 | Rank 3
 *
 * Voisins: left, right, top, bottom
 */
struct Cartesian2D {
    struct Config {
        int nx_global = 0;
        int ny_global = 0;
        int px = -1;                 // Nombre de ranks en X (-1 = auto)
        int py = -1;                 // Nombre de ranks en Y (-1 = auto)
        int padding = 1;

        bool validate() const {
            return nx_global > 0 && ny_global > 0 && padding >= 0;
        }
    };

    struct DecompositionInfo {
        int rank = 0;
        int nranks = 1;
        int nx_local = 0;
        int ny_local = 0;
        int x_offset = 0;
        int y_offset = 0;

        // Position dans la grille
        int grid_x = 0;
        int grid_y = 0;
        int grid_nx = 1;
        int grid_ny = 1;

        // Voisins: {left, right, bottom, top}
        std::array<int, 4> neighbors{-1, -1, -1, -1};
    };

    static DecompositionInfo init(const Config& cfg, MPI_Comm comm);
    static std::array<int, 4> find_neighbors(const DecompositionInfo& info);
    static bool is_on_boundary(const DecompositionInfo& info, Boundary side);
};

// ============================================================================
// SPACE-FILLING CURVE DECOMPOSITION
// ============================================================================

/**
 * @brief Décomposition selon une courbe de Hilbert/Morton
 *
 * Garantit que les cellules proches dans l'espace sont sur le même rank
 * ou sur des ranks voisins.
 */
struct SpaceFillingCurve {
    struct Config {
        int nx_global = 0;
        int ny_global = 0;

        enum CurveType {
            Morton,      // Z-order curve (plus simple)
            Hilbert     // Hilbert curve (meilleure localité)
        };
        CurveType curve_type = Hilbert;

        int order = 0;               // Ordre de la courbe (-1 = auto)

        bool validate() const {
            return nx_global > 0 && ny_global > 0;
        }
    };

    struct DecompositionInfo {
        int rank = 0;
        int nranks = 1;
        int nx_local = 0;
        int ny_local = 0;
        int x_offset = 0;
        int y_offset = 0;

        // Liste des voisins (nombre arbitraire)
        std::vector<int> neighbors;

        /**
         * @brief Mapping cellule -> rank
         *
         * @param ix Indice de cellule en X
         * @param iy Indice de cellule en Y
         * @param cfg Configuration
         * @return int Rank contenant cette cellule
         */
        static int cell_to_rank(int ix, int iy, const Config& cfg);
    };

    static DecompositionInfo init(const Config& cfg, MPI_Comm comm);
    static bool is_on_boundary(const DecompositionInfo& info, Boundary side);
};

// ============================================================================
// METIS DECOMPOSITION (Nombre Arbitraire de Voisins)
// ============================================================================

/**
 * @brief Décomposition via Metis (graph partitioning)
 *
 * Permet un nombre arbitraire de voisins par rank.
 * Optimal pour des géométries complexes et AMR.
 */
struct MetisDecomposition {
    struct Config {
        // Graphe explicite (optionnel)
        struct GraphInput {
            std::vector<int> adjacency;  // Liste d'adjacence
            std::vector<int> offsets;     // Offsets dans adjacency
            std::vector<float> weights;  // Poids des arêtes (optionnel)
        };
        std::optional<GraphInput> graph;

        // Géométrie pour construction auto du graphe
        struct GeometryInput {
            int nx = 0;
            int ny = 0;
            int halo_width = 1;
        };
        std::optional<GeometryInput> geometry;

        // Options Metis
        int nparts = 0;               // Nombre de partitions (0 = nranks)
        float imbalance = 1.05f;      // Tolérance de déséquilibre
        int options[METIS_NOPTIONS];  // Options Metis

        bool validate() const {
            return graph.has_value() || geometry.has_value();
        }
    };

    struct DecompositionInfo {
        int rank = 0;
        int nranks = 1;
        int nx_local = 0;
        int ny_local = 0;
        int x_offset = 0;
        int y_offset = 0;

        // Nombre arbitraire de voisins
        std::vector<int> neighbors;

        // Pour chaque voisin: liste des cellules frontière
        std::vector<std::vector<int>> boundary_cells;
    };

    static DecompositionInfo init(const Config& cfg, MPI_Comm comm);
    static bool is_on_boundary(const DecompositionInfo& info, Boundary side);
};

// ============================================================================
// STATIC DECOMPOSITION (User-Defined)
// ============================================================================

/**
 * @brief Décomposition statique définie par l'utilisateur
 *
 * Pour les cas où l'utilisateur veut un contrôle total.
 */
struct StaticDecomposition {
    struct Config {
        // Tableau rank -> [x_min, x_max, y_min, y_max]
        std::vector<std::array<int, 4>> rank_domains;

        bool validate() const {
            return !rank_domains.empty();
        }
    };

    struct DecompositionInfo {
        int rank = 0;
        int nranks = 1;
        int nx_local = 0;
        int ny_local = 0;
        int x_offset = 0;
        int y_offset = 0;
    };

    static DecompositionInfo init(const Config& cfg, MPI_Comm comm);
    static bool is_on_boundary(const DecompositionInfo& info, Boundary side);
};

// ============================================================================
// GENERIC DECOMPOSITION INFO (Type-erased)
// ============================================================================

/**
 * @brief Structure générique pour stocker n'importe quelle info de décomposition
 */
struct GenericDecompositionInfo {
    int rank = 0;
    int nranks = 1;
    int nx_local = 0;
    int ny_local = 0;
    int x_offset = 0;
    int y_offset = 0;
    std::vector<int> neighbors;  // Liste des voisins

    // Type de décomposition
    enum class Type {
        Cartesian1D,
        Cartesian2D,
        SpaceFilling,
        Metis,
        Static
    };
    Type type = Type::Cartesian1D;
};

} // namespace subsetix::fvd::mpi
