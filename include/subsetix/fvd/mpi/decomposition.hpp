// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#pragma once

#include <Kokkos_Core.hpp>
#include <array>
#include <optional>
#include "../system/concepts_v2.hpp"
#include "mpi_config.hpp"
#include "mpi_stub.hpp"

namespace subsetix::fvd::mpi {

// ============================================================================
// CARTESIAN 1D DECOMPOSITION
// ============================================================================

/**
 * @brief Decomposition along the X axis (vertical bands)
 *
 * Exemple avec 4 ranks:
 *   Rank 0 | Rank 1 | Rank 2 | Rank 3
 *
 * Neighbors: left (rank-1), right (rank+1), except at boundaries
 */
struct Cartesian1D {
    struct Config {
        int nx_global = 0;           // Global domain size
        int ny_global = 0;
        int padding = 1;             // Padding for ghost cells

        // Validation
        KOKKOS_FUNCTION
        bool validate() const {
            return nx_global > 0 && ny_global > 0 && padding >= 0;
        }
    };

    struct DecompositionInfo {
        int rank = 0;                // My rank
        int nranks = 1;              // Total number of ranks
        int nx_local = 0;            // Local size of my domain
        int ny_local = 0;
        int x_offset = 0;            // Position of my domain in the global
        int y_offset = 0;
        int left_neighbor = -1;      // Rank of my left neighbor (-1 if boundary)
        int right_neighbor = -1;     // Rank of my right neighbor (-1 if boundary)
    };

    /**
     * @brief Initialization of the decomposition
     *
     * @param cfg Decomposition configuration
     * @param comm MPI communicator
     * @return DecompositionInfo Information about the decomposition
     */
    static DecompositionInfo init(const Config& cfg, MPI_Comm comm);

    /**
     * @brief Find the neighbors
     *
     * @param info Decomposition information
     * @return std::array<int, 4> {left, right, bottom, top} (-1 if no neighbor)
     */
    static std::array<int, 4> find_neighbors(const DecompositionInfo& info);

    /**
     * @brief Check if a rank is on the global boundary
     *
     * @param info Decomposition information
     * @param side Direction to check
     * @return true If on the boundary
     */
    static bool is_on_boundary(const DecompositionInfo& info, Boundary side);
};

// ============================================================================
// CARTESIAN 2D DECOMPOSITION
// ============================================================================

/**
 * @brief Decomposition in 2D Cartesian grid
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
        int px = -1;                 // Number of ranks in X (-1 = auto)
        int py = -1;                 // Number of ranks in Y (-1 = auto)
        int padding = 1;

        KOKKOS_FUNCTION
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

        // Position in the grid
        int grid_x = 0;
        int grid_y = 0;
        int grid_nx = 1;
        int grid_ny = 1;

        // Neighbors: {left, right, bottom, top}
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
 * @brief Decomposition along a Hilbert/Morton curve
 *
 * Guarantees that cells close in space are on the same rank
 * or on neighboring ranks.
 */
struct SpaceFillingCurve {
    struct Config {
        int nx_global = 0;
        int ny_global = 0;

        enum CurveType {
            Morton,      // Z-order curve (simpler)
            Hilbert     // Hilbert curve (better locality)
        };
        CurveType curve_type = Hilbert;

        int order = 0;               // Order of the curve (-1 = auto)

        KOKKOS_FUNCTION
        bool validate() const {
            return nx_global > 0 && ny_global > 0;
        }
    };

    // Kokkos::View for cell -> rank mapping (device-accessible)
    struct DecompositionInfo {
        int rank = 0;
        int nranks = 1;
        int nx_local = 0;
        int ny_local = 0;
        int x_offset = 0;
        int y_offset = 0;

        // Maximum number of neighbors (fixed at compile-time)
        static constexpr int max_neighbors = 8;
        int num_neighbors = 0;
        std::array<int, max_neighbors> neighbors{};  // Fixed-size array

        /**
         * @brief Cell -> rank mapping
         *
         * @param ix Cell index in X
         * @param iy Cell index in Y
         * @param cfg Configuration
         * @return int Rank containing this cell
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
 * @brief Decomposition via Metis (graph partitioning)
 *
 * Allows an arbitrary number of neighbors per rank.
 * Optimal for complex geometries and AMR.
 */
struct MetisDecomposition {
    struct Config {
        // Explicit graph (optional) - uses Kokkos::View
        struct GraphInput {
            Kokkos::View<int*> adjacency;  // Adjacency list
            Kokkos::View<int*> offsets;     // Offsets in adjacency
            Kokkos::View<float*> weights;   // Edge weights (optional)
            int num_vertices = 0;
            int num_edges = 0;
        };
        std::optional<GraphInput> graph;

        // Geometry for automatic graph construction
        struct GeometryInput {
            int nx = 0;
            int ny = 0;
            int halo_width = 1;
        };
        std::optional<GeometryInput> geometry;

        // Metis options
        int nparts = 0;               // Number of partitions (0 = nranks)
        float imbalance = 1.05f;      // Imbalance tolerance
        int options[METIS_NOPTIONS];  // Metis options

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

        // Maximum number of neighbors (fixed at compile-time for GPU)
        static constexpr int max_neighbors = 16;
        int num_neighbors = 0;
        std::array<int, max_neighbors> neighbors{};  // Fixed-size array

        // For each neighbor: list of boundary cells
        // Uses Kokkos::View for device allocation
        Kokkos::View<int*[max_neighbors]> boundary_cells;  // boundary_cells[neighbor_rank][cell_idx]
        Kokkos::View<int[max_neighbors]> boundary_counts;  // Number of cells per neighbor
    };

    static DecompositionInfo init(const Config& cfg, MPI_Comm comm);
    static bool is_on_boundary(const DecompositionInfo& info, Boundary side);
};

// ============================================================================
// STATIC DECOMPOSITION (User-Defined)
// ============================================================================

/**
 * @brief Static decomposition defined by the user
 *
 * For cases where the user wants full control.
 */
struct StaticDecomposition {
    struct Config {
        // Kokkos::View to store domains (device-accessible)
        Kokkos::View<int*[4]> rank_domains;  // rank_domains[rank] = {x_min, x_max, y_min, y_max}
        int num_ranks = 0;

        KOKKOS_FUNCTION
        bool validate() const {
            return num_ranks > 0;
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
 * @brief Generic structure to store any decomposition info
 *
 * Uses Kokkos::View for device-accessible data.
 */
struct GenericDecompositionInfo {
    int rank = 0;
    int nranks = 1;
    int nx_local = 0;
    int ny_local = 0;
    int x_offset = 0;
    int y_offset = 0;

    // Neighbors - uses Kokkos::View with fixed size
    static constexpr int max_neighbors = 16;
    Kokkos::View<int*> neighbors;      // [num_neighbors]
    int num_neighbors = 0;

    // Decomposition type
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
