// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

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
 * @brief Information about the MPI topology
 *
 * Contains information about neighbors, boundaries, etc.
 */
template<typename Real = float>
struct TopologyInfo {
    int rank = 0;                    // Mon rank
    int nranks = 1;                  // Nombre total de ranks

    // Local domain
    int nx_local = 0;
    int ny_local = 0;
    int x_offset = 0;                // Position in the global domain
    int y_offset = 0;

    // Neighbors
    std::vector<int> neighbors;      // List of neighbor ranks
    std::vector<std::vector<int>> boundary_cells;  // Boundary cells per neighbor

    // Decomposition type
    GenericDecompositionInfo::Type decomp_type = GenericDecompositionInfo::Type::Cartesian1D;

    // For Cartesian grid
    struct {
        int grid_x = 0;
        int grid_y = 0;
        int grid_nx = 1;
        int grid_ny = 1;
    } cartesian;

    /**
     * @brief Return the number of neighbors
     */
    int num_neighbors() const {
        return static_cast<int>(neighbors.size());
    }

    /**
     * @brief Check if a rank is my neighbor
     *
     * @param other_rank Rank to check
     * @return true If it is a neighbor
     */
    bool is_neighbor(int other_rank) const {
        return std::find(neighbors.begin(), neighbors.end(), other_rank) != neighbors.end();
    }

    /**
     * @brief Check if I am on a global boundary
     *
     * @param side Direction to check
     * @return true If on the boundary
     */
    bool is_on_boundary(Boundary side) const;

    /**
     * @brief Return the boundary cells with a neighbor
     *
     * @param neighbor_rank Neighbor rank
     * @return const std::vector<int>@return const std::vector<int>@return const std::vector<int>& Liste des cellules List of cells frontières List of boundary cells
     */
    const std::vector<int>& boundary_cells_with(int neighbor_rank) const;
};

// ============================================================================
// TOPOLOGY QUERY
// ============================================================================

/**
 * @brief Interface to query the MPI topology
 *
 * Allows the user to ask: "what are my neighbors?",
 * "am I on a boundary?", etc.
 */
class TopologyQuery {
public:
    using Real = float;

    /**
     * @brief Constructor from decomposition info
     *
     * @param decomp_info Decomposition information
     * @param comm MPI communicator
     */
    template<typename DecompositionInfo>
    explicit TopologyQuery(const DecompositionInfo& decomp_info, MPI_Comm comm = MPI_COMM_WORLD);

    /**
     * @brief Default constructor (single rank)
     */
    TopologyQuery();

    // ========================================================================
    // BASIC QUERIES
    // ========================================================================

    /**
     * @brief Return my rank
     */
    int rank() const { return info_.rank; }

    /**
     * @brief Return the total number of ranks
     */
    int nranks() const { return info_.nranks; }

    /**
     * @brief Return true if I am rank 0
     */
    bool is_rank0() const { return info_.rank == 0; }

    // ========================================================================
    // NEIGHBOR QUERIES
    // ========================================================================

    /**
     * @brief Return the list of my neighbors
     *
     * @return const std::vector<int>@return const std::vector<int>& Vecteur des rangs voisins Vector of neighbor ranks
     */
    const std::vector<int>& neighbors() const {
        return info_.neighbors;
    }

    /**
     * @brief Return the number of neighbors
     */
    int num_neighbors() const {
        return info_.num_neighbors();
    }

    /**
     * @brief Check if a rank is my neighbor
     *
     * @param other_rank Rank to check
     * @return true If it is a neighbor
     */
    bool is_neighbor(int other_rank) const {
        return info_.is_neighbor(other_rank);
    }

    /**
     * @brief Return the boundary cells with a neighbor
     *
     * @param neighbor_rank Neighbor rank
     * @return const std::vector<int>@return const std::vector<int>& Liste des cellules List of cells
     */
    const std::vector<int>& boundary_cells_with(int neighbor_rank) const {
        return info_.boundary_cells_with(neighbor_rank);
    }

    // ========================================================================
    // DOMAIN QUERIES
    // ========================================================================

    /**
     * @brief Return the size of my local domain
     *
     * @return std::array<int, 2> {nx_local, ny_local}
     */
    std::array<int, 2> local_size() const {
        return {info_.nx_local, info_.ny_local};
    }

    /**
     * @brief Return the offset of my domain in the global domain
     *
     * @return std::array<int, 2> {x_offset, y_offset}
     */
    std::array<int, 2> local_offset() const {
        return {info_.x_offset, info_.y_offset};
    }

    /**
     * @brief Check if I am on a global boundary
     *
     * @param side Direction to check
     * @return true If on the boundary
     */
    bool is_on_boundary(Boundary side) const {
        return info_.is_on_boundary(side);
    }

    /**
     * @brief Return the directions where I am on a boundary
     *
     * @return std::vector<Boundary> List of directions
     */
    std::vector<Boundary> boundaries() const;

    // ========================================================================
    // CARTESIAN GRID QUERIES (si applicable)
    // ========================================================================

    /**
     * @brief Return my position in the Cartesian grid
     *
     * @return std::array<int, 2> {grid_x, grid_y}
     */
    std::array<int, 2> grid_position() const {
        return {info_.cartesian.grid_x, info_.cartesian.grid_y};
    }

    /**
     * @brief Return the size of the Cartesian grid
     *
     * @return std::array<int, 2> {grid_nx, grid_ny}
     */
    std::array<int, 2> grid_size() const {
        return {info_.cartesian.grid_nx, info_.cartesian.grid_ny};
    }

    /**
     * @brief Return the neighbor in a direction (Cartesian grid)
     *
     * @param side Direction
     * @return int Neighbor rank (-1 if no neighbor)
     */
    int cartesian_neighbor(Boundary side) const;

    // ========================================================================
    // UTILITY
    // ========================================================================

    /**
     * @brief Display the topology information
     */
    void print() const;

    /**
     * @brief Return the topology information
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
 * @brief Information about halo cells
 */
struct HaloInfo {
    int width = 1;                       // Halo width
    std::vector<int> send_ranks;         // Ranks to send to
    std::vector<int> recv_ranks;         // Ranks to receive from
    std::vector<std::vector<int>> send_cells;   // Cells to send per rank
    std::vector<std::vector<int>> recv_cells;   // Cells to receive per rank

    /**
     * @brief Return the total number of cells to send
     */
    std::size_t total_send_cells() const {
        std::size_t total = 0;
        for (const auto& cells : send_cells) {
            total += cells.size();
        }
        return total;
    }

    /**
     * @brief Return the total number of cells to receive
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
 * @brief Build halo information from the topology
 */
class HaloBuilder {
public:
    /**
     * @brief Build halo info for a decomposition
     *
     * @param topology Topology
     * @param halo_width Halo width
     * @return HaloInfo Halo information
     */
    static HaloInfo build(const TopologyQuery& topology, int halo_width = 1);

    /**
     * @brief Build halo info for Cartesian grid
     *
     * @param nx_local Local size in X
     * @param ny_local Local size in Y
     * @param neighbors Neighbors {left, right, bottom, top}
     * @param halo_width Halo width
     * @return HaloInfo Halo information
     */
    static HaloInfo build_cartesian(
        int nx_local, int ny_local,
        const std::array<int, 4>& neighbors,
        int halo_width = 1
    );
};

} // namespace subsetix::fvd::mpi
