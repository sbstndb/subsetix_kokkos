// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#pragma once

#include <experimental/subsetix/csr/set_algebra/v3.hpp>
#include <experimental/subsetix/csr/detail/utils.hpp>
#include <vector>
#include <algorithm>

namespace experimental::subsetix::csr::successive::workspace {

// ============================================================================
// IntersectionWorkspace - Reusable buffer pool for successive intersection
// ============================================================================

/**
 * @brief Workspace for successive mesh intersection operations.
 *
 * This workspace pre-allocates temporary buffers used during successive
 * intersection operations, allowing memory reuse across iterations.
 *
 * The workspace uses a ping-pong pattern between two temporary meshes,
 * avoiding repeated allocations for intermediate results.
 *
 * @tparam DIM Dimension (2 or 3)
 * @tparam MemorySpace Kokkos memory space (e.g., Kokkos::DefaultExecutionSpace::memory_space)
 */
template<int DIM, class MemorySpace>
struct IntersectionWorkspace {
  // Mesh type for this dimension and memory space
  using MeshType = v3::Mesh<DIM, MemorySpace, int32_t, std::size_t>;
  using CoordType = int32_t;
  using IndexType = std::size_t;

  // Temporary mesh buffers (ping-pong)
  MeshType temp_mesh_0;
  MeshType temp_mesh_1;

  // Index buffers for row mapping
  Kokkos::View<int*, MemorySpace> row_index_buf_0;
  Kokkos::View<int*, MemorySpace> row_index_buf_1;

  // Flag buffer for marking matching rows
  Kokkos::View<int*, MemorySpace> flags_buf;

  // Positions buffer for scan operations
  Kokkos::View<std::size_t*, MemorySpace> positions_buf;

  // Capacity tracking
  std::size_t max_rows = 0;
  std::size_t max_intervals = 0;
  double growth_factor = 1.5;

  /**
   * @brief Ensure workspace has sufficient capacity for the given mesh sizes.
   *
   * If current capacity is insufficient, buffers are grown by growth_factor.
   * Existing data is NOT preserved (for scratch buffers).
   *
   * @param rows Required number of rows
   * @param intervals Required number of intervals
   * @return true if buffers were (re)allocated, false if capacity was sufficient
   */
  bool ensure_capacity(std::size_t rows, std::size_t intervals) {
    bool reallocated = false;

    // Check if we need to grow
    if (rows > max_rows || intervals > max_intervals) {
      // Compute new capacities with growth factor
      std::size_t new_rows = max_rows;
      std::size_t new_intervals = max_intervals;

      while (new_rows < rows) {
        new_rows = static_cast<std::size_t>(new_rows * growth_factor);
        if (new_rows == 0) new_rows = std::max(rows, static_cast<std::size_t>(64));
        else new_rows = std::max(new_rows, rows);
      }

      while (new_intervals < intervals) {
        new_intervals = static_cast<std::size_t>(new_intervals * growth_factor);
        if (new_intervals == 0) new_intervals = std::max(intervals, static_cast<std::size_t>(256));
        else new_intervals = std::max(new_intervals, intervals);
      }

      // Reallocate buffers with new capacities
      reallocate_buffers(new_rows, new_intervals);
      reallocated = true;
    }

    return reallocated;
  }

  /**
   * @brief Reallocate all buffers with the specified capacities.
   *
   * This method performs the actual buffer allocation. Previous contents
   * are discarded (scratch buffers).
   *
   * @param rows New row capacity
   * @param intervals New interval capacity
   */
  void reallocate_buffers(std::size_t rows, std::size_t intervals) {
    using RowKey = typename MeshType::RowKey;
    using Interval = csr::Interval<CoordType>;

    // Reallocate index buffers
    row_index_buf_0 = Kokkos::View<int*, MemorySpace>("ws_row_idx_0", rows);
    row_index_buf_1 = Kokkos::View<int*, MemorySpace>("ws_row_idx_1", rows);

    // Reallocate flag buffer
    flags_buf = Kokkos::View<int*, MemorySpace>("ws_flags", rows);

    // Reallocate positions buffer
    positions_buf = Kokkos::View<std::size_t*, MemorySpace>("ws_positions", rows);

    // Update capacity tracking
    max_rows = rows;
    max_intervals = intervals;

    // Note: temp_mesh_0 and temp_mesh_1 are allocated on-demand during
    // intersection operations to their exact required sizes
  }

  /**
   * @brief Clear all workspace buffers.
   *
   * Resets all views to empty, freeing device memory.
   */
  void clear() {
    temp_mesh_0 = MeshType{};
    temp_mesh_1 = MeshType{};

    row_index_buf_0 = Kokkos::View<int*, MemorySpace>();
    row_index_buf_1 = Kokkos::View<int*, MemorySpace>();
    flags_buf = Kokkos::View<int*, MemorySpace>();
    positions_buf = Kokkos::View<std::size_t*, MemorySpace>();

    max_rows = 0;
    max_intervals = 0;
  }
};

// ============================================================================
// Successive intersection with workspace reuse
// ============================================================================

/**
 * @brief Compute the intersection of multiple meshes using workspace reuse.
 *
 * This function performs successive intersection operations on a vector of meshes,
 * reusing temporary buffers across iterations to minimize allocations.
 *
 * Algorithm:
 * 1. Initialize workspace based on smallest input mesh (pessimistic estimate)
 * 2. Ping-pong between temp_mesh_0 and temp_mesh_1 for intermediate results
 * 3. Use v3::intersect_meshes for core intersection logic
 * 4. Reuse all auxiliary buffers (row indices, flags, positions)
 *
 * @tparam DIM Dimension (2 or 3)
 * @tparam MemorySpace Kokkos memory space
 * @tparam CoordType Coordinate type (default: int32_t)
 * @tparam IndexType Index type (default: std::size_t)
 *
 * @param meshes Vector of meshes to intersect sequentially
 * @param workspace Reusable workspace (allocated if needed)
 * @return Intersection of all meshes (empty if input is empty or any intersection is empty)
 */
template<int DIM, class MemorySpace = Kokkos::DefaultExecutionSpace::memory_space,
         class CoordType = int32_t, class IndexType = std::size_t>
inline v3::Mesh<DIM, MemorySpace, CoordType, IndexType>
intersect_successive(
  const std::vector<v3::Mesh<DIM, MemorySpace, CoordType, IndexType>>& meshes,
  IntersectionWorkspace<DIM, MemorySpace>& workspace)
{
  using MeshType = v3::Mesh<DIM, MemorySpace, CoordType, IndexType>;
  using ExecSpace = typename Kokkos::DefaultExecutionSpace;

  // Edge cases
  if (meshes.empty()) {
    return MeshType{};
  }

  if (meshes.size() == 1) {
    // Single mesh - return a copy
    return v3::mesh_to<DIM, CoordType, IndexType, MemorySpace, MemorySpace>(meshes[0]);
  }

  // Find smallest mesh for workspace initialization (pessimistic but reasonable)
  std::size_t min_rows = meshes[0].num_rows;
  std::size_t min_intervals = meshes[0].num_intervals;

  for (const auto& mesh : meshes) {
    min_rows = std::min(min_rows, mesh.num_rows);
    min_intervals = std::min(min_intervals, mesh.num_intervals);
  }

  // Initialize workspace with pessimistic capacity estimate
  // Note: Intersection can never have more rows/intervals than the smallest input
  workspace.ensure_capacity(min_rows, min_intervals);

  // Initialize result with first mesh
  MeshType result = meshes[0];

  // Successive intersection: ping-pong between temp buffers
  bool use_temp_0 = true;

  for (std::size_t i = 1; i < meshes.size(); ++i) {
    const auto& next_mesh = meshes[i];

    // Early exit if current result is empty
    if (result.num_rows == 0 || result.num_intervals == 0) {
      return MeshType{};
    }

    // Early exit if next mesh is empty
    if (next_mesh.num_rows == 0 || next_mesh.num_intervals == 0) {
      return MeshType{};
    }

    // Compute intersection into appropriate temp buffer
    if (use_temp_0) {
      workspace.temp_mesh_0 = v3::intersect_meshes<DIM, CoordType, IndexType>(result, next_mesh);
      result = workspace.temp_mesh_0;
    } else {
      workspace.temp_mesh_1 = v3::intersect_meshes<DIM, CoordType, IndexType>(result, next_mesh);
      result = workspace.temp_mesh_1;
    }

    // Alternate ping-pong buffers
    use_temp_0 = !use_temp_0;

    // Ensure workspace capacity for next iteration (if not last)
    if (i + 1 < meshes.size()) {
      workspace.ensure_capacity(result.num_rows, result.num_intervals);
    }
  }

  return result;
}

// ============================================================================
// Convenience overloads
// ============================================================================

/**
 * @brief Compute successive intersection with automatic workspace creation.
 *
 * This overload creates a temporary workspace for the operation.
 * For repeated calls, prefer the version with a workspace parameter.
 *
 * @tparam DIM Dimension (2 or 3)
 * @tparam MemorySpace Kokkos memory space
 * @tparam CoordType Coordinate type
 * @tparam IndexType Index type
 *
 * @param meshes Vector of meshes to intersect
 * @return Intersection of all meshes
 */
template<int DIM, class MemorySpace = Kokkos::DefaultExecutionSpace::memory_space,
         class CoordType = int32_t, class IndexType = std::size_t>
inline v3::Mesh<DIM, MemorySpace, CoordType, IndexType>
intersect_successive(
  const std::vector<v3::Mesh<DIM, MemorySpace, CoordType, IndexType>>& meshes)
{
  IntersectionWorkspace<DIM, MemorySpace> workspace;
  return intersect_successive(meshes, workspace);
}

/**
 * @brief Compute successive intersection of 2D meshes with workspace.
 */
template<class MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
inline v3::Mesh2D<int32_t, std::size_t>
intersect_successive_2d(
  const std::vector<v3::Mesh2D<int32_t, std::size_t>>& meshes,
  IntersectionWorkspace<2, MemorySpace>& workspace)
{
  return intersect_successive<2, MemorySpace, int32_t, std::size_t>(meshes, workspace);
}

/**
 * @brief Compute successive intersection of 3D meshes with workspace.
 */
template<class MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
inline v3::Mesh3D<int32_t, std::size_t>
intersect_successive_3d(
  const std::vector<v3::Mesh3D<int32_t, std::size_t>>& meshes,
  IntersectionWorkspace<3, MemorySpace>& workspace)
{
  return intersect_successive<3, MemorySpace, int32_t, std::size_t>(meshes, workspace);
}

} // namespace experimental::subsetix::csr::successive::workspace
