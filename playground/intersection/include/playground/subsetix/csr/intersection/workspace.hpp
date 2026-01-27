// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#pragma once

#ifdef SUBSETIX_ENABLE_PLAYGROUND

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <playground/subsetix/csr/intersection/types.hpp>

namespace playground::subsetix::csr::intersection {

/**
 * @brief Reusable workspace for 2D intersection algorithms to eliminate per-call allocations
 *
 * This workspace pre-allocates all temporary buffers needed for 2D mesh intersection.
 * Buffers are sized based on max(A.rows, B.rows) and max(A.intervals, B.intervals)
 * since the intersection cannot be larger than the inputs.
 *
 * Usage pattern:
 * ```cpp
 * IntersectionWorkspace2D<Kokkos::Cuda> ws;
 *
 * // In benchmark SetUp():
 * std::size_t max_rows = std::max(mesh_a.num_rows, mesh_b.num_rows);
 * std::size_t max_intervals = std::max(mesh_a.num_intervals, mesh_b.num_intervals);
 * ws.ensure_capacity(max_rows, max_intervals);
 *
 * // In benchmark loop (no allocations!):
 * for (auto _ : state) {
 *     intersect_meshes_2d_in_place(a, b, result, ws);
 * }
 * ```
 */
template <typename ExecSpace, typename IndexType = std::size_t, typename CoordType = int32_t>
struct IntersectionWorkspace2D {
  using MemorySpace = typename ExecSpace::memory_space;
  using IntView = Kokkos::View<int*, MemorySpace>;
  using SizeTView = Kokkos::View<IndexType*, MemorySpace>;
  using ScalarView = Kokkos::View<IndexType, MemorySpace>;
  using RowKeyView = Kokkos::View<RowKey2D<CoordType>*, MemorySpace>;

  /// Chunk size used for hierarchical row scans
  static constexpr std::size_t rows_per_team = 4096;

  // ============================================================================
  // Phase 1: Row Mapping Buffers
  // ============================================================================

  /// Marks which rows match during binary search phase
  IntView flags;

  /// Temporary indices for mesh A row mapping
  IntView tmp_idx_a;

  /// Temporary indices for mesh B row mapping
  IntView tmp_idx_b;

  /// Positions for compaction after row mapping
  SizeTView positions;

  /// Per-team match counts for hierarchical scans
  SizeTView team_counts;

  /// Per-team offsets for hierarchical scans
  SizeTView team_offsets;

  // ============================================================================
  // Phase 2: Row Scan & Compaction Buffers
  // ============================================================================

  /// Scalar view for reduction result (number of output rows) - 0D view
  ScalarView num_rows_out_view;

  /// Compacted row keys from A (2D version)
  RowKeyView out_rows;

  /// Compacted indices for A
  IntView out_idx_a;

  /// Compacted indices for B
  IntView out_idx_b;

  // ============================================================================
  // Phase 3: Interval Counting Buffers
  // ============================================================================

  /// Counts intervals per output row
  SizeTView row_counts;

  // ============================================================================
  // Phase 4: Scan Buffers
  // ============================================================================

  /// Scalar view for total interval count - 0D view
  ScalarView total_view;

  // ============================================================================
  // Phase 5: Final Compaction Buffers
  // ============================================================================

  /// Marks rows that have intervals (non-empty)
  IntView has_intervals;

  /// New positions for re-compaction
  SizeTView new_positions;

  /// Scalar view for final row count - 0D view
  ScalarView final_num_rows_view;

  // ============================================================================
  // Metadata
  // ============================================================================

  /// Current allocated capacity (rows)
  std::size_t capacity_rows = 0;

  /// Current allocated capacity (intervals)
  std::size_t capacity_intervals = 0;

  /// Current allocated capacity (teams)
  std::size_t capacity_teams = 0;

  /**
   * @brief Ensure workspace has sufficient capacity
   *
   * Only reallocates if requested size exceeds current capacity.
   * This allows reuse across multiple benchmark iterations.
   *
   * @param max_rows Maximum number of rows needed (max of input meshes)
   * @param max_intervals Maximum number of intervals needed (max of input meshes)
   */
  void ensure_capacity(std::size_t max_rows, std::size_t max_intervals) {
    if (max_rows <= capacity_rows && max_intervals <= capacity_intervals) {
      return;  // Already allocated, reuse existing buffers
    }

    // Allocate with new capacity (grow-only, never shrink)
    std::size_t new_capacity_rows = std::max(max_rows, capacity_rows);
    std::size_t new_capacity_intervals = std::max(max_intervals, capacity_intervals);
    std::size_t new_capacity_teams =
        std::max((new_capacity_rows + rows_per_team - 1) / rows_per_team, capacity_teams);

    // Phase 1 buffers
    flags = IntView("workspace_flags", new_capacity_rows);
    tmp_idx_a = IntView("workspace_tmp_idx_a", new_capacity_rows);
    tmp_idx_b = IntView("workspace_tmp_idx_b", new_capacity_rows);
    positions = SizeTView("workspace_positions", new_capacity_rows);
    team_counts = SizeTView("workspace_team_counts", new_capacity_teams);
    team_offsets = SizeTView("workspace_team_offsets", new_capacity_teams);

    // Phase 2 buffers
    num_rows_out_view = ScalarView("workspace_num_rows_out");  // 0D scalar view
    out_rows = RowKeyView("workspace_out_rows", new_capacity_rows);
    out_idx_a = IntView("workspace_out_idx_a", new_capacity_rows);
    out_idx_b = IntView("workspace_out_idx_b", new_capacity_rows);

    // Phase 3 buffers
    row_counts = SizeTView("workspace_row_counts", new_capacity_rows);

    // Phase 4 buffers
    total_view = ScalarView("workspace_total_intervals");  // 0D scalar view

    // Phase 5 buffers
    has_intervals = IntView("workspace_has_intervals", new_capacity_rows);
    new_positions = SizeTView("workspace_new_positions", new_capacity_rows);
    final_num_rows_view = ScalarView("workspace_final_num_rows");  // 0D scalar view

    capacity_rows = new_capacity_rows;
    capacity_intervals = new_capacity_intervals;
    capacity_teams = new_capacity_teams;
  }

  /**
   * @brief Reset workspace state between uses
   *
   * Note: This is a no-op for correctness - the kernels will overwrite
   * all data. Kept for potential future optimizations (e.g., zeroing
   * if needed for certain algorithms).
   */
  void reset() {
    // No-op - kernels will overwrite all buffer contents
    // Kept for API symmetry and potential future use
  }

  /**
   * @brief Get current allocated capacity
   *
   * @return Pair of (capacity_rows, capacity_intervals)
   */
  std::pair<std::size_t, std::size_t> capacity() const {
    return {capacity_rows, capacity_intervals};
  }
};

/**
 * @brief Reusable workspace for 3D intersection algorithms
 */
template <typename ExecSpace, typename IndexType = std::size_t, typename CoordType = int32_t>
struct IntersectionWorkspace3D {
  using MemorySpace = typename ExecSpace::memory_space;
  using IntView = Kokkos::View<int*, MemorySpace>;
  using SizeTView = Kokkos::View<IndexType*, MemorySpace>;
  using ScalarView = Kokkos::View<IndexType, MemorySpace>;
  using RowKeyView = Kokkos::View<RowKey3D<CoordType>*, MemorySpace>;

  static constexpr std::size_t rows_per_team = 4096;

  // ============================================================================
  // Phase 1: Row Mapping Buffers
  // ============================================================================

  IntView flags;
  IntView tmp_idx_a;
  IntView tmp_idx_b;
  SizeTView positions;
  SizeTView team_counts;
  SizeTView team_offsets;

  // ============================================================================
  // Phase 2: Row Scan & Compaction Buffers
  // ============================================================================

  ScalarView num_rows_out_view;
  RowKeyView out_rows;
  IntView out_idx_a;
  IntView out_idx_b;

  // ============================================================================
  // Phase 3: Interval Counting Buffers
  // ============================================================================

  SizeTView row_counts;

  // ============================================================================
  // Phase 4: Scan Buffers
  // ============================================================================

  ScalarView total_view;

  // ============================================================================
  // Phase 5: Final Compaction Buffers
  // ============================================================================

  IntView has_intervals;
  SizeTView new_positions;
  ScalarView final_num_rows_view;

  // ============================================================================
  // Metadata
  // ============================================================================

  std::size_t capacity_rows = 0;
  std::size_t capacity_intervals = 0;
  std::size_t capacity_teams = 0;

  void ensure_capacity(std::size_t max_rows, std::size_t max_intervals) {
    if (max_rows <= capacity_rows && max_intervals <= capacity_intervals) {
      return;
    }

    std::size_t new_capacity_rows = std::max(max_rows, capacity_rows);
    std::size_t new_capacity_intervals = std::max(max_intervals, capacity_intervals);
    std::size_t new_capacity_teams =
        std::max((new_capacity_rows + rows_per_team - 1) / rows_per_team, capacity_teams);

    flags = IntView("workspace_flags", new_capacity_rows);
    tmp_idx_a = IntView("workspace_tmp_idx_a", new_capacity_rows);
    tmp_idx_b = IntView("workspace_tmp_idx_b", new_capacity_rows);
    positions = SizeTView("workspace_positions", new_capacity_rows);
    team_counts = SizeTView("workspace_team_counts", new_capacity_teams);
    team_offsets = SizeTView("workspace_team_offsets", new_capacity_teams);

    num_rows_out_view = ScalarView("workspace_num_rows_out");
    out_rows = RowKeyView("workspace_out_rows", new_capacity_rows);
    out_idx_a = IntView("workspace_out_idx_a", new_capacity_rows);
    out_idx_b = IntView("workspace_out_idx_b", new_capacity_rows);

    row_counts = SizeTView("workspace_row_counts", new_capacity_rows);
    total_view = ScalarView("workspace_total_intervals");

    has_intervals = IntView("workspace_has_intervals", new_capacity_rows);
    new_positions = SizeTView("workspace_new_positions", new_capacity_rows);
    final_num_rows_view = ScalarView("workspace_final_num_rows");

    capacity_rows = new_capacity_rows;
    capacity_intervals = new_capacity_intervals;
    capacity_teams = new_capacity_teams;
  }

  void reset() {
    // No-op
  }

  std::pair<std::size_t, std::size_t> capacity() const {
    return {capacity_rows, capacity_intervals};
  }
};

// Generic workspace alias for backward compatibility (defaults to 2D)
template <typename ExecSpace, typename IndexType = std::size_t, typename CoordType = int32_t>
using IntersectionWorkspace = IntersectionWorkspace2D<ExecSpace, IndexType, CoordType>;

} // namespace playground::subsetix::csr::intersection

#endif // SUBSETIX_ENABLE_PLAYGROUND
