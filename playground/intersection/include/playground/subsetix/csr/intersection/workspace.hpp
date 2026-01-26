// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#pragma once

#ifdef SUBSETIX_ENABLE_PLAYGROUND

#include <Kokkos_Core.hpp>
#include <cstddef>

namespace playground::subsetix::csr::intersection {

/**
 * @brief Reusable workspace for intersection algorithms to eliminate per-call allocations
 *
 * This workspace pre-allocates all temporary buffers needed for mesh intersection.
 * Buffers are sized based on max(A.rows, B.rows) and max(A.intervals, B.intervals)
 * since the intersection cannot be larger than the inputs.
 *
 * Usage pattern:
 * ```cpp
 * IntersectionWorkspace<Kokkos::Cuda> ws;
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
template <typename ExecSpace, typename IndexType = std::size_t>
struct IntersectionWorkspace {
  using MemorySpace = typename ExecSpace::memory_space;
  using IntView = Kokkos::View<int*, MemorySpace>;
  using SizeTView = Kokkos::View<IndexType*, MemorySpace>;

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

  // ============================================================================
  // Phase 2: Row Scan & Compaction Buffers
  // ============================================================================

  /// Scalar view for reduction result (number of output rows)
  SizeTView num_rows_out_view;

  /// Compacted row keys from A
  IntView out_rows;

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

  /// Scalar view for total interval count
  SizeTView total_view;

  // ============================================================================
  // Phase 5: Final Compaction Buffers
  // ============================================================================

  /// Marks rows that have intervals (non-empty)
  IntView has_intervals;

  /// New positions for re-compaction
  SizeTView new_positions;

  /// Scalar view for final row count
  SizeTView final_num_rows_view;

  // ============================================================================
  // Metadata
  // ============================================================================

  /// Current allocated capacity (rows)
  std::size_t capacity_rows = 0;

  /// Current allocated capacity (intervals)
  std::size_t capacity_intervals = 0;

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

    // Phase 1 buffers
    flags = IntView("workspace_flags", new_capacity_rows);
    tmp_idx_a = IntView("workspace_tmp_idx_a", new_capacity_rows);
    tmp_idx_b = IntView("workspace_tmp_idx_b", new_capacity_rows);
    positions = SizeTView("workspace_positions", new_capacity_rows);

    // Phase 2 buffers
    num_rows_out_view = SizeTView("workspace_num_rows_out");
    out_rows = IntView("workspace_out_rows", new_capacity_rows);
    out_idx_a = IntView("workspace_out_idx_a", new_capacity_rows);
    out_idx_b = IntView("workspace_out_idx_b", new_capacity_rows);

    // Phase 3 buffers
    row_counts = SizeTView("workspace_row_counts", new_capacity_rows);

    // Phase 4 buffers
    total_view = SizeTView("workspace_total_intervals");

    // Phase 5 buffers
    has_intervals = IntView("workspace_has_intervals", new_capacity_rows);
    new_positions = SizeTView("workspace_new_positions", new_capacity_rows);
    final_num_rows_view = SizeTView("workspace_final_num_rows");

    capacity_rows = new_capacity_rows;
    capacity_intervals = new_capacity_intervals;
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

} // namespace playground::subsetix::csr::intersection

#endif // SUBSETIX_ENABLE_PLAYGROUND
