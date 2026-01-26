// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#pragma once

#ifdef SUBSETIX_ENABLE_PLAYGROUND

#include <playground/subsetix/csr/intersection/algorithm/baseline.hpp>
#include <playground/subsetix/csr/intersection/workspace.hpp>

namespace playground::subsetix::csr::intersection::baseline {

// ============================================================================
// Mesh intersection with pre-allocated workspace - Implementation
// ============================================================================

/**
 * @brief Internal implementation of 2D intersection with workspace
 *
 * This is a refactored version of intersect_meshes<2> that uses
 * pre-allocated buffers instead of allocating new views for each call.
 *
 * The algorithm is identical to the original, but all temporary buffers
 * (flags, tmp_idx_a, tmp_idx_b, positions, etc.) are provided by the
 * workspace parameter.
 */
template <typename ExecSpace>
void intersect_meshes_2d_in_place_impl(
    const Mesh2DDevice& A,
    const Mesh2DDevice& B,
    Mesh2DDevice& out,
    IntersectionWorkspace<ExecSpace>& ws)
{
  using DeviceMemorySpace = typename ExecSpace::memory_space;
  using CoordType = int32_t;
  using IndexType = std::size_t;
  using Interval = intersection::Interval<CoordType>;
  using RowKey = RowKey2D<CoordType>;

  if (A.num_rows == 0 || B.num_rows == 0) {
    out.num_rows = 0;
    out.num_intervals = 0;
    return;
  }

  const std::size_t num_rows_a = A.num_rows;

  // Use workspace buffers instead of allocating
  auto flags = ws.flags;
  auto tmp_idx_a = ws.tmp_idx_a;
  auto tmp_idx_b = ws.tmp_idx_b;
  auto positions = ws.positions;

  auto rows_a = A.row_keys;
  auto rows_b = B.row_keys;
  auto row_ptr_a = A.row_ptr;
  auto row_ptr_b = B.row_ptr;
  auto intervals_a = A.intervals;
  auto intervals_b = B.intervals;

  // Phase 1: Row Mapping
  // Find matching rows and record their positions
  Kokkos::parallel_for(
      "intersection_row_mapping_2d",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
      KOKKOS_LAMBDA(const std::size_t i) {
        const RowKey row_a = rows_a(i);

        // Binary search for row_a in rows_b
        const RowKey* rows_b_data = rows_b.data();
        std::size_t idx_b = find_row_by_y(rows_b_data, 0, B.num_rows, row_a.y);

        if (idx_b < B.num_rows && rows_b(idx_b).y == row_a.y) {
          flags(i) = 1;
          tmp_idx_a(i) = static_cast<int>(i);
          tmp_idx_b(i) = static_cast<int>(idx_b);
        } else {
          flags(i) = 0;
          tmp_idx_a(i) = -1;
          tmp_idx_b(i) = -1;
        }
      });

  Kokkos::fence();

  // Phase 2: Row Scan - count matching rows
  Kokkos::parallel_scan(
      "intersection_count_matching_rows",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
        const std::size_t count = flags(i);
        if (final_pass) {
          positions(i) = update;
        }
        update += count;
      },
      ws.num_rows_out_view);

  Kokkos::fence();

  // Get number of output rows
  std::size_t num_rows_out_host = 0;
  Kokkos::deep_copy(num_rows_out_host, ws.num_rows_out_view);
  const std::size_t num_rows_out = num_rows_out_host;

  if (num_rows_out == 0) {
    out.num_rows = 0;
    out.num_intervals = 0;
    return;
  }

  // Phase 3: Compact Rows - extract matching rows
  auto out_rows = ws.out_rows;
  auto out_idx_a = ws.out_idx_a;
  auto out_idx_b = ws.out_idx_b;

  Kokkos::parallel_for(
      "intersection_compact_rows",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
      KOKKOS_LAMBDA(const std::size_t i) {
        if (flags(i) == 1) {
          const std::size_t pos = positions(i);
          out_rows(pos) = static_cast<int>(rows_a(i).y);
          out_idx_a(pos) = tmp_idx_a(i);
          out_idx_b(pos) = tmp_idx_b(i);
        }
      });

  Kokkos::fence();

  // Phase 4: Count Intervals per row
  auto row_counts = ws.row_counts;

  Kokkos::parallel_for(
      "intersection_count_intervals",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i_out) {
        const int idx_a = out_idx_a(i_out);
        const int idx_b = out_idx_b(i_out);

        const std::size_t start_a = row_ptr_a(idx_a);
        const std::size_t end_a = row_ptr_a(idx_a + 1);
        const std::size_t start_b = row_ptr_b(idx_b);
        const std::size_t end_b = row_ptr_b(idx_b + 1);

        std::size_t count = 0;
        count_intervals_in_range<true>(
            intervals_a, start_a, end_a,
            intervals_b, start_b, end_b,
            count);

        row_counts(i_out) = count;
      });

  // Phase 5: Scan - compute row_ptr offsets
  Kokkos::parallel_scan(
      "intersection_compute_row_ptr",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
        const std::size_t count = row_counts(i);
        if (final_pass) {
          row_counts(i) = update;  // Reuse row_counts to store offsets
        }
        update += count;
      },
      ws.total_view);

  Kokkos::fence();

  // Get total interval count
  std::size_t num_intervals_host = 0;
  Kokkos::deep_copy(num_intervals_host, ws.total_view);
  const std::size_t num_intervals = num_intervals_host;

  // Phase 6: Fill Intervals - compute actual intersections
  // Reuse row_counts array as row_ptr (now stores offsets)
  out.num_rows = num_rows_out;
  out.num_intervals = num_intervals;

  // Set row_ptr from offsets (row_counts was repurposed)
  Kokkos::parallel_for(
      "intersection_set_row_ptr",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out + 1),
      KOKKOS_LAMBDA(const std::size_t i) {
        if (i < num_rows_out) {
          out.row_ptr(i) = row_counts(i);
        } else {
          out.row_ptr(i) = num_intervals;
        }
      });

  Kokkos::parallel_for(
      "intersection_fill_intervals",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i_out) {
        const int idx_a = out_idx_a(i_out);
        const int idx_b = out_idx_b(i_out);

        const std::size_t start_a = row_ptr_a(idx_a);
        const std::size_t end_a = row_ptr_a(idx_a + 1);
        const std::size_t start_b = row_ptr_b(idx_b);
        const std::size_t end_b = row_ptr_b(idx_b + 1);

        const std::size_t out_offset = row_counts(i_out);

        std::size_t count = 0;
        count_intervals_in_range<false>(
            intervals_a, start_a, end_a,
            intervals_b, start_b, end_b,
            out.intervals, out_offset, count);
      });

  Kokkos::fence();

  // Phase 7: Final Compaction - remove empty rows
  auto has_intervals = ws.has_intervals;
  auto new_positions = ws.new_positions;

  Kokkos::parallel_for(
      "intersection_mark_nonempty_rows",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        const std::size_t start = out.row_ptr(i);
        const std::size_t end = out.row_ptr(i + 1);
        has_intervals(i) = (start < end) ? 1 : 0;
      });

  Kokkos::parallel_scan(
      "intersection_compute_final_positions",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
        const std::size_t count = has_intervals(i);
        if (final_pass) {
          new_positions(i) = update;
        }
        update += count;
      },
      ws.final_num_rows_view);

  Kokkos::fence();

  // Get final row count
  std::size_t final_num_rows_host = 0;
  Kokkos::deep_copy(final_num_rows_host, ws.final_num_rows_view);
  const std::size_t final_num_rows = final_num_rows_host;

  if (final_num_rows < num_rows_out) {
    // Compact the mesh
    auto compacted_row_keys = ws.out_rows;  // Reuse buffer
    auto compacted_row_ptr = ws.row_counts;  // Reuse buffer

    Kokkos::parallel_for(
        "intersection_compact_row_keys",
        Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
        KOKKOS_LAMBDA(const std::size_t i) {
          if (has_intervals(i) == 1) {
            const std::size_t new_pos = new_positions(i);
            compacted_row_keys(new_pos) = out_rows(i);
            compacted_row_ptr(new_pos) = out.row_ptr(i);
          }
        });

    Kokkos::fence();

    // Copy compacted data back to out
    Kokkos::parallel_for(
        "intersection_copy_compacted",
        Kokkos::RangePolicy<ExecSpace>(0, final_num_rows),
        KOKKOS_LAMBDA(const std::size_t i) {
          out.row_keys(i) = compacted_row_keys(i);
          out.row_ptr(i) = compacted_row_ptr(i);
        });

    // Set final row_ptr element
    Kokkos::parallel_for(
        "intersection_set_final_ptr",
        Kokkos::RangePolicy<ExecSpace>(0, 1),
        KOKKOS_LAMBDA(const std::size_t) {
          out.row_ptr(final_num_rows) = out.row_ptr(final_num_rows - 1);
        });

    Kokkos::fence();

    // Compact intervals
    Kokkos::parallel_for(
        "intersection_compact_intervals",
        Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
        KOKKOS_LAMBDA(const std::size_t i) {
          if (has_intervals(i) == 1) {
            const std::size_t new_pos = new_positions(i);
            const std::size_t start = out.row_ptr(i);
            const std::size_t end = out.row_ptr(i + 1);
            const std::size_t new_start = compacted_row_ptr(new_pos);

            for (std::size_t j = 0; j < (end - start); ++j) {
              out.intervals(new_start + j) = out.intervals(start + j);
            }
          }
        });

    out.num_rows = final_num_rows;
    // num_intervals stays the same
  }
}

// ============================================================================
// Public API wrappers
// ============================================================================

template <typename ExecSpace>
void intersect_meshes_2d_in_place(
    const Mesh2DDevice& A,
    const Mesh2DDevice& B,
    Mesh2DDevice& result_out,
    IntersectionWorkspace<ExecSpace>& workspace)
{
  intersect_meshes_2d_in_place_impl(A, B, result_out, workspace);
}

template <typename ExecSpace>
void intersect_meshes_3d_in_place(
    const Mesh3DDevice& A,
    const Mesh3DDevice& B,
    Mesh3DDevice& result_out,
    IntersectionWorkspace<ExecSpace>& workspace)
{
  using DeviceMemorySpace = typename ExecSpace::memory_space;
  using CoordType = int32_t;
  using IndexType = std::size_t;
  using Interval = intersection::Interval<CoordType>;
  using RowKey = RowKey3D<CoordType>;

  if (A.num_rows == 0 || B.num_rows == 0) {
    result_out.num_rows = 0;
    result_out.num_intervals = 0;
    return;
  }

  const std::size_t num_rows_a = A.num_rows;

  // Use workspace buffers instead of allocating
  auto flags = workspace.flags;
  auto tmp_idx_a = workspace.tmp_idx_a;
  auto tmp_idx_b = workspace.tmp_idx_b;
  auto positions = workspace.positions;

  auto rows_a = A.row_keys;
  auto rows_b = B.row_keys;
  const std::size_t num_rows_b = B.num_rows;
  auto row_ptr_a = A.row_ptr;
  auto row_ptr_b = B.row_ptr;
  auto intervals_a = A.intervals;
  auto intervals_b = B.intervals;

  // Phase 1: Row Mapping (3D version - search by y,z)
  Kokkos::parallel_for(
      "intersection_row_mapping_3d",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
      KOKKOS_LAMBDA(const std::size_t i) {
        const RowKey key = rows_a(i);
        const int idx_b = intersection::detail::find_row_by_yz(rows_b, num_rows_b, key.y, key.z);

        if (idx_b >= 0) {
          flags(i) = 1;
          tmp_idx_a(i) = static_cast<int>(i);
          tmp_idx_b(i) = idx_b;
        } else {
          flags(i) = 0;
          tmp_idx_a(i) = -1;
          tmp_idx_b(i) = -1;
        }
      });

  Kokkos::fence();

  // Phase 2: Row Scan - count matching rows
  Kokkos::parallel_scan(
      "intersection_count_matching_rows_3d",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
        const std::size_t count = static_cast<std::size_t>(flags(i));
        if (final_pass) {
          positions(i) = update;
        }
        update += count;
      },
      workspace.num_rows_out_view);

  Kokkos::fence();

  // Get number of output rows
  std::size_t num_rows_out_host = 0;
  Kokkos::deep_copy(num_rows_out_host, workspace.num_rows_out_view);
  const std::size_t num_rows_out = num_rows_out_host;

  if (num_rows_out == 0) {
    result_out.num_rows = 0;
    result_out.num_intervals = 0;
    return;
  }

  // Phase 3: Compact Rows
  auto out_rows = workspace.out_rows;
  auto out_idx_a = workspace.out_idx_a;
  auto out_idx_b = workspace.out_idx_b;

  Kokkos::parallel_for(
      "intersection_compact_rows_3d",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
      KOKKOS_LAMBDA(const std::size_t i) {
        if (flags(i) == 1) {
          const std::size_t pos = positions(i);
          out_rows(pos) = rows_a(i);  // RowKey3D includes y and z
          out_idx_a(pos) = tmp_idx_a(i);
          out_idx_b(pos) = tmp_idx_b(i);
        }
      });

  Kokkos::fence();

  // Phase 4: Count Intervals per row
  auto row_counts = workspace.row_counts;

  Kokkos::parallel_for(
      "intersection_count_intervals_3d",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i_out) {
        const int idx_a = out_idx_a(i_out);
        const int idx_b = out_idx_b(i_out);

        const std::size_t start_a = row_ptr_a(idx_a);
        const std::size_t end_a = row_ptr_a(idx_a + 1);
        const std::size_t start_b = row_ptr_b(idx_b);
        const std::size_t end_b = row_ptr_b(idx_b + 1);

        std::size_t count = 0;
        count_intervals_in_range<true>(
            intervals_a, start_a, end_a,
            intervals_b, start_b, end_b,
            count);

        row_counts(i_out) = count;
      });

  // Phase 5: Scan - compute row_ptr offsets
  Kokkos::parallel_scan(
      "intersection_compute_row_ptr_3d",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
        const std::size_t count = row_counts(i);
        if (final_pass) {
          row_counts(i) = update;  // Reuse row_counts to store offsets
        }
        update += count;
      },
      workspace.total_view);

  Kokkos::fence();

  // Get total interval count
  std::size_t num_intervals_host = 0;
  Kokkos::deep_copy(num_intervals_host, workspace.total_view);
  const std::size_t num_intervals = num_intervals_host;

  // Phase 6: Fill Intervals
  result_out.num_rows = num_rows_out;
  result_out.num_intervals = num_intervals;

  Kokkos::parallel_for(
      "intersection_set_row_ptr_3d",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out + 1),
      KOKKOS_LAMBDA(const std::size_t i) {
        if (i < num_rows_out) {
          result_out.row_ptr(i) = row_counts(i);
        } else {
          result_out.row_ptr(i) = num_intervals;
        }
      });

  Kokkos::parallel_for(
      "intersection_fill_intervals_3d",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i_out) {
        const int idx_a = out_idx_a(i_out);
        const int idx_b = out_idx_b(i_out);

        const std::size_t start_a = row_ptr_a(idx_a);
        const std::size_t end_a = row_ptr_a(idx_a + 1);
        const std::size_t start_b = row_ptr_b(idx_b);
        const std::size_t end_b = row_ptr_b(idx_b + 1);

        const std::size_t out_offset = row_counts(i_out);

        std::size_t count = 0;
        count_intervals_in_range<false>(
            intervals_a, start_a, end_a,
            intervals_b, start_b, end_b,
            result_out.intervals, out_offset, count);
      });

  Kokkos::fence();

  // Phase 7: Final Compaction
  auto has_intervals = workspace.has_intervals;
  auto new_positions = workspace.new_positions;

  Kokkos::parallel_for(
      "intersection_mark_nonempty_rows_3d",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        const std::size_t start = result_out.row_ptr(i);
        const std::size_t end = result_out.row_ptr(i + 1);
        has_intervals(i) = (start < end) ? 1 : 0;
      });

  Kokkos::parallel_scan(
      "intersection_compute_final_positions_3d",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
        const std::size_t count = has_intervals(i);
        if (final_pass) {
          new_positions(i) = update;
        }
        update += count;
      },
      workspace.final_num_rows_view);

  Kokkos::fence();

  // Get final row count
  std::size_t final_num_rows_host = 0;
  Kokkos::deep_copy(final_num_rows_host, workspace.final_num_rows_view);
  const std::size_t final_num_rows = final_num_rows_host;

  if (final_num_rows < num_rows_out) {
    // Compact the mesh
    auto compacted_row_keys = workspace.out_rows;  // Reuse buffer
    auto compacted_row_ptr = workspace.row_counts;  // Reuse buffer

    Kokkos::parallel_for(
        "intersection_compact_row_keys_3d",
        Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
        KOKKOS_LAMBDA(const std::size_t i) {
          if (has_intervals(i) == 1) {
            const std::size_t new_pos = new_positions(i);
            compacted_row_keys(new_pos) = out_rows(i);
            compacted_row_ptr(new_pos) = result_out.row_ptr(i);
          }
        });

    Kokkos::fence();

    // Copy compacted data back to result_out
    Kokkos::parallel_for(
        "intersection_copy_compacted_3d",
        Kokkos::RangePolicy<ExecSpace>(0, final_num_rows),
        KOKKOS_LAMBDA(const std::size_t i) {
          result_out.row_keys(i) = compacted_row_keys(i);
          result_out.row_ptr(i) = compacted_row_ptr(i);
        });

    // Set final row_ptr element
    Kokkos::parallel_for(
        "intersection_set_final_ptr_3d",
        Kokkos::RangePolicy<ExecSpace>(0, 1),
        KOKKOS_LAMBDA(const std::size_t) {
          result_out.row_ptr(final_num_rows) = result_out.row_ptr(final_num_rows - 1);
        });

    Kokkos::fence();

    // Compact intervals
    Kokkos::parallel_for(
        "intersection_compact_intervals_3d",
        Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
        KOKKOS_LAMBDA(const std::size_t i) {
          if (has_intervals(i) == 1) {
            const std::size_t new_pos = new_positions(i);
            const std::size_t start = result_out.row_ptr(i);
            const std::size_t end = result_out.row_ptr(i + 1);
            const std::size_t new_start = compacted_row_ptr(new_pos);

            for (std::size_t j = 0; j < (end - start); ++j) {
              result_out.intervals(new_start + j) = result_out.intervals(start + j);
            }
          }
        });

    result_out.num_rows = final_num_rows;
  }
}

} // namespace playground::subsetix::csr::intersection::baseline

#endif // SUBSETIX_ENABLE_PLAYGROUND
