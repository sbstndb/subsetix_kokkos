// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#pragma once

#ifdef SUBSETIX_ENABLE_PLAYGROUND

#include <playground/subsetix/csr/intersection/algorithm/baseline.hpp>
#include <playground/subsetix/csr/intersection/workspace.hpp>

// This file is included from within namespace playground::subsetix::csr::intersection::baseline

// ============================================================================
// Mesh intersection with pre-allocated workspace - Implementation
// ============================================================================

/**
 * @brief Internal implementation of 2D intersection with workspace
 *
 * This is a direct adaptation of intersect_meshes<2>() that uses pre-allocated
 * buffers from the workspace instead of allocating new views for each call.
 *
 * The algorithm is IDENTICAL to the original, only the buffer allocation is changed.
 */
template <typename ExecSpace>
void intersect_meshes_2d_in_place_impl(
    const Mesh2DDevice& A,
    const Mesh2DDevice& B,
    Mesh2DDevice& result_out,
    IntersectionWorkspace2D<ExecSpace>& ws)
{
  using DeviceMemorySpace = typename ExecSpace::memory_space;
  using CoordType = int32_t;
  using IndexType = std::size_t;

  if (A.num_rows == 0 || B.num_rows == 0) {
    result_out.num_rows = 0;
    result_out.num_intervals = 0;
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
  const std::size_t num_rows_b = B.num_rows;

  // Phase 1: Row mapping - find rows of A that exist in B (2D: search by y only)
  Kokkos::parallel_for(
      "intersection_row_map_2d",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
      KOKKOS_LAMBDA(const std::size_t i) {
        const RowKey2D<CoordType> key = rows_a(i);
        const int idx_b = intersection::detail::find_row_by_y(rows_b, num_rows_b, key.y);
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

  // Scan to count matching rows and compute positions
  auto num_rows_out_view = ws.num_rows_out_view;

  Kokkos::parallel_scan(
      "intersection_row_scan",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
        const std::size_t count = static_cast<std::size_t>(flags(i));
        if (final_pass) {
          positions(i) = update;
          if (i + 1 == num_rows_a) {
            num_rows_out_view() = update + count;
          }
        }
        update += count;
      });

  Kokkos::fence();

  std::size_t num_rows_out_host = 0;
  Kokkos::deep_copy(num_rows_out_host, num_rows_out_view);
  const std::size_t num_rows_out = num_rows_out_host;

  if (num_rows_out == 0) {
    result_out.num_rows = 0;
    result_out.num_intervals = 0;
    return;
  }

  // Use workspace buffers for compacted row data
  auto out_rows = ws.out_rows;
  auto out_idx_a = ws.out_idx_a;
  auto out_idx_b = ws.out_idx_b;

  // Compact matching rows
  Kokkos::parallel_for(
      "intersection_row_compact",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
      KOKKOS_LAMBDA(const std::size_t i) {
        if (!flags(i)) {
          return;
        }
        const std::size_t pos = positions(i);
        out_rows(pos) = rows_a(i);
        out_idx_a(pos) = tmp_idx_a(i);
        out_idx_b(pos) = tmp_idx_b(i);
      });

  Kokkos::fence();

  // Copy row keys to result mesh (deep_copy since we're writing to pre-allocated result)
  Kokkos::deep_copy(result_out.row_keys, out_rows);

  // Set result metadata
  result_out.num_rows = num_rows_out;

  auto row_ptr_a = A.row_ptr;
  auto row_ptr_b = B.row_ptr;
  auto intervals_a = A.intervals;
  auto intervals_b = B.intervals;

  // Phase 2: Count intervals per row
  auto row_counts = ws.row_counts;

  Kokkos::parallel_for(
      "intersection_count",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        const int ia = out_idx_a(i);
        const int ib = out_idx_b(i);

        if (ib < 0) {
          row_counts(i) = 0;
          return;
        }

        const auto r = intersection::detail::extract_row_ranges(ia, ib, row_ptr_a, row_ptr_b);

        if (r.begin_a == r.end_a || r.begin_b == r.end_b) {
          row_counts(i) = 0;
          return;
        }

        row_counts(i) = detail::row_intersection_impl<true>(
            intervals_a, r.begin_a, r.end_a,
            intervals_b, r.begin_b, r.end_b,
            Kokkos::View<intersection::Interval<CoordType>*, DeviceMemorySpace>(), 0);
      });

  // Phase 3: Scan to compute row_ptr offsets
  auto total_view = ws.total_view;

  Kokkos::parallel_scan(
      "intersection_scan",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
        const std::size_t count = row_counts(i);
        if (final_pass) {
          result_out.row_ptr(i) = static_cast<IndexType>(update);
          if (i + 1 == num_rows_out) {
            result_out.row_ptr(num_rows_out) = static_cast<IndexType>(update + count);
            total_view() = update + count;
          }
        }
        update += count;
      });

  Kokkos::fence();

  std::size_t num_intervals_host = 0;
  Kokkos::deep_copy(num_intervals_host, total_view);
  result_out.num_intervals = num_intervals_host;

  if (result_out.num_intervals == 0) {
    return;
  }

  // Phase 4: Fill intersected intervals
  Kokkos::parallel_for(
      "intersection_fill",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        const int ia = out_idx_a(i);
        const int ib = out_idx_b(i);

        if (ib < 0) {
          return;
        }

        const auto r = intersection::detail::extract_row_ranges(ia, ib, row_ptr_a, row_ptr_b);

        if (r.begin_a == r.end_a || r.begin_b == r.end_b) {
          return;
        }

        // Use result_out.row_ptr(i) as the offset (NOT r.begin_a!)
        detail::row_intersection_impl<false>(
            intervals_a, r.begin_a, r.end_a,
            intervals_b, r.begin_b, r.end_b,
            result_out.intervals, result_out.row_ptr(i));
      });

  Kokkos::fence();

  // Phase 5: Final compaction - remove rows with no intervals
  auto has_intervals = ws.has_intervals;
  auto new_positions = ws.new_positions;
  auto final_num_rows_view = ws.final_num_rows_view;

  Kokkos::parallel_for(
      "intersection_mark_rows",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        has_intervals(i) = (result_out.row_ptr(i) < result_out.row_ptr(i + 1)) ? 1 : 0;
      });

  Kokkos::parallel_scan(
      "intersection_compact_scan",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
        const std::size_t count = static_cast<std::size_t>(has_intervals(i));
        if (final_pass) {
          new_positions(i) = update;
          if (i + 1 == num_rows_out) {
            final_num_rows_view() = update + count;
          }
        }
        update += count;
      });

  Kokkos::fence();

  std::size_t final_num_rows = 0;
  Kokkos::deep_copy(final_num_rows, final_num_rows_view);

  if (final_num_rows == num_rows_out) {
    // No compaction needed - result is already compact
    return;
  }

  if (final_num_rows == 0) {
    result_out.num_rows = 0;
    result_out.num_intervals = 0;
    return;
  }

  // In-place compaction: shift row_keys and row_ptr
  Kokkos::parallel_for(
      "intersection_compact_copy",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t j) {
        if (has_intervals(j)) {
          const std::size_t new_pos = new_positions(j);
          result_out.row_keys(new_pos) = result_out.row_keys(j);
          result_out.row_ptr(new_pos) = result_out.row_ptr(j);
        }
      });

  // Set final row_ptr value
  Kokkos::parallel_for(
      "intersection_compact_final_ptr",
      Kokkos::RangePolicy<ExecSpace>(0, 1),
      KOKKOS_LAMBDA(const std::size_t) {
        result_out.row_ptr(final_num_rows) = result_out.row_ptr(num_rows_out);
      });

  result_out.num_rows = final_num_rows;
}

// ============================================================================
// 3D version
// ============================================================================

template <typename ExecSpace>
void intersect_meshes_3d_in_place_impl(
    const Mesh3DDevice& A,
    const Mesh3DDevice& B,
    Mesh3DDevice& result_out,
    IntersectionWorkspace3D<ExecSpace>& ws)
{
  using DeviceMemorySpace = typename ExecSpace::memory_space;
  using CoordType = int32_t;
  using IndexType = std::size_t;

  if (A.num_rows == 0 || B.num_rows == 0) {
    result_out.num_rows = 0;
    result_out.num_intervals = 0;
    return;
  }

  const std::size_t num_rows_a = A.num_rows;

  auto flags = ws.flags;
  auto tmp_idx_a = ws.tmp_idx_a;
  auto tmp_idx_b = ws.tmp_idx_b;
  auto positions = ws.positions;

  auto rows_a = A.row_keys;
  auto rows_b = B.row_keys;
  const std::size_t num_rows_b = B.num_rows;

  // Phase 1: Row mapping (3D: search by y,z)
  Kokkos::parallel_for(
      "intersection_row_map_3d",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
      KOKKOS_LAMBDA(const std::size_t i) {
        const RowKey3D<CoordType> key = rows_a(i);
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

  auto num_rows_out_view = ws.num_rows_out_view;

  Kokkos::parallel_scan(
      "intersection_row_scan_3d",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
        const std::size_t count = static_cast<std::size_t>(flags(i));
        if (final_pass) {
          positions(i) = update;
          if (i + 1 == num_rows_a) {
            num_rows_out_view() = update + count;
          }
        }
        update += count;
      });

  Kokkos::fence();

  std::size_t num_rows_out_host = 0;
  Kokkos::deep_copy(num_rows_out_host, num_rows_out_view);
  const std::size_t num_rows_out = num_rows_out_host;

  if (num_rows_out == 0) {
    result_out.num_rows = 0;
    result_out.num_intervals = 0;
    return;
  }

  auto out_rows = ws.out_rows;
  auto out_idx_a = ws.out_idx_a;
  auto out_idx_b = ws.out_idx_b;

  Kokkos::parallel_for(
      "intersection_row_compact_3d",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
      KOKKOS_LAMBDA(const std::size_t i) {
        if (!flags(i)) {
          return;
        }
        const std::size_t pos = positions(i);
        out_rows(pos) = rows_a(i);
        out_idx_a(pos) = tmp_idx_a(i);
        out_idx_b(pos) = tmp_idx_b(i);
      });

  Kokkos::fence();

  // Copy row keys to result mesh
  Kokkos::deep_copy(result_out.row_keys, out_rows);

  result_out.num_rows = num_rows_out;

  auto row_ptr_a = A.row_ptr;
  auto row_ptr_b = B.row_ptr;
  auto intervals_a = A.intervals;
  auto intervals_b = B.intervals;
  auto row_counts = ws.row_counts;

  Kokkos::parallel_for(
      "intersection_count_3d",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        const int ia = out_idx_a(i);
        const int ib = out_idx_b(i);

        if (ib < 0) {
          row_counts(i) = 0;
          return;
        }

        const auto r = intersection::detail::extract_row_ranges(ia, ib, row_ptr_a, row_ptr_b);

        if (r.begin_a == r.end_a || r.begin_b == r.end_b) {
          row_counts(i) = 0;
          return;
        }

        row_counts(i) = detail::row_intersection_impl<true>(
            intervals_a, r.begin_a, r.end_a,
            intervals_b, r.begin_b, r.end_b,
            Kokkos::View<intersection::Interval<CoordType>*, DeviceMemorySpace>(), 0);
      });

  auto total_view = ws.total_view;

  Kokkos::parallel_scan(
      "intersection_scan_3d",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
        const std::size_t count = row_counts(i);
        if (final_pass) {
          result_out.row_ptr(i) = static_cast<IndexType>(update);
          if (i + 1 == num_rows_out) {
            result_out.row_ptr(num_rows_out) = static_cast<IndexType>(update + count);
            total_view() = update + count;
          }
        }
        update += count;
      });

  Kokkos::fence();

  std::size_t num_intervals_host = 0;
  Kokkos::deep_copy(num_intervals_host, total_view);
  result_out.num_intervals = num_intervals_host;

  if (result_out.num_intervals == 0) {
    return;
  }

  Kokkos::parallel_for(
      "intersection_fill_3d",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        const int ia = out_idx_a(i);
        const int ib = out_idx_b(i);

        if (ib < 0) {
          return;
        }

        const auto r = intersection::detail::extract_row_ranges(ia, ib, row_ptr_a, row_ptr_b);

        if (r.begin_a == r.end_a || r.begin_b == r.end_b) {
          return;
        }

        // Use result_out.row_ptr(i) as the offset
        detail::row_intersection_impl<false>(
            intervals_a, r.begin_a, r.end_a,
            intervals_b, r.begin_b, r.end_b,
            result_out.intervals, result_out.row_ptr(i));
      });

  Kokkos::fence();

  // Phase 5: Final compaction
  auto has_intervals = ws.has_intervals;
  auto new_positions = ws.new_positions;
  auto final_num_rows_view = ws.final_num_rows_view;

  Kokkos::parallel_for(
      "intersection_mark_rows_3d",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        has_intervals(i) = (result_out.row_ptr(i) < result_out.row_ptr(i + 1)) ? 1 : 0;
      });

  Kokkos::parallel_scan(
      "intersection_compact_scan_3d",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
        const std::size_t count = static_cast<std::size_t>(has_intervals(i));
        if (final_pass) {
          new_positions(i) = update;
          if (i + 1 == num_rows_out) {
            final_num_rows_view() = update + count;
          }
        }
        update += count;
      });

  Kokkos::fence();

  std::size_t final_num_rows = 0;
  Kokkos::deep_copy(final_num_rows, final_num_rows_view);

  if (final_num_rows == num_rows_out) {
    return;
  }

  if (final_num_rows == 0) {
    result_out.num_rows = 0;
    result_out.num_intervals = 0;
    return;
  }

  Kokkos::parallel_for(
      "intersection_compact_copy_3d",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t j) {
        if (has_intervals(j)) {
          const std::size_t new_pos = new_positions(j);
          result_out.row_keys(new_pos) = result_out.row_keys(j);
          result_out.row_ptr(new_pos) = result_out.row_ptr(j);
        }
      });

  Kokkos::parallel_for(
      "intersection_compact_final_ptr_3d",
      Kokkos::RangePolicy<ExecSpace>(0, 1),
      KOKKOS_LAMBDA(const std::size_t) {
        result_out.row_ptr(final_num_rows) = result_out.row_ptr(num_rows_out);
      });

  result_out.num_rows = final_num_rows;
}

// ============================================================================
// Public API wrappers
// ============================================================================

template <typename ExecSpace>
void intersect_meshes_2d_in_place(
    const Mesh2DDevice& A,
    const Mesh2DDevice& B,
    Mesh2DDevice& result_out,
    IntersectionWorkspace2D<ExecSpace>& workspace)
{
  intersect_meshes_2d_in_place_impl(A, B, result_out, workspace);
}

template <typename ExecSpace>
void intersect_meshes_3d_in_place(
    const Mesh3DDevice& A,
    const Mesh3DDevice& B,
    Mesh3DDevice& result_out,
    IntersectionWorkspace3D<ExecSpace>& workspace)
{
  intersect_meshes_3d_in_place_impl(A, B, result_out, workspace);
}

#endif // SUBSETIX_ENABLE_PLAYGROUND
