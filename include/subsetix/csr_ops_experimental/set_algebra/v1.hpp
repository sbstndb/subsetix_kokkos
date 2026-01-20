// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <subsetix/csr_ops_experimental/geometry/mesh.hpp>
#include <subsetix/csr_ops_experimental/detail/utils.hpp>

namespace subsetix::experimental::v1 {

namespace detail {

/**
 * @brief Core row intersection algorithm (two-pointer merge).
 *
 * When CountOnly=true, only counts intervals without writing.
 * When CountOnly=false, writes intervals to intervals_out.
 *
 * This is dimension-agnostic: works for both 2D and 3D meshes.
 */
template <bool CountOnly, class IntervalViewIn, class IntervalViewOut>
KOKKOS_INLINE_FUNCTION
std::size_t row_intersection_impl(const IntervalViewIn& intervals_a,
                                  std::size_t begin_a,
                                  std::size_t end_a,
                                  const IntervalViewIn& intervals_b,
                                  std::size_t begin_b,
                                  std::size_t end_b,
                                  const IntervalViewOut& intervals_out,
                                  std::size_t out_offset) {
  std::size_t ia = begin_a;
  std::size_t ib = begin_b;
  std::size_t count = 0;

  while (ia < end_a && ib < end_b) {
    const auto a = intervals_a(ia);
    const auto b = intervals_b(ib);

    // Compute intersection: [max(begin), min(end))
    const Coord start = (a.begin > b.begin) ? a.begin : b.begin;
    const Coord end = (a.end < b.end) ? a.end : b.end;

    // Add non-empty intersection
    if (start < end) {
      if constexpr (!CountOnly) {
        intervals_out(out_offset + count) = Interval{start, end};
      }
      ++count;
    }

    // Advance the interval that ends first
    if (a.end < b.end) {
      ++ia;
    } else if (b.end < a.end) {
      ++ib;
    } else {
      ++ia;
      ++ib;
    }
  }

  return count;
}

} // namespace detail

// ============================================================================
// Mesh intersection (2D and 3D)
// ============================================================================

/**
 * @brief Compute the intersection of two meshes (2D or 3D).
 *
 * Returns a new mesh containing only the cells that exist in BOTH input meshes.
 *
 * Algorithm:
 * 1. Row mapping - find common rows via binary search
 * 2. Count - count intersecting X-intervals per row
 * 3. Scan - compute CSR offsets
 * 4. Fill - write intersected intervals
 * 5. Compact - filter rows with no intersections
 *
 * @tparam DIM Dimension (2 for 2D, 3 for 3D)
 * @param A First input mesh
 * @param B Second input mesh
 * @return Intersection mesh
 */
template <int DIM>
inline Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space>
intersect_meshes(const Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space>& A,
                const Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space>& B) {
  using DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using MeshType = Mesh<DIM, DeviceMemorySpace>;
  using RowKey = typename MeshType::RowKey;

  if (A.num_rows == 0 || B.num_rows == 0) {
    return MeshType{};
  }

  const std::size_t num_rows_a = A.num_rows;
  Kokkos::View<int*, DeviceMemorySpace> flags("flags", num_rows_a);
  Kokkos::View<int*, DeviceMemorySpace> tmp_idx_a("tmp_idx_a", num_rows_a);
  Kokkos::View<int*, DeviceMemorySpace> tmp_idx_b("tmp_idx_b", num_rows_a);
  Kokkos::View<std::size_t*, DeviceMemorySpace> positions("positions", num_rows_a);

  auto rows_a = A.row_keys;
  auto rows_b = B.row_keys;
  const std::size_t num_rows_b = B.num_rows;

  // Phase 1: Row mapping - find rows of A that exist in B
  if constexpr (DIM == 2) {
    // 2D: search by y only
    Kokkos::parallel_for(
        "intersection_row_map_2d",
        Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          const RowKey key = rows_a(i);
          const int idx_b = subsetix::experimental::detail::find_row_by_y(rows_b, num_rows_b, key.y);
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
  } else {
    // 3D: search by (y,z)
    Kokkos::parallel_for(
        "intersection_row_map_3d",
        Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          const RowKey key = rows_a(i);
          const int idx_b = subsetix::experimental::detail::find_row_by_yz(rows_b, num_rows_b, key.y, key.z);
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
  }

  Kokkos::fence();

  // Scan to count matching rows and compute positions
  Kokkos::View<std::size_t, DeviceMemorySpace> num_rows_out_view("num_rows_out");
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
    return MeshType{};
  }

  // Allocate output buffers for row mapping
  Kokkos::View<RowKey*, DeviceMemorySpace> out_rows("out_rows", num_rows_out);
  Kokkos::View<int*, DeviceMemorySpace> out_idx_a("out_idx_a", num_rows_out);
  Kokkos::View<int*, DeviceMemorySpace> out_idx_b("out_idx_b", num_rows_out);

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

  // Allocate output mesh
  MeshType out;
  if (num_rows_out > 0) {
    out.row_keys = typename MeshType::RowKeyView("mesh_row_keys", num_rows_out);
    out.row_ptr = typename MeshType::IndexView("mesh_row_ptr", num_rows_out + 1);
    out.intervals = typename MeshType::IntervalView(
        "mesh_intervals", A.num_intervals + B.num_intervals);
  }

  // Copy row keys
  Kokkos::deep_copy(out.row_keys, out_rows);

  // Allocate row counts buffer
  Kokkos::View<std::size_t*, DeviceMemorySpace> row_counts("row_counts", num_rows_out);

  auto row_ptr_a = A.row_ptr;
  auto row_ptr_b = B.row_ptr;
  auto intervals_a = A.intervals;
  auto intervals_b = B.intervals;

  // Phase 2: Count intervals per row
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

        const auto r = subsetix::experimental::detail::extract_row_ranges(ia, ib, row_ptr_a, row_ptr_b);

        if (r.begin_a == r.end_a || r.begin_b == r.end_b) {
          row_counts(i) = 0;
          return;
        }

        row_counts(i) = detail::row_intersection_impl<true>(
            intervals_a, r.begin_a, r.end_a,
            intervals_b, r.begin_b, r.end_b,
            Kokkos::View<Interval*, DeviceMemorySpace>(), 0);
      });

  // Phase 3: Scan to compute row_ptr offsets
  Kokkos::View<std::size_t, DeviceMemorySpace> total_view("total_intervals");
  Kokkos::parallel_scan(
      "intersection_scan",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
        const std::size_t count = row_counts(i);
        if (final_pass) {
          out.row_ptr(i) = update;
          if (i + 1 == num_rows_out) {
            out.row_ptr(num_rows_out) = update + count;
            total_view() = update + count;
          }
        }
        update += count;
      });

  std::size_t num_intervals_host = 0;
  Kokkos::deep_copy(num_intervals_host, total_view);
  out.num_intervals = num_intervals_host;
  out.num_rows = num_rows_out;

  if (out.num_intervals == 0) {
    return MeshType{};
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

        const auto r = subsetix::experimental::detail::extract_row_ranges(ia, ib, row_ptr_a, row_ptr_b);

        if (r.begin_a == r.end_a || r.begin_b == r.end_b) {
          return;
        }

        detail::row_intersection_impl<false>(
            intervals_a, r.begin_a, r.end_a,
            intervals_b, r.begin_b, r.end_b,
            out.intervals, out.row_ptr(i));
      });

  // Phase 5: Compact - remove rows with no intervals
  Kokkos::View<int*, DeviceMemorySpace> has_intervals("has_intervals", num_rows_out);
  Kokkos::parallel_for(
      "intersection_mark_rows",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        has_intervals(i) = (out.row_ptr(i) < out.row_ptr(i + 1)) ? 1 : 0;
      });

  Kokkos::View<std::size_t*, DeviceMemorySpace> new_positions("new_positions", num_rows_out);
  Kokkos::View<std::size_t, DeviceMemorySpace> final_num_rows_view("final_num_rows");
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

  std::size_t final_num_rows = 0;
  Kokkos::deep_copy(final_num_rows, final_num_rows_view);

  if (final_num_rows == num_rows_out) {
    return out;  // No compaction needed
  }

  if (final_num_rows == 0) {
    return MeshType{};
  }

  // Allocate compacted output
  MeshType compacted;
  compacted.row_keys = typename MeshType::RowKeyView("compacted_row_keys", final_num_rows);
  compacted.row_ptr = typename MeshType::IndexView("compacted_row_ptr", final_num_rows + 1);
  compacted.intervals = typename MeshType::IntervalView("compacted_intervals", out.num_intervals);
  compacted.num_rows = final_num_rows;
  compacted.num_intervals = out.num_intervals;

  // Copy non-empty rows
  Kokkos::parallel_for(
      "intersection_compact_copy",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t j) {
        if (has_intervals(j)) {
          const std::size_t new_pos = new_positions(j);
          compacted.row_keys(new_pos) = out.row_keys(j);
          compacted.row_ptr(new_pos) = out.row_ptr(j);
        }
      });

  // Set final row_ptr value
  Kokkos::parallel_for(
      "intersection_compact_final_ptr",
      Kokkos::RangePolicy<ExecSpace>(0, 1),
      KOKKOS_LAMBDA(const std::size_t) {
        compacted.row_ptr(final_num_rows) = out.row_ptr(num_rows_out);
      });

  // Copy intervals
  Kokkos::parallel_for(
      "intersection_compact_intervals",
      Kokkos::RangePolicy<ExecSpace>(0, out.num_intervals),
      KOKKOS_LAMBDA(const std::size_t i) {
        compacted.intervals(i) = out.intervals(i);
      });

  return compacted;
}

// Convenience aliases for 2D and 3D
inline Mesh2DDevice intersect_meshes_2d(const Mesh2DDevice& A, const Mesh2DDevice& B) {
  return intersect_meshes<2>(A, B);
}

inline Mesh3DDevice intersect_meshes_3d(const Mesh3DDevice& A, const Mesh3DDevice& B) {
  return intersect_meshes<3>(A, B);
}

// ============================================================================
// Conversion between memory spaces
// ============================================================================

/**
 * @brief Convert a mesh between memory spaces (e.g., Device -> Host).
 */
template <int DIM, class ToSpace, class FromSpace>
inline Mesh<DIM, ToSpace> mesh_to(const Mesh<DIM, FromSpace>& src) {
  Mesh<DIM, ToSpace> dst;

  if (src.num_rows == 0) {
    return dst;
  }

  dst.num_rows = src.num_rows;
  dst.num_intervals = src.num_intervals;

  dst.row_keys = Kokkos::create_mirror_view_and_copy(ToSpace{}, src.row_keys);
  dst.row_ptr = Kokkos::create_mirror_view_and_copy(ToSpace{}, src.row_ptr);
  dst.intervals = Kokkos::create_mirror_view_and_copy(ToSpace{}, src.intervals);

  return dst;
}

} // namespace subsetix::experimental::v1
