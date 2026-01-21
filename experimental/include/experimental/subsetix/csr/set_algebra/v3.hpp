// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <experimental/subsetix/csr/mesh.hpp>
#include <experimental/subsetix/csr/detail/utils.hpp>
#include <experimental/subsetix/csr/set_algebra/v3_helpers.hpp>
#include <Kokkos_Core.hpp>

namespace experimental::subsetix::csr::v3 {

// ============================================================================
// v3 Algorithm: Workqueue + Bounding Box + Optimized Intersection
//
// Key improvements over v1:
// 1. Bounding box early termination (quick reject for non-overlapping meshes)
// 2. Workqueue compaction (eliminates warp divergence on GPU)
// 3. Cached row ranges in registers (reduces memory loads)
// 4. Semi-branchless intersection logic (boolean arithmetic instead of nested if/else)
//
// Implementation notes:
// - Row mapping uses binary search O(log n) - hash map is not implemented due to CUDA complexity
// - "Branchless" reduces but does not eliminate all branches (one if remains for intersection check)
// - Workqueue filters matching rows upfront, allowing all threads in the kernel to do useful work
//
// Expected performance:
// - 2D/3D: 1.2-1.5x faster on GPU (workqueue + cached ranges)
// - CPU: Similar to v1 (workqueue overhead may not justify benefits)
// - Best case: Non-overlapping meshes rejected immediately by bbox check
// ============================================================================

namespace detail {

// ============================================================================
// Semi-branchless intersection loop
// Uses boolean arithmetic for advance logic instead of nested if/else
// Note: Still has one conditional branch for the intersection check (start < end)
// ============================================================================

template <class IntervalViewIn, class IntervalViewOut>
KOKKOS_INLINE_FUNCTION
std::size_t row_intersection_branchless(
    const IntervalViewIn& intervals_a,
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

    // Branchless max/min using ternary (compiles to conditional move)
    const Coord start = (a.begin > b.begin) ? a.begin : b.begin;
    const Coord end = (a.end < b.end) ? a.end : b.end;

    // Branchless write: single predictable branch
    const bool has_intersection = (start < end);
    if (has_intersection) {
      intervals_out(out_offset + count) = Interval{start, end};
      ++count;
    }

    // Branchless advance: compute both, select with comparison
    const bool advance_a = (a.end < b.end);
    const bool advance_b = (b.end < a.end);
    const bool advance_both = !(advance_a || advance_b);

    ia += (advance_a || advance_both) ? 1 : 0;
    ib += (advance_b || advance_both) ? 1 : 0;
  }

  return count;
}

// Count-only version (no output writes)
template <class IntervalViewIn>
KOKKOS_INLINE_FUNCTION
std::size_t row_intersection_count(
    const IntervalViewIn& intervals_a,
    std::size_t begin_a,
    std::size_t end_a,
    const IntervalViewIn& intervals_b,
    std::size_t begin_b,
    std::size_t end_b) {
  std::size_t ia = begin_a;
  std::size_t ib = begin_b;
  std::size_t count = 0;

  while (ia < end_a && ib < end_b) {
    const auto a = intervals_a(ia);
    const auto b = intervals_b(ib);

    const Coord start = (a.begin > b.begin) ? a.begin : b.begin;
    const Coord end = (a.end < b.end) ? a.end : b.end;

    if (start < end) {
      ++count;
    }

    const bool advance_a = (a.end < b.end);
    const bool advance_b = (b.end < a.end);
    const bool advance_both = !(advance_a || advance_b);

    ia += (advance_a || advance_both) ? 1 : 0;
    ib += (advance_b || advance_both) ? 1 : 0;
  }

  return count;
}

} // namespace detail

// ============================================================================
// v3 Mesh intersection with workqueue and bounding box optimization
// ============================================================================

template <int DIM, class MemorySpace>
inline Mesh<DIM, MemorySpace>
intersect_meshes(
    const Mesh<DIM, MemorySpace>& A,
    const Mesh<DIM, MemorySpace>& B) {

  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using MeshType = Mesh<DIM, MemorySpace>;
  using RowKey = typename MeshType::RowKey;

  // ========================================================================
  // Phase 0: Early termination with bounding box check
  // ========================================================================

  // Quick check for empty inputs
  if (A.num_rows == 0 || B.num_rows == 0) {
    return MeshType{};
  }

  // Compute bounding boxes and check overlap
  auto bbox_a = detail::compute_mesh_bbox(A);
  auto bbox_b = detail::compute_mesh_bbox(B);

  if (!detail::bboxes_overlap(bbox_a, bbox_b)) {
    return MeshType{};  // No spatial overlap
  }

  const std::size_t num_rows_a = A.num_rows;
  const std::size_t num_rows_b = B.num_rows;

  // ========================================================================
  // Phase 1: Find matching rows using binary search O(log n)
  // ========================================================================

  Kokkos::View<int*, MemorySpace> tmp_idx_a("tmp_idx_a", num_rows_a);
  Kokkos::View<int*, MemorySpace> tmp_idx_b("tmp_idx_b", num_rows_a);
  Kokkos::View<int*, MemorySpace> flags("flags", num_rows_a);

  if constexpr (DIM == 2) {
    Kokkos::parallel_for(
        "v3_row_map",
        Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          const RowKey key = A.row_keys(i);
          int idx_b = csr::detail::find_row_by_y(B.row_keys, num_rows_b, key.y);
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
    Kokkos::parallel_for(
        "v3_row_map",
        Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          const RowKey key = A.row_keys(i);
          int idx_b = -1;
          idx_b = csr::detail::find_row_by_yz(B.row_keys, num_rows_b, key.y, key.z);
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

  // ========================================================================
  // Phase 3: Create compacted workqueue (eliminates warp divergence)
  // ========================================================================

  Kokkos::View<std::size_t*, MemorySpace> positions("positions", num_rows_a);
  Kokkos::View<std::size_t, MemorySpace> num_rows_out_view("num_rows_out");
  Kokkos::deep_copy(num_rows_out_view, std::size_t(0));

  Kokkos::parallel_scan(
      "v3_count_matches",
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

  std::size_t num_rows_out = 0;
  Kokkos::deep_copy(num_rows_out, num_rows_out_view);

  if (num_rows_out == 0) {
    return MeshType{};
  }

  // Create workqueue with only matching rows
  Kokkos::View<int*, MemorySpace> workqueue("workqueue", num_rows_out);

  Kokkos::parallel_for(
      "v3_build_workqueue",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
      KOKKOS_LAMBDA(const std::size_t i) {
        if (flags(i)) {
          const std::size_t out_pos = positions(i);
          workqueue(out_pos) = static_cast<int>(i);
        }
      });

  // ========================================================================
  // Phase 4: Count intersections (only for matching rows)
  // ========================================================================

  Kokkos::View<std::size_t*, MemorySpace> row_counts("row_counts", num_rows_out);

  Kokkos::parallel_for(
      "v3_count",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t j) {
        const std::size_t i = workqueue(j);  // No early return - all rows match!
        const int ia = tmp_idx_a(i);
        const int ib = tmp_idx_b(i);

        // Cache row ranges in registers (eliminates redundant lookups)
        const std::size_t begin_a = A.row_ptr(ia);
        const std::size_t end_a = A.row_ptr(ia + 1);
        const std::size_t begin_b = B.row_ptr(ib);
        const std::size_t end_b = B.row_ptr(ib + 1);

        // Count intersections with branchless logic
        row_counts(j) = detail::row_intersection_count(
            A.intervals, begin_a, end_a,
            B.intervals, begin_b, end_b);
      });

  // ========================================================================
  // Phase 5: Scan to compute row_ptr offsets
  // ========================================================================

  MeshType out;
  out.row_keys = typename MeshType::RowKeyView("out_row_keys", num_rows_out);
  out.row_ptr = typename MeshType::IndexView("out_row_ptr", num_rows_out + 1);
  out.num_rows = num_rows_out;

  Kokkos::View<std::size_t, MemorySpace> total_view("total");
  Kokkos::parallel_scan(
      "v3_scan",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t j, std::size_t& update, const bool final_pass) {
        const std::size_t count = row_counts(j);
        if (final_pass) {
          out.row_ptr(j) = update;
        }
        update += count;
        if (j + 1 == num_rows_out && final_pass) {
          out.row_ptr(num_rows_out) = update;
          total_view() = update;
        }
      });

  Kokkos::fence();

  std::size_t num_intervals = 0;
  Kokkos::deep_copy(num_intervals, total_view);
  out.num_intervals = num_intervals;

  if (num_intervals == 0) {
    return MeshType{};
  }

  out.intervals = typename MeshType::IntervalView("out_intervals", num_intervals);

  // ========================================================================
  // Phase 6: Fill row keys and intersected intervals (branchless)
  // ========================================================================

  Kokkos::parallel_for(
      "v3_fill",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t j) {
        const std::size_t i = workqueue(j);  // No divergence here!
        const int ia = tmp_idx_a(i);
        const int ib = tmp_idx_b(i);

        out.row_keys(j) = A.row_keys(i);

        // Cached row ranges
        const std::size_t begin_a = A.row_ptr(ia);
        const std::size_t end_a = A.row_ptr(ia + 1);
        const std::size_t begin_b = B.row_ptr(ib);
        const std::size_t end_b = B.row_ptr(ib + 1);

        // Branchless intersection
        detail::row_intersection_branchless(
            A.intervals, begin_a, end_a,
            B.intervals, begin_b, end_b,
            out.intervals, out.row_ptr(j));
      });

  Kokkos::fence();

  return out;
}

// ============================================================================
// Convenience wrappers
// ============================================================================

inline Mesh2DDevice intersect_meshes_2d(const Mesh2DDevice& A, const Mesh2DDevice& B) {
  return intersect_meshes<2>(A, B);
}

inline Mesh3DDevice intersect_meshes_3d(const Mesh3DDevice& A, const Mesh3DDevice& B) {
  return intersect_meshes<3>(A, B);
}

} // namespace experimental::subsetix::csr::v3
