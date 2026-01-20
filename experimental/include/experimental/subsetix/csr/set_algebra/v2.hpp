// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <experimental/subsetix/csr/mesh.hpp>
#include <experimental/subsetix/csr/detail/utils.hpp>
#include <experimental/subsetix/csr/set_algebra/v2_workspace.hpp>
#include <Kokkos_Core.hpp>

namespace experimental::subsetix::csr::v2 {

namespace detail {

// ============================================================================
// Single-pass row intersection (writes directly to output buffer)
// ============================================================================

/**
 * @brief Single-pass intersection that writes directly to output buffer.
 *
 * Unlike v1's row_intersection_impl<CountOnly> which requires two passes,
 * this version writes directly during the first pass.
 *
 * Returns the number of intervals written.
 */
template <class IntervalViewIn, class IntervalViewOut>
KOKKOS_INLINE_FUNCTION
std::size_t row_intersection_single_pass(
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

    // Compute intersection: [max(begin), min(end))
    const Coord start = (a.begin > b.begin) ? a.begin : b.begin;
    const Coord end = (a.end < b.end) ? a.end : b.end;

    // Add non-empty intersection (write immediately!)
    if (start < end) {
      intervals_out(out_offset + count) = Interval{start, end};
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
// Mesh intersection (2D and 3D) - v2 Algorithm
// ============================================================================

/**
 * @brief v2 mesh intersection with workspace.
 *
 * v2 improvements over v1:
 * - Single-pass interval intersection - no separate count+fill phases
 * - Reusable workspace - eliminates per-operation allocations
 *
 * Algorithm:
 * 1. Find matching rows via binary search (same as v1)
 * 2. Single-pass intersection with workspace buffers
 * 3. Compact output
 *
 * @tparam DIM Dimension (2 for 2D, 3 for 3D)
 * @param A First input mesh
 * @param B Second input mesh
 * @param workspace Reusable workspace (will grow as needed)
 * @return Intersection mesh
 */
template <int DIM, class MemorySpace>
inline Mesh<DIM, MemorySpace>
intersect_meshes(
    const Mesh<DIM, MemorySpace>& A,
    const Mesh<DIM, MemorySpace>& B,
    MeshIntersectionWorkspace<MemorySpace>& workspace) {

  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using MeshType = Mesh<DIM, MemorySpace>;
  using RowKey = typename MeshType::RowKey;

  // Early exit for empty inputs
  if (A.num_rows == 0 || B.num_rows == 0) {
    return MeshType{};
  }

  const std::size_t num_rows_a = A.num_rows;
  const std::size_t num_rows_b = B.num_rows;

  // ========================================================================
  // Phase 1: Row mapping - find rows of A that exist in B (using binary search)
  // ========================================================================

  // FIX: Allocate Views directly instead of using workspace to isolate the issue
  Kokkos::View<int*, MemorySpace> flags("flags", num_rows_a);
  Kokkos::View<int*, MemorySpace> tmp_idx_a("tmp_idx_a", num_rows_a);
  Kokkos::View<int*, MemorySpace> tmp_idx_b("tmp_idx_b", num_rows_a);
  Kokkos::View<std::size_t*, MemorySpace> positions("positions", num_rows_a);

  // Capture B.row_keys and num_rows_b once to avoid CUDA constexpr-if capture issues
  const auto B_row_keys = B.row_keys;
  const std::size_t num_rows_b_val = num_rows_b;

  if constexpr (DIM == 2) {
    Kokkos::parallel_for(
        "v2_row_map_2d",
        Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          const RowKey key = A.row_keys(i);
          const int idx_b = csr::detail::find_row_by_y(B_row_keys, num_rows_b_val, key.y);
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
        "v2_row_map_3d",
        Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          const RowKey key = A.row_keys(i);
          const int idx_b = csr::detail::find_row_by_yz(B_row_keys, num_rows_b_val, key.y, key.z);
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
  // Phase 2: Scan to count matching rows and compute positions
  // ========================================================================

  Kokkos::View<std::size_t, MemorySpace> num_rows_out_view("num_rows_out");
  Kokkos::deep_copy(num_rows_out_view, std::size_t(0));

  Kokkos::parallel_scan(
      "v2_row_scan",
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

  Kokkos::fence();  // FIX: Use Kokkos::fence() instead of ExecSpace().fence()

  std::size_t num_rows_out = 0;
  Kokkos::deep_copy(num_rows_out, num_rows_out_view);

  if (num_rows_out == 0) {
    return MeshType{};
  }

  // ========================================================================
  // Phase 3: Allocate output mesh and count intersections
  // ========================================================================

  MeshType out;
  out.row_keys = typename MeshType::RowKeyView("out_row_keys", num_rows_out);
  out.row_ptr = typename MeshType::IndexView("out_row_ptr", num_rows_out + 1);
  out.intervals = typename MeshType::IntervalView("out_intervals", A.num_intervals + B.num_intervals);
  out.num_rows = num_rows_out;

  // FIX: Allocate directly instead of using workspace
  Kokkos::View<std::size_t*, MemorySpace> row_counts("row_counts", num_rows_a);

  Kokkos::parallel_for(
      "v2_count",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
      KOKKOS_LAMBDA(const std::size_t i) {
        if (!flags(i)) {
          row_counts(i) = 0;
          return;
        }
        const int ia = tmp_idx_a(i);
        const int ib = tmp_idx_b(i);

        auto r = csr::detail::extract_row_ranges(ia, ib, A.row_ptr, B.row_ptr);

        // Count intersections using counting-only logic
        std::size_t int_a = r.begin_a;
        std::size_t int_b = r.begin_b;
        std::size_t count = 0;

        while (int_a < r.end_a && int_b < r.end_b) {
          const auto a = A.intervals(int_a);
          const auto b = B.intervals(int_b);

          const Coord start = (a.begin > b.begin) ? a.begin : b.begin;
          const Coord end = (a.end < b.end) ? a.end : b.end;

          if (start < end) {
            ++count;
          }

          if (a.end < b.end) {
            ++int_a;
          } else if (b.end < a.end) {
            ++int_b;
          } else {
            ++int_a;
            ++int_b;
          }
        }

        row_counts(i) = count;
      });

  // ========================================================================
  // Phase 4: Scan to compute row_ptr offsets
  // ========================================================================

  Kokkos::View<std::size_t, MemorySpace> total_view("total");
  Kokkos::parallel_scan(
      "v2_scan",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
        // FIX: Follow v1's pattern exactly - no early return, proper handling
        const std::size_t count = static_cast<std::size_t>(row_counts(i));

        // Only process matching rows
        if (flags(i)) {
          const std::size_t out_pos = positions(i);
          if (final_pass) {
            out.row_ptr(out_pos) = update;
          }
          update += count;
        }
        if (i + 1 == num_rows_a && final_pass) {
          out.row_ptr(num_rows_out) = update;
          total_view() = update;
        }
      });

  Kokkos::fence();  // FIX: Use Kokkos::fence()

  std::size_t num_intervals = 0;
  Kokkos::deep_copy(num_intervals, total_view);
  out.num_intervals = num_intervals;

  if (num_intervals == 0) {
    return MeshType{};
  }

  // ========================================================================
  // Phase 5: Fill row keys and intersected intervals
  // ========================================================================

  Kokkos::parallel_for(
      "v2_fill",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
      KOKKOS_LAMBDA(const std::size_t i) {
        if (!flags(i)) return;

        const std::size_t out_pos = positions(i);
        const int ia = tmp_idx_a(i);
        const int ib = tmp_idx_b(i);  // FIX: No cast needed

        out.row_keys(out_pos) = A.row_keys(i);

        auto r = csr::detail::extract_row_ranges(ia, ib, A.row_ptr, B.row_ptr);

        detail::row_intersection_single_pass(
            A.intervals, r.begin_a, r.end_a,
            B.intervals, r.begin_b, r.end_b,
            out.intervals, out.row_ptr(out_pos));
      });

  Kokkos::fence();  // FIX: Use Kokkos::fence()

  // ========================================================================
  // Phase 6: Compact - remove rows with no intervals
  // ========================================================================

  // FIX: Allocate directly instead of reusing workspace buffers
  Kokkos::View<int*, MemorySpace> has_intervals("has_intervals", num_rows_out);
  Kokkos::View<std::size_t*, MemorySpace> new_positions("new_positions", num_rows_out);

  Kokkos::parallel_for(
      "v2_mark_rows",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        has_intervals(i) = (out.row_ptr(i) < out.row_ptr(i + 1)) ? 1 : 0;
      });

  Kokkos::View<std::size_t, MemorySpace> final_num_rows_view("final_num_rows");
  Kokkos::parallel_scan(
      "v2_compact_scan",
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

  ExecSpace().fence();

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
      "v2_compact_copy",
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
      "v2_compact_final_ptr",
      Kokkos::RangePolicy<ExecSpace>(0, 1),
      KOKKOS_LAMBDA(const std::size_t) {
        compacted.row_ptr(final_num_rows) = out.row_ptr(num_rows_out);
      });

  // Copy intervals
  Kokkos::parallel_for(
      "v2_compact_intervals",
      Kokkos::RangePolicy<ExecSpace>(0, out.num_intervals),
      KOKKOS_LAMBDA(const std::size_t i) {
        compacted.intervals(i) = out.intervals(i);
      });

  Kokkos::fence();  // FIX: Use Kokkos::fence()

  return compacted;
}

// ========================================================================
// Convenience wrappers (without workspace parameter)
// ========================================================================

/**
 * @brief v2 intersection with automatic workspace creation.
 *
 * Creates a temporary workspace for the operation.
 * For repeated operations, use the workspace version for better performance.
 */
template <int DIM>
inline Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space>
intersect_meshes(
    const Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space>& A,
    const Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space>& B) {

  using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
  MeshIntersectionWorkspace<MemorySpace> workspace;
  return intersect_meshes<DIM>(A, B, workspace);
}

// Convenience aliases for 2D and 3D
inline Mesh2DDevice intersect_meshes_2d(const Mesh2DDevice& A, const Mesh2DDevice& B) {
  return intersect_meshes<2>(A, B);
}

inline Mesh3DDevice intersect_meshes_3d(const Mesh3DDevice& A, const Mesh3DDevice& B) {
  return intersect_meshes<3>(A, B);
}

// Workspace versions
inline Mesh2DDevice intersect_meshes_2d(const Mesh2DDevice& A, const Mesh2DDevice& B,
                                        MeshIntersectionWorkspace<Kokkos::DefaultExecutionSpace::memory_space>& ws) {
  return intersect_meshes<2>(A, B, ws);
}

inline Mesh3DDevice intersect_meshes_3d(const Mesh3DDevice& A, const Mesh3DDevice& B,
                                        MeshIntersectionWorkspace<Kokkos::DefaultExecutionSpace::memory_space>& ws) {
  return intersect_meshes<3>(A, B, ws);
}

} // namespace experimental::subsetix::csr::v2
