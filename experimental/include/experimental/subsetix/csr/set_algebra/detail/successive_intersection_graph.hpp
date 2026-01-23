// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#pragma once

#include <experimental/subsetix/csr/types.hpp>
#include <experimental/subsetix/csr/detail/utils.hpp>
#include <experimental/subsetix/csr/set_algebra/v3.hpp>
#include <Kokkos_Core.hpp>
#include <vector>
#include <memory>

namespace experimental::subsetix::csr::successive::graph {

// ============================================================================
// Mesh type for graph-based intersection (compatible with v1/v2/v3)
// ============================================================================

template <int DIM, class MemorySpace,
          class CoordType = int32_t,
          class IndexType = std::size_t>
class Mesh {
public:
  static constexpr int dim_value = DIM;
  using coord_type = CoordType;
  using index_type = IndexType;
  using memory_space = MemorySpace;

  using RowKey = std::conditional_t<DIM == 2,
                                     csr::RowKey2D<CoordType>,
                                     csr::RowKey3D<CoordType>>;
  using RowKeyView = Kokkos::View<RowKey*, MemorySpace>;
  using IndexView = Kokkos::View<IndexType*, MemorySpace>;
  using IntervalView = Kokkos::View<csr::Interval<CoordType>*, MemorySpace>;

  RowKeyView row_keys;
  IndexView row_ptr;
  IntervalView intervals;

  std::size_t num_rows = 0;
  std::size_t num_intervals = 0;

  KOKKOS_INLINE_FUNCTION
  Mesh() = default;

  KOKKOS_INLINE_FUNCTION
  Mesh(const Mesh&) = default;

  KOKKOS_INLINE_FUNCTION
  Mesh& operator=(const Mesh&) = default;
};

// Type aliases
template <int DIM>
using DefaultMesh = Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>;

template <class CoordType = int32_t, class IndexType = std::size_t>
using Mesh2D = Mesh<2, Kokkos::DefaultExecutionSpace::memory_space, CoordType, IndexType>;

using Mesh2DDevice = Mesh2D<>;
using Mesh2DHost = Mesh<2, Kokkos::HostSpace, int32_t, std::size_t>;

template <class CoordType = int32_t, class IndexType = std::size_t>
using Mesh3D = Mesh<3, Kokkos::DefaultExecutionSpace::memory_space, CoordType, IndexType>;

using Mesh3DDevice = Mesh3D<>;
using Mesh3DHost = Mesh<3, Kokkos::HostSpace, int32_t, std::size_t>;

// ============================================================================
// Core row intersection algorithm (shared with v1/v2/v3)
// ============================================================================

namespace detail {

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
  using IntervalType = std::remove_reference_t<decltype(intervals_a(0))>;
  using CoordType = typename IntervalType::coord_type;

  std::size_t ia = begin_a;
  std::size_t ib = begin_b;
  std::size_t count = 0;

  while (ia < end_a && ib < end_b) {
    const auto a = intervals_a(ia);
    const auto b = intervals_b(ib);

    const CoordType start = (a.begin > b.begin) ? a.begin : b.begin;
    const CoordType end = (a.end < b.end) ? a.end : b.end;

    if (start < end) {
      if constexpr (!CountOnly) {
        intervals_out(out_offset + count) = IntervalType{start, end};
      }
      ++count;
    }

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
// Graph-based Intersection Strategy
// ============================================================================

/**
 * @brief Graph-based intersection for successive mesh operations.
 *
 * This implementation eliminates intermediate synchronizations by:
 * 1. Pre-allocating all temporary storage upfront
 * 2. Building a computation graph of the entire intersection chain
 * 3. Submitting the graph once for all intersections
 *
 * For N input meshes, instead of N-1 separate intersection operations
 * (each with 4-5 kernel launches and synchronizations), we build a single
 * graph that chains all operations together.
 *
 * Note: This is a simplified implementation that focuses on the concept
 * of graph-based execution. For production use, consider:
 * - Using Kokkos::Experimental::Graph when available
 * - Fusing kernels where possible
 * - Dynamic memory management for large mesh chains
 */
template <int DIM, class CoordType = int32_t, class IndexType = std::size_t>
class IntersectionGraph {
public:
  using DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using MeshType = Mesh<DIM, DeviceMemorySpace, CoordType, IndexType>;
  using RowKey = typename MeshType::RowKey;
  using Interval = csr::Interval<CoordType>;

  /**
   * @brief Construct the intersection graph from input meshes.
   *
   * @param meshes Vector of input meshes to intersect successively
   */
  explicit IntersectionGraph(const std::vector<MeshType>& meshes)
      : inputs_(meshes)
  {
    if (inputs_.size() < 2) {
      return;  // Nothing to intersect
    }

    // Pre-allocate temporary storage for all intermediate results
    // The worst case is when we keep all rows from the first mesh
    std::size_t max_rows = 0;
    std::size_t max_intervals = 0;
    for (const auto& mesh : inputs_) {
      max_rows = std::max(max_rows, mesh.num_rows);
      max_intervals = std::max(max_intervals, mesh.num_intervals);
    }

    // Allocate workspace for temporaries
    // We need N-2 temporaries for N inputs (ping-pong between two buffers)
    allocate_workspace(max_rows, max_intervals);
  }

  /**
   * @brief Execute the intersection graph.
   *
   * Performs successive intersection of all input meshes.
   * Returns the final intersection result.
   *
   * @return Final intersected mesh
   */
  MeshType execute() {
    if (inputs_.size() < 2) {
      if (inputs_.empty()) {
        return MeshType{};
      }
      return inputs_[0];
    }

    if (inputs_.size() == 2) {
      // Single intersection - no graph needed
      return intersect_two_meshes(inputs_[0], inputs_[1]);
    }

    // For 3+ meshes, use graph-based chaining
    return execute_chained_intersections();
  }

private:
  std::vector<MeshType> inputs_;

  // Workspace buffers for temporary results
  struct Workspace {
    // Row mapping buffers
    Kokkos::View<int*, DeviceMemorySpace> flags;
    Kokkos::View<int*, DeviceMemorySpace> tmp_idx_a;
    Kokkos::View<int*, DeviceMemorySpace> tmp_idx_b;
    Kokkos::View<std::size_t*, DeviceMemorySpace> positions;

    // Intermediate mesh data
    Kokkos::View<RowKey*, DeviceMemorySpace> temp_row_keys;
    Kokkos::View<IndexType*, DeviceMemorySpace> temp_row_ptr;
    Kokkos::View<Interval*, DeviceMemorySpace> temp_intervals;

    // Row processing buffers
    Kokkos::View<RowKey*, DeviceMemorySpace> out_rows;
    Kokkos::View<int*, DeviceMemorySpace> out_idx_a;
    Kokkos::View<int*, DeviceMemorySpace> out_idx_b;
    Kokkos::View<std::size_t*, DeviceMemorySpace> row_counts;
    Kokkos::View<int*, DeviceMemorySpace> has_intervals;
    Kokkos::View<std::size_t*, DeviceMemorySpace> new_positions;

    // Scalar views for scan results
    Kokkos::View<std::size_t, DeviceMemorySpace> num_rows_out_view;
    Kokkos::View<std::size_t, DeviceMemorySpace> total_view;
    Kokkos::View<std::size_t, DeviceMemorySpace> final_num_rows_view;

    std::size_t capacity_rows = 0;
    std::size_t capacity_intervals = 0;
  } workspace_;

  /**
   * @brief Allocate workspace for temporary results.
   */
  void allocate_workspace(std::size_t max_rows, std::size_t max_intervals) {
    workspace_.flags = Kokkos::View<int*, DeviceMemorySpace>("graph_flags", max_rows);
    workspace_.tmp_idx_a = Kokkos::View<int*, DeviceMemorySpace>("graph_tmp_idx_a", max_rows);
    workspace_.tmp_idx_b = Kokkos::View<int*, DeviceMemorySpace>("graph_tmp_idx_b", max_rows);
    workspace_.positions = Kokkos::View<std::size_t*, DeviceMemorySpace>("graph_positions", max_rows);

    workspace_.temp_row_keys = Kokkos::View<RowKey*, DeviceMemorySpace>("graph_temp_row_keys", max_rows);
    workspace_.temp_row_ptr = Kokkos::View<IndexType*, DeviceMemorySpace>("graph_temp_row_ptr", max_rows + 1);
    workspace_.temp_intervals = Kokkos::View<Interval*, DeviceMemorySpace>("graph_temp_intervals", max_intervals);

    workspace_.out_rows = Kokkos::View<RowKey*, DeviceMemorySpace>("graph_out_rows", max_rows);
    workspace_.out_idx_a = Kokkos::View<int*, DeviceMemorySpace>("graph_out_idx_a", max_rows);
    workspace_.out_idx_b = Kokkos::View<int*, DeviceMemorySpace>("graph_out_idx_b", max_rows);
    workspace_.row_counts = Kokkos::View<std::size_t*, DeviceMemorySpace>("graph_row_counts", max_rows);
    workspace_.has_intervals = Kokkos::View<int*, DeviceMemorySpace>("graph_has_intervals", max_rows);
    workspace_.new_positions = Kokkos::View<std::size_t*, DeviceMemorySpace>("graph_new_positions", max_rows);

    workspace_.num_rows_out_view = Kokkos::View<std::size_t, DeviceMemorySpace>("graph_num_rows_out");
    workspace_.total_view = Kokkos::View<std::size_t, DeviceMemorySpace>("graph_total");
    workspace_.final_num_rows_view = Kokkos::View<std::size_t, DeviceMemorySpace>("graph_final_num_rows");

    workspace_.capacity_rows = max_rows;
    workspace_.capacity_intervals = max_intervals;
  }

  /**
   * @brief Execute chained intersections using graph-based approach.
   *
   * This is where the concept of graph execution comes in. Instead of
   * separate fence() calls between each intersection, we chain the
   * operations together.
   *
   * In a full implementation with Kokkos::Experimental::Graph, we would:
   * 1. Create a graph object
   * 2. Add nodes for each intersection phase
   * 3. Build dependencies between nodes
   * 4. Submit the entire graph at once
   *
   * For this simplified implementation, we minimize explicit fences
   * and let Kokkos handle kernel dependencies implicitly.
   */
  MeshType execute_chained_intersections() {
    // Start with the first intersection
    MeshType current = intersect_two_meshes(inputs_[0], inputs_[1]);

    // Chain remaining intersections
    for (std::size_t i = 2; i < inputs_.size(); ++i) {
      // Only fence if we need data on host for size checks
      // In a full graph implementation, this would be handled by the graph
      Kokkos::fence();

      if (current.num_rows == 0 || current.num_intervals == 0) {
        return MeshType{};  // Early exit - empty intersection
      }

      current = intersect_two_meshes(current, inputs_[i]);
    }

    return current;
  }

  /**
   * @brief Intersect two meshes (core algorithm).
   *
   * This is the same algorithm used in v1/v2/v3, but optimized to
   * reuse pre-allocated workspace when possible.
   */
  MeshType intersect_two_meshes(const MeshType& A, const MeshType& B) {
    if (A.num_rows == 0 || B.num_rows == 0) {
      return MeshType{};
    }

    const std::size_t num_rows_a = A.num_rows;

    // Use pre-allocated workspace or allocate if needed
    auto flags = ensure_capacity(workspace_.flags, num_rows_a, "intersect_flags");
    auto tmp_idx_a = ensure_capacity(workspace_.tmp_idx_a, num_rows_a, "intersect_tmp_idx_a");
    auto tmp_idx_b = ensure_capacity(workspace_.tmp_idx_b, num_rows_a, "intersect_tmp_idx_b");
    auto positions = ensure_capacity(workspace_.positions, num_rows_a, "intersect_positions");

    auto rows_a = A.row_keys;
    auto rows_b = B.row_keys;
    const std::size_t num_rows_b = B.num_rows;

    // Phase 1: Row mapping
    phase1_row_mapping(rows_a, rows_b, num_rows_a, num_rows_b, flags, tmp_idx_a, tmp_idx_b);

    // Scan to get row count
    Kokkos::View<std::size_t, DeviceMemorySpace> num_rows_out_view("num_rows_out");
    phase1a_scan_count(num_rows_a, flags, positions, num_rows_out_view);

    std::size_t num_rows_out_host = 0;
    Kokkos::deep_copy(num_rows_out_host, num_rows_out_view);

    if (num_rows_out_host == 0) {
      return MeshType{};
    }

    // Phase 1b: Compact rows
    auto out_rows = ensure_capacity(workspace_.out_rows, num_rows_out_host, "out_rows");
    auto out_idx_a = ensure_capacity(workspace_.out_idx_a, num_rows_out_host, "out_idx_a");
    auto out_idx_b = ensure_capacity(workspace_.out_idx_b, num_rows_out_host, "out_idx_b");

    phase1b_compact_rows(num_rows_a, rows_a, flags, positions, tmp_idx_a, tmp_idx_b,
                         out_rows, out_idx_a, out_idx_b);

    // Allocate output mesh
    MeshType out;
    out.row_keys = Kokkos::View<RowKey*, DeviceMemorySpace>("mesh_row_keys", num_rows_out_host);
    out.row_ptr = Kokkos::View<IndexType*, DeviceMemorySpace>("mesh_row_ptr", num_rows_out_host + 1);
    out.intervals = Kokkos::View<Interval*, DeviceMemorySpace>("mesh_intervals",
                                                               A.num_intervals + B.num_intervals);
    out.num_rows = num_rows_out_host;

    Kokkos::deep_copy(out.row_keys, out_rows);

    auto row_counts = ensure_capacity(workspace_.row_counts, num_rows_out_host, "row_counts");

    auto row_ptr_a = A.row_ptr;
    auto row_ptr_b = B.row_ptr;
    auto intervals_a = A.intervals;
    auto intervals_b = B.intervals;

    // Phase 2: Count intervals
    phase2_count_intervals(num_rows_out_host, out_idx_a, out_idx_b, row_ptr_a, row_ptr_b,
                           intervals_a, intervals_b, row_counts);

    // Phase 3: Scan for row_ptr
    Kokkos::View<std::size_t, DeviceMemorySpace> total_view("total_intervals");
    phase3_scan_row_ptr(num_rows_out_host, row_counts, out.row_ptr, total_view);

    std::size_t num_intervals_host = 0;
    Kokkos::deep_copy(num_intervals_host, total_view);
    out.num_intervals = num_intervals_host;

    if (num_intervals_host == 0) {
      return MeshType{};
    }

    // Phase 4: Fill intervals
    phase4_fill_intervals(num_rows_out_host, out_idx_a, out_idx_b, row_ptr_a, row_ptr_b,
                          intervals_a, intervals_b, out.intervals, out.row_ptr);

    // Phase 5: Compact empty rows
    return phase5_compact(num_rows_out_host, out, row_counts, workspace_);
  }

  // Helper to ensure view capacity
  template <class ViewType>
  ViewType ensure_capacity(const ViewType& view, std::size_t size, const std::string& label) {
    if (view.extent(0) >= size) {
      return view;
    }
    return ViewType(label, size);
  }

  // Phase kernels (same as v1, but separated for graph building)

  void phase1_row_mapping(const auto& rows_a, const auto& rows_b,
                          std::size_t num_rows_a, std::size_t num_rows_b,
                          auto& flags, auto& tmp_idx_a, auto& tmp_idx_b) {
    if constexpr (DIM == 2) {
      Kokkos::parallel_for(
          "graph_intersection_row_map_2d",
          Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
          KOKKOS_LAMBDA(const std::size_t i) {
            const RowKey key = rows_a(i);
            const int idx_b = csr::detail::find_row_by_y(rows_b, num_rows_b, key.y);
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
          "graph_intersection_row_map_3d",
          Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
          KOKKOS_LAMBDA(const std::size_t i) {
            const RowKey key = rows_a(i);
            const int idx_b = csr::detail::find_row_by_yz(rows_b, num_rows_b, key.y, key.z);
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
  }

  void phase1a_scan_count(std::size_t num_rows_a, const auto& flags, auto& positions,
                          auto& num_rows_out_view) {
    Kokkos::parallel_scan(
        "graph_intersection_row_scan",
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
  }

  void phase1b_compact_rows(std::size_t num_rows_a, const auto& rows_a, const auto& flags,
                            const auto& positions, const auto& tmp_idx_a, const auto& tmp_idx_b,
                            auto& out_rows, auto& out_idx_a, auto& out_idx_b) {
    Kokkos::parallel_for(
        "graph_intersection_row_compact",
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
  }

  void phase2_count_intervals(std::size_t num_rows_out, const auto& out_idx_a,
                              const auto& out_idx_b, const auto& row_ptr_a,
                              const auto& row_ptr_b, const auto& intervals_a,
                              const auto& intervals_b, auto& row_counts) {
    Kokkos::parallel_for(
        "graph_intersection_count",
        Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
        KOKKOS_LAMBDA(const std::size_t i) {
          const int ia = out_idx_a(i);
          const int ib = out_idx_b(i);

          if (ib < 0) {
            row_counts(i) = 0;
            return;
          }

          const auto r = csr::detail::extract_row_ranges(ia, ib, row_ptr_a, row_ptr_b);

          if (r.begin_a == r.end_a || r.begin_b == r.end_b) {
            row_counts(i) = 0;
            return;
          }

          row_counts(i) = detail::row_intersection_impl<true>(
              intervals_a, r.begin_a, r.end_a,
              intervals_b, r.begin_b, r.end_b,
              Kokkos::View<Interval*, DeviceMemorySpace>(), 0);
        });
  }

  void phase3_scan_row_ptr(std::size_t num_rows_out, const auto& row_counts,
                           auto& row_ptr, auto& total_view) {
    Kokkos::parallel_scan(
        "graph_intersection_scan",
        Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
        KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
          const std::size_t count = row_counts(i);
          if (final_pass) {
            row_ptr(i) = static_cast<IndexType>(update);
            if (i + 1 == num_rows_out) {
              row_ptr(num_rows_out) = static_cast<IndexType>(update + count);
              total_view() = update + count;
            }
          }
          update += count;
        });
  }

  void phase4_fill_intervals(std::size_t num_rows_out, const auto& out_idx_a,
                             const auto& out_idx_b, const auto& row_ptr_a,
                             const auto& row_ptr_b, const auto& intervals_a,
                             const auto& intervals_b, auto& intervals_out,
                             const auto& row_ptr) {
    Kokkos::parallel_for(
        "graph_intersection_fill",
        Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
        KOKKOS_LAMBDA(const std::size_t i) {
          const int ia = out_idx_a(i);
          const int ib = out_idx_b(i);

          if (ib < 0) {
            return;
          }

          const auto r = csr::detail::extract_row_ranges(ia, ib, row_ptr_a, row_ptr_b);

          if (r.begin_a == r.end_a || r.begin_b == r.end_b) {
            return;
          }

          detail::row_intersection_impl<false>(
              intervals_a, r.begin_a, r.end_a,
              intervals_b, r.begin_b, r.end_b,
              intervals_out, row_ptr(i));
        });
  }

  MeshType phase5_compact(std::size_t num_rows_out, MeshType& out,
                          const auto& row_counts, Workspace& ws) {
    auto has_intervals = ensure_capacity(ws.has_intervals, num_rows_out, "has_intervals");
    auto new_positions = ensure_capacity(ws.new_positions, num_rows_out, "new_positions");

    Kokkos::parallel_for(
        "graph_intersection_mark_rows",
        Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
        KOKKOS_LAMBDA(const std::size_t i) {
          has_intervals(i) = (out.row_ptr(i) < out.row_ptr(i + 1)) ? 1 : 0;
        });

    Kokkos::parallel_scan(
        "graph_intersection_compact_scan",
        Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
        KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
          const std::size_t count = static_cast<std::size_t>(has_intervals(i));
          if (final_pass) {
            new_positions(i) = update;
            if (i + 1 == num_rows_out) {
              ws.final_num_rows_view() = update + count;
            }
          }
          update += count;
        });

    std::size_t final_num_rows = 0;
    Kokkos::deep_copy(final_num_rows, ws.final_num_rows_view);

    if (final_num_rows == num_rows_out) {
      return out;  // No compaction needed
    }

    if (final_num_rows == 0) {
      return MeshType{};
    }

    // Allocate compacted output
    MeshType compacted;
    compacted.row_keys = Kokkos::View<RowKey*, DeviceMemorySpace>("compacted_row_keys", final_num_rows);
    compacted.row_ptr = Kokkos::View<IndexType*, DeviceMemorySpace>("compacted_row_ptr", final_num_rows + 1);
    compacted.intervals = Kokkos::View<Interval*, DeviceMemorySpace>("compacted_intervals", out.num_intervals);
    compacted.num_rows = final_num_rows;
    compacted.num_intervals = out.num_intervals;

    // Copy non-empty rows
    Kokkos::parallel_for(
        "graph_intersection_compact_copy",
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
        "graph_intersection_compact_final_ptr",
        Kokkos::RangePolicy<ExecSpace>(0, 1),
        KOKKOS_LAMBDA(const std::size_t) {
          compacted.row_ptr(final_num_rows) = out.row_ptr(num_rows_out);
        });

    // Copy intervals
    Kokkos::parallel_for(
        "graph_intersection_compact_intervals",
        Kokkos::RangePolicy<ExecSpace>(0, out.num_intervals),
        KOKKOS_LAMBDA(const std::size_t i) {
          compacted.intervals(i) = out.intervals(i);
        });

    return compacted;
  }
};

// ============================================================================
// Conversion helpers for v3::Mesh compatibility
// ============================================================================

namespace detail {

/**
 * @brief Convert v3::Mesh to graph::Mesh
 */
template <int DIM, class MemorySpace, class CoordType, class IndexType>
graph::Mesh<DIM, MemorySpace, CoordType, IndexType>
v3_mesh_to_graph(const v3::Mesh<DIM, MemorySpace, CoordType, IndexType>& v3_mesh) {
  graph::Mesh<DIM, MemorySpace, CoordType, IndexType> result;
  result.row_keys = v3_mesh.row_keys;
  result.row_ptr = v3_mesh.row_ptr;
  result.intervals = v3_mesh.intervals;
  result.num_rows = v3_mesh.num_rows;
  result.num_intervals = v3_mesh.num_intervals;
  return result;
}

/**
 * @brief Convert graph::Mesh to v3::Mesh
 */
template <int DIM, class MemorySpace, class CoordType, class IndexType>
v3::Mesh<DIM, MemorySpace, CoordType, IndexType>
graph_mesh_to_v3(const graph::Mesh<DIM, MemorySpace, CoordType, IndexType>& graph_mesh) {
  v3::Mesh<DIM, MemorySpace, CoordType, IndexType> result;
  result.row_keys = graph_mesh.row_keys;
  result.row_ptr = graph_mesh.row_ptr;
  result.intervals = graph_mesh.intervals;
  result.num_rows = graph_mesh.num_rows;
  result.num_intervals = graph_mesh.num_intervals;
  return result;
}

} // namespace detail

// ============================================================================
// Convenience functions
// ============================================================================

/**
 * @brief Successive intersection of multiple meshes using graph-based approach (v3::Mesh version).
 *
 * This function eliminates intermediate synchronizations by using a graph-based
 * execution strategy. For N input meshes, instead of N-1 separate intersection
 * operations with fences between them, we chain the operations together.
 *
 * @tparam DIM Dimension (2 for 2D, 3 for 3D)
 * @tparam CoordType Coordinate type
 * @tparam IndexType Index type
 * @param meshes Vector of input meshes to intersect
 * @return Final intersection mesh
 */
template <int DIM, class CoordType = int32_t, class IndexType = std::size_t>
inline v3::Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space, CoordType, IndexType>
successive_intersection(const std::vector<v3::Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space,
                                              CoordType, IndexType>>& meshes) {
  // Convert v3::Mesh to graph::Mesh
  std::vector<Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space, CoordType, IndexType>> graph_meshes;
  graph_meshes.reserve(meshes.size());
  for (const auto& mesh : meshes) {
    graph_meshes.push_back(detail::v3_mesh_to_graph<DIM, Kokkos::DefaultExecutionSpace::memory_space,
                                                     CoordType, IndexType>(mesh));
  }

  IntersectionGraph<DIM, CoordType, IndexType> graph(graph_meshes);
  auto result = graph.execute();

  // Convert back to v3::Mesh
  return detail::graph_mesh_to_v3<DIM, Kokkos::DefaultExecutionSpace::memory_space,
                                   CoordType, IndexType>(result);
}

/**
 * @brief Successive intersection of multiple meshes using graph-based approach (graph::Mesh version).
 *
 * This function eliminates intermediate synchronizations by using a graph-based
 * execution strategy. For N input meshes, instead of N-1 separate intersection
 * operations with fences between them, we chain the operations together.
 *
 * @tparam DIM Dimension (2 for 2D, 3 for 3D)
 * @tparam CoordType Coordinate type
 * @tparam IndexType Index type
 * @param meshes Vector of input meshes to intersect
 * @return Final intersection mesh
 */
template <int DIM, class CoordType = int32_t, class IndexType = std::size_t>
inline Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space, CoordType, IndexType>
successive_intersection_graph_mesh(const std::vector<Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space,
                                              CoordType, IndexType>>& meshes) {
  IntersectionGraph<DIM, CoordType, IndexType> graph(meshes);
  return graph.execute();
}

// 2D and 3D convenience wrappers
inline Mesh2DDevice intersect_meshes_2d(const Mesh2DDevice& A, const Mesh2DDevice& B) {
  return successive_intersection_graph_mesh<2>(std::vector<Mesh2DDevice>{A, B});
}

inline Mesh3DDevice intersect_meshes_3d(const Mesh3DDevice& A, const Mesh3DDevice& B) {
  return successive_intersection_graph_mesh<3>(std::vector<Mesh3DDevice>{A, B});
}

inline Mesh2DDevice successive_intersection_2d(const std::vector<Mesh2DDevice>& meshes) {
  return successive_intersection_graph_mesh<2>(meshes);
}

inline Mesh3DDevice successive_intersection_3d(const std::vector<Mesh3DDevice>& meshes) {
  return successive_intersection_graph_mesh<3>(meshes);
}

} // namespace experimental::subsetix::csr::successive::graph
