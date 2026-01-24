// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#pragma once

#include <playground/subsetix/csr/intersection/types.hpp>
#include <playground/subsetix/csr/intersection/detail/utils.hpp>
#include <algorithm>
#include <vector>

namespace playground::subsetix::csr::intersection::hybrid_cpu_gpu {

// ============================================================================
// hybrid_cpu_gpu Mesh type (identical to optimized, see optimized.hpp for full documentation)
// ============================================================================

/** @brief CSR mesh for hybrid CPU-GPU algorithm. Identical to optimized::Mesh. */
template <int DIM, class MemorySpace,
          class CoordType = int32_t,
          class IndexType = std::size_t>
class Mesh {
public:
  static constexpr int dim_value = DIM;
  using coord_type = CoordType;
  using index_type = IndexType;
  using memory_space = MemorySpace;

  // Row key type based on dimension
  using RowKey = std::conditional_t<DIM == 2,
                                     intersection::RowKey2D<CoordType>,
                                     intersection::RowKey3D<CoordType>>;

  // View types
  using RowKeyView = Kokkos::View<RowKey*, MemorySpace>;
  using IndexView = Kokkos::View<IndexType*, MemorySpace>;
  using IntervalView = Kokkos::View<intersection::Interval<CoordType>*, MemorySpace>;

  // Mesh data
  RowKeyView row_keys;     // [num_rows] - row coordinates
  IndexView row_ptr;       // [num_rows + 1] - CSR offsets
  IntervalView intervals;  // [num_intervals] - X-intervals

  std::size_t num_rows = 0;
  std::size_t num_intervals = 0;

  KOKKOS_INLINE_FUNCTION
  Mesh() = default;

  KOKKOS_INLINE_FUNCTION
  Mesh(const Mesh&) = default;

  KOKKOS_INLINE_FUNCTION
  Mesh& operator=(const Mesh&) = default;
};

// ============================================================================
// Type aliases for common configurations
// ============================================================================

// Default configurations
template <int DIM>
using DefaultMesh = Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space, int32_t, std::size_t>;

// 2D aliases
template <class CoordType = int32_t, class IndexType = std::size_t>
using Mesh2D = Mesh<2, Kokkos::DefaultExecutionSpace::memory_space, CoordType, IndexType>;

using Mesh2DDevice = Mesh2D<>;  // Default types
using Mesh2DHost = Mesh<2, Kokkos::HostSpace, int32_t, std::size_t>;

// 3D aliases
template <class CoordType = int32_t, class IndexType = std::size_t>
using Mesh3D = Mesh<3, Kokkos::DefaultExecutionSpace::memory_space, CoordType, IndexType>;

using Mesh3DDevice = Mesh3D<>;  // Default types
using Mesh3DHost = Mesh<3, Kokkos::HostSpace, int32_t, std::size_t>;

// ============================================================================
// Core row intersection (identical to optimized, see optimized.hpp)
// ============================================================================

namespace detail {

/** @brief Row intersection. Identical to optimized::detail::row_intersection_impl. */
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

    // Compute intersection: [max(begin), min(end))
    const CoordType start = (a.begin > b.begin) ? a.begin : b.begin;
    const CoordType end = (a.end < b.end) ? a.end : b.end;

    // Add non-empty intersection
    if (start < end) {
      if constexpr (!CountOnly) {
        intervals_out(out_offset + count) = IntervalType{start, end};
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
// Decision heuristic for hybrid vs pure GPU
// ============================================================================

/**
 * @brief Decision heuristic for when to use hybrid CPU-GPU approach.
 *
 * Hybrid approach is beneficial for medium-to-large meshes where:
 * - CPU row mapping overhead is amortized over many rows
 * - GPU parallel processing benefits from reduced work (only matching rows)
 *
 * @tparam DIM Dimension (2 or 3)
 * @tparam CoordType Coordinate type
 * @tparam IndexType Index type
 * @param n_a Number of rows in mesh A
 * @param n_b Number of rows in mesh B
 * @return true if hybrid approach should be used, false for pure GPU
 */
template <int DIM, class CoordType = int32_t, class IndexType = std::size_t>
inline bool should_use_hybrid(std::size_t n_a, std::size_t n_b) {
  constexpr std::size_t BREAKOVER_THRESHOLD = 1000;
  return (n_a >= BREAKOVER_THRESHOLD && n_b >= BREAKOVER_THRESHOLD);
}

// ============================================================================
// CPU row mapping (2D)
// ============================================================================

namespace detail {

/**
 * @brief CPU-side row matching for 2D meshes.
 *
 * Uses std::set_intersection pattern to find matching y-coordinates.
 * This is significantly faster than GPU binary search for large meshes.
 *
 * @tparam CoordType Coordinate type
 * @tparam IndexType Index type
 * @param host_rows_a Row keys from mesh A (on host)
 * @param host_rows_b Row keys from mesh B (on host)
 * @param host_row_ptr_a Row pointers from mesh A (on host)
 * @param host_row_ptr_b Row pointers from mesh B (on host)
 * @param num_rows_a Number of rows in A
 * @param num_rows_b Number of rows in B
 * @return Tuple of (idx_a_vector, idx_b_vector, row_keys_vector) for matching rows
 */
template <class CoordType, class IndexType, class RowKey, class RowKeyHostView, class RowPtrHostView>
inline auto cpu_row_mapping_2d(const RowKeyHostView& host_rows_a,
                               const RowKeyHostView& host_rows_b,
                               const RowPtrHostView& host_row_ptr_a,
                               const RowPtrHostView& host_row_ptr_b,
                               std::size_t num_rows_a,
                               std::size_t num_rows_b) {
  struct RowPair {
    int idx_a;
    int idx_b;
  };

  std::vector<RowPair> cpu_matches;

  // Use pointers for efficient traversal
  const RowKey* it_a = host_rows_a.data();
  const RowKey* it_a_end = host_rows_a.data() + num_rows_a;
  const RowKey* it_b = host_rows_b.data();
  const RowKey* it_b_end = host_rows_b.data() + num_rows_b;

  // std::set_intersection pattern for 2D (compare by y only)
  while (it_a != it_a_end && it_b != it_b_end) {
    if (it_a->y < it_b->y) {
      ++it_a;
    } else if (it_b->y < it_a->y) {
      ++it_b;
    } else {
      // Match found: y coordinates are equal
      cpu_matches.push_back({
        static_cast<int>(it_a - host_rows_a.data()),
        static_cast<int>(it_b - host_rows_b.data())
      });
      ++it_a;
      ++it_b;
    }
  }

  return cpu_matches;
}

/**
 * @brief CPU-side row matching for 3D meshes.
 *
 * Uses std::set_intersection pattern with lexicographic (y, z) comparison.
 *
 * @tparam CoordType Coordinate type
 * @tparam IndexType Index type
 * @param host_rows_a Row keys from mesh A (on host)
 * @param host_rows_b Row keys from mesh B (on host)
 * @param host_row_ptr_a Row pointers from mesh A (on host)
 * @param host_row_ptr_b Row pointers from mesh B (on host)
 * @param num_rows_a Number of rows in A
 * @param num_rows_b Number of rows in B
 * @return Tuple of (idx_a_vector, idx_b_vector, row_keys_vector) for matching rows
 */
template <class CoordType, class IndexType, class RowKey, class RowKeyHostView, class RowPtrHostView>
inline auto cpu_row_mapping_3d(const RowKeyHostView& host_rows_a,
                               const RowKeyHostView& host_rows_b,
                               const RowPtrHostView& host_row_ptr_a,
                               const RowPtrHostView& host_row_ptr_b,
                               std::size_t num_rows_a,
                               std::size_t num_rows_b) {
  struct RowPair {
    int idx_a;
    int idx_b;
  };

  std::vector<RowPair> cpu_matches;

  // Use pointers for efficient traversal
  const RowKey* it_a = host_rows_a.data();
  const RowKey* it_a_end = host_rows_a.data() + num_rows_a;
  const RowKey* it_b = host_rows_b.data();
  const RowKey* it_b_end = host_rows_b.data() + num_rows_b;

  // std::set_intersection pattern for 3D (lexicographic compare by y, then z)
  while (it_a != it_a_end && it_b != it_b_end) {
    const RowKey& key_a = *it_a;
    const RowKey& key_b = *it_b;

    // Lexicographic comparison: (y, z) < (y', z') iff y < y' or (y == y' and z < z')
    bool a_less = (key_a.y < key_b.y) || (key_a.y == key_b.y && key_a.z < key_b.z);
    bool b_less = (key_b.y < key_a.y) || (key_b.y == key_a.y && key_b.z < key_a.z);

    if (a_less) {
      ++it_a;
    } else if (b_less) {
      ++it_b;
    } else {
      // Match found: both y and z are equal
      cpu_matches.push_back({
        static_cast<int>(it_a - host_rows_a.data()),
        static_cast<int>(it_b - host_rows_b.data())
      });
      ++it_a;
      ++it_b;
    }
  }

  return cpu_matches;
}

} // namespace detail

// ============================================================================
// Mesh intersection (2D and 3D) - Hybrid CPU-GPU Algorithm
// ============================================================================

/**
 * @brief Hybrid CPU-GPU mesh intersection.
 *
 * Algorithm phases:
 * - Phase 0: Decision heuristic (check if hybrid is beneficial)
 * - Phase 1: CPU row mapping (find matching rows on CPU)
 * - Phase 2: Transfer matching rows to GPU
 * - Phase 3-5: GPU interval processing (count, scan, fill)
 * - Phase 6: GPU compaction (remove empty rows)
 *
 * For small meshes, falls back to pure GPU approach.
 *
 * @tparam DIM Dimension (2 or 3)
 * @tparam CoordType Coordinate type
 * @tparam IndexType Index type
 * @param A First input mesh (device memory)
 * @param B Second input mesh (device memory)
 * @return Intersected mesh (device memory)
 */
template <int DIM, class CoordType = int32_t, class IndexType = std::size_t>
inline Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space, CoordType, IndexType>
intersect_meshes(const Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space, CoordType, IndexType>& A,
                const Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space, CoordType, IndexType>& B) {
  using DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
  using HostMemorySpace = Kokkos::HostSpace;
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using MeshType = Mesh<DIM, DeviceMemorySpace, CoordType, IndexType>;
  using RowKey = typename MeshType::RowKey;
  using Interval = intersection::Interval<CoordType>;

  // Early exit for empty inputs
  if (A.num_rows == 0 || B.num_rows == 0) {
    return MeshType{};
  }

  // Phase 0: Decision heuristic - fall back to pure GPU for small meshes
  if (!should_use_hybrid<DIM>(A.num_rows, B.num_rows)) {
    // Fall back to pure GPU implementation
    return optimized::intersect_meshes<DIM, CoordType, IndexType>(A, B);
  }

  // ========================================================================
  // Phase 1: CPU Row Mapping
  // ========================================================================

  // 1a. Copy row_keys and row_ptr to host
  auto host_rows_a = Kokkos::create_mirror_view_and_copy(HostMemorySpace{}, A.row_keys);
  auto host_rows_b = Kokkos::create_mirror_view_and_copy(HostMemorySpace{}, B.row_keys);
  auto host_row_ptr_a = Kokkos::create_mirror_view_and_copy(HostMemorySpace{}, A.row_ptr);
  auto host_row_ptr_b = Kokkos::create_mirror_view_and_copy(HostMemorySpace{}, B.row_ptr);

  // 1b. Perform std::set_intersection pattern on CPU
  std::vector<typename detail::RowPair> cpu_matches;

  if constexpr (DIM == 2) {
    cpu_matches = detail::cpu_row_mapping_2d<CoordType, IndexType>(
        host_rows_a, host_rows_b, host_row_ptr_a, host_row_ptr_b,
        A.num_rows, B.num_rows);
  } else {
    cpu_matches = detail::cpu_row_mapping_3d<CoordType, IndexType>(
        host_rows_a, host_rows_b, host_row_ptr_a, host_row_ptr_b,
        A.num_rows, B.num_rows);
  }

  const std::size_t n_match = cpu_matches.size();

  // Early exit if no matching rows
  if (n_match == 0) {
    return MeshType{};
  }

  // ========================================================================
  // Phase 2: Transfer Matching Rows to GPU
  // ========================================================================

  // Prepare host vectors for transfer
  std::vector<int> host_idx_a;
  std::vector<int> host_idx_b;
  std::vector<RowKey> host_row_keys;

  host_idx_a.reserve(n_match);
  host_idx_b.reserve(n_match);
  host_row_keys.reserve(n_match);

  for (const auto& match : cpu_matches) {
    host_idx_a.push_back(match.idx_a);
    host_idx_b.push_back(match.idx_b);
    host_row_keys.push_back(host_rows_a(match.idx_a));
  }

  // Deep copy to device
  Kokkos::View<int*, DeviceMemorySpace> gpu_idx_a("gpu_idx_a", n_match);
  Kokkos::View<int*, DeviceMemorySpace> gpu_idx_b("gpu_idx_b", n_match);
  Kokkos::View<RowKey*, DeviceMemorySpace> gpu_row_keys("gpu_row_keys", n_match);

  Kokkos::deep_copy(gpu_idx_a, Kokkos::View<int*, HostMemorySpace>(host_idx_a.data(), n_match));
  Kokkos::deep_copy(gpu_idx_b, Kokkos::View<int*, HostMemorySpace>(host_idx_b.data(), n_match));
  Kokkos::deep_copy(gpu_row_keys, Kokkos::View<RowKey*, HostMemorySpace>(host_row_keys.data(), n_match));

  // ========================================================================
  // Phase 3-5: GPU Interval Processing (same as optimized, but on matched rows only)
  // ========================================================================

  // Allocate output mesh
  MeshType out;
  out.row_keys = Kokkos::View<RowKey*, DeviceMemorySpace>("mesh_row_keys", n_match);
  out.row_ptr = Kokkos::View<IndexType*, DeviceMemorySpace>("mesh_row_ptr", n_match + 1);
  out.intervals = Kokkos::View<Interval*, DeviceMemorySpace>(
      "mesh_intervals", A.num_intervals + B.num_intervals);
  out.num_rows = n_match;

  // Copy row keys from GPU
  Kokkos::deep_copy(out.row_keys, gpu_row_keys);

  // Allocate row counts buffer
  Kokkos::View<std::size_t*, DeviceMemorySpace> row_counts("row_counts", n_match);

  auto row_ptr_a = A.row_ptr;
  auto row_ptr_b = B.row_ptr;
  auto intervals_a = A.intervals;
  auto intervals_b = B.intervals;

  // Phase 3: Count intervals per matched row
  Kokkos::parallel_for(
      "hybrid_count",
      Kokkos::RangePolicy<ExecSpace>(0, n_match),
      KOKKOS_LAMBDA(const std::size_t i) {
        const int ia = gpu_idx_a(i);
        const int ib = gpu_idx_b(i);

        const auto r = intersection::detail::extract_row_ranges(ia, ib, row_ptr_a, row_ptr_b);

        if (r.begin_a == r.end_a || r.begin_b == r.end_b) {
          row_counts(i) = 0;
          return;
        }

        row_counts(i) = detail::row_intersection_impl<true>(
            intervals_a, r.begin_a, r.end_a,
            intervals_b, r.begin_b, r.end_b,
            Kokkos::View<Interval*, DeviceMemorySpace>(), 0);
      });

  // Phase 4: Scan to compute row_ptr offsets
  Kokkos::View<std::size_t, DeviceMemorySpace> total_view("total_intervals");
  Kokkos::parallel_scan(
      "hybrid_scan",
      Kokkos::RangePolicy<ExecSpace>(0, n_match),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
        const std::size_t count = row_counts(i);
        if (final_pass) {
          out.row_ptr(i) = static_cast<IndexType>(update);
          if (i + 1 == n_match) {
            out.row_ptr(n_match) = static_cast<IndexType>(update + count);
            total_view() = update + count;
          }
        }
        update += count;
      });

  Kokkos::fence();

  std::size_t num_intervals_host = 0;
  Kokkos::deep_copy(num_intervals_host, total_view);
  out.num_intervals = num_intervals_host;

  if (out.num_intervals == 0) {
    return MeshType{};
  }

  // Phase 5: Fill intersected intervals
  Kokkos::parallel_for(
      "hybrid_fill",
      Kokkos::RangePolicy<ExecSpace>(0, n_match),
      KOKKOS_LAMBDA(const std::size_t i) {
        const int ia = gpu_idx_a(i);
        const int ib = gpu_idx_b(i);

        const auto r = intersection::detail::extract_row_ranges(ia, ib, row_ptr_a, row_ptr_b);

        if (r.begin_a == r.end_a || r.begin_b == r.end_b) {
          return;
        }

        detail::row_intersection_impl<false>(
            intervals_a, r.begin_a, r.end_a,
            intervals_b, r.begin_b, r.end_b,
            out.intervals, out.row_ptr(i));
      });

  // ========================================================================
  // Phase 6: Compact - Remove rows with no intervals (same as optimized)
  // ========================================================================

  Kokkos::View<int*, DeviceMemorySpace> has_intervals("has_intervals", n_match);
  Kokkos::parallel_for(
      "hybrid_mark_rows",
      Kokkos::RangePolicy<ExecSpace>(0, n_match),
      KOKKOS_LAMBDA(const std::size_t i) {
        has_intervals(i) = (out.row_ptr(i) < out.row_ptr(i + 1)) ? 1 : 0;
      });

  Kokkos::View<std::size_t*, DeviceMemorySpace> new_positions("new_positions", n_match);
  Kokkos::View<std::size_t, DeviceMemorySpace> final_num_rows_view("final_num_rows");
  Kokkos::parallel_scan(
      "hybrid_compact_scan",
      Kokkos::RangePolicy<ExecSpace>(0, n_match),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
        const std::size_t count = static_cast<std::size_t>(has_intervals(i));
        if (final_pass) {
          new_positions(i) = update;
          if (i + 1 == n_match) {
            final_num_rows_view() = update + count;
          }
        }
        update += count;
      });

  Kokkos::fence();

  std::size_t final_num_rows = 0;
  Kokkos::deep_copy(final_num_rows, final_num_rows_view);

  if (final_num_rows == n_match) {
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
      "hybrid_compact_copy",
      Kokkos::RangePolicy<ExecSpace>(0, n_match),
      KOKKOS_LAMBDA(const std::size_t j) {
        if (has_intervals(j)) {
          const std::size_t new_pos = new_positions(j);
          compacted.row_keys(new_pos) = out.row_keys(j);
          compacted.row_ptr(new_pos) = out.row_ptr(j);
        }
      });

  // Set final row_ptr value
  Kokkos::parallel_for(
      "hybrid_compact_final_ptr",
      Kokkos::RangePolicy<ExecSpace>(0, 1),
      KOKKOS_LAMBDA(const std::size_t) {
        compacted.row_ptr(final_num_rows) = out.row_ptr(n_match);
      });

  // Copy intervals
  Kokkos::parallel_for(
      "hybrid_compact_intervals",
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
template <int DIM, class CoordType, class IndexType, class ToSpace, class FromSpace>
inline Mesh<DIM, ToSpace, CoordType, IndexType>
mesh_to(const Mesh<DIM, FromSpace, CoordType, IndexType>& src) {
  Mesh<DIM, ToSpace, CoordType, IndexType> dst;

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

} // namespace playground::subsetix::csr::intersection::hybrid_cpu_gpu
