// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#pragma once

#include <playground/subsetix/csr/intersection/types.hpp>
#include <playground/subsetix/csr/intersection/detail/utils.hpp>

namespace playground::subsetix::csr::intersection::direct_index {

// ============================================================================
// Direct Index Mesh type (identical to baseline/optimized, see optimized.hpp)
// ============================================================================

/** @brief CSR mesh for direct index algorithm. Identical to optimized::Mesh. */
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
// Row mapping strategy enumeration
// ============================================================================

/** @brief Strategy for mapping rows from mesh A to mesh B. */
enum class RowMapStrategy {
  BINARY_SEARCH,   // Fallback to binary search (O(log n))
  DIRECT_DENSE,    // Consecutive coordinates (O(1))
  DIRECT_STRIDE,   // Uniform spacing (O(1))
  LOOKUP_TABLE     // Small coordinate range (O(1))
};

// ============================================================================
// Row mapper configuration
// ============================================================================

/**
 * @brief Configuration for direct index row mapping.
 *
 * This structure is filled by pattern detection on the host and then
 * copied to device for use in the row mapping kernel.
 */
template <class MemorySpace, class CoordType = int32_t>
struct RowMapperConfig {
  RowMapStrategy strategy = RowMapStrategy::BINARY_SEARCH;

  // Coordinate range (computed from mesh B)
  CoordType y_min = 0;
  CoordType y_max = 0;

  // For DIRECT_STRIDE strategy
  CoordType stride = 1;

  // For LOOKUP_TABLE strategy
  Kokkos::View<int*, MemorySpace> lookup_table;

  KOKKOS_INLINE_FUNCTION
  RowMapperConfig() = default;
};

// ============================================================================
// Host-side pattern detection
// ============================================================================

namespace detail {

/**
 * @brief Detect coordinate pattern on mesh B for 2D.
 *
 * This function runs on the host and analyzes the row keys of mesh B
 * to determine the optimal row mapping strategy.
 *
 * @param mesh_b Mesh B (device memory)
 * @return Configuration for the detected strategy
 */
template <class CoordType, class IndexType, class DeviceMemorySpace>
inline RowMapperConfig<Kokkos::HostSpace, CoordType>
detect_row_pattern_2d_host(const Mesh<2, DeviceMemorySpace, CoordType, IndexType>& mesh_b) {
  using HostMemorySpace = Kokkos::HostSpace;
  using RowKey = typename Mesh<2, DeviceMemorySpace, CoordType, IndexType>::RowKey;

  RowMapperConfig<HostMemorySpace, CoordType> config;

  if (mesh_b.num_rows == 0) {
    config.strategy = RowMapStrategy::BINARY_SEARCH;
    return config;
  }

  // Create host mirror of row keys
  auto row_keys_host = Kokkos::create_mirror_view_and_copy(HostMemorySpace{}, mesh_b.row_keys);

  // Get coordinate range
  config.y_min = row_keys_host(0).y;
  config.y_max = row_keys_host(mesh_b.num_rows - 1).y;

  // Compute y range (handle potential overflow)
  const std::size_t y_range = static_cast<std::size_t>(config.y_max) -
                              static_cast<std::size_t>(config.y_min) + 1;

  // Strategy 1: Check for DENSE (consecutive coordinates)
  if (y_range == mesh_b.num_rows) {
    config.strategy = RowMapStrategy::DIRECT_DENSE;
    return config;
  }

  // Strategy 2: Check for UNIFORM stride
  // Sample first 100 rows (or all if fewer)
  const std::size_t sample_size = std::min(static_cast<std::size_t>(100), mesh_b.num_rows);

  if (sample_size >= 2) {
    // Compute stride from first two rows
    CoordType stride = row_keys_host(1).y - row_keys_host(0).y;

    // Verify stride is positive
    if (stride > 0) {
      bool is_uniform = true;

      // Check if all sampled rows follow the same stride
      for (std::size_t i = 2; i < sample_size; ++i) {
        CoordType current_stride = row_keys_host(i).y - row_keys_host(i - 1).y;
        if (current_stride != stride) {
          is_uniform = false;
          break;
        }
      }

      // Verify uniform stride covers full range
      if (is_uniform) {
        // Check if (y_max - y_min) is divisible by stride
        const CoordType total_span = config.y_max - config.y_min;
        if (total_span % stride == 0) {
          const std::size_t expected_count = (static_cast<std::size_t>(total_span) / stride) + 1;

          if (expected_count == mesh_b.num_rows) {
            config.strategy = RowMapStrategy::DIRECT_STRIDE;
            config.stride = stride;
            return config;
          }
        }
      }
    }
  }

  // Strategy 3: Check for SMALL RANGE (lookup table feasible)
  // Lookup table is beneficial if:
  // - Coordinate range is small (< 10000)
  // - Density is high (> 50%)
  const double density = static_cast<double>(mesh_b.num_rows) / static_cast<double>(y_range);

  if (y_range < 10000 && density > 0.5) {
    config.strategy = RowMapStrategy::LOOKUP_TABLE;

    // Allocate and build lookup table
    config.lookup_table = Kokkos::View<int*, HostMemorySpace>("lookup_table", y_range);

    // Initialize with -1 (not found)
    for (std::size_t i = 0; i < y_range; ++i) {
      config.lookup_table(i) = -1;
    }

    // Fill lookup table
    for (std::size_t i = 0; i < mesh_b.num_rows; ++i) {
      const CoordType y = row_keys_host(i).y;
      const std::size_t idx = static_cast<std::size_t>(y) - static_cast<std::size_t>(config.y_min);
      config.lookup_table(idx) = static_cast<int>(i);
    }

    return config;
  }

  // Strategy 4: Fallback to binary search
  config.strategy = RowMapStrategy::BINARY_SEARCH;
  return config;
}

/**
 * @brief Detect coordinate pattern on mesh B for 3D.
 *
 * For 3D, we only detect patterns on the y-coordinate.
 * The (y, z) lexicographic search uses binary search for z.
 *
 * @param mesh_b Mesh B (device memory)
 * @return Configuration for the detected strategy
 */
template <class CoordType, class IndexType, class DeviceMemorySpace>
inline RowMapperConfig<Kokkos::HostSpace, CoordType>
detect_row_pattern_3d_host(const Mesh<3, DeviceMemorySpace, CoordType, IndexType>& mesh_b) {
  // For 3D, we use the same 2D detection on y-coordinate
  // The z-coordinate is handled via binary search within y-groups

  using HostMemorySpace = Kokkos::HostSpace;
  using RowKey = typename Mesh<3, DeviceMemorySpace, CoordType, IndexType>::RowKey;

  RowMapperConfig<HostMemorySpace, CoordType> config;

  if (mesh_b.num_rows == 0) {
    config.strategy = RowMapStrategy::BINARY_SEARCH;
    return config;
  }

  // Create host mirror of row keys
  auto row_keys_host = Kokkos::create_mirror_view_and_copy(HostMemorySpace{}, mesh_b.row_keys);

  // Get coordinate range
  config.y_min = row_keys_host(0).y;
  config.y_max = row_keys_host(mesh_b.num_rows - 1).y;

  // Compute y range
  const std::size_t y_range = static_cast<std::size_t>(config.y_max) -
                              static_cast<std::size_t>(config.y_min) + 1;

  // Strategy 1: Check for DENSE (consecutive y-coordinates)
  if (y_range == mesh_b.num_rows) {
    config.strategy = RowMapStrategy::DIRECT_DENSE;
    return config;
  }

  // Strategy 2: Check for UNIFORM stride in y
  const std::size_t sample_size = std::min(static_cast<std::size_t>(100), mesh_b.num_rows);

  if (sample_size >= 2) {
    CoordType stride = row_keys_host(1).y - row_keys_host(0).y;

    if (stride > 0) {
      bool is_uniform = true;

      for (std::size_t i = 2; i < sample_size; ++i) {
        CoordType current_stride = row_keys_host(i).y - row_keys_host(i - 1).y;
        if (current_stride != stride) {
          is_uniform = false;
          break;
        }
      }

      if (is_uniform) {
        const CoordType total_span = config.y_max - config.y_min;
        if (total_span % stride == 0) {
          const std::size_t expected_count = (static_cast<std::size_t>(total_span) / stride) + 1;

          if (expected_count == mesh_b.num_rows) {
            config.strategy = RowMapStrategy::DIRECT_STRIDE;
            config.stride = stride;
            return config;
          }
        }
      }
    }
  }

  // Strategy 3: Check for SMALL RANGE (lookup table)
  const double density = static_cast<double>(mesh_b.num_rows) / static_cast<double>(y_range);

  if (y_range < 10000 && density > 0.5) {
    config.strategy = RowMapStrategy::LOOKUP_TABLE;

    config.lookup_table = Kokkos::View<int*, HostMemorySpace>("lookup_table", y_range);

    for (std::size_t i = 0; i < y_range; ++i) {
      config.lookup_table(i) = -1;
    }

    for (std::size_t i = 0; i < mesh_b.num_rows; ++i) {
      const CoordType y = row_keys_host(i).y;
      const std::size_t idx = static_cast<std::size_t>(y) - static_cast<std::size_t>(config.y_min);
      config.lookup_table(idx) = static_cast<int>(i);
    }

    return config;
  }

  // Strategy 4: Fallback
  config.strategy = RowMapStrategy::BINARY_SEARCH;
  return config;
}

/**
 * @brief Convert RowMapperConfig from HostSpace to DeviceSpace.
 */
template <class CoordType, class DeviceMemorySpace>
inline RowMapperConfig<DeviceMemorySpace, CoordType>
config_to_device(const RowMapperConfig<Kokkos::HostSpace, CoordType>& host_config) {
  RowMapperConfig<DeviceMemorySpace, CoordType> device_config;

  device_config.strategy = host_config.strategy;
  device_config.y_min = host_config.y_min;
  device_config.y_max = host_config.y_max;
  device_config.stride = host_config.stride;

  // Copy lookup table if present
  if (host_config.lookup_table.size() > 0) {
    device_config.lookup_table = Kokkos::create_mirror_view_and_copy(
        DeviceMemorySpace{}, host_config.lookup_table);
  }

  return device_config;
}

} // namespace detail

// ============================================================================
// Device-side direct index lookup
// ============================================================================

namespace detail {

/**
 * @brief Find row index in mesh B using direct index strategy.
 *
 * This function runs on the device and uses the pre-computed configuration
 * to perform O(1) lookup when possible, falling back to binary search.
 *
 * @param config Row mapper configuration (device memory)
 * @param row_keys_b Row keys of mesh B
 * @param num_rows_b Number of rows in mesh B
 * @param y Y-coordinate to search for
 * @return Row index in mesh B, or -1 if not found
 */
template <class CoordType = int32_t>
KOKKOS_INLINE_FUNCTION
int find_row_direct_2d(const RowMapperConfig<Kokkos::DefaultExecutionSpace::memory_space, CoordType>& config,
                       const Kokkos::View<intersection::RowKey2D<CoordType>*,
                                          Kokkos::DefaultExecutionSpace::memory_space>& row_keys_b,
                       std::size_t num_rows_b,
                       int32_t y) {
  switch (config.strategy) {
    case RowMapStrategy::DIRECT_DENSE:
      {
        // Direct index: consecutive coordinates
        const std::size_t idx = static_cast<std::size_t>(y) -
                                static_cast<std::size_t>(config.y_min);
        if (idx < num_rows_b) {
          return static_cast<int>(idx);
        }
        return -1;
      }

    case RowMapStrategy::DIRECT_STRIDE:
      {
        // Direct index with uniform stride
        const int32_t offset = y - config.y_min;
        if (offset >= 0 && offset % config.stride == 0) {
          const std::size_t idx = static_cast<std::size_t>(offset) /
                                  static_cast<std::size_t>(config.stride);
          if (idx < num_rows_b) {
            return static_cast<int>(idx);
          }
        }
        return -1;
      }

    case RowMapStrategy::LOOKUP_TABLE:
      {
        // Lookup table (already verified bounds)
        if (y < config.y_min || y > config.y_max) {
          return -1;
        }
        const std::size_t idx = static_cast<std::size_t>(y) -
                                static_cast<std::size_t>(config.y_min);
        return config.lookup_table(idx);
      }

    case RowMapStrategy::BINARY_SEARCH:
      {
        // Fallback: standard binary search
        std::size_t lo = 0, hi = num_rows_b;
        while (lo < hi) {
          const std::size_t mid = lo + (hi - lo) / 2;
          if (row_keys_b(mid).y < y) {
            lo = mid + 1;
          } else {
            hi = mid;
          }
        }
        if (lo < num_rows_b && row_keys_b(lo).y == y) {
          return static_cast<int>(lo);
        }
        return -1;
      }

    default:
      return -1;
  }
}

/**
 * @brief Find row index in mesh B for 3D using direct index.
 *
 * For 3D, we use direct index for y-coordinate, then binary search
 * within the y-group for z-coordinate.
 *
 * @param config Row mapper configuration
 * @param row_keys_b Row keys of mesh B
 * @param num_rows_b Number of rows in mesh B
 * @param y Y-coordinate to search for
 * @param z Z-coordinate to search for
 * @return Row index in mesh B, or -1 if not found
 */
template <class CoordType = int32_t>
KOKKOS_INLINE_FUNCTION
int find_row_direct_3d(const RowMapperConfig<Kokkos::DefaultExecutionSpace::memory_space>& config,
                       const Kokkos::View<intersection::RowKey3D<int32_t>*,
                                          Kokkos::DefaultExecutionSpace::memory_space>& row_keys_b,
                       std::size_t num_rows_b,
                       int32_t y,
                       int32_t z) {
  // First, find the y-group using direct index strategy
  std::size_t y_group_start, y_group_end;

  switch (config.strategy) {
    case RowMapStrategy::DIRECT_DENSE:
      {
        // Direct index for y
        const std::size_t idx = static_cast<std::size_t>(y) -
                                static_cast<std::size_t>(config.y_min);
        if (idx >= num_rows_b) {
          return -1;
        }
        // For dense 2D, each y has exactly one row
        y_group_start = idx;
        y_group_end = idx + 1;
        break;
      }

    case RowMapStrategy::DIRECT_STRIDE:
      {
        // Direct index with stride for y
        const int32_t offset = y - config.y_min;
        if (offset < 0 || offset % config.stride != 0) {
          return -1;
        }
        const std::size_t idx = static_cast<std::size_t>(offset) /
                                static_cast<std::size_t>(config.stride);
        if (idx >= num_rows_b) {
          return -1;
        }
        y_group_start = idx;
        y_group_end = idx + 1;
        break;
      }

    case RowMapStrategy::LOOKUP_TABLE:
      {
        // Lookup table for y
        if (y < config.y_min || y > config.y_max) {
          return -1;
        }
        const std::size_t y_idx = static_cast<std::size_t>(y) -
                                   static_cast<std::size_t>(config.y_min);
        const int row_idx = config.lookup_table(y_idx);
        if (row_idx < 0) {
          return -1;
        }
        // Assume each y has one row (simplest case)
        y_group_start = static_cast<std::size_t>(row_idx);
        y_group_end = y_group_start + 1;
        break;
      }

    case RowMapStrategy::BINARY_SEARCH:
      {
        // Binary search for y-group
        std::size_t lo = 0, hi = num_rows_b;
        while (lo < hi) {
          const std::size_t mid = lo + (hi - lo) / 2;
          if (row_keys_b(mid).y < y) {
            lo = mid + 1;
          } else {
            hi = mid;
          }
        }

        // Find the range of rows with this y-coordinate
        y_group_start = lo;
        while (y_group_start < num_rows_b && row_keys_b(y_group_start).y < y) {
          ++y_group_start;
        }

        y_group_end = y_group_start;
        while (y_group_end < num_rows_b && row_keys_b(y_group_end).y == y) {
          ++y_group_end;
        }

        if (y_group_start >= num_rows_b || row_keys_b(y_group_start).y != y) {
          return -1;
        }
        break;
      }

    default:
      return -1;
  }

  // Binary search within y-group for z-coordinate
  std::size_t lo = y_group_start, hi = y_group_end;
  while (lo < hi) {
    const std::size_t mid = lo + (hi - lo) / 2;
    if (row_keys_b(mid).z < z) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  if (lo < y_group_end && row_keys_b(lo).z == z) {
    return static_cast<int>(lo);
  }

  return -1;
}

} // namespace detail

// ============================================================================
// Core row intersection (identical to optimized)
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
// Mesh intersection (2D and 3D) - Direct Index Algorithm
// ============================================================================

/**
 * @brief Mesh intersection with direct index row mapping.
 *
 * This algorithm detects coordinate patterns on mesh B and uses O(1) direct
 * indexing when possible, falling back to binary search otherwise.
 */
template <int DIM, class CoordType = int32_t, class IndexType = std::size_t>
inline Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space, CoordType, IndexType>
intersect_meshes(const Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space, CoordType, IndexType>& A,
                const Mesh<DIM, Kokkos::DefaultExecutionSpace::memory_space, CoordType, IndexType>& B) {
  using DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using MeshType = Mesh<DIM, DeviceMemorySpace, CoordType, IndexType>;
  using RowKey = typename MeshType::RowKey;
  using Interval = intersection::Interval<CoordType>;

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

  // Detect pattern on mesh B (host-side)
  auto config_host = [&]() {
    if constexpr (DIM == 2) {
      return detail::detect_row_pattern_2d_host(B);
    } else {
      return detail::detect_row_pattern_3d_host(B);
    }
  }();

  // Copy configuration to device
  auto config_device = detail::config_to_device<CoordType, DeviceMemorySpace>(config_host);

  // Phase 1: Row mapping - find rows of A that exist in B
  if constexpr (DIM == 2) {
    Kokkos::parallel_for(
        "direct_index_row_map_2d",
        Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          const RowKey key = rows_a(i);
          const int idx_b = detail::find_row_direct_2d(config_device, rows_b, num_rows_b, key.y);
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
        "direct_index_row_map_3d",
        Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          const RowKey key = rows_a(i);
          const int idx_b = detail::find_row_direct_3d(config_device, rows_b, num_rows_b, key.y, key.z);
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
  Kokkos::View<typename MeshType::RowKey*, DeviceMemorySpace> out_rows("out_rows", num_rows_out);
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
    out.row_keys = Kokkos::View<typename MeshType::RowKey*, DeviceMemorySpace>("mesh_row_keys", num_rows_out);
    out.row_ptr = Kokkos::View<IndexType*, DeviceMemorySpace>("mesh_row_ptr", num_rows_out + 1);
    out.intervals = Kokkos::View<intersection::Interval<CoordType>*, DeviceMemorySpace>(
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
  Kokkos::View<std::size_t, DeviceMemorySpace> total_view("total_intervals");
  Kokkos::parallel_scan(
      "intersection_scan",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
        const std::size_t count = row_counts(i);
        if (final_pass) {
          out.row_ptr(i) = static_cast<IndexType>(update);
          if (i + 1 == num_rows_out) {
            out.row_ptr(num_rows_out) = static_cast<IndexType>(update + count);
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

        const auto r = intersection::detail::extract_row_ranges(ia, ib, row_ptr_a, row_ptr_b);

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
    return out;
  }

  if (final_num_rows == 0) {
    return MeshType{};
  }

  // Allocate compacted output
  MeshType compacted;
  compacted.row_keys = Kokkos::View<typename MeshType::RowKey*, DeviceMemorySpace>("compacted_row_keys", final_num_rows);
  compacted.row_ptr = Kokkos::View<IndexType*, DeviceMemorySpace>("compacted_row_ptr", final_num_rows + 1);
  compacted.intervals = Kokkos::View<intersection::Interval<CoordType>*, DeviceMemorySpace>("compacted_intervals", out.num_intervals);
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

} // namespace playground::subsetix::csr::intersection::direct_index
