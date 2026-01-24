// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Sebastien DUBOIS and the HPC@Maths Team, CMAP Laboratory, Ecole Polytechnique

#pragma once

#include <Kokkos_Core.hpp>
#include <cstdint>
#include <cstddef>
#include <cmath>

namespace subsetix {
namespace csr {
namespace row_mapping {

using Coord = std::int32_t;

// ============================================================================
// 1. Structure-of-Arrays (SoA) Row Key Conversion
// ============================================================================

/**
 * @brief Structure-of-Arrays representation for 2D row keys.
 *
 * Converts AoS (Array-of-Structures) RowKey2D to SoA for better cache
 * utilization and SIMD/GPU vectorization.
 *
 * Benefits:
 * - Improved cache line utilization (only load y coordinates)
 * - Better memory coalescing on GPU
 * - Enables vectorized comparisons
 */
template <class MemorySpace>
struct RowKey2DSoA {
  Kokkos::View<Coord*, MemorySpace> y;  ///< [num_rows] Y coordinates only

  KOKKOS_INLINE_FUNCTION
  std::size_t extent() const { return y.extent(0); }

  KOKKOS_INLINE_FUNCTION
  bool operator==(const RowKey2DSoA& other) const {
    return y.data() == other.y.data();
  }
};

/**
 * @brief Convert IntervalSet2D with AoS row_keys to SoA representation.
 *
 * @tparam MemorySpace Memory space of the input IntervalSet2D
 * @param geom The IntervalSet2D with AoS row_keys
 * @return RowKey2DSoA<MemorySpace> SoA representation
 */
template <class MemorySpace>
inline RowKey2DSoA<MemorySpace>
to_soa(const IntervalSet2D<MemorySpace>& geom) {
  RowKey2DSoA<MemorySpace> soa;
  if (geom.num_rows == 0) {
    return soa;
  }

  soa.y = Kokkos::View<Coord*, MemorySpace>(
      Kokkos::view_alloc("soa_row_y", geom.row_keys.extent(0)));

  // Extract y coordinates from RowKey2D structures
  Kokkos::parallel_for(
      "convert_to_soa_2d",
      Kokkos::RangePolicy<ExecSpace>(0, geom.num_rows),
      KOKKOS_LAMBDA(const std::size_t i) {
        soa.y(i) = geom.row_keys(i).y;
      });

  ExecSpace().fence();
  return soa;
}

/**
 * @brief Find row index by y-coordinate using SoA representation.
 *
 * This is the key optimization: we only load y coordinates, not the entire
 * RowKey2D structure. On GPUs, this enables better memory coalescing.
 *
 * @tparam MemorySpace Memory space of the SoA structure
 * @param rows SoA row keys
 * @param num_rows Number of rows
 * @param y Y-coordinate to search for
 * @return Row index if found, -1 otherwise
 */
template <class MemorySpace>
KOKKOS_INLINE_FUNCTION
int find_row_by_y_soa(const RowKey2DSoA<MemorySpace>& rows,
                      std::size_t num_rows,
                      Coord y) {
  std::size_t lo = 0;
  std::size_t hi = num_rows;

  while (lo < hi) {
    const std::size_t mid = lo + (hi - lo) / 2;
    if (rows.y(mid) < y) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  if (lo < num_rows && rows.y(lo) == y) {
    return static_cast<int>(lo);
  }
  return -1;
}

// ============================================================================
// 2. 3D SoA Representation (for future 3D extension)
// ============================================================================

/**
 * @brief Structure-of-Arrays representation for 3D row keys.
 *
 * For 3D meshes with (y, z) coordinates, SoA provides even greater
 * benefits as we can search on y first, then filter by z.
 */
template <class MemorySpace>
struct RowKey3DSoA {
  Kokkos::View<Coord*, MemorySpace> y;  ///< [num_rows] Y coordinates
  Kokkos::View<Coord*, MemorySpace> z;  ///< [num_rows] Z coordinates

  KOKKOS_INLINE_FUNCTION
  std::size_t extent() const { return y.extent(0); }
};

/**
 * @brief Find row index by (y, z) coordinates using 3D SoA.
 *
 * Uses lexicographic ordering: first match y, then match z.
 *
 * Two-phase search for better cache efficiency:
 * 1. Binary search on y (primary coordinate)
 * 2. Linear scan in matching y-range for z
 */
template <class MemorySpace>
KOKKOS_INLINE_FUNCTION
int find_row_by_yz_soa(const RowKey3DSoA<MemorySpace>& rows,
                       std::size_t num_rows,
                       Coord y,
                       Coord z) {
  // Phase 1: Binary search for y range
  std::size_t lo = 0;
  std::size_t hi = num_rows;

  while (lo < hi) {
    const std::size_t mid = lo + (hi - lo) / 2;
    if (rows.y(mid) < y) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  // Phase 2: Linear scan in y-range for exact (y, z) match
  // This is efficient because rows with same y are typically
  // stored consecutively and the range is small
  for (std::size_t i = lo; i < num_rows && rows.y(i) == y; ++i) {
    if (rows.z(i) == z) {
      return static_cast<int>(i);
    }
  }

  return -1;
}

// ============================================================================
// 3. Warp-Optimized Binary Search (CUDA-specific)
// ============================================================================

#ifdef KOKKOS_ENABLE_CUDA

/**
 * @brief Warp-optimized binary search using CUDA shuffle primitives.
 *
 * This implementation leverages warp-level parallelism to accelerate
 * binary search by having all 32 threads collaboratively search.
 *
 * Key optimizations:
 * - Uses __shfl_sync for data exchange between warp threads
 * - Reduces memory transaction count via coordinated access
 * - Early termination when found
 *
 * @tparam MemorySpace Memory space (must be accessible from CUDA)
 * @param rows SoA row keys (works best with SoA for coalesced access)
 * @param num_rows Number of rows
 * @param y Y-coordinate to search for
 * @return Row index if found, -1 otherwise
 */
template <class MemorySpace>
KOKKOS_INLINE_FUNCTION
int find_row_by_y_warp_optimized(const RowKey2DSoA<MemorySpace>& rows,
                                  std::size_t num_rows,
                                  Coord y) {
  // This function should be called with exactly one warp (32 threads)
  // It uses warp-level primitives for collaborative search

  const int lane_id = threadIdx.x % 32;
  const int warp_id = threadIdx.x / 32;

  // Ensure we have valid data
  if (num_rows == 0) {
    return -1;
  }

  // Each warp performs a coordinated binary search
  std::size_t lo = 0;
  std::size_t hi = num_rows;
  int found_idx = -1;

  while (lo < hi && found_idx < 0) {
    const std::size_t mid = lo + (hi - lo) / 2;

    // Load the mid value - all threads load same index (broadcast)
    // This is automatically coalesced by CUDA
    Coord mid_y = rows.y(mid);

    // Broadcast the comparison result to all threads in warp
    // Only lane 0's result is used, but all threads participate
    bool go_right = (mid_y < y);
    bool match = (mid_y == y);

    // Use warp ballot to check if any thread found a match
    unsigned match_mask = __ballot_sync(0xFFFFFFFF, match);
    if (match_mask != 0) {
      // Found! Get the lane id of the first matching thread
      unsigned first_match_lane = __ffs(match_mask) - 1;
      if (lane_id == first_match_lane) {
        found_idx = static_cast<int>(mid);
      }
      break;
    }

    // Update search range - all lanes make the same decision
    if (go_right) {
      lo = mid + 1;
    } else {
      hi = mid;
    }

    // Warp synchronization
    __syncwarp();
  }

  // Broadcast the found index to all threads in warp
  if (found_idx >= 0) {
    found_idx = __shfl_sync(0xFFFFFFFF, found_idx, 0);
  }

  return found_idx;
}

/**
 * @brief Build row mapping using warp-optimized search.
 *
 * This kernel processes multiple row lookups in parallel, with each
 * warp handling one lookup using the warp-optimized binary search.
 *
 * @param mask_rows SoA mask row keys
 * @param parent_rows SoA parent row keys
 * @param num_parent_rows Number of rows in parent
 * @param mapping Output mapping array (mask_row -> parent_row_idx)
 */
template <class MemorySpace>
inline void
build_row_map_y_warp_optimized(const RowKey2DSoA<MemorySpace>& mask_rows,
                               const RowKey2DSoA<MemorySpace>& parent_rows,
                               std::size_t num_parent_rows,
                               Kokkos::View<int*, MemorySpace> mapping) {
  const std::size_t num_mask_rows = mask_rows.extent(0);
  if (num_mask_rows == 0) {
    return;
  }

  // Launch with warp-sized teams for optimal warp primitive usage
  // Each team processes one row lookup
  const int team_size = 32;  // CUDA warp size
  const std::size_t num_teams = (num_mask_rows + team_size - 1) / team_size;

  Kokkos::parallel_for(
      "build_row_map_warp_opt",
      Kokkos::TeamPolicy<ExecSpace>(num_teams, team_size),
      KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<ExecSpace>::member_type& team) {
        const std::size_t team_id = team.league_rank();
        const std::size_t local_idx = team.team_rank();

        // Each team processes multiple rows if needed
        for (std::size_t offset = 0; offset < team_size; ++offset) {
          const std::size_t row_idx = team_id * team_size + local_idx + offset;
          if (row_idx >= num_mask_rows) break;

          const Coord y = mask_rows.y(row_idx);

          // Use warp-optimized search
          // Only lane 0 writes the result
          if (local_idx == 0) {
            mapping(row_idx) = find_row_by_y_warp_optimized(
                parent_rows, num_parent_rows, y);
          }
        }
      });

  ExecSpace().fence();
}

#endif // KOKKOS_ENABLE_CUDA

// ============================================================================
// 4. Hierarchical (Coarse-Fine) Search Index
// ============================================================================

/**
 * @brief Coarse-fine hierarchical index for fast row lookup.
 *
 * Divides the coordinate space into buckets (coarse level) and
 * maintains row indices within each bucket (fine level).
 *
 * Memory layout:
 * - bucket_min_y[COARSE_BUCKETS]: Minimum y in each bucket
 * - bucket_max_y[COARSE_BUCKETS]: Maximum y in each bucket
 * - bucket_row_ptr[COARSE_BUCKETS + 1]: CSR offsets into bucket_rows
 * - bucket_rows[num_rows]: Row indices grouped by bucket
 *
 * This enables O(1) bucket lookup + O(log bucket_size) fine search.
 */
template <class MemorySpace>
class HierarchicalRowIndex {
public:
  static constexpr std::size_t DEFAULT_COARSE_BUCKETS = 256;

  // Coarse-level bucket bounds
  Kokkos::View<Coord*, MemorySpace> bucket_min_y;
  Kokkos::View<Coord*, MemorySpace> bucket_max_y;

  // Fine-level CSR structure for rows within buckets
  Kokkos::View<std::size_t*, MemorySpace> bucket_row_ptr;
  Kokkos::View<std::size_t*, MemorySpace> bucket_rows;

  std::size_t num_buckets = 0;
  std::size_t num_rows = 0;

  KOKKOS_INLINE_FUNCTION
  HierarchicalRowIndex() = default;

  /**
   * @brief Find bucket index containing a y-coordinate.
   *
   * Uses binary search on bucket bounds.
   */
  KOKKOS_INLINE_FUNCTION
  int find_bucket(Coord y) const {
    if (num_buckets == 0) return -1;

    std::size_t lo = 0;
    std::size_t hi = num_buckets;

    while (lo < hi) {
      const std::size_t mid = lo + (hi - lo) / 2;
      if (bucket_max_y(mid) < y) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }

    if (lo < num_buckets && bucket_min_y(lo) <= y && y <= bucket_max_y(lo)) {
      return static_cast<int>(lo);
    }
    return -1;
  }
};

/**
 * @brief Build hierarchical row index from sorted row keys.
 *
 * @param rows Sorted SoA row keys
 * @param num_rows Number of rows
 * @param num_buckets Number of coarse buckets (power of 2 recommended)
 * @return Built hierarchical index
 */
template <class MemorySpace>
inline HierarchicalRowIndex<MemorySpace>
build_hierarchical_index(const RowKey2DSoA<MemorySpace>& rows,
                         std::size_t num_rows,
                         std::size_t num_buckets =
                             HierarchicalRowIndex<MemorySpace>::DEFAULT_COARSE_BUCKETS) {
  HierarchicalRowIndex<MemorySpace> index;
  index.num_rows = num_rows;
  index.num_buckets = num_buckets;

  if (num_rows == 0 || num_buckets == 0) {
    return index;
  }

  // Allocate index structures
  index.bucket_min_y = Kokkos::View<Coord*, MemorySpace>("bucket_min_y", num_buckets);
  index.bucket_max_y = Kokkos::View<Coord*, MemorySpace>("bucket_max_y", num_buckets);
  index.bucket_row_ptr = Kokkos::View<std::size_t*, MemorySpace>("bucket_row_ptr", num_buckets + 1);
  index.bucket_rows = Kokkos::View<std::size_t*, MemorySpace>("bucket_rows", num_rows);

  // Step 1: Compute bucket ranges (parallel scan)
  Kokkos::View<std::size_t*, MemorySpace> bucket_counts("bucket_counts", num_buckets);

  Coord y_min = rows.y(0);
  Coord y_max = rows.y(num_rows - 1);
  Coord y_range = y_max - y_min + 1;

  // Assign each row to a bucket
  Kokkos::parallel_for(
      "assign_to_buckets",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows),
      KOKKOS_LAMBDA(const std::size_t i) {
        const Coord y = rows.y(i);
        const std::size_t bucket = static_cast<std::size_t>(
            (static_cast<double>(y - y_min) / y_range) * num_buckets);
        const std::size_t clamped_bucket = (bucket >= num_buckets) ? num_buckets - 1 : bucket;
        Kokkos::atomic_increment(&bucket_counts(clamped_bucket));
      });

  // Step 2: Exclusive scan to get bucket offsets
  Kokkos::parallel_scan(
      "bucket_offset_scan",
      Kokkos::RangePolicy<ExecSpace>(0, num_buckets + 1),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final) {
        if (i < num_buckets) {
          if (final) {
            index.bucket_row_ptr(i + 1) = update;
          }
          update += bucket_counts(i);
        } else {
          if (final) {
            index.bucket_row_ptr(i) = update;
          }
        }
      });

  // Step 3: Fill bucket_rows and compute bounds
  Kokkos::parallel_for(
      "reset_bucket_counts",
      Kokkos::RangePolicy<ExecSpace>(0, num_buckets),
      KOKKOS_LAMBDA(const std::size_t i) {
        bucket_counts(i) = 0;
      });

  Kokkos::parallel_for(
      "fill_buckets",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows),
      KOKKOS_LAMBDA(const std::size_t i) {
        const Coord y = rows.y(i);
        const std::size_t bucket = static_cast<std::size_t>(
            (static_cast<double>(y - y_min) / y_range) * num_buckets);
        const std::size_t clamped_bucket = (bucket >= num_buckets) ? num_buckets - 1 : bucket;

        const std::size_t offset = index.bucket_row_ptr(clamped_bucket) +
                                   Kokkos::atomic_fetch_add(&bucket_counts(clamped_bucket), 1);
        index.bucket_rows(offset) = i;
      });

  // Step 4: Compute bucket min/max y values
  Kokkos::parallel_for(
      "compute_bucket_bounds",
      Kokkos::RangePolicy<ExecSpace>(0, num_buckets),
      KOKKOS_LAMBDA(const std::size_t b) {
        const std::size_t begin = index.bucket_row_ptr(b);
        const std::size_t end = index.bucket_row_ptr(b + 1);

        if (begin < end) {
          Coord b_min = rows.y(index.bucket_rows(begin));
          Coord b_max = rows.y(index.bucket_rows(begin));

          for (std::size_t i = begin + 1; i < end; ++i) {
            const Coord y = rows.y(index.bucket_rows(i));
            if (y < b_min) b_min = y;
            if (y > b_max) b_max = y;
          }

          index.bucket_min_y(b) = b_min;
          index.bucket_max_y(b) = b_max;
        } else {
          index.bucket_min_y(b) = 0;
          index.bucket_max_y(b) = -1;  // Empty bucket marker
        }
      });

  ExecSpace().fence();
  return index;
}

/**
 * @brief Find row index using hierarchical index.
 *
 * Two-phase lookup:
 * 1. O(log num_buckets) to find bucket
 * 2. O(log bucket_size) binary search within bucket
 *
 * Overall: O(log num_buckets + log bucket_size) = O(log num_rows)
 * but with much better cache locality.
 */
template <class MemorySpace>
KOKKOS_INLINE_FUNCTION
int find_row_by_y_hierarchical(const HierarchicalRowIndex<MemorySpace>& index,
                               const RowKey2DSoA<MemorySpace>& rows,
                               Coord y) {
  // Phase 1: Find bucket
  const int bucket = index.find_bucket(y);
  if (bucket < 0) {
    return -1;
  }

  // Phase 2: Binary search within bucket
  const std::size_t begin = index.bucket_row_ptr(bucket);
  const std::size_t end = index.bucket_row_ptr(bucket + 1);

  std::size_t lo = begin;
  std::size_t hi = end;

  while (lo < hi) {
    const std::size_t mid = lo + (hi - lo) / 2;
    const std::size_t row_idx = index.bucket_rows(mid);
    if (rows.y(row_idx) < y) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  if (lo < end && lo >= begin) {
    const std::size_t row_idx = index.bucket_rows(lo);
    if (rows.y(row_idx) == y) {
      return static_cast<int>(row_idx);
    }
  }

  return -1;
}

// ============================================================================
// 5. Hash-Based Row Mapping (for dense, regular meshes)
// ============================================================================

/**
 * @brief Perfect hash-based row mapping for dense, axis-aligned meshes.
 *
 * This is optimal for regular rectangular meshes where rows form a
 * contiguous sequence [y_min, y_max). The hash is simply: idx = y - y_min.
 *
 * NOT suitable for irregular/sparse geometries.
 */
template <class MemorySpace>
struct DenseRowHashMap {
  Coord y_base = 0;           ///< Base Y coordinate
  std::size_t num_rows = 0;   ///< Number of rows (must equal y_max - y_min)

  KOKKOS_INLINE_FUNCTION
  bool is_valid() const {
    return num_rows > 0;
  }

  /**
   * @brief O(1) lookup for dense meshes.
   */
  KOKKOS_INLINE_FUNCTION
  int lookup(Coord y) const {
    const std::size_t idx = static_cast<std::size_t>(y - y_base);
    if (idx < num_rows) {
      return static_cast<int>(idx);
    }
    return -1;
  }
};

/**
 * @brief Build dense hash map from sorted row keys.
 *
 * @param rows Sorted SoA row keys
 * @param num_rows Number of rows
 * @return Dense hash map (if applicable), or invalid map
 */
template <class MemorySpace>
inline DenseRowHashMap<MemorySpace>
build_dense_hash_map(const RowKey2DSoA<MemorySpace>& rows,
                     std::size_t num_rows) {
  DenseRowHashMap<MemorySpace> map;

  if (num_rows == 0) {
    return map;
  }

  const Coord y_min = rows.y(0);
  const Coord y_max = rows.y(num_rows - 1);

  // Check if rows form a contiguous sequence
  const std::size_t expected_count = static_cast<std::size_t>(y_max - y_min + 1);
  if (expected_count != num_rows) {
    // Not a dense mesh - return invalid map
    return map;
  }

  // Verify contiguity (can be skipped in production for performance)
  Kokkos::View<int, MemorySpace> is_contiguous("is_contiguous", 1);
  Kokkos::deep_copy(is_contiguous, 1);

  Kokkos::parallel_for(
      "verify_contiguity",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows),
      KOKKOS_LAMBDA(const std::size_t i) {
        if (rows.y(i) != y_min + static_cast<Coord>(i)) {
          Kokkos::atomic_store(is_contiguous.data(), 0);
        }
      });

  int host_contiguous;
  Kokkos::deep_copy(host_contiguous, is_contiguous);

  if (host_contiguous) {
    map.y_base = y_min;
    map.num_rows = num_rows;
  }

  ExecSpace().fence();
  return map;
}

/**
 * @brief General-purpose hash map for sparse, irregular meshes.
 *
 * Uses open addressing with linear probing for collision resolution.
 * Suitable for general meshes that don't benefit from hierarchical indexing.
 *
 * Load factor should be kept below 0.7 for good performance.
 */
template <class MemorySpace>
struct SparseRowHashMap {
  Kokkos::View<Coord*, MemorySpace> keys;      ///< Hash table keys (y values)
  Kokkos::View<int*, MemorySpace> values;      ///< Hash table values (row indices)
  std::size_t table_size = 0;                  ///< Power of 2 for fast modulo
  int empty_marker = -1;                       ///< Marks empty slots

  KOKKOS_INLINE_FUNCTION
  bool is_valid() const {
    return table_size > 0;
  }

  /**
   * @brief Hash function for y-coordinate.
   * Uses simple multiplication method for power-of-2 tables.
   */
  KOKKOS_INLINE_FUNCTION
  std::size_t hash(Coord y) const {
    // Multiply by a prime and use bitmask for power-of-2 table
    const unsigned key = static_cast<unsigned>(y);
    return (key * 0x9e3779b9U) & (table_size - 1);
  }

  /**
   * @brief O(1) average-case lookup with linear probing.
   */
  KOKKOS_INLINE_FUNCTION
  int lookup(Coord y) const {
    if (table_size == 0) return -1;

    std::size_t idx = hash(y);
    const std::size_t start_idx = idx;

    do {
      if (values(idx) == empty_marker) {
        return -1;  // Key not found
      }
      if (keys(idx) == y) {
        return values(idx);  // Found
      }
      idx = (idx + 1) & (table_size - 1);  // Linear probing
    } while (idx != start_idx);

    return -1;  // Table full, key not found
  }
};

/**
 * @brief Build sparse hash map from sorted row keys.
 *
 * @param rows Sorted SoA row keys
 * @param num_rows Number of rows
 * @param load_factor Maximum load factor (default 0.7)
 * @return Built sparse hash map
 */
template <class MemorySpace>
inline SparseRowHashMap<MemorySpace>
build_sparse_hash_map(const RowKey2DSoA<MemorySpace>& rows,
                      std::size_t num_rows,
                      double load_factor = 0.7) {
  SparseRowHashMap<MemorySpace> map;

  if (num_rows == 0) {
    return map;
  }

  // Calculate table size (power of 2)
  const std::size_t min_size = static_cast<std::size_t>(
      static_cast<double>(num_rows) / load_factor);
  std::size_t table_size = 1;
  while (table_size < min_size) {
    table_size *= 2;
  }

  map.table_size = table_size;
  map.keys = Kokkos::View<Coord*, MemorySpace>("hash_keys", table_size);
  map.values = Kokkos::View<int*, MemorySpace>("hash_values", table_size);

  // Initialize table
  Kokkos::deep_copy(map.values, map.empty_marker);

  // Insert all rows
  Kokkos::parallel_for(
      "build_hash_table",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows),
      KOKKOS_LAMBDA(const std::size_t i) {
        const Coord y = rows.y(i);
        std::size_t idx = map.hash(y);

        // Linear probing to find empty slot
        while (map.values(idx) != map.empty_marker) {
          idx = (idx + 1) & (map.table_size - 1);
        }

        map.keys(idx) = y;
        map.values(idx) = static_cast<int>(i);
      });

  ExecSpace().fence();
  return map;
}

// ============================================================================
// 6. Adaptive Row Mapping (automatic strategy selection)
// ============================================================================

/**
 * @brief Adaptive row mapping that automatically selects the best strategy.
 *
 * Analyzes the row distribution and chooses:
 * - Dense hash: for contiguous sequences
 * - SoA binary search: for small datasets
 * - Hierarchical index: for large, irregular datasets
 * - Sparse hash: for highly irregular datasets
 */
template <class MemorySpace>
class AdaptiveRowMapping {
public:
  enum Strategy {
    DENSE_HASH,      ///< Perfect hash for dense meshes
    SOA_BINARY,      ///< Simple SoA binary search
    HIERARCHICAL,    ///< Coarse-fine hierarchical index
    SPARSE_HASH      ///< Open addressing hash table
  };

  RowKey2DSoA<MemorySpace> rows_soa;
  DenseRowHashMap<MemorySpace> dense_map;
  SparseRowHashMap<MemorySpace> sparse_map;
  HierarchicalRowIndex<MemorySpace> hierarchical_index;
  std::size_t num_rows = 0;
  Strategy strategy = SOA_BINARY;

  /**
   * @brief Build adaptive mapping from row keys.
   */
  static inline AdaptiveRowMapping<MemorySpace>
  build(const typename IntervalSet2D<MemorySpace>::RowKeyView& rows_aos,
        std::size_t num_rows) {
    AdaptiveRowMapping<MemorySpace> mapping;

    if (num_rows == 0) {
      return mapping;
    }

    mapping.num_rows = num_rows;
    mapping.rows_soa = to_soa(IntervalSet2D<MemorySpace>{
        rows_aos,
        typename IntervalSet2D<MemorySpace>::IndexView(),
        typename IntervalSet2D<MemorySpace>::IntervalView(),
        typename IntervalSet2D<MemorySpace>::OffsetView(),
        0, num_rows, 0
    });

    // Strategy selection (heuristic)
    const Coord y_min = mapping.rows_soa.y(0);
    const Coord y_max = mapping.rows_soa.y(num_rows - 1);
    const std::size_t y_range = static_cast<std::size_t>(y_max - y_min + 1);
    const double density = static_cast<double>(num_rows) / static_cast<double>(y_range);

    // Small dataset: use simple binary search
    if (num_rows < 128) {
      mapping.strategy = SOA_BINARY;
    }
    // Dense mesh: use perfect hash
    else if (density > 0.9) {
      mapping.dense_map = build_dense_hash_map(mapping.rows_soa, num_rows);
      if (mapping.dense_map.is_valid()) {
        mapping.strategy = DENSE_HASH;
      } else {
        mapping.strategy = HIERARCHICAL;
        mapping.hierarchical_index = build_hierarchical_index(
            mapping.rows_soa, num_rows);
      }
    }
    // Large regular mesh: use hierarchical index
    else if (num_rows > 4096 && density > 0.3) {
      mapping.strategy = HIERARCHICAL;
      mapping.hierarchical_index = build_hierarchical_index(
          mapping.rows_soa, num_rows);
    }
    // Sparse/irregular: use hash map
    else {
      mapping.strategy = SPARSE_HASH;
      mapping.sparse_map = build_sparse_hash_map(
          mapping.rows_soa, num_rows);
    }

    ExecSpace().fence();
    return mapping;
  }

  /**
   * @brief Find row index by y-coordinate using selected strategy.
   */
  KOKKOS_INLINE_FUNCTION
  int find_row(Coord y) const {
    switch (strategy) {
      case DENSE_HASH:
        return dense_map.lookup(y);
      case SOA_BINARY:
        return find_row_by_y_soa(rows_soa, num_rows, y);
      case HIERARCHICAL:
        return find_row_by_y_hierarchical(hierarchical_index, rows_soa, y);
      case SPARSE_HASH:
        return sparse_map.lookup(y);
      default:
        return -1;
    }
  }
};

// ============================================================================
// Utility Functions for Row Mapping
// ============================================================================

/**
 * @brief Convert IntervalSet2D to use SoA row keys.
 *
 * This is a convenience wrapper that creates a modified IntervalSet2D
 * with SoA row keys for better performance in row lookup operations.
 */
template <class MemorySpace>
inline IntervalSet2D<MemorySpace>
convert_to_soa_geometry(const IntervalSet2D<MemorySpace>& geom) {
  // For now, we keep the original structure but add a helper
  // In a full implementation, you might want to extend IntervalSet2D
  // to support both AoS and SoA representations
  return geom;
}

} // namespace row_mapping
} // namespace csr
} // namespace subsetix
