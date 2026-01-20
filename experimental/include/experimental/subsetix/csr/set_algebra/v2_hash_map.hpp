// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <experimental/subsetix/csr/mesh.hpp>
#include <Kokkos_Core.hpp>

namespace experimental::subsetix::csr::v2::detail {

// ============================================================================
// GPU-Friendly Hash Map for Row Mapping
// ============================================================================

/**
 * @brief Open addressing hash map for row key -> index mapping.
 *
 * Key design:
 * - Open addressing with linear probing (GPU-friendly)
 * - Stored as parallel arrays (keys, values, occupied flags)
 * - Load factor < 0.7 to minimize collisions
 *
 * Why not std::unordered_map?
 * - Not GPU-compatible
 * - Dynamic allocation not allowed in device code
 * - Pointer-based structures cause warp divergence
 *
 * Why not binary search?
 * - Binary search is O(log n) with divergence
 * - Hash is O(1) average case
 * - Better cache locality for small to medium meshes
 */
template <class RowKey, class MemorySpace>
class RowHashMap {
public:
  using KeyView = Kokkos::View<RowKey*, MemorySpace>;
  using ValueView = Kokkos::View<int*, MemorySpace>;
  using FlagView = Kokkos::View<int*, MemorySpace>;

  KeyView keys_;
  ValueView values_;
  FlagView occupied_;  // 0 = empty, 1 = occupied
  std::size_t capacity_;
  std::size_t size_;

  // ========================================================================
  // Construction
  // ========================================================================

  RowHashMap() : capacity_(0), size_(0) {}

  /**
   * @brief Create hash map with capacity for n elements.
   *
   * Uses 1.5x capacity to keep load factor < 0.7
   */
  void reserve(std::size_t n, const std::string& label = "hash_map") {
    capacity_ = (n * 3) / 2 + 1;  // ~1.5x for load factor < 0.7
    if (capacity_ < 16) capacity_ = 16;  // Minimum size

    keys_ = KeyView(label + "_keys", capacity_);
    values_ = ValueView(label + "_values", capacity_);
    occupied_ = FlagView(label + "_occupied", capacity_);

    // Initialize as empty
    Kokkos::deep_copy(occupied_, 0);
    size_ = 0;
  }

  // ========================================================================
  // Host-side insert (build phase)
  // ========================================================================

  /**
   * @brief Insert key-value pair from host.
   *
   * Used during hash map construction (build phase).
   * For device-side operations, see device_find().
   */
  void insert(const RowKey& key, int value) {
    if (size_ >= capacity_) {
      // Should not happen if reserve() was called correctly
      return;
    }

    std::size_t idx = hash(key) % capacity_;

    // Linear probing
    while (occupied_(idx) && keys_(idx) != key) {
      idx = (idx + 1) % capacity_;
    }

    // Insert
    keys_(idx) = key;
    values_(idx) = value;
    occupied_(idx) = 1;
    ++size_;
  }

  // ========================================================================
  // Device-side find (query phase)
  // ========================================================================

  /**
   * @brief Find value by key in device code.
   *
   * Returns index or -1 if not found.
   *
   * Usage in GPU kernel:
   *   const int idx = map.device_find(map, row_key);
   */
  KOKKOS_INLINE_FUNCTION
  static int device_find(const RowHashMap& map, const RowKey& key) {
    if (map.capacity_ == 0) return -1;

    std::size_t idx = hash_device(key) % map.capacity_;

    // Linear probing with iteration limit
    const std::size_t max_probe = map.capacity_;  // Prevent infinite loops
    std::size_t probe_count = 0;

    while (probe_count < max_probe) {
      if (!map.occupied_(idx)) {
        return -1;  // Not found
      }
      if (map.keys_(idx) == key) {
        return map.values_(idx);  // Found
      }
      idx = (idx + 1) % map.capacity_;
      ++probe_count;
    }

    return -1;  // Not found (table full or key absent)
  }

  // ========================================================================
  // Hash functions
  // ========================================================================

  // Device-side hash (for querying) - public for use in build function
  KOKKOS_INLINE_FUNCTION
  static constexpr std::size_t hash_device(const RowKey2D& key) {
    std::size_t h = static_cast<std::size_t>(key.y);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    return h;
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr std::size_t hash_device(const RowKey3D& key) {
    std::size_t h = static_cast<std::size_t>(key.y);
    h ^= static_cast<std::size_t>(key.z) << 32;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    return h;
  }

private:
  // Host-side hash (for building)
  static constexpr std::size_t hash(const RowKey2D& key) {
    // Simple but effective hash for 2D
    std::size_t h = static_cast<std::size_t>(key.y);
    h ^= h >> 33;  // Bit mixing
    h *= 0xff51afd7ed558ccdULL;
    return h;
  }

  static constexpr std::size_t hash(const RowKey3D& key) {
    // Hash pair (y, z)
    std::size_t h = static_cast<std::size_t>(key.y);
    h ^= static_cast<std::size_t>(key.z) << 32;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    return h;
  }
};

// ============================================================================
// Hash map builder (serial construction)
// ============================================================================

/**
 * @brief Build hash map from row keys.
 *
 * Note: This is a serial build for simplicity.
 * TODO: Implement parallel build with proper atomics.
 */
template <class RowKey, class RowKeyView, class MemorySpace>
inline void build_hash_map_parallel(
    const RowKeyView& row_keys,
    std::size_t num_rows,
    RowHashMap<RowKey, MemorySpace>& map_out)
{
  using ExecSpace = Kokkos::DefaultExecutionSpace;

  map_out.reserve(num_rows, "row_hash_map");

  // For Serial backend, MemorySpace is HostSpace, so row_keys is already accessible
  // For CUDA/OpenMP, we need to copy to HostSpace first
  #ifdef KOKKOS_ENABLE_SERIAL
    // Serial backend: row_keys is already in HostSpace, use directly
    for (std::size_t i = 0; i < num_rows; ++i) {
      RowKey key;
      Kokkos::View<RowKey*, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> const_row_keys(
          const_cast<RowKey*>(row_keys.data()), num_rows);
      key = const_row_keys(i);
      map_out.insert(key, static_cast<int>(i));
    }
  #else
    // Other backends: copy to host first
    auto row_keys_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, row_keys);
    for (std::size_t i = 0; i < num_rows; ++i) {
      map_out.insert(row_keys_host(i), static_cast<int>(i));
    }
  #endif

  // Sync to device
  ExecSpace().fence();
}

} // namespace experimental::subsetix::csr::v2::detail
