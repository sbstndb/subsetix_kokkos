#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

#include <Kokkos_Core.hpp>

namespace subsetix {
namespace tiled {

// Basic coordinate and row key types for 2D tiled interval sets.
using Coord = std::int32_t;

struct Interval {
  Coord begin = 0;  // Inclusive
  Coord end = 0;    // Exclusive
};

struct RowKey2D {
  Coord y = 0;
};

/**
 * @brief Host-side tiled representation of a 2D interval set.
 *
 * The Y axis is partitioned into fixed-height tiles. Each Tile stores its own
 * CSR structure (row_keys / row_ptr / intervals) restricted to that Y band.
 *
 * Invariants (by construction, not enforced here):
 *  - row_keys are sorted by y within each tile,
 *  - intervals in each row are sorted, non-overlapping,
 *  - tiles cover a contiguous Y band of height rows_per_tile, starting at min_y.
 */
struct TiledIntervalSet2DHost {
  Coord min_y = 0;
  std::size_t rows_per_tile = 0;

  struct Tile {
    std::vector<RowKey2D> row_keys;   // Non-empty rows in this tile
    std::vector<std::size_t> row_ptr; // CSR pointers (size = rows+1, row_ptr[0] = 0)
    std::vector<Interval> intervals;  // All intervals for this tile, concatenated
  };

  std::vector<Tile> tiles;
};

/**
 * @brief Device-friendly tiled representation using Kokkos::View.
 *
 * All tiles are flattened into global CSR arrays. Per-tile offsets and sizes
 * are stored so that kernels can navigate tiles on device.
 *
 * row_ptr stores per-tile LOCAL offsets; to access the intervals of a row
 * belonging to tile t:
 *
 *  - base_row_ptr = tile_row_ptr_offset[t]
 *  - base_interval = tile_intervals_offset[t]
 *  - local_begin = row_ptr(base_row_ptr + local_row)
 *  - local_end   = row_ptr(base_row_ptr + local_row + 1)
 *  - global_begin = base_interval + local_begin
 *  - global_end   = base_interval + local_end
 */
template <class MemorySpace>
struct TiledIntervalSet2DView {
  using RowKeyView = Kokkos::View<RowKey2D*, MemorySpace>;
  using IndexView = Kokkos::View<std::size_t*, MemorySpace>;
  using IntervalView = Kokkos::View<Interval*, MemorySpace>;

  RowKeyView row_keys;     ///< [total_rows]
  IndexView row_ptr;       ///< [total_row_ptr_entries]
  IntervalView intervals;  ///< [total_intervals]

  IndexView tile_row_keys_offset;   ///< [num_tiles]
  IndexView tile_row_ptr_offset;    ///< [num_tiles]
  IndexView tile_intervals_offset;  ///< [num_tiles]
  IndexView tile_num_rows;          ///< [num_tiles]
  IndexView tile_num_intervals;     ///< [num_tiles]

  std::size_t total_rows = 0;
  std::size_t total_row_ptr_entries = 0;
  std::size_t total_intervals = 0;

  Coord min_y = 0;
  std::size_t rows_per_tile = 0;
  std::size_t num_tiles = 0;
};

using DeviceMemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space;
using HostMemorySpace = Kokkos::HostSpace;

using TiledIntervalSet2DDevice = TiledIntervalSet2DView<DeviceMemorySpace>;
using TiledIntervalSet2DHostView = TiledIntervalSet2DView<HostMemorySpace>;

/**
 * @brief Build a device tiled interval set from a host tiled representation.
 *
 * The layout is:
 *  - tiles are concatenated in order 0..num_tiles-1,
 *  - for each tile, row_keys / row_ptr / intervals are appended in the
 *    same order as in the host Tile,
 *  - row_ptr entries remain tile-local (start at 0 for each tile).
 */
inline TiledIntervalSet2DDevice
build_device_from_host(const TiledIntervalSet2DHost& host) {
  TiledIntervalSet2DDevice dev;

  dev.min_y = host.min_y;
  dev.rows_per_tile = host.rows_per_tile;
  dev.num_tiles = host.tiles.size();

  if (dev.num_tiles == 0) {
    dev.total_rows = 0;
    dev.total_row_ptr_entries = 0;
    dev.total_intervals = 0;
    return dev;
  }

  std::size_t total_rows = 0;
  std::size_t total_row_ptr_entries = 0;
  std::size_t total_intervals = 0;

  for (const auto& tile : host.tiles) {
    total_rows += tile.row_keys.size();
    total_row_ptr_entries += tile.row_ptr.size();
    total_intervals += tile.intervals.size();
  }

  dev.total_rows = total_rows;
  dev.total_row_ptr_entries = total_row_ptr_entries;
  dev.total_intervals = total_intervals;

  if (total_rows == 0 || total_row_ptr_entries == 0) {
    return dev;
  }

  typename TiledIntervalSet2DDevice::RowKeyView row_keys(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "subsetix_tiled_row_keys"),
      total_rows);
  typename TiledIntervalSet2DDevice::IndexView row_ptr(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "subsetix_tiled_row_ptr"),
      total_row_ptr_entries);
  typename TiledIntervalSet2DDevice::IntervalView intervals(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "subsetix_tiled_intervals"),
      total_intervals);

  typename TiledIntervalSet2DDevice::IndexView tile_row_keys_offset(
      "subsetix_tiled_tile_row_keys_offset", dev.num_tiles);
  typename TiledIntervalSet2DDevice::IndexView tile_row_ptr_offset(
      "subsetix_tiled_tile_row_ptr_offset", dev.num_tiles);
  typename TiledIntervalSet2DDevice::IndexView tile_intervals_offset(
      "subsetix_tiled_tile_intervals_offset", dev.num_tiles);
  typename TiledIntervalSet2DDevice::IndexView tile_num_rows(
      "subsetix_tiled_tile_num_rows", dev.num_tiles);
  typename TiledIntervalSet2DDevice::IndexView tile_num_intervals(
      "subsetix_tiled_tile_num_intervals", dev.num_tiles);

  // Host mirrors to fill the flattened arrays and per-tile metadata.
  Kokkos::View<RowKey2D*, HostMemorySpace> h_row_keys(
      "subsetix_tiled_row_keys_host", total_rows);
  Kokkos::View<std::size_t*, HostMemorySpace> h_row_ptr(
      "subsetix_tiled_row_ptr_host", total_row_ptr_entries);
  Kokkos::View<Interval*, HostMemorySpace> h_intervals(
      "subsetix_tiled_intervals_host", total_intervals);

  Kokkos::View<std::size_t*, HostMemorySpace> h_tile_row_keys_offset(
      "subsetix_tiled_tile_row_keys_offset_host", dev.num_tiles);
  Kokkos::View<std::size_t*, HostMemorySpace> h_tile_row_ptr_offset(
      "subsetix_tiled_tile_row_ptr_offset_host", dev.num_tiles);
  Kokkos::View<std::size_t*, HostMemorySpace> h_tile_intervals_offset(
      "subsetix_tiled_tile_intervals_offset_host", dev.num_tiles);
  Kokkos::View<std::size_t*, HostMemorySpace> h_tile_num_rows(
      "subsetix_tiled_tile_num_rows_host", dev.num_tiles);
  Kokkos::View<std::size_t*, HostMemorySpace> h_tile_num_intervals(
      "subsetix_tiled_tile_num_intervals_host", dev.num_tiles);

  std::size_t row_acc = 0;
  std::size_t row_ptr_acc = 0;
  std::size_t intervals_acc = 0;

  for (std::size_t t = 0; t < dev.num_tiles; ++t) {
    const auto& tile = host.tiles[t];

    const std::size_t rows = tile.row_keys.size();
    const std::size_t row_ptr_entries = tile.row_ptr.size();
    const std::size_t ivs = tile.intervals.size();

    h_tile_row_keys_offset(t) = row_acc;
    h_tile_row_ptr_offset(t) = row_ptr_acc;
    h_tile_intervals_offset(t) = intervals_acc;
    h_tile_num_rows(t) = rows;
    h_tile_num_intervals(t) = ivs;

    for (std::size_t i = 0; i < rows; ++i) {
      h_row_keys(row_acc + i) = tile.row_keys[i];
    }
    for (std::size_t i = 0; i < row_ptr_entries; ++i) {
      h_row_ptr(row_ptr_acc + i) = tile.row_ptr[i];
    }
    for (std::size_t i = 0; i < ivs; ++i) {
      h_intervals(intervals_acc + i) = tile.intervals[i];
    }

    row_acc += rows;
    row_ptr_acc += row_ptr_entries;
    intervals_acc += ivs;
  }

  Kokkos::deep_copy(row_keys, h_row_keys);
  Kokkos::deep_copy(row_ptr, h_row_ptr);
  Kokkos::deep_copy(intervals, h_intervals);

  Kokkos::deep_copy(tile_row_keys_offset, h_tile_row_keys_offset);
  Kokkos::deep_copy(tile_row_ptr_offset, h_tile_row_ptr_offset);
  Kokkos::deep_copy(tile_intervals_offset, h_tile_intervals_offset);
  Kokkos::deep_copy(tile_num_rows, h_tile_num_rows);
  Kokkos::deep_copy(tile_num_intervals, h_tile_num_intervals);

  dev.row_keys = row_keys;
  dev.row_ptr = row_ptr;
  dev.intervals = intervals;
  dev.tile_row_keys_offset = tile_row_keys_offset;
  dev.tile_row_ptr_offset = tile_row_ptr_offset;
  dev.tile_intervals_offset = tile_intervals_offset;
  dev.tile_num_rows = tile_num_rows;
  dev.tile_num_intervals = tile_num_intervals;

  return dev;
}

/**
 * @brief Rebuild a host tiled representation from a device tiled set.
 *
 * This is primarily for inspection/debugging and tests; performance-critical
 * code should work directly with TiledIntervalSet2DDevice on the device.
 */
inline TiledIntervalSet2DHost
build_host_from_device(const TiledIntervalSet2DDevice& dev) {
  TiledIntervalSet2DHost host;

  host.min_y = dev.min_y;
  host.rows_per_tile = dev.rows_per_tile;
  host.tiles.resize(dev.num_tiles);

  if (dev.num_tiles == 0 || dev.total_rows == 0 ||
      dev.total_row_ptr_entries == 0) {
    return host;
  }

  auto h_row_keys =
      Kokkos::create_mirror_view_and_copy(HostMemorySpace{}, dev.row_keys);
  auto h_row_ptr =
      Kokkos::create_mirror_view_and_copy(HostMemorySpace{}, dev.row_ptr);
  auto h_intervals =
      Kokkos::create_mirror_view_and_copy(HostMemorySpace{}, dev.intervals);

  auto h_tile_row_keys_offset = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace{}, dev.tile_row_keys_offset);
  auto h_tile_row_ptr_offset = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace{}, dev.tile_row_ptr_offset);
  auto h_tile_intervals_offset = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace{}, dev.tile_intervals_offset);
  auto h_tile_num_rows = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace{}, dev.tile_num_rows);
  auto h_tile_num_intervals = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace{}, dev.tile_num_intervals);

  for (std::size_t t = 0; t < dev.num_tiles; ++t) {
    auto& tile = host.tiles[t];

    const std::size_t rows = h_tile_num_rows(t);
    const std::size_t rk_off = h_tile_row_keys_offset(t);
    const std::size_t rp_off = h_tile_row_ptr_offset(t);
    const std::size_t iv_off = h_tile_intervals_offset(t);
    const std::size_t ivs = h_tile_num_intervals(t);

    tile.row_keys.resize(rows);
    for (std::size_t i = 0; i < rows; ++i) {
      tile.row_keys[i] = h_row_keys(rk_off + i);
    }

    if (rows > 0) {
      const std::size_t row_ptr_entries = rows + 1;
      tile.row_ptr.resize(row_ptr_entries);
      for (std::size_t i = 0; i < row_ptr_entries; ++i) {
        tile.row_ptr[i] = h_row_ptr(rp_off + i);
      }
    } else {
      tile.row_ptr.clear();
    }

    tile.intervals.resize(ivs);
    for (std::size_t i = 0; i < ivs; ++i) {
      tile.intervals[i] = h_intervals(iv_off + i);
    }
  }

  return host;
}

} // namespace tiled
} // namespace subsetix

