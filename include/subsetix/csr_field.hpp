#pragma once

#include <cstddef>
#include <vector>

#include <Kokkos_Core.hpp>

#include <subsetix/csr_interval_set.hpp>

namespace subsetix {
namespace csr {

/**
 * @brief Per-interval metadata for fields: geometry + offset into values array.
 *
 * Each interval stores its [begin, end) coordinates on X and the starting
 * offset of its cell values in the linear values buffer.
 */
struct FieldInterval {
  Coord begin = 0;
  Coord end = 0;
  std::size_t value_offset = 0;

  Coord size() const { return end - begin; }
};

/**
 * @brief Host-side sparse 2D field using CSR-of-intervals layout.
 *
 * Layout:
 *   - row_keys: one entry per non-empty row (Y coordinate)
 *   - row_ptr: CSR pointers into intervals (size = num_rows + 1)
 *   - intervals: FieldInterval entries (geometry + offset into values)
 *   - values: concatenation of all cell values (per interval, per row)
 *
 * Intervals must be appended in non-decreasing Y, and for a given Y in
 * non-decreasing X (begin). Intervals within a row are expected to be
 * non-overlapping.
 */
template <typename T>
struct IntervalField2DHost {
  std::vector<RowKey2D> row_keys;
  std::vector<std::size_t> row_ptr;
  std::vector<FieldInterval> intervals;
  std::vector<T> values;

  IntervalField2DHost() {
    row_ptr.push_back(0);
  }

  std::size_t num_rows() const { return row_keys.size(); }
  std::size_t num_intervals() const { return intervals.size(); }
  std::size_t value_count() const { return values.size(); }

  /**
   * @brief Append an interval [begin, begin+vals.size()) on row y with values.
   *
   * Rows must be appended in strictly increasing Y. For a given row, X
   * coordinates must be non-decreasing.
   */
  void append_interval(Coord y, Coord begin, const std::vector<T>& vals) {
    if (vals.empty()) {
      return;
    }

    const Coord end = static_cast<Coord>(begin + static_cast<Coord>(vals.size()));

    if (row_keys.empty() || row_keys.back().y != y) {
      // New row: enforce increasing Y and start a new CSR row.
      if (!row_keys.empty()) {
        if (!(row_keys.back().y < y)) {
          // Invalid ordering: ignore in release builds.
          return;
        }
      }
      row_keys.push_back(RowKey2D{y});
      row_ptr.push_back(intervals.size());
    } else {
      // Same row: enforce non-decreasing X.
      const std::size_t last_idx = intervals.empty() ? 0 : intervals.size() - 1;
      if (!intervals.empty()) {
        const Coord last_begin = intervals[last_idx].begin;
        if (!(last_begin <= begin)) {
          return;
        }
      }
    }

    const std::size_t offset = values.size();
    FieldInterval fi;
    fi.begin = begin;
    fi.end = end;
    fi.value_offset = offset;
    intervals.push_back(fi);

    values.insert(values.end(), vals.begin(), vals.end());

    // Update CSR pointer for current row to point past all intervals added so far.
    row_ptr.back() = intervals.size();
  }
};

/**
 * @brief Device-friendly field representation using Kokkos::View.
 *
 * Geometry is stored in CSR form similarly to IntervalSet2DDevice, but each
 * interval also carries an offset into the linear values array.
 */
template <typename T, class MemorySpace>
struct IntervalField2DView {
  using RowKeyView = Kokkos::View<RowKey2D*, MemorySpace>;
  using IndexView = Kokkos::View<std::size_t*, MemorySpace>;
  using IntervalView = Kokkos::View<FieldInterval*, MemorySpace>;
  using ValueView = Kokkos::View<T*, MemorySpace>;

  RowKeyView row_keys;    ///< [num_rows]
  IndexView row_ptr;      ///< [num_rows + 1]
  IntervalView intervals; ///< [num_intervals]
  ValueView values;       ///< [value_count]

  std::size_t num_rows = 0;
  std::size_t num_intervals = 0;
  std::size_t value_count = 0;
};

template <typename T>
using IntervalField2DDevice = IntervalField2DView<T, DeviceMemorySpace>;

template <typename T>
using IntervalField2DHostView = IntervalField2DView<T, HostMemorySpace>;

/**
 * @brief Build a device field from a host field.
 */
template <typename T>
inline IntervalField2DDevice<T>
build_device_field_from_host(const IntervalField2DHost<T>& host) {
  IntervalField2DDevice<T> dev;

  const std::size_t num_rows = host.row_keys.size();
  const std::size_t num_row_ptr = host.row_ptr.size();
  const std::size_t num_intervals = host.intervals.size();
  const std::size_t value_count = host.values.size();

  if (num_rows == 0 || num_row_ptr == 0) {
    return dev;
  }

  dev.row_keys = typename IntervalField2DDevice<T>::RowKeyView(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "subsetix_csr_field_row_keys"),
      num_rows);
  dev.row_ptr = typename IntervalField2DDevice<T>::IndexView(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "subsetix_csr_field_row_ptr"),
      num_row_ptr);
  dev.intervals = typename IntervalField2DDevice<T>::IntervalView(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "subsetix_csr_field_intervals"),
      num_intervals);
  dev.values = typename IntervalField2DDevice<T>::ValueView(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "subsetix_csr_field_values"),
      value_count);

  Kokkos::View<RowKey2D*, HostMemorySpace> h_row_keys(
      "subsetix_csr_field_row_keys_host", num_rows);
  Kokkos::View<std::size_t*, HostMemorySpace> h_row_ptr(
      "subsetix_csr_field_row_ptr_host", num_row_ptr);
  Kokkos::View<FieldInterval*, HostMemorySpace> h_intervals(
      "subsetix_csr_field_intervals_host", num_intervals);
  Kokkos::View<T*, HostMemorySpace> h_values(
      "subsetix_csr_field_values_host", value_count);

  for (std::size_t i = 0; i < num_rows; ++i) {
    h_row_keys(i) = host.row_keys[i];
  }
  for (std::size_t i = 0; i < num_row_ptr; ++i) {
    h_row_ptr(i) = host.row_ptr[i];
  }
  for (std::size_t i = 0; i < num_intervals; ++i) {
    h_intervals(i) = host.intervals[i];
  }
  for (std::size_t i = 0; i < value_count; ++i) {
    h_values(i) = host.values[i];
  }

  Kokkos::deep_copy(dev.row_keys, h_row_keys);
  Kokkos::deep_copy(dev.row_ptr, h_row_ptr);
  Kokkos::deep_copy(dev.intervals, h_intervals);
  Kokkos::deep_copy(dev.values, h_values);

  dev.num_rows = num_rows;
  dev.num_intervals = num_intervals;
  dev.value_count = value_count;

  return dev;
}

/**
 * @brief Rebuild a host field from a device field.
 */
template <typename T>
inline IntervalField2DHost<T>
build_host_field_from_device(const IntervalField2DDevice<T>& dev) {
  IntervalField2DHost<T> host;

  const std::size_t num_rows = dev.num_rows;
  const std::size_t num_intervals = dev.num_intervals;
  const std::size_t value_count = dev.value_count;

  if (num_rows == 0) {
    return host;
  }

  // Create mirrors of the device views. Note that device views may have larger capacity
  // than the actual size (num_rows, etc.). We must copy only the valid parts.
  // Or copy everything and resize on host.
  // Kokkos::create_mirror_view_and_copy copies the WHOLE view extent.
  // If capacity >> size, this is wasteful and potentially wrong if we assume size == extent.
  
  auto h_row_keys_full = Kokkos::create_mirror_view_and_copy(HostMemorySpace{}, dev.row_keys);
  auto h_row_ptr_full = Kokkos::create_mirror_view_and_copy(HostMemorySpace{}, dev.row_ptr);
  auto h_intervals_full = Kokkos::create_mirror_view_and_copy(HostMemorySpace{}, dev.intervals);
  auto h_values_full = Kokkos::create_mirror_view_and_copy(HostMemorySpace{}, dev.values);

  host.row_keys.resize(num_rows);
  host.row_ptr.resize(num_rows + 1);
  host.intervals.resize(num_intervals);
  host.values.resize(value_count);

  // Copy only valid data
  for (std::size_t i = 0; i < num_rows; ++i) {
    host.row_keys[i] = h_row_keys_full(i);
  }
  for (std::size_t i = 0; i < num_rows + 1; ++i) {
    host.row_ptr[i] = h_row_ptr_full(i);
  }
  for (std::size_t i = 0; i < num_intervals; ++i) {
    host.intervals[i] = h_intervals_full(i);
  }
  for (std::size_t i = 0; i < value_count; ++i) {
    host.values[i] = h_values_full(i);
  }

  return host;
}

/**
 * @brief Build a host field filled with a constant value, using an existing
 *        IntervalSet2DHost geometry as template.
 */
template <typename T>
inline IntervalField2DHost<T>
make_field_like_geometry(const IntervalSet2DHost& geom,
                         const T& init_value) {
  IntervalField2DHost<T> field;

  const std::size_t num_rows = geom.row_keys.size();
  if (num_rows == 0 || geom.row_ptr.size() != num_rows + 1) {
    return field;
  }

  field.row_keys.clear();
  field.row_ptr.clear();
  field.intervals.clear();
  field.values.clear();

  field.row_keys.reserve(num_rows);
  field.row_ptr.reserve(num_rows + 1);

  field.row_ptr.push_back(0);

  for (std::size_t i = 0; i < num_rows; ++i) {
    const Coord y = geom.row_keys[i].y;
    field.row_keys.push_back(RowKey2D{y});

    const std::size_t begin = geom.row_ptr[i];
    const std::size_t end = geom.row_ptr[i + 1];

    for (std::size_t k = begin; k < end; ++k) {
      const Interval& iv = geom.intervals[k];
      const Coord len = static_cast<Coord>(iv.end - iv.begin);
      if (len <= 0) {
        continue;
      }

      FieldInterval fi;
      fi.begin = iv.begin;
      fi.end = iv.end;
      fi.value_offset = field.values.size();

      field.intervals.push_back(fi);
      field.values.insert(field.values.end(),
                          static_cast<std::size_t>(len),
                          init_value);
    }

    field.row_ptr.push_back(field.intervals.size());
  }

  return field;
}

} // namespace csr
} // namespace subsetix
