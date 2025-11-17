#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

#include <Kokkos_Core.hpp>

namespace subsetix {
namespace csr {

// Basic coordinate and row key types for 2D CSR interval sets.
using Coord = std::int32_t;

struct Interval {
  Coord begin = 0;  // Inclusive
  Coord end = 0;    // Exclusive
};

struct RowKey2D {
  Coord y = 0;
};

/**
 * @brief Host-side CSR representation of a 2D interval set.
 *
 * Invariants (par convention) :
 *  - row_keys.size() == num_rows,
 *  - row_ptr.size() == num_rows + 1,
 *  - intervals.size() == row_ptr.back(),
 *  - pour chaque ligne, les intervalles sont triés et non chevauchants.
 */
struct IntervalSet2DHost {
  std::vector<RowKey2D> row_keys;
  std::vector<std::size_t> row_ptr;
  std::vector<Interval> intervals;

  std::size_t num_rows() const { return row_keys.size(); }
  std::size_t num_intervals() const { return intervals.size(); }
};

/**
 * @brief Device-friendly CSR representation using Kokkos::View.
 *
 * Layout identique au host : un seul CSR global (pas de tuiles).
 */
template <class MemorySpace>
struct IntervalSet2DView {
  using RowKeyView = Kokkos::View<RowKey2D*, MemorySpace>;
  using IndexView = Kokkos::View<std::size_t*, MemorySpace>;
  using IntervalView = Kokkos::View<Interval*, MemorySpace>;

  RowKeyView row_keys;    ///< [num_rows]
  IndexView row_ptr;      ///< [num_rows + 1]
  IntervalView intervals; ///< [num_intervals]

  std::size_t num_rows = 0;
  std::size_t num_intervals = 0;
};

using DeviceMemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space;
using HostMemorySpace = Kokkos::HostSpace;

using IntervalSet2DDevice = IntervalSet2DView<DeviceMemorySpace>;
using IntervalSet2DHostView = IntervalSet2DView<HostMemorySpace>;

/**
 * @brief Build a device CSR interval set from a host CSR representation.
 */
inline IntervalSet2DDevice
build_device_from_host(const IntervalSet2DHost& host) {
  IntervalSet2DDevice dev;

  const std::size_t num_rows = host.row_keys.size();
  if (num_rows == 0) {
    return dev;
  }

  const std::size_t row_ptr_size = host.row_ptr.size();
  if (row_ptr_size != num_rows + 1) {
    return dev;
  }

  const std::size_t num_intervals = host.intervals.size();

  dev.num_rows = num_rows;
  dev.num_intervals = num_intervals;

  typename IntervalSet2DDevice::RowKeyView row_keys(
      "subsetix_csr_row_keys", num_rows);
  typename IntervalSet2DDevice::IndexView row_ptr(
      "subsetix_csr_row_ptr", row_ptr_size);
  typename IntervalSet2DDevice::IntervalView intervals(
      "subsetix_csr_intervals", num_intervals);

  Kokkos::View<RowKey2D*, HostMemorySpace> h_row_keys(
      "subsetix_csr_row_keys_host", num_rows);
  Kokkos::View<std::size_t*, HostMemorySpace> h_row_ptr(
      "subsetix_csr_row_ptr_host", row_ptr_size);
  Kokkos::View<Interval*, HostMemorySpace> h_intervals(
      "subsetix_csr_intervals_host", num_intervals);

  for (std::size_t i = 0; i < num_rows; ++i) {
    h_row_keys(i) = host.row_keys[i];
  }
  for (std::size_t i = 0; i < row_ptr_size; ++i) {
    h_row_ptr(i) = host.row_ptr[i];
  }
  for (std::size_t i = 0; i < num_intervals; ++i) {
    h_intervals(i) = host.intervals[i];
  }

  Kokkos::deep_copy(row_keys, h_row_keys);
  Kokkos::deep_copy(row_ptr, h_row_ptr);
  Kokkos::deep_copy(intervals, h_intervals);

  dev.row_keys = row_keys;
  dev.row_ptr = row_ptr;
  dev.intervals = intervals;

  return dev;
}

/**
 * @brief Rebuild a host CSR representation from a device CSR interval set.
 */
inline IntervalSet2DHost
build_host_from_device(const IntervalSet2DDevice& dev) {
  IntervalSet2DHost host;

  const std::size_t num_rows = dev.num_rows;
  const std::size_t num_intervals = dev.num_intervals;

  if (num_rows == 0) {
    return host;
  }

  auto h_row_keys =
      Kokkos::create_mirror_view_and_copy(HostMemorySpace{}, dev.row_keys);
  auto h_row_ptr =
      Kokkos::create_mirror_view_and_copy(HostMemorySpace{}, dev.row_ptr);
  auto h_intervals =
      Kokkos::create_mirror_view_and_copy(HostMemorySpace{}, dev.intervals);

  host.row_keys.resize(num_rows);
  host.row_ptr.resize(num_rows + 1);
  host.intervals.resize(num_intervals);

  for (std::size_t i = 0; i < num_rows; ++i) {
    host.row_keys[i] = h_row_keys(i);
  }
  for (std::size_t i = 0; i < num_rows + 1; ++i) {
    host.row_ptr[i] = h_row_ptr(i);
  }
  for (std::size_t i = 0; i < num_intervals; ++i) {
    host.intervals[i] = h_intervals(i);
  }

  return host;
}

} // namespace csr
} // namespace subsetix

