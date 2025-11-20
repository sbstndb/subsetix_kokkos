#pragma once

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

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
  std::vector<std::size_t> cell_offsets;
  std::size_t total_cells = 0;

  std::size_t num_rows() const { return row_keys.size(); }
  std::size_t num_intervals() const { return intervals.size(); }

  void rebuild_mapping() {
    cell_offsets.resize(intervals.size());
    std::size_t accum = 0;
    for (std::size_t i = 0; i < intervals.size(); ++i) {
      cell_offsets[i] = accum;
      accum += static_cast<std::size_t>(intervals[i].end - intervals[i].begin);
    }
    total_cells = accum;
  }
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
  using OffsetView = Kokkos::View<std::size_t*, MemorySpace>;

  RowKeyView row_keys;    ///< [num_rows]
  IndexView row_ptr;      ///< [num_rows + 1]
  IntervalView intervals; ///< [num_intervals]
  OffsetView cell_offsets; ///< [num_intervals]
  std::size_t total_cells = 0;
  std::size_t num_rows = 0;
  std::size_t num_intervals = 0;
};

using DeviceMemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space;
using HostMemorySpace = Kokkos::HostSpace;

using IntervalSet2DDevice = IntervalSet2DView<DeviceMemorySpace>;
using IntervalSet2DHostView = IntervalSet2DView<HostMemorySpace>;

inline IntervalSet2DDevice
allocate_interval_set_device(std::size_t row_capacity,
                             std::size_t interval_capacity) {
  IntervalSet2DDevice dev;

  const std::size_t row_ptr_size =
      (row_capacity > 0) ? (row_capacity + 1) : std::size_t(1);

  dev.row_keys = IntervalSet2DDevice::RowKeyView(
      "subsetix_csr_prealloc_row_keys", row_capacity);
  dev.row_ptr = IntervalSet2DDevice::IndexView(
      "subsetix_csr_prealloc_row_ptr", row_ptr_size);
  dev.intervals = IntervalSet2DDevice::IntervalView(
      "subsetix_csr_prealloc_intervals", interval_capacity);
  dev.cell_offsets = IntervalSet2DDevice::OffsetView(
      "subsetix_csr_prealloc_offsets", interval_capacity);

  dev.num_rows = 0;
  dev.num_intervals = 0;
  dev.total_cells = 0;
  return dev;
}

inline void compute_cell_offsets_device(IntervalSet2DDevice& dev) {
  using ExecSpace = Kokkos::DefaultExecutionSpace;

  if (dev.num_intervals == 0) {
    dev.total_cells = 0;
    return;
  }

  if (dev.cell_offsets.extent(0) < dev.num_intervals) {
    dev.cell_offsets = IntervalSet2DDevice::OffsetView(
        "subsetix_csr_offsets", dev.num_intervals);
  }

  auto intervals = dev.intervals;
  auto offsets = dev.cell_offsets;
  Kokkos::View<std::size_t, DeviceMemorySpace> total("subsetix_csr_total_cells");

  Kokkos::parallel_scan(
      "subsetix_csr_compute_offsets",
      Kokkos::RangePolicy<ExecSpace>(0, dev.num_intervals),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& update,
                    const bool final_pass) {
        const Interval iv = intervals(i);
        const std::size_t len =
            static_cast<std::size_t>(iv.end - iv.begin);
        if (final_pass) {
          offsets(i) = update;
          if (i + 1 == dev.num_intervals) {
            total() = update + len;
          }
        }
        update += len;
      });

  ExecSpace().fence();

  std::size_t total_cells = 0;
  Kokkos::deep_copy(total_cells, total);
  dev.total_cells = total_cells;
}

/**
 * @brief Build a device CSR interval set from a host CSR representation.
 */
inline IntervalSet2DDevice
build_device_from_host(const IntervalSet2DHost& host_in) {
  IntervalSet2DHost host = host_in;
  IntervalSet2DDevice dev;

  const std::size_t num_rows = host.row_keys.size();
  if (num_rows == 0) {
    return dev;
  }

  const std::size_t row_ptr_size = host.row_ptr.size();
  if (row_ptr_size != num_rows + 1) {
    return dev;
  }

  if (host.cell_offsets.size() != host.intervals.size()) {
    host.rebuild_mapping();
  }

  const std::size_t num_intervals = host.intervals.size();

  dev.num_rows = num_rows;
  dev.num_intervals = num_intervals;
  dev.total_cells = host.total_cells;

  typename IntervalSet2DDevice::RowKeyView row_keys(
      "subsetix_csr_row_keys", num_rows);
  typename IntervalSet2DDevice::IndexView row_ptr(
      "subsetix_csr_row_ptr", row_ptr_size);
  typename IntervalSet2DDevice::IntervalView intervals(
      "subsetix_csr_intervals", num_intervals);
  typename IntervalSet2DDevice::OffsetView cell_offsets(
      "subsetix_csr_offsets", num_intervals);

  Kokkos::View<RowKey2D*, HostMemorySpace> h_row_keys(
      "subsetix_csr_row_keys_host", num_rows);
  Kokkos::View<std::size_t*, HostMemorySpace> h_row_ptr(
      "subsetix_csr_row_ptr_host", row_ptr_size);
  Kokkos::View<Interval*, HostMemorySpace> h_intervals(
      "subsetix_csr_intervals_host", num_intervals);
  Kokkos::View<std::size_t*, HostMemorySpace> h_offsets(
      "subsetix_csr_offsets_host", num_intervals);

  for (std::size_t i = 0; i < num_rows; ++i) {
    h_row_keys(i) = host.row_keys[i];
  }
  for (std::size_t i = 0; i < row_ptr_size; ++i) {
    h_row_ptr(i) = host.row_ptr[i];
  }
  for (std::size_t i = 0; i < num_intervals; ++i) {
    h_intervals(i) = host.intervals[i];
    h_offsets(i) = host.cell_offsets[i];
  }

  Kokkos::deep_copy(row_keys, h_row_keys);
  Kokkos::deep_copy(row_ptr, h_row_ptr);
  Kokkos::deep_copy(intervals, h_intervals);
  Kokkos::deep_copy(cell_offsets, h_offsets);

  dev.row_keys = row_keys;
  dev.row_ptr = row_ptr;
  dev.intervals = intervals;
  dev.cell_offsets = cell_offsets;

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

  host.rebuild_mapping();

  return host;
}

// ---------------------------------------------------------------------------
// Geometry builders (CSR) - rectangles, disks, random on domain
// ---------------------------------------------------------------------------

struct Box2D {
  Coord x_min = 0;
  Coord x_max = 0; // half-open: [x_min, x_max)
  Coord y_min = 0;
  Coord y_max = 0; // half-open: [y_min, y_max)
};

struct HollowBox2D {
  Box2D outer;
  Box2D inner; // must be inside outer; empty frame if not.
};

struct Disk2D {
  Coord cx = 0;
  Coord cy = 0;
  Coord radius = 0; // radius in cell units (integer)
};

struct HollowDisk2D {
  Coord cx = 0;
  Coord cy = 0;
  Coord outer_radius = 0;
  Coord inner_radius = 0; // hole radius; if <= 0 behaves like a solid disk.
};

struct Domain2D {
  Coord x_min = 0;
  Coord x_max = 0; // half-open
  Coord y_min = 0;
  Coord y_max = 0; // half-open
};

using ExecSpace = Kokkos::DefaultExecutionSpace;

template <class ComputeRowFunctor, class FillRowFunctor>
inline IntervalSet2DDevice
build_interval_set_from_rows(std::size_t num_rows,
                             const std::string& label_prefix,
                             ComputeRowFunctor compute_row,
                             FillRowFunctor fill_row) {
  IntervalSet2DDevice dev;

  if (num_rows == 0) {
    return dev;
  }

  dev.num_rows = num_rows;

  const auto make_label = [&](const char* suffix) {
    return label_prefix + suffix;
  };

  typename IntervalSet2DDevice::RowKeyView row_keys(
      make_label("_row_keys"), num_rows);
  typename IntervalSet2DDevice::IndexView row_ptr(
      make_label("_row_ptr"), num_rows + 1);
  Kokkos::View<std::size_t*, DeviceMemorySpace> row_counts(
      make_label("_row_counts"), num_rows);

  Kokkos::parallel_for(
      make_label("_rows"),
      Kokkos::RangePolicy<ExecSpace>(0, num_rows),
      KOKKOS_LAMBDA(const std::size_t i) {
        RowKey2D key{};
        std::size_t count = 0;
        compute_row(i, key, count);
        row_keys(i) = key;
        row_counts(i) = count;
      });

  Kokkos::View<std::size_t, DeviceMemorySpace> total_intervals(
      make_label("_total_intervals"));

  Kokkos::parallel_scan(
      make_label("_scan"),
      Kokkos::RangePolicy<ExecSpace>(0, num_rows),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& update,
                    const bool final_pass) {
        const std::size_t c = row_counts(i);
        if (final_pass) {
          row_ptr(i) = update;
          if (i + 1 == num_rows) {
            const std::size_t end = update + c;
            row_ptr(num_rows) = end;
            total_intervals() = end;
          }
        }
        update += c;
      });

  ExecSpace().fence();

  std::size_t num_intervals_host = 0;
  Kokkos::deep_copy(num_intervals_host, total_intervals);

  dev.num_intervals = num_intervals_host;

  if (num_intervals_host == 0) {
    dev.row_keys = row_keys;
    dev.row_ptr = row_ptr;
    dev.intervals = typename IntervalSet2DDevice::IntervalView();
    return dev;
  }

  typename IntervalSet2DDevice::IntervalView intervals(
      make_label("_intervals"), num_intervals_host);

  Kokkos::parallel_for(
      make_label("_fill"),
      Kokkos::RangePolicy<ExecSpace>(0, num_rows),
      KOKKOS_LAMBDA(const std::size_t i) {
        const std::size_t count = row_counts(i);
        if (count == 0) {
          return;
        }

        const std::size_t offset = row_ptr(i);
        fill_row(i, count, offset, intervals);
      });

  ExecSpace().fence();

  dev.row_keys = row_keys;
  dev.row_ptr = row_ptr;
  dev.intervals = intervals;
  compute_cell_offsets_device(dev);

  return dev;
}

/**
 * @brief Build a filled axis-aligned rectangle on device.
 *
 * All rows y in [box.y_min, box.y_max) contain a single interval [x_min, x_max).
 */
inline IntervalSet2DDevice
make_box_device(const Box2D& box) {
  IntervalSet2DDevice dev;

  if (box.x_min >= box.x_max || box.y_min >= box.y_max) {
    return dev;
  }

  const std::size_t num_rows =
      static_cast<std::size_t>(box.y_max - box.y_min);

  dev.num_rows = num_rows;
  dev.num_intervals = num_rows;

  typename IntervalSet2DDevice::RowKeyView row_keys(
      "subsetix_csr_box_row_keys", num_rows);
  typename IntervalSet2DDevice::IndexView row_ptr(
      "subsetix_csr_box_row_ptr", num_rows + 1);
  typename IntervalSet2DDevice::IntervalView intervals(
      "subsetix_csr_box_intervals", num_rows);

  // Initialize row_ptr to 0, then fill row_ptr(i+1) = i+1 in parallel.
  Kokkos::deep_copy(row_ptr, std::size_t(0));

  Kokkos::parallel_for(
      "subsetix_csr_box_fill",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows),
      KOKKOS_LAMBDA(const std::size_t i) {
        const Coord y = static_cast<Coord>(box.y_min + static_cast<Coord>(i));
        row_keys(i) = RowKey2D{y};
        row_ptr(i + 1) = i + 1;
        intervals(i) = Interval{box.x_min, box.x_max};
      });

  ExecSpace().fence();

  dev.row_keys = row_keys;
  dev.row_ptr = row_ptr;
  dev.intervals = intervals;
  compute_cell_offsets_device(dev);
  return dev;
}

inline IntervalSet2DDevice
make_hollow_box_device(const HollowBox2D& frame) {
  IntervalSet2DDevice dev;

  const Box2D& outer = frame.outer;
  const Box2D& inner = frame.inner;

  if (outer.x_min >= outer.x_max || outer.y_min >= outer.y_max) {
    return dev;
  }

  // Inner box must be strictly inside to produce a frame; otherwise fall back to solid box.
  const bool inner_valid =
      (inner.x_min > outer.x_min && inner.x_max < outer.x_max &&
       inner.y_min > outer.y_min && inner.y_max < outer.y_max &&
       inner.x_min < inner.x_max && inner.y_min < inner.y_max);

  if (!inner_valid) {
    return make_box_device(outer);
  }

  const std::size_t num_rows =
      static_cast<std::size_t>(outer.y_max - outer.y_min);

  const Coord x_min = outer.x_min;
  const Coord x_max = outer.x_max;
  const Coord y_min = outer.y_min;
  const Coord inner_x_min = inner.x_min;
  const Coord inner_x_max = inner.x_max;
  const Coord inner_y_min = inner.y_min;
  const Coord inner_y_max = inner.y_max;
  const std::string prefix = "subsetix_csr_hollow_box";

  Kokkos::View<Coord*, DeviceMemorySpace> left_begin(
      (prefix + "_left_begin"), num_rows);
  Kokkos::View<Coord*, DeviceMemorySpace> left_end(
      (prefix + "_left_end"), num_rows);
  Kokkos::View<Coord*, DeviceMemorySpace> right_begin(
      (prefix + "_right_begin"), num_rows);
  Kokkos::View<Coord*, DeviceMemorySpace> right_end(
      (prefix + "_right_end"), num_rows);

  auto compute_row = [=] KOKKOS_FUNCTION(const std::size_t i,
                                         RowKey2D& key,
                                         std::size_t& count) {
    const Coord y = static_cast<Coord>(y_min + static_cast<Coord>(i));
    key = RowKey2D{y};

    const bool in_vertical_hole = (y >= inner_y_min && y < inner_y_max);

    if (!in_vertical_hole) {
      left_begin(i) = x_min;
      left_end(i) = x_max;
      count = 1;
      return;
    }

    std::size_t local_count = 0;
    if (x_min < inner_x_min) {
      left_begin(i) = x_min;
      left_end(i) = inner_x_min;
      ++local_count;
    }
    if (inner_x_max < x_max) {
      right_begin(i) = inner_x_max;
      right_end(i) = x_max;
      ++local_count;
    }

    count = local_count;
  };

  auto fill_row = [=] KOKKOS_FUNCTION(
                       const std::size_t i,
                       const std::size_t count,
                       const std::size_t offset,
                       IntervalSet2DDevice::IntervalView intervals) {
    if (count == 0) {
      return;
    }

    std::size_t idx = 0;
    const bool has_left = (left_end(i) > left_begin(i));
    if (has_left) {
      intervals(offset + idx) = Interval{left_begin(i), left_end(i)};
      ++idx;
    }

    const bool has_right = (count > idx);
    if (has_right) {
      intervals(offset + idx) = Interval{right_begin(i), right_end(i)};
    }
  };

  return build_interval_set_from_rows(
      num_rows, prefix, compute_row, fill_row);
}

/**
 * @brief Build a discrete disk (filled circle) on device.
 *
 * For each integer row y in [cy-radius, cy+radius], we add at most one
 * interval [x_begin, x_end) where (x,y) lies inside the disk.
 */
inline IntervalSet2DDevice
make_disk_device(const Disk2D& disk) {
  IntervalSet2DDevice dev;

  if (disk.radius <= 0) {
    return dev;
  }

  const Coord y_min = disk.cy - disk.radius;
  const Coord y_max = disk.cy + disk.radius + 1; // half-open
  const std::size_t num_rows =
      static_cast<std::size_t>(y_max - y_min);

  Kokkos::View<Coord*, DeviceMemorySpace> x_begin(
      "subsetix_csr_disk_x_begin", num_rows);
  Kokkos::View<Coord*, DeviceMemorySpace> x_end(
      "subsetix_csr_disk_x_end", num_rows);

  const Coord cx = disk.cx;
  const Coord cy = disk.cy;
  const Coord radius = disk.radius;
  const std::string prefix = "subsetix_csr_disk";

  auto compute_row = [=] KOKKOS_FUNCTION(const std::size_t i,
                                         RowKey2D& key,
                                         std::size_t& count) {
    const Coord y = static_cast<Coord>(y_min + static_cast<Coord>(i));
    key = RowKey2D{y};

    const Coord dy = y - cy;
    const long long d2 =
        static_cast<long long>(dy) * static_cast<long long>(dy);
    const long long r2 =
        static_cast<long long>(radius) * static_cast<long long>(radius);

    if (d2 > r2) {
      count = 0;
      return;
    }

    const double dx = std::sqrt(static_cast<double>(r2 - d2));
    const Coord half_width = static_cast<Coord>(dx);
    x_begin(i) = cx - half_width;
    x_end(i) = cx + half_width + 1;
    count = 1;
  };

  auto fill_row = [=] KOKKOS_FUNCTION(
                       const std::size_t i,
                       const std::size_t count,
                       const std::size_t offset,
                       IntervalSet2DDevice::IntervalView intervals) {
    (void)count;
    intervals(offset) = Interval{x_begin(i), x_end(i)};
  };

  return build_interval_set_from_rows(
      num_rows, prefix, compute_row, fill_row);
}

inline IntervalSet2DDevice
make_hollow_disk_device(const HollowDisk2D& disk) {
  IntervalSet2DDevice dev;

  if (disk.outer_radius <= 0) {
    return dev;
  }

  const Coord inner_radius = (disk.inner_radius < 0) ? 0 : disk.inner_radius;
  if (inner_radius >= disk.outer_radius) {
    return dev;
  }

  const Coord y_min = disk.cy - disk.outer_radius;
  const Coord y_max = disk.cy + disk.outer_radius + 1;
  const std::size_t num_rows =
      static_cast<std::size_t>(y_max - y_min);

  Kokkos::View<Coord*, DeviceMemorySpace> left_begin(
      "subsetix_csr_hollow_disk_left_begin", num_rows);
  Kokkos::View<Coord*, DeviceMemorySpace> left_end(
      "subsetix_csr_hollow_disk_left_end", num_rows);
  Kokkos::View<Coord*, DeviceMemorySpace> right_begin(
      "subsetix_csr_hollow_disk_right_begin", num_rows);
  Kokkos::View<Coord*, DeviceMemorySpace> right_end(
      "subsetix_csr_hollow_disk_right_end", num_rows);

  const Coord cx = disk.cx;
  const Coord cy = disk.cy;
  const Coord outer_radius = disk.outer_radius;
  const std::string prefix = "subsetix_csr_hollow_disk";

  auto compute_row = [=] KOKKOS_FUNCTION(const std::size_t i,
                                         RowKey2D& key,
                                         std::size_t& count) {
    const Coord y = static_cast<Coord>(y_min + static_cast<Coord>(i));
    key = RowKey2D{y};

    const Coord dy = y - cy;
    const long long dy2 =
        static_cast<long long>(dy) * static_cast<long long>(dy);
    const long long outer2 =
        static_cast<long long>(outer_radius) * static_cast<long long>(outer_radius);

    if (dy2 > outer2) {
      count = 0;
      return;
    }

    const long long inner2 =
        static_cast<long long>(inner_radius) * static_cast<long long>(inner_radius);

    const Coord outer_half =
        static_cast<Coord>(std::sqrt(static_cast<double>(outer2 - dy2)));
    const Coord outer_x0 = cx - outer_half;
    const Coord outer_x1 = cx + outer_half + 1;

    if (inner_radius == 0 || dy2 >= inner2) {
      left_begin(i) = outer_x0;
      left_end(i) = outer_x1;
      count = 1;
      return;
    }

    const Coord inner_half =
        static_cast<Coord>(std::sqrt(static_cast<double>(inner2 - dy2)));
    const Coord inner_x0 = cx - inner_half;
    const Coord inner_x1 = cx + inner_half + 1;

    std::size_t local_count = 0;
    if (outer_x0 < inner_x0) {
      left_begin(i) = outer_x0;
      left_end(i) = inner_x0;
      ++local_count;
    }
    if (inner_x1 < outer_x1) {
      right_begin(i) = inner_x1;
      right_end(i) = outer_x1;
      ++local_count;
    }

    count = local_count;
  };

  auto fill_row = [=] KOKKOS_FUNCTION(
                       const std::size_t i,
                       const std::size_t count,
                       const std::size_t offset,
                       IntervalSet2DDevice::IntervalView intervals) {
    if (count == 0) {
      return;
    }

    std::size_t idx = 0;
    const bool has_left = (left_end(i) > left_begin(i));
    if (has_left) {
      intervals(offset + idx) = Interval{left_begin(i), left_end(i)};
      ++idx;
    }

    const bool has_right = (count > idx);
    if (has_right) {
      intervals(offset + idx) = Interval{right_begin(i), right_end(i)};
    }
  };

  return build_interval_set_from_rows(
      num_rows, prefix, compute_row, fill_row);
}

/**
 * @brief Build a random CSR geometry on a rectangular domain.
 *
 * Pour chaque ligne y dans [domain.y_min, domain.y_max), on crée une
 * intervalle aléatoire avec probabilité fill_probability, en utilisant
 * un RNG Kokkos sur le device. Les bornes de l'intervalle sont tirées
 * uniformément dans [x_min, x_max).
 */
inline IntervalSet2DDevice
make_random_device(const Domain2D& domain,
                   double fill_probability,
                   std::uint64_t seed) {
  IntervalSet2DDevice dev;

  if (domain.x_min >= domain.x_max || domain.y_min >= domain.y_max) {
    return dev;
  }

  if (fill_probability <= 0.0) {
    return dev;
  }
  if (fill_probability > 1.0) {
    fill_probability = 1.0;
  }

  const std::size_t num_rows =
      static_cast<std::size_t>(domain.y_max - domain.y_min);

  Kokkos::View<Coord*, DeviceMemorySpace> x_begin(
      "subsetix_csr_rand_x_begin", num_rows);
  Kokkos::View<Coord*, DeviceMemorySpace> x_end(
      "subsetix_csr_rand_x_end", num_rows);

  Kokkos::Random_XorShift64_Pool<ExecSpace> pool(seed);

  const Coord x_min = domain.x_min;
  const Coord x_max = domain.x_max;
  const Coord y_min = domain.y_min;
  const Coord width = x_max - x_min;
  const std::string prefix = "subsetix_csr_rand";

  auto compute_row = [=] KOKKOS_FUNCTION(const std::size_t i,
                                         RowKey2D& key,
                                         std::size_t& count) {
    const Coord y = static_cast<Coord>(y_min + static_cast<Coord>(i));
    key = RowKey2D{y};

    auto state = pool.get_state();
    const double p = state.drand();

    if (p < fill_probability) {
      count = 1;
      const Coord w = static_cast<Coord>(
          1 + (state.urand() % static_cast<unsigned int>(width)));
      const Coord max_start = width - w;
      const Coord offset =
          (max_start > 0)
              ? static_cast<Coord>(state.urand() %
                                   (static_cast<unsigned int>(max_start) +
                                    1U))
              : 0;
      const Coord xb = x_min + offset;
      x_begin(i) = xb;
      x_end(i) = xb + w;
    } else {
      count = 0;
    }

    pool.free_state(state);
  };

  auto fill_row = [=] KOKKOS_FUNCTION(
                       const std::size_t i,
                       const std::size_t count,
                       const std::size_t offset,
                       IntervalSet2DDevice::IntervalView intervals) {
    (void)count;
    intervals(offset) = Interval{x_begin(i), x_end(i)};
  };

  return build_interval_set_from_rows(
      num_rows, prefix, compute_row, fill_row);
}

/**
 * @brief Build a checkerboard pattern on a rectangular domain.
 *
 * Cells are grouped into squares of size cell_size x cell_size. A square at
 * block coordinates (bx, by) (with bx, by >= 0) is filled iff (bx + by) is even.
 * Each filled square contributes contiguous intervals of length cell_size along X
 * for each of its cell_size rows.
 */
inline IntervalSet2DDevice
make_checkerboard_device(const Domain2D& domain, Coord cell_size) {
  IntervalSet2DDevice dev;

  if (domain.x_min >= domain.x_max || domain.y_min >= domain.y_max) {
    return dev;
  }

  if (cell_size <= 0) {
    return dev;
  }

  const std::size_t num_rows =
      static_cast<std::size_t>(domain.y_max - domain.y_min);

  const Coord x_min = domain.x_min;
  const Coord x_max = domain.x_max;
  const Coord y_min = domain.y_min;
  const Coord width = x_max - x_min;
  const std::string prefix = "subsetix_csr_chk";

  auto compute_row = [=] KOKKOS_FUNCTION(const std::size_t i,
                                         RowKey2D& key,
                                         std::size_t& count) {
    const Coord y = static_cast<Coord>(y_min + static_cast<Coord>(i));
    key = RowKey2D{y};

    const Coord local_y = static_cast<Coord>(y - y_min);
    const std::size_t block_y =
        static_cast<std::size_t>(local_y / cell_size);

    count = 0;
    if (width > 0) {
      const std::size_t num_blocks_x =
          static_cast<std::size_t>(
              (static_cast<long long>(width) + cell_size - 1) /
              cell_size);
      for (std::size_t bx = 0; bx < num_blocks_x; ++bx) {
        const bool filled = (((bx + block_y) & 1U) == 0U);
        if (filled) {
          ++count;
        }
      }
    }
  };

  auto fill_row = [=] KOKKOS_FUNCTION(
                       const std::size_t i,
                       const std::size_t count,
                       const std::size_t offset,
                       IntervalSet2DDevice::IntervalView intervals) {
    if (count == 0) {
      return;
    }

    const Coord y = static_cast<Coord>(y_min + static_cast<Coord>(i));
    const Coord local_y = static_cast<Coord>(y - y_min);
    const std::size_t block_y =
        static_cast<std::size_t>(local_y / cell_size);

    std::size_t write_idx = 0;
    if (width > 0) {
      const std::size_t num_blocks_x =
          static_cast<std::size_t>(
              (static_cast<long long>(width) + cell_size - 1) /
              cell_size);
      for (std::size_t bx = 0; bx < num_blocks_x; ++bx) {
        const bool filled = (((bx + block_y) & 1U) == 0U);
        if (!filled) {
          continue;
        }
        const Coord x0 = static_cast<Coord>(x_min + bx * cell_size);
        if (x0 >= x_max) {
          continue;
        }
        const Coord x1_candidate =
            static_cast<Coord>(x0 + cell_size);
        const Coord x1 = (x1_candidate > x_max) ? x_max : x1_candidate;
        intervals(offset + write_idx) = Interval{x0, x1};
        ++write_idx;
      }
    }
  };

  return build_interval_set_from_rows(
      num_rows, prefix, compute_row, fill_row);
}

} // namespace csr
} // namespace subsetix
