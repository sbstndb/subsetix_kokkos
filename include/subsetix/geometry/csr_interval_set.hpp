#pragma once

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <subsetix/geometry/csr_backend.hpp>
#include <subsetix/detail/scan_utils.hpp>

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
 * @brief Unified CSR representation of a 2D interval set.
 *
 * Templated on MemorySpace to support both Host and Device.
 * Uses Kokkos::View for all storage.
 *
 * Invariants (par convention) :
 *  - row_keys.extent(0) == num_rows,
 *  - row_ptr.extent(0) == num_rows + 1,
 *  - intervals.extent(0) >= num_intervals,
 *  - pour chaque ligne, les intervalles sont triés et non chevauchants.
 */
template <class MemorySpace>
struct IntervalSet2D {
  using RowKeyView = Kokkos::View<RowKey2D*, MemorySpace>;
  using IndexView = Kokkos::View<std::size_t*, MemorySpace>;
  using IntervalView = Kokkos::View<Interval*, MemorySpace>;
  using OffsetView = Kokkos::View<std::size_t*, MemorySpace>;

  RowKeyView row_keys;     ///< [num_rows]
  IndexView row_ptr;       ///< [num_rows + 1]
  IntervalView intervals;  ///< [num_intervals]
  OffsetView cell_offsets; ///< [num_intervals]
  std::size_t total_cells = 0;
  std::size_t num_rows = 0;
  std::size_t num_intervals = 0;
};

// Primary type aliases
using IntervalSet2DDevice = IntervalSet2D<DeviceMemorySpace>;
using IntervalSet2DHost = IntervalSet2D<HostMemorySpace>;

// Backward compatibility aliases
template <class MemorySpace>
using IntervalSet2DView = IntervalSet2D<MemorySpace>;
using IntervalSet2DHostView = IntervalSet2D<HostMemorySpace>;

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
 * @brief Compute cell offsets for a host IntervalSet2D.
 */
inline void compute_cell_offsets_host(IntervalSet2DHost& h) {
  if (h.num_intervals == 0) {
    h.total_cells = 0;
    return;
  }

  if (h.cell_offsets.extent(0) < h.num_intervals) {
    h.cell_offsets = IntervalSet2DHost::OffsetView(
        "subsetix_csr_offsets_host", h.num_intervals);
  }

  std::size_t accum = 0;
  for (std::size_t i = 0; i < h.num_intervals; ++i) {
    h.cell_offsets(i) = accum;
    accum += static_cast<std::size_t>(h.intervals(i).end - h.intervals(i).begin);
  }
  h.total_cells = accum;
}

/**
 * @brief Convert an IntervalSet2D between memory spaces.
 *
 * Usage:
 *   auto device_set = to<DeviceMemorySpace>(host_set);
 *   auto host_set = to<HostMemorySpace>(device_set);
 */
template <class ToSpace, class FromSpace>
inline IntervalSet2D<ToSpace> to(const IntervalSet2D<FromSpace>& src) {
  IntervalSet2D<ToSpace> dst;

  if (src.num_rows == 0) {
    return dst;
  }

  dst.num_rows = src.num_rows;
  dst.num_intervals = src.num_intervals;
  dst.total_cells = src.total_cells;

  // Use subviews to handle cases where view extent > num_*
  auto src_row_keys = Kokkos::subview(src.row_keys, std::make_pair(std::size_t(0), src.num_rows));
  auto src_row_ptr = Kokkos::subview(src.row_ptr, std::make_pair(std::size_t(0), src.num_rows + 1));
  auto src_intervals = Kokkos::subview(src.intervals, std::make_pair(std::size_t(0), src.num_intervals));
  auto src_offsets = Kokkos::subview(src.cell_offsets, std::make_pair(std::size_t(0), src.num_intervals));

  dst.row_keys = Kokkos::create_mirror_view_and_copy(ToSpace{}, src_row_keys);
  dst.row_ptr = Kokkos::create_mirror_view_and_copy(ToSpace{}, src_row_ptr);
  dst.intervals = Kokkos::create_mirror_view_and_copy(ToSpace{}, src_intervals);
  dst.cell_offsets = Kokkos::create_mirror_view_and_copy(ToSpace{}, src_offsets);

  return dst;
}

/**
 * @brief Create an IntervalSet2D on host from initializer lists.
 *
 * Useful for tests and simple constructions.
 *
 * Usage:
 *   auto h = make_interval_set_host(
 *       {{0}, {5}},           // row_keys (y values)
 *       {0, 1, 2},            // row_ptr
 *       {{0, 10}, {5, 8}}     // intervals
 *   );
 */
inline IntervalSet2DHost make_interval_set_host(
    std::initializer_list<RowKey2D> row_keys_init,
    std::initializer_list<std::size_t> row_ptr_init,
    std::initializer_list<Interval> intervals_init) {
  IntervalSet2DHost h;

  const auto nrows = row_keys_init.size();
  const auto nints = intervals_init.size();

  if (nrows == 0) {
    return h;
  }

  h.row_keys = IntervalSet2DHost::RowKeyView("row_keys", nrows);
  h.row_ptr = IntervalSet2DHost::IndexView("row_ptr", row_ptr_init.size());
  h.intervals = IntervalSet2DHost::IntervalView("intervals", nints);
  h.cell_offsets = IntervalSet2DHost::OffsetView("cell_offsets", nints);

  std::copy(row_keys_init.begin(), row_keys_init.end(), h.row_keys.data());
  std::copy(row_ptr_init.begin(), row_ptr_init.end(), h.row_ptr.data());
  std::copy(intervals_init.begin(), intervals_init.end(), h.intervals.data());

  h.num_rows = nrows;
  h.num_intervals = nints;

  compute_cell_offsets_host(h);

  return h;
}

// ---------------------------------------------------------------------------
// Backward compatibility wrappers (deprecated)
// ---------------------------------------------------------------------------

/**
 * @brief Build a device CSR interval set from a host CSR representation.
 * @deprecated Use to<DeviceMemorySpace>(host) instead.
 */
[[deprecated("Use to<DeviceMemorySpace>(host) instead")]]
inline IntervalSet2DDevice
build_device_from_host(const IntervalSet2DHost& host) {
  return to<DeviceMemorySpace>(host);
}

/**
 * @brief Rebuild a host CSR representation from a device CSR interval set.
 * @deprecated Use to<HostMemorySpace>(dev) instead.
 */
[[deprecated("Use to<HostMemorySpace>(dev) instead")]]
inline IntervalSet2DHost
build_host_from_device(const IntervalSet2DDevice& dev) {
  return to<HostMemorySpace>(dev);
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

struct Disk2D {
  Coord cx = 0;
  Coord cy = 0;
  Coord radius = 0; // radius in cell units (integer)
};

struct Domain2D {
  Coord x_min = 0;
  Coord x_max = 0; // half-open
  Coord y_min = 0;
  Coord y_max = 0; // half-open
};

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

  std::size_t num_intervals_host = detail::exclusive_scan_csr_row_ptr<std::size_t>(
      make_label("_scan"),
      num_rows,
      row_counts,
      row_ptr);

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
  Kokkos::Experimental::fill(ExecSpace(), row_ptr, std::size_t(0));

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

template <class MaskView>
inline IntervalSet2DDevice
make_bitmap_device(const MaskView& mask,
                   Coord x_min,
                   Coord y_min,
                   std::uint8_t on_value = 1) {
  using MaskSpace = typename MaskView::memory_space;
  static_assert(Kokkos::SpaceAccessibility<ExecSpace, MaskSpace>::accessible,
                "Mask view must be accessible from the default execution space");

  IntervalSet2DDevice dev;

  const std::size_t height = mask.extent(0);
  const std::size_t width = mask.extent(1);
  if (height == 0 || width == 0) {
    return dev;
  }

  const std::size_t run_capacity = height * width;
  const std::string prefix = "subsetix_csr_bitmap";

  Kokkos::View<Coord*, DeviceMemorySpace> run_begin(
      (prefix + "_begin"), run_capacity);
  Kokkos::View<Coord*, DeviceMemorySpace> run_end(
      (prefix + "_end"), run_capacity);

  auto compute_row = [=] KOKKOS_FUNCTION(const std::size_t i,
                                         RowKey2D& key,
                                         std::size_t& count) {
    key = RowKey2D{static_cast<Coord>(y_min + static_cast<Coord>(i))};
    std::size_t base = i * width;
    std::size_t runs = 0;
    bool in_run = false;
    Coord start = 0;

    for (std::size_t j = 0; j < width; ++j) {
      const bool filled = (mask(i, j) == on_value);
      if (filled && !in_run) {
        in_run = true;
        start = static_cast<Coord>(x_min + static_cast<Coord>(j));
      } else if (!filled && in_run) {
        run_begin(base + runs) = start;
        run_end(base + runs) =
            static_cast<Coord>(x_min + static_cast<Coord>(j));
        ++runs;
        in_run = false;
      }
    }
    if (in_run) {
      run_begin(base + runs) = start;
      run_end(base + runs) =
          static_cast<Coord>(x_min + static_cast<Coord>(width));
      ++runs;
    }
    count = runs;
  };

  auto fill_row = [=] KOKKOS_FUNCTION(
                       const std::size_t i,
                       const std::size_t count,
                       const std::size_t offset,
                       IntervalSet2DDevice::IntervalView intervals) {
    if (count == 0) {
      return;
    }
    const std::size_t base = i * width;
    for (std::size_t k = 0; k < count; ++k) {
      intervals(offset + k) =
          Interval{run_begin(base + k), run_end(base + k)};
    }
  };

  return build_interval_set_from_rows(
      height, prefix, compute_row, fill_row);
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
