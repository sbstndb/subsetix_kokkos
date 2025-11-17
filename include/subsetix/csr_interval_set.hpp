#pragma once

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <cmath>

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

  dev.num_rows = 0;
  dev.num_intervals = 0;
  return dev;
}

inline IntervalSet2DDevice
allocate_union_output_buffer(const IntervalSet2DDevice& A,
                             const IntervalSet2DDevice& B) {
  return allocate_interval_set_device(A.num_rows + B.num_rows,
                                      A.num_intervals + B.num_intervals);
}

inline IntervalSet2DDevice
allocate_intersection_output_buffer(const IntervalSet2DDevice& A,
                                    const IntervalSet2DDevice& B) {
  const std::size_t row_capacity =
      std::min(A.num_rows, B.num_rows);
  const std::size_t interval_capacity =
      A.num_intervals + B.num_intervals;
  return allocate_interval_set_device(row_capacity, interval_capacity);
}

inline IntervalSet2DDevice
allocate_difference_output_buffer(const IntervalSet2DDevice& lhs,
                                  const IntervalSet2DDevice& rhs) {
  return allocate_interval_set_device(lhs.num_rows,
                                      lhs.num_intervals + rhs.num_intervals);
}

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

using ExecSpace = Kokkos::DefaultExecutionSpace;

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

  dev.num_rows = num_rows;

  typename IntervalSet2DDevice::RowKeyView row_keys(
      "subsetix_csr_disk_row_keys", num_rows);
  typename IntervalSet2DDevice::IndexView row_ptr(
      "subsetix_csr_disk_row_ptr", num_rows + 1);

  // Per-row markers and temporary per-row interval bounds.
  Kokkos::View<int*, DeviceMemorySpace> has_interval(
      "subsetix_csr_disk_has_interval", num_rows);
  Kokkos::View<Coord*, DeviceMemorySpace> x_begin(
      "subsetix_csr_disk_x_begin", num_rows);
  Kokkos::View<Coord*, DeviceMemorySpace> x_end(
      "subsetix_csr_disk_x_end", num_rows);

  const Coord cx = disk.cx;
  const Coord cy = disk.cy;
  const Coord radius = disk.radius;

  // 1) Decide for each row whether it intersects the disk.
  Kokkos::parallel_for(
      "subsetix_csr_disk_rows",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows),
      KOKKOS_LAMBDA(const std::size_t i) {
        const Coord y = static_cast<Coord>(y_min + static_cast<Coord>(i));
        row_keys(i) = RowKey2D{y};

        const Coord dy = y - cy;
        const long long d2 =
            static_cast<long long>(dy) * static_cast<long long>(dy);
        const long long r2 =
            static_cast<long long>(radius) * static_cast<long long>(radius);

        if (d2 > r2) {
          has_interval(i) = 0;
          return;
        }

        has_interval(i) = 1;
        const double dx = std::sqrt(static_cast<double>(r2 - d2));
        const Coord half_width = static_cast<Coord>(dx);
        const Coord xb = cx - half_width;
        const Coord xe = cx + half_width + 1;
        x_begin(i) = xb;
        x_end(i) = xe;
      });

  // 2) Exclusive scan over rows to build row_ptr and get total intervals.
  Kokkos::View<std::size_t, DeviceMemorySpace> total_intervals(
      "subsetix_csr_disk_total_intervals");

  Kokkos::parallel_scan(
      "subsetix_csr_disk_scan",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& update,
                    const bool final_pass) {
        const std::size_t c =
            static_cast<std::size_t>(has_interval(i) ? 1 : 0);
        if (final_pass) {
          row_ptr(i) = update;
          if (i + 1 == num_rows) {
            row_ptr(num_rows) = update + c;
            total_intervals() = update + c;
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
      "subsetix_csr_disk_intervals", num_intervals_host);

  // 3) Fill intervals at the offsets determined by row_ptr.
  Kokkos::parallel_for(
      "subsetix_csr_disk_fill",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows),
      KOKKOS_LAMBDA(const std::size_t i) {
        if (!has_interval(i)) {
          return;
        }
        const std::size_t offset = row_ptr(i);
        intervals(offset) = Interval{x_begin(i), x_end(i)};
      });

  ExecSpace().fence();

  dev.row_keys = row_keys;
  dev.row_ptr = row_ptr;
  dev.intervals = intervals;

  return dev;
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

  dev.num_rows = num_rows;

  typename IntervalSet2DDevice::RowKeyView row_keys(
      "subsetix_csr_rand_row_keys", num_rows);
  typename IntervalSet2DDevice::IndexView row_ptr(
      "subsetix_csr_rand_row_ptr", num_rows + 1);

  Kokkos::View<int*, DeviceMemorySpace> has_interval(
      "subsetix_csr_rand_has_interval", num_rows);
  Kokkos::View<Coord*, DeviceMemorySpace> x_begin(
      "subsetix_csr_rand_x_begin", num_rows);
  Kokkos::View<Coord*, DeviceMemorySpace> x_end(
      "subsetix_csr_rand_x_end", num_rows);

  Kokkos::Random_XorShift64_Pool<ExecSpace> pool(seed);

  const Coord x_min = domain.x_min;
  const Coord x_max = domain.x_max;
  const Coord y_min = domain.y_min;
  const Coord width = x_max - x_min;

  // 1) Generate per-row random intervals (or emptiness) on device.
  Kokkos::parallel_for(
      "subsetix_csr_rand_rows",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows),
      KOKKOS_LAMBDA(const std::size_t i) {
        const Coord y = static_cast<Coord>(y_min + static_cast<Coord>(i));
        row_keys(i) = RowKey2D{y};

        auto state = pool.get_state();
        const double p = state.drand();

        if (p < fill_probability) {
          has_interval(i) = 1;
          // Choose a random width in [1, width].
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
          const Coord xe = xb + w;
          x_begin(i) = xb;
          x_end(i) = xe;
        } else {
          has_interval(i) = 0;
        }

        pool.free_state(state);
      });

  // 2) Exclusive scan to build row_ptr and total number of intervals.
  Kokkos::View<std::size_t, DeviceMemorySpace> total_intervals(
      "subsetix_csr_rand_total_intervals");

  Kokkos::parallel_scan(
      "subsetix_csr_rand_scan",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& update,
                    const bool final_pass) {
        const std::size_t c =
            static_cast<std::size_t>(has_interval(i) ? 1 : 0);
        if (final_pass) {
          row_ptr(i) = update;
          if (i + 1 == num_rows) {
            row_ptr(num_rows) = update + c;
            total_intervals() = update + c;
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
      "subsetix_csr_rand_intervals", num_intervals_host);

  Kokkos::parallel_for(
      "subsetix_csr_rand_fill",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows),
      KOKKOS_LAMBDA(const std::size_t i) {
        if (!has_interval(i)) {
          return;
        }
        const std::size_t offset = row_ptr(i);
        intervals(offset) = Interval{x_begin(i), x_end(i)};
      });

  ExecSpace().fence();

  dev.row_keys = row_keys;
  dev.row_ptr = row_ptr;
  dev.intervals = intervals;

  return dev;
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

  dev.num_rows = num_rows;

  typename IntervalSet2DDevice::RowKeyView row_keys(
      "subsetix_csr_chk_row_keys", num_rows);
  typename IntervalSet2DDevice::IndexView row_ptr(
      "subsetix_csr_chk_row_ptr", num_rows + 1);

  Kokkos::View<std::size_t*, DeviceMemorySpace> row_counts(
      "subsetix_csr_chk_row_counts", num_rows);

  const Coord x_min = domain.x_min;
  const Coord x_max = domain.x_max;
  const Coord y_min = domain.y_min;
  const Coord width = x_max - x_min;

  // 1) Compute per-row counts and row keys.
  Kokkos::parallel_for(
      "subsetix_csr_chk_rows",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows),
      KOKKOS_LAMBDA(const std::size_t i) {
        const Coord y = static_cast<Coord>(y_min + static_cast<Coord>(i));
        row_keys(i) = RowKey2D{y};

        const Coord local_y = static_cast<Coord>(y - y_min);
        const std::size_t block_y =
            static_cast<std::size_t>(local_y / cell_size);

        std::size_t count = 0;
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

        row_counts(i) = count;
      });

  // 2) Exclusive scan on row_counts to build row_ptr and total intervals.
  Kokkos::View<std::size_t, DeviceMemorySpace> total_intervals(
      "subsetix_csr_chk_total_intervals");

  Kokkos::parallel_scan(
      "subsetix_csr_chk_scan",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& update,
                    const bool final_pass) {
        const std::size_t c = row_counts(i);
        if (final_pass) {
          row_ptr(i) = update;
          if (i + 1 == num_rows) {
            row_ptr(num_rows) = update + c;
            total_intervals() = update + c;
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
      "subsetix_csr_chk_intervals", num_intervals_host);

  // 3) Fill intervals on each row using the offsets defined by row_ptr.
  Kokkos::parallel_for(
      "subsetix_csr_chk_fill",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows),
      KOKKOS_LAMBDA(const std::size_t i) {
        const std::size_t count = row_counts(i);
        if (count == 0) {
          return;
        }

        const Coord y = static_cast<Coord>(y_min + static_cast<Coord>(i));
        const std::size_t offset = row_ptr(i);

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
      });

  ExecSpace().fence();

  dev.row_keys = row_keys;
  dev.row_ptr = row_ptr;
  dev.intervals = intervals;

  return dev;
}

} // namespace csr
} // namespace subsetix
