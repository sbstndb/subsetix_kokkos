#pragma once

#include <stdexcept>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include <subsetix/csr_interval_set.hpp>
#include <subsetix/csr_ops/workspace.hpp>
#include <subsetix/csr_ops/core.hpp>

namespace subsetix {
namespace csr {

/**
 * @brief Device-friendly view describing a subset of an IntervalSet2D.
 *
 * Each entry in the subset references an interval from the parent geometry
 * along with the restricted [x_begin, x_end) range inside that interval.
 */
template <class MemorySpace>
struct IntervalSubSet2DView {
  using GeometryView = IntervalSet2DView<MemorySpace>;
  using IndexView = Kokkos::View<std::size_t*, MemorySpace>;
  using CoordView = Kokkos::View<Coord*, MemorySpace>;
  using RowIndexView = Kokkos::View<int*, MemorySpace>;

  GeometryView parent;
  IndexView interval_indices;
  CoordView x_begin;
  CoordView x_end;
  RowIndexView row_indices;
  std::size_t num_entries = 0;
  std::size_t total_cells = 0;

  KOKKOS_INLINE_FUNCTION
  bool valid() const {
    return parent.num_rows > 0 && num_entries > 0;
  }
};

using IntervalSubSet2DDevice = IntervalSubSet2DView<DeviceMemorySpace>;
using IntervalSubSet2DHostView = IntervalSubSet2DView<HostMemorySpace>;

inline void reset_interval_subset(IntervalSubSet2DDevice& subset) {
  subset.interval_indices = IntervalSubSet2DDevice::IndexView();
  subset.x_begin = IntervalSubSet2DDevice::CoordView();
  subset.x_end = IntervalSubSet2DDevice::CoordView();
  subset.row_indices = IntervalSubSet2DDevice::RowIndexView();
  subset.parent = IntervalSet2DDevice{};
  subset.num_entries = 0;
  subset.total_cells = 0;
}

namespace detail {

inline Kokkos::View<int*, DeviceMemorySpace>
build_subset_row_map(const IntervalSet2DDevice& geom,
                     const IntervalSet2DDevice& mask) {
  Kokkos::View<int*, DeviceMemorySpace> row_map(
      "subsetix_interval_subset_row_map", mask.num_rows);
  if (mask.num_rows == 0) {
    return row_map;
  }

  if (geom.num_rows == 0) {
    throw std::runtime_error(
        "IntervalSubSet2D requires a non-empty parent geometry");
  }

  auto mask_rows = mask.row_keys;
  auto geom_rows = geom.row_keys;
  const std::size_t num_geom_rows = geom.num_rows;

  Kokkos::parallel_for(
      "subsetix_interval_subset_row_map",
      Kokkos::RangePolicy<ExecSpace>(0, mask.num_rows),
      KOKKOS_LAMBDA(const std::size_t i) {
        const Coord ym = mask_rows(i).y;

        std::size_t lo = 0;
        std::size_t hi = num_geom_rows;
        while (lo < hi) {
          const std::size_t mid = lo + (hi - lo) / 2;
          if (geom_rows(mid).y < ym) {
            lo = mid + 1;
          } else {
            hi = mid;
          }
        }

        if (lo < num_geom_rows && geom_rows(lo).y == ym) {
          row_map(i) = static_cast<int>(lo);
        } else {
          row_map(i) = -1;
        }
      });

  ExecSpace().fence();

  int min_val = 0;
  Kokkos::parallel_reduce(
      "subsetix_interval_subset_row_map_min",
      Kokkos::RangePolicy<ExecSpace>(0, mask.num_rows),
      KOKKOS_LAMBDA(const std::size_t i, int& lmin) {
        if (row_map(i) < lmin) {
          lmin = row_map(i);
        }
      },
      Kokkos::Min<int>(min_val));

  if (min_val < 0) {
    throw std::runtime_error(
        "IntervalSubSet2D mask rows must exist in parent geometry");
  }

  return row_map;
}

inline Kokkos::View<std::size_t*, DeviceMemorySpace>
allocate_or_view_size_t(std::size_t size,
                        Kokkos::View<std::size_t*, DeviceMemorySpace> reuse) {
  if (reuse.extent(0) >= size && reuse.data()) {
    return reuse;
  }
  return Kokkos::View<std::size_t*, DeviceMemorySpace>(
      "subsetix_interval_subset_size_t", size);
}

inline Kokkos::View<int*, DeviceMemorySpace>
allocate_or_view_int(std::size_t size,
                     Kokkos::View<int*, DeviceMemorySpace> reuse) {
  if (reuse.extent(0) >= size && reuse.data()) {
    return reuse;
  }
  return Kokkos::View<int*, DeviceMemorySpace>(
      "subsetix_interval_subset_int", size);
}

} // namespace detail

inline void build_interval_subset_device(
    const IntervalSet2DDevice& geom,
    const IntervalSet2DDevice& mask,
    IntervalSubSet2DDevice& subset,
    CsrSetAlgebraContext* ctx = nullptr) {
  if (mask.num_rows == 0 || mask.num_intervals == 0) {
    reset_interval_subset(subset);
    return;
  }

  if (geom.num_rows == 0) {
    throw std::runtime_error(
        "parent geometry must not be empty when building IntervalSubSet2D");
  }

  const auto row_map = detail::build_subset_row_map(geom, mask);

  const std::size_t num_rows = mask.num_rows;
  Kokkos::View<std::size_t*, DeviceMemorySpace> row_counts;
  Kokkos::View<std::size_t*, DeviceMemorySpace> row_offsets;

  if (ctx) {
    row_counts = ctx->workspace.get_size_t_buf_0(num_rows);
    row_offsets = ctx->workspace.get_size_t_buf_1(num_rows + 1);
  } else {
    row_counts = detail::allocate_or_view_size_t(
        num_rows, Kokkos::View<std::size_t*, DeviceMemorySpace>());
    row_offsets = detail::allocate_or_view_size_t(
        num_rows + 1, Kokkos::View<std::size_t*, DeviceMemorySpace>());
  }

  auto mask_row_ptr = mask.row_ptr;
  auto mask_intervals = mask.intervals;
  auto geom_row_ptr = geom.row_ptr;
  auto geom_intervals = geom.intervals;

  Kokkos::parallel_for(
      "subsetix_interval_subset_count",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows),
      KOKKOS_LAMBDA(const std::size_t row_idx) {
        const int geom_row = row_map(row_idx);
        if (geom_row < 0) {
          row_counts(row_idx) = 0;
          return;
        }

        const std::size_t mask_begin = mask_row_ptr(row_idx);
        const std::size_t mask_end = mask_row_ptr(row_idx + 1);
        const std::size_t geom_begin = geom_row_ptr(geom_row);
        const std::size_t geom_end = geom_row_ptr(geom_row + 1);

        row_counts(row_idx) = detail::row_intersection_count(
            mask_intervals, mask_begin, mask_end, geom_intervals, geom_begin,
            geom_end);
      });

  ExecSpace().fence();

  Kokkos::View<std::size_t, DeviceMemorySpace> total_entries(
      "subsetix_interval_subset_total_entries");

  Kokkos::parallel_scan(
      "subsetix_interval_subset_offsets",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows),
      KOKKOS_LAMBDA(const std::size_t row_idx, std::size_t& update,
                    const bool final_pass) {
        const std::size_t count = row_counts(row_idx);
        if (final_pass) {
          row_offsets(row_idx) = update;
          if (row_idx + 1 == num_rows) {
            row_offsets(num_rows) = update + count;
            total_entries() = update + count;
          }
        }
        update += count;
      });

  ExecSpace().fence();

  std::size_t num_entries = 0;
  Kokkos::deep_copy(num_entries, total_entries);

  if (num_entries == 0) {
    reset_interval_subset(subset);
    subset.parent = geom;
    return;
  }

  subset.parent = geom;
  subset.interval_indices = IntervalSubSet2DDevice::IndexView(
      "subsetix_interval_subset_indices", num_entries);
  subset.x_begin = IntervalSubSet2DDevice::CoordView(
      "subsetix_interval_subset_x_begin", num_entries);
  subset.x_end = IntervalSubSet2DDevice::CoordView(
      "subsetix_interval_subset_x_end", num_entries);
  subset.row_indices = IntervalSubSet2DDevice::RowIndexView(
      "subsetix_interval_subset_row_indices", num_entries);
  subset.num_entries = num_entries;

  auto subset_indices = subset.interval_indices;
  auto subset_x_begin = subset.x_begin;
  auto subset_x_end = subset.x_end;
  auto subset_rows = subset.row_indices;

  Kokkos::parallel_for(
      "subsetix_interval_subset_fill",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows),
      KOKKOS_LAMBDA(const std::size_t row_idx) {
        const int geom_row = row_map(row_idx);
        if (geom_row < 0) {
          return;
        }

        const std::size_t mask_begin = mask_row_ptr(row_idx);
        const std::size_t mask_end = mask_row_ptr(row_idx + 1);
        const std::size_t geom_begin = geom_row_ptr(geom_row);
        const std::size_t geom_end = geom_row_ptr(geom_row + 1);

        std::size_t out_offset = row_offsets(row_idx);
        std::size_t write_idx = 0;

        std::size_t ia = geom_begin;
        std::size_t ib = mask_begin;
        while (ia < geom_end && ib < mask_end) {
          const Interval geom_iv = geom_intervals(ia);
          const Interval mask_iv = mask_intervals(ib);

          const Coord start = geom_iv.begin > mask_iv.begin ? geom_iv.begin
                                                            : mask_iv.begin;
          const Coord end = geom_iv.end < mask_iv.end ? geom_iv.end
                                                      : mask_iv.end;
          if (start < end) {
            const std::size_t out_idx = out_offset + write_idx;
            subset_indices(out_idx) = ia;
            subset_rows(out_idx) = geom_row;
            subset_x_begin(out_idx) = start;
            subset_x_end(out_idx) = end;
            ++write_idx;
          }

          if (geom_iv.end < mask_iv.end) {
            ++ia;
          } else if (mask_iv.end < geom_iv.end) {
            ++ib;
          } else {
            ++ia;
            ++ib;
          }
        }
      });

  ExecSpace().fence();

  Kokkos::parallel_reduce(
      "subsetix_interval_subset_total_cells",
      Kokkos::RangePolicy<ExecSpace>(0, num_entries),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& sum) {
        const Coord begin = subset_x_begin(i);
        const Coord end = subset_x_end(i);
        sum += static_cast<std::size_t>(end - begin);
      },
      subset.total_cells);
}

inline IntervalSubSet2DDevice
build_interval_subset_device(const IntervalSet2DDevice& geom,
                             const IntervalSet2DDevice& mask,
                             CsrSetAlgebraContext* ctx = nullptr) {
  IntervalSubSet2DDevice subset;
  build_interval_subset_device(geom, mask, subset, ctx);
  return subset;
}

} // namespace csr
} // namespace subsetix
