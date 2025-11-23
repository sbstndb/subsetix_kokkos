#pragma once

#include <cstddef>

#include <Kokkos_Core.hpp>

#include <subsetix/csr_interval_set.hpp>

namespace subsetix {
namespace csr {

/**
 * @brief Public mapping between a mask IntervalSet and a field geometry.
 *
 * The mapping does not depend on the field value type; it only encodes
 * structural relations between mask rows/intervals and field rows/intervals.
 *
 * Missing rows or intervals are encoded with -1 instead of throwing.
 */
struct FieldMaskMapping {
  Kokkos::View<int*, DeviceMemorySpace> row_map;
  Kokkos::View<int*, DeviceMemorySpace> interval_to_row;
  Kokkos::View<int*, DeviceMemorySpace> interval_to_field_interval;
};

/**
 * @brief Build a row mapping between two row-key arrays (mask -> parent).
 *
 * For each mask row y, find the index of the matching row in parent_rows.
 * If no match is found, the entry is set to -1. No exception is thrown.
 */
inline Kokkos::View<int*, DeviceMemorySpace>
build_row_map_y(const IntervalSet2DDevice::RowKeyView& mask_rows,
                const IntervalSet2DDevice::RowKeyView& parent_rows,
                std::size_t num_parent_rows) {
  Kokkos::View<int*, DeviceMemorySpace> mapping(
      "subsetix_row_map_y", mask_rows.extent(0));
  if (mask_rows.extent(0) == 0) {
    return mapping;
  }

  Kokkos::parallel_for(
      "subsetix_row_map_y_kernel",
      Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(mask_rows.extent(0))),
      KOKKOS_LAMBDA(const int i) {
        const Coord ym = mask_rows(i).y;

        std::size_t lo = 0;
        std::size_t hi = num_parent_rows;
        while (lo < hi) {
          const std::size_t mid = lo + (hi - lo) / 2;
          const Coord y_parent = parent_rows(mid).y;
          if (y_parent < ym) {
            lo = mid + 1;
          } else {
            hi = mid;
          }
        }

        if (lo < num_parent_rows && parent_rows(lo).y == ym) {
          mapping(i) = static_cast<int>(lo);
        } else {
          mapping(i) = -1;
        }
      });

  ExecSpace().fence();
  return mapping;
}

/**
 * @brief Build a FieldMaskMapping between a mask IntervalSet and a field
 *        geometry IntervalSet.
 *
 * Rows or intervals that do not find a match are marked with -1. No exception
 * is thrown.
 */
inline FieldMaskMapping
build_field_mask_mapping(const IntervalSet2DDevice& mask,
                         const IntervalSet2DDevice& geom) {
  FieldMaskMapping mapping;

  if (mask.num_rows == 0 || mask.num_intervals == 0 ||
      geom.num_rows == 0 || geom.num_intervals == 0) {
    return mapping;
  }

  mapping.row_map = build_row_map_y(mask.row_keys, geom.row_keys, geom.num_rows);

  mapping.interval_to_row = Kokkos::View<int*, DeviceMemorySpace>(
      "subsetix_mask_interval_rows", mask.num_intervals);
  Kokkos::parallel_for(
      "subsetix_fill_mask_interval_rows",
      Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(mask.num_rows)),
      KOKKOS_LAMBDA(const int row_idx) {
        const std::size_t begin = mask.row_ptr(row_idx);
        const std::size_t end = mask.row_ptr(row_idx + 1);
        for (std::size_t k = begin; k < end; ++k) {
          mapping.interval_to_row(k) = row_idx;
        }
      });
  ExecSpace().fence();

  mapping.interval_to_field_interval = Kokkos::View<int*, DeviceMemorySpace>(
      "subsetix_mask_interval_to_field_interval", mask.num_intervals);
  Kokkos::deep_copy(mapping.interval_to_field_interval, -1);

  auto mask_row_ptr = mask.row_ptr;
  auto mask_intervals = mask.intervals;
  auto geom_row_ptr = geom.row_ptr;
  auto geom_intervals = geom.intervals;
  auto row_map = mapping.row_map;

  Kokkos::parallel_for(
      "subsetix_fill_mask_interval_field_interval",
      Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(mask.num_rows)),
      KOKKOS_LAMBDA(const int row_idx) {
        const int geom_row = row_map(row_idx);
        if (geom_row < 0) {
          return;
        }

        const std::size_t mask_begin = mask_row_ptr(row_idx);
        const std::size_t mask_end = mask_row_ptr(row_idx + 1);
        const std::size_t geom_begin = geom_row_ptr(geom_row);
        const std::size_t geom_end = geom_row_ptr(geom_row + 1);

        std::size_t fi = geom_begin;
        for (std::size_t mi = mask_begin; mi < mask_end; ++mi) {
          const auto mask_iv = mask_intervals(mi);
          while (fi < geom_end) {
            const auto geom_iv = geom_intervals(fi);
            if (mask_iv.begin >= geom_iv.begin &&
                mask_iv.end <= geom_iv.end) {
              mapping.interval_to_field_interval(mi) = static_cast<int>(fi);
              break;
            }
            ++fi;
          }
        }
      });

  ExecSpace().fence();
  return mapping;
}

} // namespace csr
} // namespace subsetix
