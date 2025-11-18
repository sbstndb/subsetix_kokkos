#pragma once

#include <cstddef>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include <subsetix/csr_field.hpp>
#include <subsetix/csr_interval_set.hpp>
#include <subsetix/detail/csr_utils.hpp>

namespace subsetix {
namespace csr {

using ExecSpace = Kokkos::DefaultExecutionSpace;

namespace detail {

// ---------------------------------------------------------------------------
// Helpers to map a mask IntervalSet to a field geometry
// ---------------------------------------------------------------------------

template <typename T>
struct MaskFieldMapping {
  Kokkos::View<int*, DeviceMemorySpace> row_map;
  Kokkos::View<int*, DeviceMemorySpace> interval_to_row;
  Kokkos::View<int*, DeviceMemorySpace> interval_to_field_interval;
};

template <typename T>
inline Kokkos::View<int*, DeviceMemorySpace>
build_mask_row_to_field_row_mapping(const IntervalSet2DDevice& mask,
                                    const IntervalField2DDevice<T>& field) {
  Kokkos::View<int*, DeviceMemorySpace> mapping(
      "subsetix_mask_row_to_field_row", mask.num_rows);
  if (mask.num_rows == 0) {
    return mapping;
  }

  if (field.num_rows == 0) {
    throw std::runtime_error(
        "field has no rows but the mask is non-empty");
  }

  auto mask_rows = mask.row_keys;
  auto field_rows = field.row_keys;
  const std::size_t num_field_rows = field.num_rows;

  Kokkos::parallel_for(
      "subsetix_mask_row_map_kernel",
      Kokkos::RangePolicy<ExecSpace>(0, mask.num_rows),
      KOKKOS_LAMBDA(const std::size_t i) {
        const Coord ym = mask_rows(i).y;

        std::size_t lo = 0;
        std::size_t hi = num_field_rows;
        while (lo < hi) {
          const std::size_t mid = lo + (hi - lo) / 2;
          if (field_rows(mid).y < ym) {
            lo = mid + 1;
          } else {
            hi = mid;
          }
        }

        if (lo < num_field_rows && field_rows(lo).y == ym) {
          mapping(i) = static_cast<int>(lo);
        } else {
          mapping(i) = -1;
        }
      });

  ExecSpace().fence();

  int min_val = 0;
  Kokkos::parallel_reduce(
      "subsetix_check_mask_mapping_validity",
      Kokkos::RangePolicy<ExecSpace>(0, mask.num_rows),
      KOKKOS_LAMBDA(const std::size_t i, int& lmin) {
        if (mapping(i) < lmin) lmin = mapping(i);
      },
      Kokkos::Min<int>(min_val));

  if (min_val < 0) {
    throw std::runtime_error(
        "mask row not found in field geometry; "
        "mask must be a subset of the field geometry");
  }

  return mapping;
}

inline Kokkos::View<int*, DeviceMemorySpace>
build_mask_interval_to_row_mapping(const IntervalSet2DDevice& mask) {
  Kokkos::View<int*, DeviceMemorySpace> interval_rows(
      "subsetix_mask_interval_rows", mask.num_intervals);
  if (mask.num_rows == 0 || mask.num_intervals == 0) {
    return interval_rows;
  }

  auto mask_row_ptr = mask.row_ptr;
  Kokkos::parallel_for(
      "subsetix_fill_mask_interval_rows",
      Kokkos::RangePolicy<ExecSpace>(0,
                                     static_cast<int>(mask.num_rows)),
      KOKKOS_LAMBDA(const int row_idx) {
        const std::size_t begin = mask_row_ptr(row_idx);
        const std::size_t end = mask_row_ptr(row_idx + 1);
        for (std::size_t k = begin; k < end; ++k) {
          interval_rows(k) = row_idx;
        }
      });
  ExecSpace().fence();
  return interval_rows;
}

template <typename T>
inline Kokkos::View<int*, DeviceMemorySpace>
build_mask_interval_to_field_interval_mapping(
    const IntervalSet2DDevice& mask,
    const IntervalField2DDevice<T>& field,
    const Kokkos::View<int*, DeviceMemorySpace>& row_map) {
  Kokkos::View<int*, DeviceMemorySpace> mapping(
      "subsetix_mask_interval_to_field_interval",
      mask.num_intervals);

  if (mask.num_rows == 0 || mask.num_intervals == 0) {
    return mapping;
  }

  Kokkos::deep_copy(mapping, -1);

  auto mask_row_ptr = mask.row_ptr;
  auto mask_intervals = mask.intervals;
  auto field_row_ptr = field.row_ptr;
  auto field_intervals = field.intervals;

  Kokkos::parallel_for(
      "subsetix_fill_mask_interval_field_interval",
      Kokkos::RangePolicy<ExecSpace>(0,
                                     static_cast<int>(mask.num_rows)),
      KOKKOS_LAMBDA(const int row_idx) {
        const int field_row = row_map(row_idx);
        if (field_row < 0) {
          return;
        }

        const std::size_t mask_begin = mask_row_ptr(row_idx);
        const std::size_t mask_end = mask_row_ptr(row_idx + 1);
        const std::size_t field_begin = field_row_ptr(field_row);
        const std::size_t field_end = field_row_ptr(field_row + 1);

        std::size_t fi = field_begin;
        for (std::size_t mi = mask_begin; mi < mask_end; ++mi) {
          const auto mask_iv = mask_intervals(mi);
          while (fi < field_end) {
            const auto field_iv = field_intervals(fi);
            if (mask_iv.begin >= field_iv.begin &&
                mask_iv.end <= field_iv.end) {
              mapping(mi) = static_cast<int>(fi);
              break;
            }
            ++fi;
          }
        }
      });
  ExecSpace().fence();

  return mapping;
}

template <typename T>
inline MaskFieldMapping<T>
build_mask_field_mapping(const IntervalField2DDevice<T>& field,
                         const IntervalSet2DDevice& mask) {
  MaskFieldMapping<T> mapping;
  mapping.row_map =
      build_mask_row_to_field_row_mapping(mask, field);
  mapping.interval_to_row =
      build_mask_interval_to_row_mapping(mask);
  mapping.interval_to_field_interval =
      build_mask_interval_to_field_interval_mapping(
          mask, field, mapping.row_map);
  return mapping;
}

} // namespace detail

// ---------------------------------------------------------------------------
// Phase 1 – Masked field operations
// ---------------------------------------------------------------------------

/**
 * @brief Apply a user functor on all cells of a field restricted by a mask.
 *
 * Functor signature:
 *   KOKKOS_INLINE_FUNCTION
 *   void operator()(Coord x, Coord y,
 *                   typename IntervalField2DDevice<T>::ValueView::
 *                       reference_type value,
 *                   std::size_t linear_index) const;
 *
 * The linear index corresponds to the entry in field.values().
 */
template <typename T, class Functor>
inline void apply_on_set_device(IntervalField2DDevice<T>& field,
                                const IntervalSet2DDevice& mask,
                                Functor func) {
  if (mask.num_rows == 0 || mask.num_intervals == 0) {
    return;
  }

  const auto mapping =
      detail::build_mask_field_mapping(field, mask);
  auto interval_to_row = mapping.interval_to_row;
  auto interval_to_field_interval =
      mapping.interval_to_field_interval;
  auto row_map = mapping.row_map;

  auto mask_row_keys = mask.row_keys;
  auto mask_intervals = mask.intervals;
  auto field_intervals = field.intervals;
  auto field_values = field.values;

  Kokkos::parallel_for(
      "subsetix_apply_on_set_device",
      Kokkos::RangePolicy<ExecSpace>(0,
                                     static_cast<int>(mask.num_intervals)),
      KOKKOS_LAMBDA(const int interval_idx) {
        const int row_idx = interval_to_row(interval_idx);
        const int field_row = row_map(row_idx);
        if (row_idx < 0 || field_row < 0) {
          return;
        }

        const int field_interval_idx =
            interval_to_field_interval(interval_idx);
        if (field_interval_idx < 0) {
          return;
        }

        const auto mask_iv = mask_intervals(interval_idx);
        const auto field_iv =
            field_intervals(field_interval_idx);
        const Coord y = mask_row_keys(row_idx).y;
        const std::size_t base_offset = field_iv.value_offset;
        const Coord base_begin = field_iv.begin;

        for (Coord x = mask_iv.begin; x < mask_iv.end; ++x) {
          const std::size_t linear_index =
              base_offset +
              static_cast<std::size_t>(x - base_begin);
          func(x, y, field_values(linear_index), linear_index);
        }
      });
  ExecSpace().fence();
}

template <typename T>
inline void fill_on_set_device(IntervalField2DDevice<T>& field,
                               const IntervalSet2DDevice& mask,
                               const T& value) {
  const T value_copy = value;
  apply_on_set_device(
      field, mask,
      KOKKOS_LAMBDA(
          Coord /*x*/, Coord /*y*/,
          typename IntervalField2DDevice<T>::ValueView::
              reference_type cell,
          std::size_t /*idx*/) { cell = value_copy; });
}

template <typename T>
inline void scale_on_set_device(IntervalField2DDevice<T>& field,
                                const IntervalSet2DDevice& mask,
                                const T& alpha) {
  const T alpha_copy = alpha;
  apply_on_set_device(
      field, mask,
      KOKKOS_LAMBDA(
          Coord /*x*/, Coord /*y*/,
          typename IntervalField2DDevice<T>::ValueView::
              reference_type cell,
          std::size_t /*idx*/) { cell *= alpha_copy; });
}

template <typename T>
inline void copy_on_set_device(IntervalField2DDevice<T>& dst,
                               const IntervalField2DDevice<T>& src,
                               const IntervalSet2DDevice& mask) {
  auto src_values = src.values;
  apply_on_set_device(
      dst, mask,
      KOKKOS_LAMBDA(
          Coord /*x*/, Coord /*y*/,
          typename IntervalField2DDevice<T>::ValueView::
              reference_type cell,
          std::size_t idx) { cell = src_values(idx); });
}

} // namespace csr
} // namespace subsetix

