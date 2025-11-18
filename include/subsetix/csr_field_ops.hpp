#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <Kokkos_Core.hpp>

#include <subsetix/csr_field.hpp>
#include <subsetix/csr_interval_set.hpp>
#include <subsetix/csr_set_ops.hpp>

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

// ---------------------------------------------------------------------------
// Field access helpers for stencils / AMR
// ---------------------------------------------------------------------------

template <class RowKeyView>
KOKKOS_INLINE_FUNCTION
int find_row_index(const RowKeyView& row_keys,
                   std::size_t num_rows,
                   Coord y) {
  std::size_t left = 0;
  std::size_t right = num_rows;
  while (left < right) {
    const std::size_t mid = left + (right - left) / 2;
    const Coord current = row_keys(mid).y;
    if (current == y) {
      return static_cast<int>(mid);
    }
    if (current < y) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return -1;
}

template <class IntervalView>
KOKKOS_INLINE_FUNCTION
int find_interval_index(const IntervalView& intervals,
                        std::size_t begin,
                        std::size_t end,
                        Coord x) {
  std::size_t left = begin;
  std::size_t right = end;
  while (left < right) {
    const std::size_t mid = left + (right - left) / 2;
    const auto iv = intervals(mid);
    if (x < iv.begin) {
      right = mid;
    } else if (x >= iv.end) {
      left = mid + 1;
    } else {
      return static_cast<int>(mid);
    }
  }
  return -1;
}

template <typename T>
struct FieldReadAccessor {
  typename IntervalField2DDevice<T>::RowKeyView row_keys;
  typename IntervalField2DDevice<T>::IndexView row_ptr;
  typename IntervalField2DDevice<T>::IntervalView intervals;
  typename IntervalField2DDevice<T>::ValueView values;
  std::size_t num_rows = 0;

  KOKKOS_INLINE_FUNCTION
  bool try_get(Coord x, Coord y, T& out) const {
    const int row_idx =
        find_row_index(row_keys, num_rows, y);
    if (row_idx < 0) {
      return false;
    }
    const std::size_t begin = row_ptr(row_idx);
    const std::size_t end = row_ptr(row_idx + 1);
    const int interval_idx =
        find_interval_index(intervals, begin, end, x);
    if (interval_idx < 0) {
      return false;
    }
    const auto iv = intervals(interval_idx);
    const std::size_t offset =
        iv.value_offset +
        static_cast<std::size_t>(x - iv.begin);
    out = values(offset);
    return true;
  }

  KOKKOS_INLINE_FUNCTION
  T value_at(Coord x, Coord y) const {
    T out = T();
    const bool ok = try_get(x, y, out);
    return ok ? out : T();
  }

  KOKKOS_INLINE_FUNCTION
  T value_from_linear_index(std::size_t idx) const {
    return values(idx);
  }
};

struct VerticalIntervalMapping {
  Kokkos::View<int*, DeviceMemorySpace> up_interval;
  Kokkos::View<int*, DeviceMemorySpace> down_interval;
};

struct AmrIntervalMapping {
  Kokkos::View<int*, DeviceMemorySpace> coarse_to_fine_first;
  Kokkos::View<int*, DeviceMemorySpace> coarse_to_fine_second;
  Kokkos::View<int*, DeviceMemorySpace> fine_to_coarse;
};

template <typename T>
inline VerticalIntervalMapping
build_vertical_interval_mapping(
    const IntervalField2DDevice<T>& field) {
  VerticalIntervalMapping mapping;

  const std::size_t num_rows = field.num_rows;
  const std::size_t num_intervals = field.num_intervals;

  Kokkos::View<int*, DeviceMemorySpace> up(
      "subsetix_field_interval_up", num_intervals);
  Kokkos::View<int*, DeviceMemorySpace> down(
      "subsetix_field_interval_down", num_intervals);

  mapping.up_interval = up;
  mapping.down_interval = down;

  if (num_rows == 0 || num_intervals == 0) {
    return mapping;
  }

  Kokkos::deep_copy(up, -1);
  Kokkos::deep_copy(down, -1);

  auto row_keys = field.row_keys;
  auto row_ptr = field.row_ptr;

  Kokkos::parallel_for(
      "subsetix_vertical_interval_mapping",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows),
      KOKKOS_LAMBDA(const std::size_t r) {
        const Coord y = row_keys(r).y;
        const std::size_t begin = row_ptr(r);
        const std::size_t end = row_ptr(r + 1);
        const std::size_t count = end - begin;

        if (count == 0) return;

        // Find up row (y + 1)
        int row_up = -1;
        // Optimization: check next row first
        if (r + 1 < num_rows && row_keys(r + 1).y == y + 1) {
          row_up = static_cast<int>(r + 1);
        } else {
          std::size_t lo = 0;
          std::size_t hi = num_rows;
          while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            if (row_keys(mid).y < y + 1) {
              lo = mid + 1;
            } else {
              hi = mid;
            }
          }
          if (lo < num_rows && row_keys(lo).y == y + 1) {
            row_up = static_cast<int>(lo);
          }
        }

        if (row_up >= 0) {
          const std::size_t up_begin = row_ptr(row_up);
          const std::size_t up_end = row_ptr(row_up + 1);
          if ((up_end - up_begin) == count) {
            for (std::size_t k = 0; k < count; ++k) {
              up(begin + k) = static_cast<int>(up_begin + k);
            }
          }
        }

        // Find down row (y - 1)
        int row_down = -1;
        // Optimization: check prev row first
        if (r > 0 && row_keys(r - 1).y == y - 1) {
          row_down = static_cast<int>(r - 1);
        } else {
          std::size_t lo = 0;
          std::size_t hi = num_rows;
          while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            if (row_keys(mid).y < y - 1) {
              lo = mid + 1;
            } else {
              hi = mid;
            }
          }
          if (lo < num_rows && row_keys(lo).y == y - 1) {
            row_down = static_cast<int>(lo);
          }
        }

        if (row_down >= 0) {
          const std::size_t down_begin = row_ptr(row_down);
          const std::size_t down_end = row_ptr(row_down + 1);
          if ((down_end - down_begin) == count) {
            for (std::size_t k = 0; k < count; ++k) {
              down(begin + k) = static_cast<int>(down_begin + k);
            }
          }
        }
      });

  ExecSpace().fence();
  return mapping;
}

template <typename T>
inline AmrIntervalMapping
build_amr_interval_mapping(
    const IntervalField2DDevice<T>& coarse_field,
    const IntervalField2DDevice<T>& fine_field) {
  AmrIntervalMapping mapping;

  const std::size_t num_rows_coarse = coarse_field.num_rows;
  const std::size_t num_rows_fine = fine_field.num_rows;
  const std::size_t num_intervals_coarse = coarse_field.num_intervals;
  const std::size_t num_intervals_fine = fine_field.num_intervals;

  Kokkos::View<int*, DeviceMemorySpace> coarse_to_fine_first(
      "subsetix_field_coarse_to_fine_first", num_intervals_coarse);
  Kokkos::View<int*, DeviceMemorySpace> coarse_to_fine_second(
      "subsetix_field_coarse_to_fine_second", num_intervals_coarse);
  Kokkos::View<int*, DeviceMemorySpace> fine_to_coarse(
      "subsetix_field_fine_to_coarse_interval", num_intervals_fine);

  mapping.coarse_to_fine_first = coarse_to_fine_first;
  mapping.coarse_to_fine_second = coarse_to_fine_second;
  mapping.fine_to_coarse = fine_to_coarse;

  if (num_rows_coarse == 0 || num_rows_fine == 0 ||
      num_intervals_coarse == 0 || num_intervals_fine == 0) {
    return mapping;
  }

  auto coarse_rows = coarse_field.row_keys;
  auto coarse_row_ptr = coarse_field.row_ptr;
  auto coarse_intervals = coarse_field.intervals;

  auto fine_rows = fine_field.row_keys;
  auto fine_row_ptr = fine_field.row_ptr;
  auto fine_intervals = fine_field.intervals;

  // Error flags
  Kokkos::View<int, DeviceMemorySpace> error_flag("subsetix_amr_error_flag");

  Kokkos::parallel_for(
      "subsetix_amr_interval_mapping",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_coarse),
      KOKKOS_LAMBDA(const std::size_t rc) {
        const Coord y_c = coarse_rows(rc).y;
        const Coord y_f0 = static_cast<Coord>(2 * y_c);
        const Coord y_f1 = static_cast<Coord>(2 * y_c + 1);

        // Find fine rows
        int row_f0 = -1;
        int row_f1 = -1;

        // Search for y_f0
        {
          std::size_t lo = 0;
          std::size_t hi = num_rows_fine;
          while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            if (fine_rows(mid).y < y_f0) {
              lo = mid + 1;
            } else {
              hi = mid;
            }
          }
          if (lo < num_rows_fine && fine_rows(lo).y == y_f0) {
            row_f0 = static_cast<int>(lo);
          }
        }

        // Search for y_f1
        {
          std::size_t lo = 0;
          std::size_t hi = num_rows_fine;
          while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            if (fine_rows(mid).y < y_f1) {
              lo = mid + 1;
            } else {
              hi = mid;
            }
          }
          if (lo < num_rows_fine && fine_rows(lo).y == y_f1) {
            row_f1 = static_cast<int>(lo);
          }
        }

        if (row_f0 < 0 || row_f1 < 0) {
          Kokkos::atomic_store(&error_flag(), 1);
          return;
        }

        const std::size_t begin_c = coarse_row_ptr(rc);
        const std::size_t end_c = coarse_row_ptr(rc + 1);
        const std::size_t begin_f0 = fine_row_ptr(row_f0);
        const std::size_t end_f0 = fine_row_ptr(row_f0 + 1);
        const std::size_t begin_f1 = fine_row_ptr(row_f1);
        const std::size_t end_f1 = fine_row_ptr(row_f1 + 1);

        const std::size_t count_c = end_c - begin_c;
        const std::size_t count_f0 = end_f0 - begin_f0;
        const std::size_t count_f1 = end_f1 - begin_f1;

        if (count_c != count_f0 || count_c != count_f1) {
          Kokkos::atomic_store(&error_flag(), 2);
          return;
        }

        for (std::size_t k = 0; k < count_c; ++k) {
          const std::size_t ic = begin_c + k;
          const std::size_t iff0 = begin_f0 + k;
          const std::size_t iff1 = begin_f1 + k;

          const FieldInterval ci = coarse_intervals(ic);
          const FieldInterval fi0 = fine_intervals(iff0);
          const FieldInterval fi1 = fine_intervals(iff1);

          const Coord expected_begin = static_cast<Coord>(ci.begin * 2);
          const Coord expected_end = static_cast<Coord>(ci.end * 2);

          if (!(fi0.begin == expected_begin && fi0.end == expected_end &&
                fi1.begin == expected_begin && fi1.end == expected_end)) {
            Kokkos::atomic_store(&error_flag(), 3);
            return;
          }

          coarse_to_fine_first(ic) = static_cast<int>(iff0);
          coarse_to_fine_second(ic) = static_cast<int>(iff1);
          fine_to_coarse(iff0) = static_cast<int>(ic);
          fine_to_coarse(iff1) = static_cast<int>(ic);
        }
      });

  ExecSpace().fence();

  int err = 0;
  Kokkos::deep_copy(err, error_flag);
  if (err != 0) {
    if (err == 1)
      throw std::runtime_error("AMR interval mapping: fine row not found for coarse row");
    if (err == 2)
      throw std::runtime_error("AMR interval mapping: coarse/fine interval counts differ");
    if (err == 3)
      throw std::runtime_error("AMR interval mapping: fine interval does not match refined coarse interval");
    throw std::runtime_error("AMR interval mapping: unknown error");
  }

  return mapping;
}

template <typename T>
struct FieldStencilContext {
  typename IntervalField2DDevice<T>::IntervalView intervals;
  typename IntervalField2DDevice<T>::ValueView values;
  Kokkos::View<int*, DeviceMemorySpace> up_interval;
  Kokkos::View<int*, DeviceMemorySpace> down_interval;

  KOKKOS_INLINE_FUNCTION
  T center(std::size_t idx) const {
    return values(idx);
  }

  KOKKOS_INLINE_FUNCTION
  T left(std::size_t idx) const {
    return values(idx - 1);
  }

  KOKKOS_INLINE_FUNCTION
  T right(std::size_t idx) const {
    return values(idx + 1);
  }

  KOKKOS_INLINE_FUNCTION
  T north(Coord x,
          int center_interval_idx) const {
    const int up_idx =
        up_interval(center_interval_idx);
    const FieldInterval iv = intervals(up_idx);
    const std::size_t offset =
        iv.value_offset +
        static_cast<std::size_t>(x - iv.begin);
    return values(offset);
  }

  KOKKOS_INLINE_FUNCTION
  T south(Coord x,
          int center_interval_idx) const {
    const int down_idx =
        down_interval(center_interval_idx);
    const FieldInterval iv = intervals(down_idx);
    const std::size_t offset =
        iv.value_offset +
        static_cast<std::size_t>(x - iv.begin);
    return values(offset);
  }
};

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

// ---------------------------------------------------------------------------
// Phase 2 – Simple stencils restricted to a set
// ---------------------------------------------------------------------------

/**
 * @brief Apply a stencil functor on a masked region.
 *
 * Stencil functor signature:
 *   KOKKOS_INLINE_FUNCTION
 *   T operator()(Coord x, Coord y,
 *                std::size_t linear_index,
 *                int field_interval_index,
 *                const detail::FieldStencilContext<T>& ctx) const;
 *
 * The accessor can read any coordinate from the input field.
 */
template <typename T, class StencilFunctor>
inline void apply_stencil_on_set_device(
    IntervalField2DDevice<T>& field_out,
    const IntervalField2DDevice<T>& field_in,
    const IntervalSet2DDevice& mask,
    StencilFunctor stencil) {
  if (mask.num_rows == 0 || mask.num_intervals == 0) {
    return;
  }

  if (field_out.num_rows == 0 || field_in.num_rows == 0) {
    throw std::runtime_error(
        "fields must be initialized before applying a stencil");
  }

  const auto mapping =
      detail::build_mask_field_mapping(field_out, mask);
  auto interval_to_row = mapping.interval_to_row;
  auto interval_to_field_interval =
      mapping.interval_to_field_interval;
  auto row_map = mapping.row_map;

  auto mask_row_keys = mask.row_keys;
  auto mask_intervals = mask.intervals;
  auto field_out_intervals = field_out.intervals;
  auto field_out_values = field_out.values;

  const detail::VerticalIntervalMapping vertical =
      detail::build_vertical_interval_mapping(
          field_in);

  detail::FieldStencilContext<T> ctx;
  ctx.intervals = field_in.intervals;
  ctx.values = field_in.values;
  ctx.up_interval = vertical.up_interval;
  ctx.down_interval = vertical.down_interval;

  Kokkos::parallel_for(
      "subsetix_apply_stencil_on_set_device",
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
            field_out_intervals(field_interval_idx);
        const Coord y = mask_row_keys(row_idx).y;
          const std::size_t base_offset =
              field_iv.value_offset;
          const Coord base_begin = field_iv.begin;

          for (Coord x = mask_iv.begin; x < mask_iv.end; ++x) {
            const std::size_t linear_index =
                base_offset +
                static_cast<std::size_t>(x - base_begin);
            const T value =
                stencil(x, y, linear_index,
                        field_interval_idx, ctx);
            field_out_values(linear_index) = value;
          }
      });
  ExecSpace().fence();
}

// ---------------------------------------------------------------------------
// Phase 3 – AMR field operations (restriction / prolongation)
// ---------------------------------------------------------------------------

template <typename T>
inline void restrict_field_on_set_device(
    IntervalField2DDevice<T>& coarse_field,
    const IntervalField2DDevice<T>& fine_field,
    const IntervalSet2DDevice& coarse_mask) {
  if (coarse_mask.num_rows == 0 ||
      coarse_mask.num_intervals == 0) {
    return;
  }

  const auto mapping =
      detail::build_mask_field_mapping(coarse_field, coarse_mask);
  auto interval_to_row = mapping.interval_to_row;
  auto interval_to_field_interval =
      mapping.interval_to_field_interval;
  auto row_map = mapping.row_map;

  auto mask_intervals = coarse_mask.intervals;
  auto coarse_intervals = coarse_field.intervals;
  auto coarse_values = coarse_field.values;

  const detail::AmrIntervalMapping amr_mapping =
      detail::build_amr_interval_mapping(
          coarse_field, fine_field);
  auto coarse_to_fine_first =
      amr_mapping.coarse_to_fine_first;
  auto coarse_to_fine_second =
      amr_mapping.coarse_to_fine_second;
  auto fine_intervals = fine_field.intervals;
  auto fine_values = fine_field.values;

  Kokkos::parallel_for(
      "subsetix_restrict_field_on_set_device",
      Kokkos::RangePolicy<ExecSpace>(0,
                                     static_cast<int>(coarse_mask.num_intervals)),
      KOKKOS_LAMBDA(const int interval_idx) {
        const int row_idx = interval_to_row(interval_idx);
        const int field_row = row_map(row_idx);
        if (row_idx < 0 || field_row < 0) {
          return;
        }

        const int coarse_interval_idx =
            interval_to_field_interval(interval_idx);
        if (coarse_interval_idx < 0) {
          return;
        }

        const auto mask_iv = mask_intervals(interval_idx);
        const auto coarse_iv =
            coarse_intervals(coarse_interval_idx);
        const std::size_t base_offset = coarse_iv.value_offset;
        const Coord base_begin = coarse_iv.begin;

        const int fine_interval_idx0 =
            coarse_to_fine_first(coarse_interval_idx);
        const int fine_interval_idx1 =
            coarse_to_fine_second(coarse_interval_idx);
        const auto fine_iv0 =
            fine_intervals(fine_interval_idx0);
        const auto fine_iv1 =
            fine_intervals(fine_interval_idx1);
        const std::size_t fine_base_offset0 =
            fine_iv0.value_offset;
        const std::size_t fine_base_offset1 =
            fine_iv1.value_offset;
        const Coord fine_base_begin0 =
            fine_iv0.begin;
        const Coord fine_base_begin1 =
            fine_iv1.begin;

        for (Coord x = mask_iv.begin; x < mask_iv.end; ++x) {
          const std::size_t linear_index =
              base_offset +
              static_cast<std::size_t>(x - base_begin);

          const Coord fine_x0 =
              static_cast<Coord>(2 * x);
          const Coord fine_x1 =
              static_cast<Coord>(2 * x + 1);

          const std::size_t fine_idx00 =
              fine_base_offset0 +
              static_cast<std::size_t>(
                  fine_x0 - fine_base_begin0);
          const std::size_t fine_idx01 =
              fine_base_offset0 +
              static_cast<std::size_t>(
                  fine_x1 - fine_base_begin0);
          const std::size_t fine_idx10 =
              fine_base_offset1 +
              static_cast<std::size_t>(
                  fine_x0 - fine_base_begin1);
          const std::size_t fine_idx11 =
              fine_base_offset1 +
              static_cast<std::size_t>(
                  fine_x1 - fine_base_begin1);

          const T v00 = fine_values(fine_idx00);
          const T v01 = fine_values(fine_idx01);
          const T v10 = fine_values(fine_idx10);
          const T v11 = fine_values(fine_idx11);

          const T avg =
              static_cast<T>(0.25) *
              (v00 + v01 + v10 + v11);
          coarse_values(linear_index) = avg;
        }
      });
  ExecSpace().fence();
}

template <typename T>
inline void prolong_field_on_set_device(
    IntervalField2DDevice<T>& fine_field,
    const IntervalField2DDevice<T>& coarse_field,
    const IntervalSet2DDevice& fine_mask) {
  if (fine_mask.num_rows == 0 ||
      fine_mask.num_intervals == 0) {
    return;
  }

  const auto mapping =
      detail::build_mask_field_mapping(fine_field, fine_mask);
  auto interval_to_row = mapping.interval_to_row;
  auto interval_to_field_interval =
      mapping.interval_to_field_interval;
  auto row_map = mapping.row_map;

  auto mask_intervals = fine_mask.intervals;
  auto fine_intervals = fine_field.intervals;
  auto fine_values = fine_field.values;

  const detail::AmrIntervalMapping amr_mapping =
      detail::build_amr_interval_mapping(
          coarse_field, fine_field);
  auto fine_to_coarse =
      amr_mapping.fine_to_coarse;
  auto coarse_intervals = coarse_field.intervals;
  auto coarse_values = coarse_field.values;

  Kokkos::parallel_for(
      "subsetix_prolong_field_on_set_device",
      Kokkos::RangePolicy<ExecSpace>(0,
                                     static_cast<int>(fine_mask.num_intervals)),
      KOKKOS_LAMBDA(const int interval_idx) {
        const int row_idx = interval_to_row(interval_idx);
        const int field_row = row_map(row_idx);
        if (row_idx < 0 || field_row < 0) {
          return;
        }

        const int fine_interval_idx =
            interval_to_field_interval(interval_idx);
        if (fine_interval_idx < 0) {
          return;
        }

        const auto mask_iv = mask_intervals(interval_idx);
        const auto fine_iv =
            fine_intervals(fine_interval_idx);
        const std::size_t base_offset = fine_iv.value_offset;
        const Coord base_begin = fine_iv.begin;
        const int coarse_interval_idx =
            fine_to_coarse(fine_interval_idx);
        const auto coarse_iv =
            coarse_intervals(coarse_interval_idx);
        const std::size_t coarse_base_offset =
            coarse_iv.value_offset;
        const Coord coarse_base_begin =
            coarse_iv.begin;

        for (Coord x = mask_iv.begin; x < mask_iv.end; ++x) {
          const std::size_t linear_index =
              base_offset +
              static_cast<std::size_t>(x - base_begin);

          const Coord coarse_x =
              detail::floor_div2(x);
          const std::size_t coarse_offset =
              coarse_base_offset +
              static_cast<std::size_t>(
                  coarse_x - coarse_base_begin);
          const T value =
              coarse_values(coarse_offset);
          fine_values(linear_index) = value;
        }
      });
  ExecSpace().fence();
}

} // namespace csr
} // namespace subsetix
