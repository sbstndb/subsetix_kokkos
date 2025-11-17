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

  auto mask_rows_host =
      Kokkos::create_mirror_view_and_copy(HostMemorySpace{},
                                          mask.row_keys);
  auto field_rows_host =
      Kokkos::create_mirror_view_and_copy(HostMemorySpace{},
                                          field.row_keys);

  auto mapping_host = Kokkos::create_mirror_view(mapping);
  std::size_t im = 0;
  std::size_t ifield = 0;

  while (im < mask.num_rows && ifield < field.num_rows) {
    const Coord ym = mask_rows_host(im).y;
    const Coord yf = field_rows_host(ifield).y;
    if (ym == yf) {
      mapping_host(im) = static_cast<int>(ifield);
      ++im;
      ++ifield;
    } else if (ym > yf) {
      ++ifield;
    } else {
      mapping_host(im) = -1;
      ++im;
    }
  }

  while (im < mask.num_rows) {
    mapping_host(im) = -1;
    ++im;
  }

  for (std::size_t i = 0; i < mask.num_rows; ++i) {
    if (mapping_host(i) < 0) {
      throw std::runtime_error(
          "mask row not found in field geometry; "
          "mask must be a subset of the field geometry");
    }
  }

  Kokkos::deep_copy(mapping, mapping_host);
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

  if (num_rows == 0 || num_intervals == 0) {
    mapping.up_interval = up;
    mapping.down_interval = down;
    return mapping;
  }

  auto h_row_keys =
      Kokkos::create_mirror_view_and_copy(HostMemorySpace{},
                                          field.row_keys);
  auto h_row_ptr =
      Kokkos::create_mirror_view_and_copy(HostMemorySpace{},
                                          field.row_ptr);

  std::vector<int> up_host(num_intervals, -1);
  std::vector<int> down_host(num_intervals, -1);

  std::size_t j_up = 0;
  std::size_t j_down = 0;

  for (std::size_t r = 0; r < num_rows; ++r) {
    const Coord y = h_row_keys(r).y;

    const Coord target_up =
        static_cast<Coord>(y + 1);
    while (j_up < num_rows &&
           h_row_keys(j_up).y < target_up) {
      ++j_up;
    }
    const int row_up =
        (j_up < num_rows &&
         h_row_keys(j_up).y == target_up)
            ? static_cast<int>(j_up)
            : -1;

    const Coord target_down =
        static_cast<Coord>(y - 1);
    while (j_down < num_rows &&
           h_row_keys(j_down).y < target_down) {
      ++j_down;
    }
    const int row_down =
        (j_down < num_rows &&
         h_row_keys(j_down).y == target_down)
            ? static_cast<int>(j_down)
            : -1;

    const std::size_t begin = h_row_ptr(r);
    const std::size_t end = h_row_ptr(r + 1);
    const std::size_t count = end - begin;

    if (row_up >= 0) {
      const std::size_t up_begin =
          h_row_ptr(row_up);
      const std::size_t up_end =
          h_row_ptr(row_up + 1);
      const std::size_t up_count =
          up_end - up_begin;
      if (count == up_count) {
        for (std::size_t k = 0; k < count; ++k) {
          up_host[begin + k] =
              static_cast<int>(up_begin + k);
        }
      }
    }

    if (row_down >= 0) {
      const std::size_t down_begin =
          h_row_ptr(row_down);
      const std::size_t down_end =
          h_row_ptr(row_down + 1);
      const std::size_t down_count =
          down_end - down_begin;
      if (count == down_count) {
        for (std::size_t k = 0; k < count; ++k) {
          down_host[begin + k] =
              static_cast<int>(down_begin + k);
        }
      }
    }
  }

  auto h_up = Kokkos::create_mirror_view(up);
  auto h_down = Kokkos::create_mirror_view(down);
  for (std::size_t i = 0; i < num_intervals; ++i) {
    h_up(i) = up_host[i];
    h_down(i) = down_host[i];
  }

  Kokkos::deep_copy(up, h_up);
  Kokkos::deep_copy(down, h_down);

  mapping.up_interval = up;
  mapping.down_interval = down;
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

  auto mask_row_keys = coarse_mask.row_keys;
  auto mask_intervals = coarse_mask.intervals;
  auto coarse_intervals = coarse_field.intervals;
  auto coarse_values = coarse_field.values;

  detail::FieldReadAccessor<T> fine_accessor;
  fine_accessor.row_keys = fine_field.row_keys;
  fine_accessor.row_ptr = fine_field.row_ptr;
  fine_accessor.intervals = fine_field.intervals;
  fine_accessor.values = fine_field.values;
  fine_accessor.num_rows = fine_field.num_rows;

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
        const Coord y_coarse = mask_row_keys(row_idx).y;
        const std::size_t base_offset = coarse_iv.value_offset;
        const Coord base_begin = coarse_iv.begin;

        for (Coord x = mask_iv.begin; x < mask_iv.end; ++x) {
          const std::size_t linear_index =
              base_offset +
              static_cast<std::size_t>(x - base_begin);

          const Coord fine_x0 = 2 * x;
          const Coord fine_y0 = 2 * y_coarse;
          Coord fine_coords[4][2] = {
              {fine_x0, fine_y0},
              {fine_x0 + 1, fine_y0},
              {fine_x0, fine_y0 + 1},
              {fine_x0 + 1, fine_y0 + 1}};

          T sum = T();
          int count = 0;
          for (int k = 0; k < 4; ++k) {
            T value;
            if (fine_accessor.try_get(fine_coords[k][0],
                                      fine_coords[k][1],
                                      value)) {
              sum += value;
              ++count;
            }
          }

          if (count > 0) {
            const T avg =
                sum / static_cast<T>(count);
            coarse_values(linear_index) = avg;
          }
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

  auto mask_row_keys = fine_mask.row_keys;
  auto mask_intervals = fine_mask.intervals;
  auto fine_intervals = fine_field.intervals;
  auto fine_values = fine_field.values;

  detail::FieldReadAccessor<T> coarse_accessor;
  coarse_accessor.row_keys = coarse_field.row_keys;
  coarse_accessor.row_ptr = coarse_field.row_ptr;
  coarse_accessor.intervals = coarse_field.intervals;
  coarse_accessor.values = coarse_field.values;
  coarse_accessor.num_rows = coarse_field.num_rows;

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
        const Coord y_fine = mask_row_keys(row_idx).y;
        const std::size_t base_offset = fine_iv.value_offset;
        const Coord base_begin = fine_iv.begin;

        for (Coord x = mask_iv.begin; x < mask_iv.end; ++x) {
          const std::size_t linear_index =
              base_offset +
              static_cast<std::size_t>(x - base_begin);
          const Coord coarse_x =
              detail::floor_div2(x);
          const Coord coarse_y =
              detail::floor_div2(y_fine);
          const T value =
              coarse_accessor.value_at(coarse_x, coarse_y);
          fine_values(linear_index) = value;
        }
      });
  ExecSpace().fence();
}

} // namespace csr
} // namespace subsetix
