#pragma once

#include <cstddef>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include <subsetix/csr_field.hpp>
#include <subsetix/csr_interval_set.hpp>
#include <subsetix/csr_ops/field_core.hpp>
#include <subsetix/detail/csr_utils.hpp>

namespace subsetix {
namespace csr {

using ExecSpace = Kokkos::DefaultExecutionSpace;

namespace detail {

// ---------------------------------------------------------------------------
// Field access helpers for stencils
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
  typename Field2DDevice<T>::RowKeyView row_keys;
  typename Field2DDevice<T>::IndexView row_ptr;
  typename Field2DDevice<T>::IntervalView intervals;
  Kokkos::View<std::size_t*, DeviceMemorySpace> offsets;
  typename Field2DDevice<T>::ValueView values;
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
        offsets(interval_idx) +
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
    const Field2DDevice<T>& field) {
  VerticalIntervalMapping mapping;

  const std::size_t num_rows = field.geometry.num_rows;
  const std::size_t num_intervals = field.geometry.num_intervals;

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

  auto row_keys = field.geometry.row_keys;
  auto row_ptr = field.geometry.row_ptr;

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
struct FieldStencilContext {
  typename Field2DDevice<T>::IntervalView intervals;
  Kokkos::View<std::size_t*, DeviceMemorySpace> offsets;
  typename Field2DDevice<T>::ValueView values;
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
    const Interval iv = intervals(up_idx);
    const std::size_t offset =
        offsets(up_idx) +
        static_cast<std::size_t>(x - iv.begin);
    return values(offset);
  }

  KOKKOS_INLINE_FUNCTION
  T south(Coord x,
          int center_interval_idx) const {
    const int down_idx =
        down_interval(center_interval_idx);
    const Interval iv = intervals(down_idx);
    const std::size_t offset =
        offsets(down_idx) +
        static_cast<std::size_t>(x - iv.begin);
    return values(offset);
  }
};

} // namespace detail

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
    Field2DDevice<T>& field_out,
    const Field2DDevice<T>& field_in,
    const IntervalSet2DDevice& mask,
    StencilFunctor stencil) {
  if (mask.num_rows == 0 || mask.num_intervals == 0) {
    return;
  }

  if (field_out.geometry.num_rows == 0 || field_in.geometry.num_rows == 0) {
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
  auto field_out_intervals = field_out.geometry.intervals;
  auto field_out_offsets = field_out.geometry.cell_offsets;
  auto field_out_values = field_out.values;

  const detail::VerticalIntervalMapping vertical =
      detail::build_vertical_interval_mapping(
          field_in);

  detail::FieldStencilContext<T> ctx;
  ctx.intervals = field_in.geometry.intervals;
  ctx.offsets = field_in.geometry.cell_offsets;
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
            field_out_offsets(field_interval_idx);
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

} // namespace csr
} // namespace subsetix

