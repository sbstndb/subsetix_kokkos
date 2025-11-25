#pragma once

#include <cstddef>

#include <Kokkos_Core.hpp>

#include <subsetix/field/csr_field.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/geometry/csr_interval_subset.hpp>
#include <subsetix/csr_ops/field_core.hpp>
#include <subsetix/detail/csr_utils.hpp>

namespace subsetix {
namespace csr {

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

        // Find up row (y + 1) - optimization: check adjacent row first
        const int row_up = (r + 1 < num_rows && row_keys(r + 1).y == y + 1)
            ? static_cast<int>(r + 1)
            : find_row_by_y(row_keys, num_rows, y + 1);

        if (row_up >= 0) {
          const std::size_t up_begin = row_ptr(row_up);
          const std::size_t up_end = row_ptr(row_up + 1);
          if ((up_end - up_begin) == count) {
            for (std::size_t k = 0; k < count; ++k) {
              up(begin + k) = static_cast<int>(up_begin + k);
            }
          }
        }

        // Find down row (y - 1) - optimization: check adjacent row first
        const int row_down = (r > 0 && row_keys(r - 1).y == y - 1)
            ? static_cast<int>(r - 1)
            : find_row_by_y(row_keys, num_rows, y - 1);

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
// CSR-friendly stencil helpers
// ---------------------------------------------------------------------------

template <typename T>
struct CsrStencilPoint {
  typename Field2DDevice<T>::ValueView values;
  std::size_t idx_center = 0;
  std::size_t idx_west = 0;
  std::size_t idx_east = 0;
  std::size_t idx_south = 0;
  std::size_t idx_north = 0;

  KOKKOS_INLINE_FUNCTION
  T center() const { return values(idx_center); }
  KOKKOS_INLINE_FUNCTION
  T west() const { return values(idx_west); }
  KOKKOS_INLINE_FUNCTION
  T east() const { return values(idx_east); }
  KOKKOS_INLINE_FUNCTION
  T south() const { return values(idx_south); }
  KOKKOS_INLINE_FUNCTION
  T north() const { return values(idx_north); }
};

namespace detail {

template <typename T>
struct SubsetStencilVerticalMapping {
  Kokkos::View<int*, DeviceMemorySpace> north_interval;
  Kokkos::View<int*, DeviceMemorySpace> south_interval;
};

template <typename T, typename Mapping>
inline SubsetStencilVerticalMapping<T>
build_subset_stencil_vertical_mapping(
    const Field2DDevice<T>& field,
    const IntervalSet2DDevice& subset,
    const Mapping& mapping) {
  SubsetStencilVerticalMapping<T> out;
  const std::size_t n_intervals = subset.num_intervals;
  out.north_interval = Kokkos::View<int*, DeviceMemorySpace>(
      "subsetix_subset_stencil_north", n_intervals);
  out.south_interval = Kokkos::View<int*, DeviceMemorySpace>(
      "subsetix_subset_stencil_south", n_intervals);

  if (n_intervals == 0 || subset.num_rows == 0 ||
      field.geometry.num_rows == 0 || field.geometry.num_intervals == 0) {
    return out;
  }

  Kokkos::deep_copy(out.north_interval, -1);
  Kokkos::deep_copy(out.south_interval, -1);

  auto subset_row_keys = subset.row_keys;
  auto subset_intervals = subset.intervals;

  auto field_row_keys = field.geometry.row_keys;
  auto field_row_ptr = field.geometry.row_ptr;
  auto field_intervals = field.geometry.intervals;

  auto interval_to_row = mapping.interval_to_row;
  auto row_map = mapping.row_map;

  const std::size_t num_field_rows = field.geometry.num_rows;

  Kokkos::parallel_for(
      "subsetix_build_subset_stencil_vertical_mapping",
      Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(n_intervals)),
      KOKKOS_LAMBDA(const int interval_idx) {
        const int subset_row = interval_to_row(interval_idx);
        const int field_row = row_map(subset_row);
        if (subset_row < 0 || field_row < 0) {
          return;
        }

        const Coord y = subset_row_keys(subset_row).y;
        const auto mask_iv = subset_intervals(interval_idx);

        auto find_interval_containing = [&](int row_idx) -> int {
          const std::size_t begin = field_row_ptr(row_idx);
          const std::size_t end = field_row_ptr(row_idx + 1);
          for (std::size_t k = begin; k < end; ++k) {
            const Interval iv = field_intervals(k);
            if (mask_iv.begin >= iv.begin && mask_iv.end <= iv.end) {
              return static_cast<int>(k);
            }
          }
          return -1;
        };

        const int row_up = find_row_by_y(field_row_keys, num_field_rows, y + 1);
        if (row_up >= 0) {
          const int up_interval = find_interval_containing(row_up);
          if (up_interval >= 0) {
            out.north_interval(interval_idx) = up_interval;
          }
        }

        const int row_down = find_row_by_y(field_row_keys, num_field_rows, y - 1);
        if (row_down >= 0) {
          const int down_interval = find_interval_containing(row_down);
          if (down_interval >= 0) {
            out.south_interval(interval_idx) = down_interval;
          }
        }
      });
  ExecSpace().fence();
  return out;
}

} // namespace detail

template <typename T>
struct FieldStencilMapping {
  FieldMaskMapping mask;
  detail::SubsetStencilVerticalMapping<T> vertical;
};

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
    const FieldMaskMapping& mapping,
    const detail::VerticalIntervalMapping& vertical,
    StencilFunctor stencil) {
  if (mask.num_rows == 0 || mask.num_intervals == 0) {
    return;
  }

  if (field_out.geometry.num_rows == 0 || field_in.geometry.num_rows == 0) {
    return;
  }

  auto interval_to_row = mapping.interval_to_row;
  auto interval_to_field_interval =
      mapping.interval_to_field_interval;
  auto row_map = mapping.row_map;

  auto mask_row_keys = mask.row_keys;
  auto mask_intervals = mask.intervals;
  auto field_out_intervals = field_out.geometry.intervals;
  auto field_out_offsets = field_out.geometry.cell_offsets;
  auto field_out_values = field_out.values;

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
        (void)field_row;

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
    return;
  }

  const auto mapping =
      detail::build_mask_field_mapping(field_out, mask);
  const detail::VerticalIntervalMapping vertical =
      detail::build_vertical_interval_mapping(
          field_in);
  apply_stencil_on_set_device(field_out, field_in, mask, mapping, vertical,
                              stencil);
}

// ---------------------------------------------------------------------------
// CSR-friendly stencils restricted to a set/subset
// ---------------------------------------------------------------------------

/**
 * @brief Apply a 5-point stencil on a masked region with CSR-friendly
 *        neighbour localisation.
 *
 * Functor signature:
 *   KOKKOS_INLINE_FUNCTION
 *   T operator()(Coord x, Coord y,
 *                const CsrStencilPoint<T>& p) const;
 *
 * Precondition: the mask must be an interior region such that all four
 * neighbours of every cell exist in the input field geometry.
 */
template <typename OutT, typename InT, class StencilFunctor>
inline void apply_csr_stencil_on_set_device(
    Field2DDevice<OutT>& field_out,
    const Field2DDevice<InT>& field_in,
    const IntervalSet2DDevice& mask,
    const FieldMaskMapping& mapping_out,
    const FieldMaskMapping& mapping_in,
    const detail::SubsetStencilVerticalMapping<InT>& vertical,
    StencilFunctor stencil,
    bool strict_check = false) {
  if (mask.num_rows == 0 || mask.num_intervals == 0) {
    return;
  }

  if (field_out.geometry.num_rows == 0 || field_in.geometry.num_rows == 0) {
    return;
  }

  (void)strict_check;

  auto interval_to_row_out = mapping_out.interval_to_row;
  auto interval_to_field_interval_out =
      mapping_out.interval_to_field_interval;
  auto row_map_out = mapping_out.row_map;

  auto interval_to_row_in = mapping_in.interval_to_row;
  auto interval_to_field_interval_in =
      mapping_in.interval_to_field_interval;
  auto row_map_in = mapping_in.row_map;

  auto mask_row_keys = mask.row_keys;
  auto mask_intervals = mask.intervals;

  auto out_intervals = field_out.geometry.intervals;
  auto out_offsets = field_out.geometry.cell_offsets;
  auto in_intervals = field_in.geometry.intervals;
  auto in_offsets = field_in.geometry.cell_offsets;
  auto values_in = field_in.values;
  auto values_out = field_out.values;

  auto north_interval = vertical.north_interval;
  auto south_interval = vertical.south_interval;

  using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
  using MemberType = TeamPolicy::member_type;

  const TeamPolicy policy(
      static_cast<int>(mask.num_intervals), Kokkos::AUTO);

  Kokkos::parallel_for(
      "subsetix_apply_csr_stencil_on_set_device",
      policy,
      KOKKOS_LAMBDA(const MemberType& team) {
        const int interval_idx = team.league_rank();

        const int row_idx_out = interval_to_row_out(interval_idx);
        const int field_row_out = row_map_out(row_idx_out);
        if (row_idx_out < 0 || field_row_out < 0) {
          return;
        }

        const int out_interval_idx =
            interval_to_field_interval_out(interval_idx);
        if (out_interval_idx < 0) {
          return;
        }

        const auto mask_iv = mask_intervals(interval_idx);
        const auto out_iv =
            out_intervals(out_interval_idx);
        const Coord y = mask_row_keys(row_idx_out).y;
        const std::size_t out_base =
            out_offsets(out_interval_idx);
        const Coord out_begin = out_iv.begin;

        // Input mapping
        const int row_idx_in = interval_to_row_in(interval_idx);
        const int field_row_in = row_map_in(row_idx_in);
        if (row_idx_in < 0 || field_row_in < 0) {
          return;
        }

        const int in_interval_idx =
            interval_to_field_interval_in(interval_idx);
        if (in_interval_idx < 0) {
          return;
        }

        const auto in_iv =
            in_intervals(in_interval_idx);
        const std::size_t in_base =
            in_offsets(in_interval_idx);
        const Coord in_begin = in_iv.begin;

        const int north_iv_idx = north_interval(interval_idx);
        const int south_iv_idx = south_interval(interval_idx);

        const auto north_iv = (north_iv_idx >= 0)
                                  ? in_intervals(north_iv_idx)
                                  : in_iv;
        const auto south_iv = (south_iv_idx >= 0)
                                  ? in_intervals(south_iv_idx)
                                  : in_iv;
        const std::size_t north_base =
            (north_iv_idx >= 0) ? in_offsets(north_iv_idx) : in_base;
        const std::size_t south_base =
            (south_iv_idx >= 0) ? in_offsets(south_iv_idx) : in_base;

        const Coord x_begin = mask_iv.begin;
        const Coord x_end = mask_iv.end;

        const int team_size = team.team_size();
        const int team_rank = team.team_rank();

        for (Coord x = static_cast<Coord>(x_begin + team_rank);
             x < x_end;
             x += static_cast<Coord>(team_size)) {
          const std::size_t idx_center_in =
              in_base +
              static_cast<std::size_t>(x - in_begin);
          // Sub-geometry is assumed to exclude boundaries, so idx±1 are valid.
          const std::size_t idx_west = idx_center_in - 1;
          const std::size_t idx_east = idx_center_in + 1;
          const std::size_t idx_north =
              north_base +
              static_cast<std::size_t>(x - north_iv.begin);
          const std::size_t idx_south =
              south_base +
              static_cast<std::size_t>(x - south_iv.begin);

          CsrStencilPoint<InT> p;
          p.values = values_in;
          p.idx_center = idx_center_in;
          p.idx_west = idx_west;
          p.idx_east = idx_east;
          p.idx_south = idx_south;
          p.idx_north = idx_north;

          const OutT out_val = stencil(x, y, p);
          const std::size_t idx_out =
              out_base +
              static_cast<std::size_t>(x - out_begin);
          values_out(idx_out) = out_val;
        }
      });
  ExecSpace().fence();
}

template <typename OutT, typename InT, class StencilFunctor>
inline void apply_csr_stencil_on_set_device(
    Field2DDevice<OutT>& field_out,
    const Field2DDevice<InT>& field_in,
    const IntervalSet2DDevice& mask,
    StencilFunctor stencil,
    bool strict_check = false) {
  const auto mapping_out =
      detail::build_mask_field_mapping(field_out, mask);
  const auto mapping_in =
      detail::build_mask_field_mapping(field_in, mask);
  const auto vertical = detail::build_subset_stencil_vertical_mapping(
      field_in, mask, mapping_in);
  apply_csr_stencil_on_set_device(field_out, field_in, mask,
                                  mapping_out, mapping_in, vertical,
                                  stencil, strict_check);
}

// ---------------------------------------------------------------------------
// Subset-based stencil operations
// ---------------------------------------------------------------------------

template <typename T, class StencilFunctor>
inline void apply_stencil_on_subset_device(
    Field2DDevice<T>& field_out,
    const Field2DDevice<T>& field_in,
    const IntervalSubSet2DDevice& subset,
    StencilFunctor stencil) {
  if (!subset.valid()) return;
  if (field_out.geometry.num_rows == 0 || field_in.geometry.num_rows == 0) return;

  auto mask_indices = subset.interval_indices;
  auto mask_x_begin = subset.x_begin;
  auto mask_x_end = subset.x_end;
  auto mask_rows = subset.row_indices;

  auto row_keys = field_out.geometry.row_keys;
  auto intervals = field_out.geometry.intervals;
  auto offsets = field_out.geometry.cell_offsets;
  auto values_out = field_out.values;
  auto src_rows = field_in.geometry.row_keys;
  auto src_row_ptr = field_in.geometry.row_ptr;
  auto src_intervals = field_in.geometry.intervals;
  auto src_offsets = field_in.geometry.cell_offsets;
  const std::size_t src_num_rows = field_in.geometry.num_rows;

  const detail::VerticalIntervalMapping vertical =
      detail::build_vertical_interval_mapping(field_in);

  detail::FieldStencilContext<T> ctx;
  ctx.intervals = field_in.geometry.intervals;
  ctx.offsets = field_in.geometry.cell_offsets;
  ctx.values = field_in.values;
  ctx.up_interval = vertical.up_interval;
  ctx.down_interval = vertical.down_interval;

  using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
  using MemberType = TeamPolicy::member_type;
  const TeamPolicy policy(static_cast<int>(subset.num_entries), Kokkos::AUTO);

  Kokkos::parallel_for(
      "subsetix_apply_stencil_on_subset_device", policy,
      KOKKOS_LAMBDA(const MemberType& team) {
        const int entry_idx = team.league_rank();
        const std::size_t interval_idx = mask_indices(entry_idx);
        const int row_idx = mask_rows(entry_idx);
        const Coord y = row_keys(row_idx).y;
        const Interval iv = intervals(interval_idx);
        const std::size_t base_offset = offsets(interval_idx);
        const Coord base_begin = iv.begin;
        const Coord xb = mask_x_begin(entry_idx);
        const Coord xe = mask_x_end(entry_idx);

        const int src_row_idx =
            detail::find_row_index(src_rows, src_num_rows, y);
        if (src_row_idx < 0) return;

        const std::size_t src_row_begin = src_row_ptr(src_row_idx);
        const std::size_t src_row_end = src_row_ptr(src_row_idx + 1);
        if (src_row_begin == src_row_end) return;

        int src_interval_idx =
            detail::find_interval_index(src_intervals, src_row_begin,
                                        src_row_end, xb);
        if (src_interval_idx < 0) return;

        Interval src_iv = src_intervals(src_interval_idx);
        std::size_t src_base = src_offsets(src_interval_idx);
        Coord src_begin = src_iv.begin;

        const int team_size = team.team_size();
        const int team_rank = team.team_rank();

        for (Coord x = static_cast<Coord>(xb + team_rank); x < xe;
             x += static_cast<Coord>(team_size)) {
          while (x >= src_iv.end) {
            ++src_interval_idx;
            if (src_interval_idx >= static_cast<int>(src_row_end)) return;
            src_iv = src_intervals(src_interval_idx);
            src_base = src_offsets(src_interval_idx);
            src_begin = src_iv.begin;
          }

          const std::size_t dst_linear_index =
              base_offset + static_cast<std::size_t>(x - base_begin);
          const std::size_t src_linear_index =
              src_base + static_cast<std::size_t>(x - src_begin);

          const T value = stencil(x, y, src_linear_index, src_interval_idx, ctx);
          values_out(dst_linear_index) = value;
        }
      });
  ExecSpace().fence();
}

template <typename T, class StencilFunctor>
inline void apply_stencil_on_subset_device(
    Field2DDevice<T>& field_out,
    const Field2DDevice<T>& field_in,
    const IntervalSet2DDevice& mask,
    StencilFunctor stencil,
    CsrSetAlgebraContext* ctx = nullptr) {
  if (mask.num_rows == 0 || mask.num_intervals == 0) return;
  IntervalSubSet2DDevice subset;
  build_interval_subset_device(field_out.geometry, mask, subset, ctx);
  apply_stencil_on_subset_device(field_out, field_in, subset, stencil);
}

} // namespace csr
} // namespace subsetix
