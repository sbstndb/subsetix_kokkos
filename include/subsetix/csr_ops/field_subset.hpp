#pragma once

#include <subsetix/field/csr_field.hpp>
#include <subsetix/geometry/csr_interval_subset.hpp>
#include <subsetix/csr_ops/workspace.hpp>

namespace subsetix {
namespace csr {

// ---------------------------------------------------------------------------
// Core subset operations (single kernel implementation)
// ---------------------------------------------------------------------------

/**
 * @brief Apply a user functor on all cells of a field restricted by a subset.
 *
 * Functor signature:
 *   KOKKOS_INLINE_FUNCTION
 *   void operator()(Coord x, Coord y,
 *                   typename Field2DDevice<T>::ValueView::reference_type value,
 *                   std::size_t linear_index) const;
 */
template <typename T, class Functor>
inline void apply_on_subset_device(Field2DDevice<T>& field,
                                   const IntervalSubSet2DDevice& subset,
                                   Functor func) {
  if (!subset.valid()) return;

  auto intervals = field.geometry.intervals;
  auto offsets = field.geometry.cell_offsets;
  auto row_keys = field.geometry.row_keys;
  auto values = field.values;

  auto subset_indices = subset.interval_indices;
  auto subset_x_begin = subset.x_begin;
  auto subset_x_end = subset.x_end;
  auto subset_rows = subset.row_indices;

  using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
  using MemberType = TeamPolicy::member_type;

  const TeamPolicy policy(static_cast<int>(subset.num_entries), Kokkos::AUTO);

  Kokkos::parallel_for(
      "subsetix_apply_on_subset_device", policy,
      KOKKOS_LAMBDA(const MemberType& team) {
        const int entry_idx = team.league_rank();
        const std::size_t interval_idx = subset_indices(entry_idx);
        const int row_idx = subset_rows(entry_idx);
        const Coord y = row_keys(row_idx).y;
        const Interval iv = intervals(interval_idx);
        const std::size_t base_offset = offsets(interval_idx);
        const Coord base_begin = iv.begin;
        const Coord xb = subset_x_begin(entry_idx);
        const Coord xe = subset_x_end(entry_idx);

        const int team_size = team.team_size();
        const int team_rank = team.team_rank();

        for (Coord x = static_cast<Coord>(xb + team_rank); x < xe;
             x += static_cast<Coord>(team_size)) {
          const std::size_t linear_index =
              base_offset + static_cast<std::size_t>(x - base_begin);
          func(x, y, values(linear_index), linear_index);
        }
      });
  ExecSpace().fence();
}

/** @brief Convenience overload: build subset from Set on-the-fly. */
template <typename T, class Functor>
inline void apply_on_subset_device(Field2DDevice<T>& field,
                                   const IntervalSet2DDevice& mask,
                                   Functor func,
                                   CsrSetAlgebraContext* ctx = nullptr) {
  if (mask.num_rows == 0 || mask.num_intervals == 0) return;
  IntervalSubSet2DDevice subset;
  build_interval_subset_device(field.geometry, mask, subset, ctx);
  apply_on_subset_device(field, subset, func);
}

// ---------------------------------------------------------------------------
// Fill
// ---------------------------------------------------------------------------

template <typename T>
inline void fill_on_subset_device(Field2DDevice<T>& field,
                                  const IntervalSubSet2DDevice& subset,
                                  const T& value) {
  const T value_copy = value;
  apply_on_subset_device(
      field, subset,
      KOKKOS_LAMBDA(Coord, Coord,
          typename Field2DDevice<T>::ValueView::reference_type cell,
          std::size_t) { cell = value_copy; });
}

template <typename T>
inline void fill_on_subset_device(Field2DDevice<T>& field,
                                  const IntervalSet2DDevice& mask,
                                  const T& value,
                                  CsrSetAlgebraContext* ctx = nullptr) {
  if (mask.num_rows == 0 || mask.num_intervals == 0) return;
  IntervalSubSet2DDevice subset;
  build_interval_subset_device(field.geometry, mask, subset, ctx);
  fill_on_subset_device(field, subset, value);
}

// ---------------------------------------------------------------------------
// Scale
// ---------------------------------------------------------------------------

template <typename T>
inline void scale_on_subset_device(Field2DDevice<T>& field,
                                   const IntervalSubSet2DDevice& subset,
                                   const T& alpha) {
  const T alpha_copy = alpha;
  apply_on_subset_device(
      field, subset,
      KOKKOS_LAMBDA(Coord, Coord,
          typename Field2DDevice<T>::ValueView::reference_type cell,
          std::size_t) { cell *= alpha_copy; });
}

template <typename T>
inline void scale_on_subset_device(Field2DDevice<T>& field,
                                   const IntervalSet2DDevice& mask,
                                   const T& alpha,
                                   CsrSetAlgebraContext* ctx = nullptr) {
  if (mask.num_rows == 0 || mask.num_intervals == 0) return;
  IntervalSubSet2DDevice subset;
  build_interval_subset_device(field.geometry, mask, subset, ctx);
  scale_on_subset_device(field, subset, alpha);
}

// ---------------------------------------------------------------------------
// Copy (subset kernel requires detail helpers, defined after field_stencil)
// ---------------------------------------------------------------------------

namespace detail {

template <class RowKeyView>
KOKKOS_INLINE_FUNCTION
int find_row_index_subset(const RowKeyView& row_keys,
                          std::size_t num_rows, Coord y) {
  std::size_t left = 0;
  std::size_t right = num_rows;
  while (left < right) {
    const std::size_t mid = left + (right - left) / 2;
    const Coord current = row_keys(mid).y;
    if (current == y) return static_cast<int>(mid);
    if (current < y) left = mid + 1;
    else right = mid;
  }
  return -1;
}

template <class IntervalView>
KOKKOS_INLINE_FUNCTION
int find_interval_index_subset(const IntervalView& intervals,
                               std::size_t begin, std::size_t end, Coord x) {
  std::size_t left = begin;
  std::size_t right = end;
  while (left < right) {
    const std::size_t mid = left + (right - left) / 2;
    const auto iv = intervals(mid);
    if (x < iv.begin) right = mid;
    else if (x >= iv.end) left = mid + 1;
    else return static_cast<int>(mid);
  }
  return -1;
}

} // namespace detail

template <typename T>
inline void copy_on_subset_device(Field2DDevice<T>& dst,
                                  const Field2DDevice<T>& src,
                                  const IntervalSubSet2DDevice& subset) {
  if (!subset.valid()) return;

  auto dst_values = dst.values;
  auto dst_intervals = dst.geometry.intervals;
  auto dst_offsets = dst.geometry.cell_offsets;
  auto dst_rows = dst.geometry.row_keys;

  auto src_values = src.values;
  auto src_intervals = src.geometry.intervals;
  auto src_offsets = src.geometry.cell_offsets;
  auto src_rows = src.geometry.row_keys;
  auto src_row_ptr = src.geometry.row_ptr;
  const std::size_t src_num_rows = src.geometry.num_rows;

  auto subset_indices = subset.interval_indices;
  auto subset_x_begin = subset.x_begin;
  auto subset_x_end = subset.x_end;
  auto subset_rows = subset.row_indices;

  using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
  using MemberType = TeamPolicy::member_type;

  const TeamPolicy policy(static_cast<int>(subset.num_entries), Kokkos::AUTO);

  Kokkos::parallel_for(
      "subsetix_copy_on_subset_device", policy,
      KOKKOS_LAMBDA(const MemberType& team) {
        const int entry_idx = team.league_rank();
        const std::size_t interval_idx = subset_indices(entry_idx);
        const Interval dst_iv = dst_intervals(interval_idx);
        const std::size_t dst_base = dst_offsets(interval_idx);
        const Coord dst_begin = dst_iv.begin;
        const Coord xb = subset_x_begin(entry_idx);
        const Coord xe = subset_x_end(entry_idx);
        const int dst_row_idx = subset_rows(entry_idx);
        const Coord y = dst_rows(dst_row_idx).y;

        const int src_row_idx =
            detail::find_row_index_subset(src_rows, src_num_rows, y);
        if (src_row_idx < 0) return;

        const std::size_t src_row_begin = src_row_ptr(src_row_idx);
        const std::size_t src_row_end = src_row_ptr(src_row_idx + 1);
        if (src_row_begin == src_row_end) return;

        int src_interval_idx0 = detail::find_interval_index_subset(
            src_intervals, src_row_begin, src_row_end, xb);
        if (src_interval_idx0 < 0) return;

        const int team_size = team.team_size();
        const int team_rank = team.team_rank();

        for (Coord x = static_cast<Coord>(xb + team_rank); x < xe;
             x += static_cast<Coord>(team_size)) {
          int src_interval_idx = src_interval_idx0;
          Interval src_iv = src_intervals(src_interval_idx);
          std::size_t src_base = src_offsets(src_interval_idx);
          Coord src_begin = src_iv.begin;

          while (x >= src_iv.end) {
            ++src_interval_idx;
            if (src_interval_idx >= static_cast<int>(src_row_end)) return;
            src_iv = src_intervals(src_interval_idx);
            src_base = src_offsets(src_interval_idx);
            src_begin = src_iv.begin;
          }

          const std::size_t dst_idx =
              dst_base + static_cast<std::size_t>(x - dst_begin);
          const std::size_t src_idx =
              src_base + static_cast<std::size_t>(x - src_begin);
          dst_values(dst_idx) = src_values(src_idx);
        }
      });
  ExecSpace().fence();
}

template <typename T>
inline void copy_on_subset_device(Field2DDevice<T>& dst,
                                  const Field2DDevice<T>& src,
                                  const IntervalSet2DDevice& mask,
                                  CsrSetAlgebraContext* ctx = nullptr) {
  if (mask.num_rows == 0 || mask.num_intervals == 0) return;
  IntervalSubSet2DDevice subset;
  build_interval_subset_device(dst.geometry, mask, subset, ctx);
  copy_on_subset_device(dst, src, subset);
}

// ---------------------------------------------------------------------------
// Backward-compatible aliases: _on_set_ -> _on_subset_
// ---------------------------------------------------------------------------

template <typename T, class Functor>
inline void apply_on_set_device(Field2DDevice<T>& field,
                                const IntervalSet2DDevice& mask,
                                Functor func) {
  apply_on_subset_device(field, mask, func, nullptr);
}

template <typename T>
inline void fill_on_set_device(Field2DDevice<T>& field,
                               const IntervalSet2DDevice& mask,
                               const T& value) {
  fill_on_subset_device(field, mask, value, nullptr);
}

template <typename T>
inline void scale_on_set_device(Field2DDevice<T>& field,
                                const IntervalSet2DDevice& mask,
                                const T& alpha) {
  scale_on_subset_device(field, mask, alpha, nullptr);
}

template <typename T>
inline void copy_on_set_device(Field2DDevice<T>& dst,
                               const Field2DDevice<T>& src,
                               const IntervalSet2DDevice& mask) {
  copy_on_subset_device(dst, src, mask, nullptr);
}

} // namespace csr
} // namespace subsetix
