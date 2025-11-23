#pragma once

#include <subsetix/csr_field.hpp>
#include <subsetix/csr_interval_subset.hpp>
#include <subsetix/csr_ops/field_core.hpp>
#include <subsetix/csr_ops/field_stencil.hpp>
#include <subsetix/csr_ops/field_amr.hpp>

namespace subsetix {
namespace csr {

using ExecSpace = Kokkos::DefaultExecutionSpace;

template <typename T, class Functor>
inline void apply_on_subset_device(Field2DDevice<T>& field,
                                   const IntervalSubSet2DDevice& subset,
                                   Functor func) {
  if (!subset.valid()) {
    return;
  }

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

  const TeamPolicy policy(
      static_cast<int>(subset.num_entries), Kokkos::AUTO);

  Kokkos::parallel_for(
      "subsetix_apply_on_subset_device",
      policy,
      KOKKOS_LAMBDA(const MemberType& team) {
        const int entry_idx = team.league_rank();
        const std::size_t interval_idx =
            subset_indices(entry_idx);
        const int row_idx = subset_rows(entry_idx);
        const Coord y = row_keys(row_idx).y;
        const Interval iv = intervals(interval_idx);
        const std::size_t base_offset =
            offsets(interval_idx);
        const Coord base_begin = iv.begin;
        const Coord xb = subset_x_begin(entry_idx);
        const Coord xe = subset_x_end(entry_idx);

        const int team_size = team.team_size();
        const int team_rank = team.team_rank();

        for (Coord x = static_cast<Coord>(xb + team_rank);
             x < xe;
             x += static_cast<Coord>(team_size)) {
          const std::size_t linear_index =
              base_offset +
              static_cast<std::size_t>(x - base_begin);
          func(x, y, values(linear_index), linear_index);
        }
      });
  ExecSpace().fence();
}

template <typename T>
inline void fill_on_subset_device(Field2DDevice<T>& field,
                                  const IntervalSubSet2DDevice& subset,
                                  const T& value) {
  const T value_copy = value;
  apply_on_subset_device(
      field, subset,
      KOKKOS_LAMBDA(
          Coord /*x*/, Coord /*y*/,
          typename Field2DDevice<T>::ValueView::reference_type cell,
          std::size_t /*idx*/) { cell = value_copy; });
}

template <typename T>
inline void scale_on_subset_device(Field2DDevice<T>& field,
                                   const IntervalSubSet2DDevice& subset,
                                   const T& alpha) {
  const T alpha_copy = alpha;
  apply_on_subset_device(
      field, subset,
      KOKKOS_LAMBDA(
          Coord /*x*/, Coord /*y*/,
          typename Field2DDevice<T>::ValueView::reference_type cell,
          std::size_t /*idx*/) { cell *= alpha_copy; });
}

template <typename T>
inline void copy_on_subset_device(Field2DDevice<T>& dst,
                                  const Field2DDevice<T>& src,
                                  const IntervalSubSet2DDevice& subset) {
  if (!subset.valid()) {
    return;
  }

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

  const TeamPolicy policy(
      static_cast<int>(subset.num_entries), Kokkos::AUTO);

  Kokkos::parallel_for(
      "subsetix_copy_on_subset_device",
      policy,
      KOKKOS_LAMBDA(const MemberType& team) {
        const int entry_idx = team.league_rank();

        const std::size_t interval_idx =
            subset_indices(entry_idx);
        const Interval dst_iv =
            dst_intervals(interval_idx);
        const std::size_t dst_base =
            dst_offsets(interval_idx);
        const Coord dst_begin = dst_iv.begin;
        const Coord xb = subset_x_begin(entry_idx);
        const Coord xe = subset_x_end(entry_idx);
        const int dst_row_idx = subset_rows(entry_idx);
        const Coord y = dst_rows(dst_row_idx).y;

        const int src_row_idx =
            detail::find_row_index(src_rows, src_num_rows, y);
        if (src_row_idx < 0) {
          return;
        }
        const std::size_t src_row_begin = src_row_ptr(src_row_idx);
        const std::size_t src_row_end = src_row_ptr(src_row_idx + 1);
        if (src_row_begin == src_row_end) {
          return;
        }

        // Initial interval in src row that contains xb, reused by all team threads.
        int src_interval_idx0 =
            detail::find_interval_index(src_intervals, src_row_begin,
                                        src_row_end, xb);
        if (src_interval_idx0 < 0) {
          return;
        }

        const int team_size = team.team_size();
        const int team_rank = team.team_rank();

        for (Coord x = static_cast<Coord>(xb + team_rank);
             x < xe;
             x += static_cast<Coord>(team_size)) {
          int src_interval_idx = src_interval_idx0;
          Interval src_iv = src_intervals(src_interval_idx);
          std::size_t src_base = src_offsets(src_interval_idx);
          Coord src_begin = src_iv.begin;

          while (x >= src_iv.end) {
            ++src_interval_idx;
            if (src_interval_idx >= static_cast<int>(src_row_end)) {
              return;
            }
            src_iv = src_intervals(src_interval_idx);
            src_base = src_offsets(src_interval_idx);
            src_begin = src_iv.begin;
          }

          const std::size_t dst_idx =
              dst_base +
              static_cast<std::size_t>(x - dst_begin);
          const std::size_t src_idx =
              src_base +
              static_cast<std::size_t>(x - src_begin);
          dst_values(dst_idx) = src_values(src_idx);
        }
      });
  ExecSpace().fence();
}

template <typename T, class StencilFunctor>
inline void apply_stencil_on_subset_device(
    Field2DDevice<T>& field_out,
    const Field2DDevice<T>& field_in,
    const IntervalSubSet2DDevice& subset,
    StencilFunctor stencil) {
  if (!subset.valid()) {
    return;
  }

  if (field_out.geometry.num_rows == 0 ||
      field_in.geometry.num_rows == 0) {
    throw std::runtime_error(
        "fields must be initialized before applying a stencil");
  }

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

  const TeamPolicy policy(
      static_cast<int>(subset.num_entries), Kokkos::AUTO);

  Kokkos::parallel_for(
      "subsetix_apply_stencil_on_subset_device",
      policy,
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
        if (src_row_idx < 0) {
          return;
        }
        const std::size_t src_row_begin = src_row_ptr(src_row_idx);
        const std::size_t src_row_end = src_row_ptr(src_row_idx + 1);
        if (src_row_begin == src_row_end) {
          return;
        }

        int src_interval_idx =
            detail::find_interval_index(src_intervals, src_row_begin,
                                        src_row_end, xb);
        if (src_interval_idx < 0) {
          return;
        }

        Interval src_iv = src_intervals(src_interval_idx);
        std::size_t src_base = src_offsets(src_interval_idx);
        Coord src_begin = src_iv.begin;

        const int team_size = team.team_size();
        const int team_rank = team.team_rank();

        for (Coord x = static_cast<Coord>(xb + team_rank);
             x < xe;
             x += static_cast<Coord>(team_size)) {
          while (x >= src_iv.end) {
            ++src_interval_idx;
            if (src_interval_idx >= static_cast<int>(src_row_end)) {
              return;
            }
            src_iv = src_intervals(src_interval_idx);
            src_base = src_offsets(src_interval_idx);
            src_begin = src_iv.begin;
          }

          const std::size_t dst_linear_index =
              base_offset +
              static_cast<std::size_t>(x - base_begin);
          const std::size_t src_linear_index =
              src_base +
              static_cast<std::size_t>(x - src_begin);

          const T value =
              stencil(x, y, src_linear_index,
                      src_interval_idx, ctx);
          values_out(dst_linear_index) = value;
        }
      });
  ExecSpace().fence();
}

template <typename T>
inline void restrict_field_on_subset_device(
    Field2DDevice<T>& coarse_field,
    const Field2DDevice<T>& fine_field,
    const IntervalSubSet2DDevice& coarse_subset) {
  if (!coarse_subset.valid()) {
    return;
  }

  const detail::AmrIntervalMapping mapping =
      detail::build_amr_interval_mapping(coarse_field,
                                         fine_field);

  auto coarse_intervals = coarse_field.geometry.intervals;
  auto coarse_offsets = coarse_field.geometry.cell_offsets;
  auto coarse_values = coarse_field.values;

  auto fine_intervals = fine_field.geometry.intervals;
  auto fine_offsets = fine_field.geometry.cell_offsets;
  auto fine_values = fine_field.values;

  auto subset_indices = coarse_subset.interval_indices;
  auto subset_x_begin = coarse_subset.x_begin;
  auto subset_x_end = coarse_subset.x_end;

  auto coarse_to_fine_first = mapping.coarse_to_fine_first;
  auto coarse_to_fine_second = mapping.coarse_to_fine_second;

  using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
  using MemberType = TeamPolicy::member_type;

  const TeamPolicy policy(
      static_cast<int>(coarse_subset.num_entries), Kokkos::AUTO);

  Kokkos::parallel_for(
      "subsetix_restrict_field_on_subset_device",
      policy,
      KOKKOS_LAMBDA(const MemberType& team) {
        const int entry_idx = team.league_rank();
        const std::size_t coarse_interval_idx =
            subset_indices(entry_idx);
        const int fine_interval_idx0 =
            coarse_to_fine_first(coarse_interval_idx);
        const int fine_interval_idx1 =
            coarse_to_fine_second(coarse_interval_idx);

        if (fine_interval_idx0 < 0 || fine_interval_idx1 < 0) {
          return;
        }

        const Interval coarse_iv =
            coarse_intervals(coarse_interval_idx);
        const std::size_t coarse_base =
            coarse_offsets(coarse_interval_idx);
        const Coord coarse_begin = coarse_iv.begin;

        const Interval fine_iv0 =
            fine_intervals(fine_interval_idx0);
        const Interval fine_iv1 =
            fine_intervals(fine_interval_idx1);
        const std::size_t fine_base0 =
            fine_offsets(fine_interval_idx0);
        const std::size_t fine_base1 =
            fine_offsets(fine_interval_idx1);
        const Coord fine_begin0 = fine_iv0.begin;
        const Coord fine_begin1 = fine_iv1.begin;

        const Coord xb = subset_x_begin(entry_idx);
        const Coord xe = subset_x_end(entry_idx);

        const int team_size = team.team_size();
        const int team_rank = team.team_rank();

        for (Coord x = static_cast<Coord>(xb + team_rank);
             x < xe;
             x += static_cast<Coord>(team_size)) {
          const std::size_t coarse_index =
              coarse_base +
              static_cast<std::size_t>(x - coarse_begin);

          const Coord fine_x0 =
              static_cast<Coord>(2 * x);
          const Coord fine_x1 =
              static_cast<Coord>(2 * x + 1);

          const std::size_t fine_idx00 =
              fine_base0 +
              static_cast<std::size_t>(fine_x0 - fine_begin0);
          const std::size_t fine_idx01 =
              fine_base0 +
              static_cast<std::size_t>(fine_x1 - fine_begin0);
          const std::size_t fine_idx10 =
              fine_base1 +
              static_cast<std::size_t>(fine_x0 - fine_begin1);
          const std::size_t fine_idx11 =
              fine_base1 +
              static_cast<std::size_t>(fine_x1 - fine_begin1);

          const T v00 = fine_values(fine_idx00);
          const T v01 = fine_values(fine_idx01);
          const T v10 = fine_values(fine_idx10);
          const T v11 = fine_values(fine_idx11);

          coarse_values(coarse_index) =
              static_cast<T>(0.25) *
              (v00 + v01 + v10 + v11);
        }
      });
  ExecSpace().fence();
}

template <typename T, typename Reconstructor>
inline void prolong_field_on_subset_generic_device(
    Field2DDevice<T>& fine_field,
    const Field2DDevice<T>& coarse_field,
    const IntervalSubSet2DDevice& fine_subset,
    Reconstructor reconstructor) {
  if (!fine_subset.valid()) {
    return;
  }

  const detail::AmrIntervalMapping mapping =
      detail::build_amr_interval_mapping(coarse_field,
                                         fine_field);

  auto fine_intervals = fine_field.geometry.intervals;
  auto fine_offsets = fine_field.geometry.cell_offsets;
  auto fine_values = fine_field.values;
  auto fine_rows = fine_field.geometry.row_keys;

  auto coarse_intervals = coarse_field.geometry.intervals;
  auto coarse_offsets = coarse_field.geometry.cell_offsets;
  auto coarse_values = coarse_field.values;

  auto subset_indices = fine_subset.interval_indices;
  auto subset_rows = fine_subset.row_indices;
  auto subset_x_begin = fine_subset.x_begin;
  auto subset_x_end = fine_subset.x_end;

  auto fine_to_coarse = mapping.fine_to_coarse;

  using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
  using MemberType = TeamPolicy::member_type;

  const TeamPolicy policy(
      static_cast<int>(fine_subset.num_entries), Kokkos::AUTO);

  Kokkos::parallel_for(
      "subsetix_prolong_field_on_subset_generic",
      policy,
      KOKKOS_LAMBDA(const MemberType& team) {
        const int entry_idx = team.league_rank();
        const std::size_t fine_interval_idx =
            subset_indices(entry_idx);
        const int coarse_interval_idx =
            fine_to_coarse(fine_interval_idx);
        if (coarse_interval_idx < 0) {
          return;
        }

        const Interval fine_iv =
            fine_intervals(fine_interval_idx);
        const Interval coarse_iv =
            coarse_intervals(coarse_interval_idx);
        const std::size_t fine_base =
            fine_offsets(fine_interval_idx);
        const std::size_t coarse_base =
            coarse_offsets(coarse_interval_idx);
        const Coord fine_begin = fine_iv.begin;
        const Coord coarse_begin = coarse_iv.begin;
        const Coord fine_y = fine_rows(subset_rows(entry_idx)).y;

        const Coord xb = subset_x_begin(entry_idx);
        const Coord xe = subset_x_end(entry_idx);

        const int team_size = team.team_size();
        const int team_rank = team.team_rank();

        for (Coord x = static_cast<Coord>(xb + team_rank);
             x < xe;
             x += static_cast<Coord>(team_size)) {
          const std::size_t fine_index =
              fine_base +
              static_cast<std::size_t>(x - fine_begin);
          const Coord coarse_x = detail::floor_div2(x);
          const std::size_t coarse_idx =
              coarse_base +
              static_cast<std::size_t>(coarse_x - coarse_begin);

          const T value =
              reconstructor(x, fine_y, coarse_x,
                            coarse_idx, coarse_interval_idx,
                            coarse_intervals, coarse_values);
          fine_values(fine_index) = value;
        }
      });
  ExecSpace().fence();
}

template <typename T>
inline void prolong_field_on_subset_device(
    Field2DDevice<T>& fine_field,
    const Field2DDevice<T>& coarse_field,
    const IntervalSubSet2DDevice& fine_subset) {
  prolong_field_on_subset_generic_device(
      fine_field, coarse_field, fine_subset,
      InjectionReconstructor<T>{});
}

template <typename T>
inline void prolong_field_prediction_on_subset_device(
    Field2DDevice<T>& fine_field,
    const Field2DDevice<T>& coarse_field,
    const IntervalSubSet2DDevice& fine_subset) {
  const detail::VerticalIntervalMapping vertical =
      detail::build_vertical_interval_mapping(coarse_field);

  LinearPredictionReconstructor<T> recon;
  recon.coarse_intervals_view = coarse_field.geometry.intervals;
  recon.coarse_offsets_view = coarse_field.geometry.cell_offsets;
  recon.coarse_values_view = coarse_field.values;
  recon.up_interval = vertical.up_interval;
  recon.down_interval = vertical.down_interval;

  prolong_field_on_subset_generic_device(fine_field, coarse_field,
                                         fine_subset, recon);
}

} // namespace csr
} // namespace subsetix
