#pragma once

#include <cstddef>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include <subsetix/csr_field.hpp>
#include <subsetix/csr_interval_set.hpp>
#include <subsetix/csr_ops/field_core.hpp>
#include <subsetix/csr_ops/field_stencil.hpp>
#include <subsetix/detail/csr_utils.hpp>

namespace subsetix {
namespace csr {

using ExecSpace = Kokkos::DefaultExecutionSpace;

namespace detail {

// ---------------------------------------------------------------------------
// AMR field operations helpers
// ---------------------------------------------------------------------------

struct AmrIntervalMapping {
  Kokkos::View<int*, DeviceMemorySpace> coarse_to_fine_first;
  Kokkos::View<int*, DeviceMemorySpace> coarse_to_fine_second;
  Kokkos::View<int*, DeviceMemorySpace> fine_to_coarse;
};

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

} // namespace detail

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

template <typename T>
inline void prolong_field_prediction_device(
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
  auto fine_row_keys = fine_field.row_keys;
  auto fine_intervals = fine_field.intervals;
  auto fine_values = fine_field.values;

  const detail::AmrIntervalMapping amr_mapping =
      detail::build_amr_interval_mapping(
          coarse_field, fine_field);
  auto fine_to_coarse =
      amr_mapping.fine_to_coarse;
  auto coarse_intervals = coarse_field.intervals;
  auto coarse_values = coarse_field.values;

  const detail::VerticalIntervalMapping vertical =
      detail::build_vertical_interval_mapping(
          coarse_field);
  auto up_interval = vertical.up_interval;
  auto down_interval = vertical.down_interval;

  Kokkos::parallel_for(
      "subsetix_prolong_field_prediction_device",
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

        const Coord y = fine_row_keys(field_row).y;

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
          const T u_center =
              coarse_values(coarse_offset);

          // Left neighbor
          T u_left = u_center;
          if (coarse_x > coarse_iv.begin) {
            u_left = coarse_values(coarse_offset - 1);
          }

          // Right neighbor
          T u_right = u_center;
          if (coarse_x + 1 < coarse_iv.end) {
            u_right = coarse_values(coarse_offset + 1);
          }

          // Up neighbor (y+1)
          T u_up = u_center;
          const int up_idx = up_interval(coarse_interval_idx);
          if (up_idx >= 0) {
            const auto iv_up = coarse_intervals(up_idx);
            if (coarse_x >= iv_up.begin && coarse_x < iv_up.end) {
              u_up = coarse_values(iv_up.value_offset +
                                   static_cast<std::size_t>(coarse_x - iv_up.begin));
            }
          }

          // Down neighbor (y-1)
          T u_down = u_center;
          const int down_idx = down_interval(coarse_interval_idx);
          if (down_idx >= 0) {
            const auto iv_down = coarse_intervals(down_idx);
            if (coarse_x >= iv_down.begin && coarse_x < iv_down.end) {
              u_down = coarse_values(iv_down.value_offset +
                                     static_cast<std::size_t>(coarse_x - iv_down.begin));
            }
          }

          const T grad_x =
              static_cast<T>(0.125) * (u_right - u_left);
          const T grad_y =
              static_cast<T>(0.125) * (u_up - u_down);

          const T sign_x =
              (x % 2 == 0) ? static_cast<T>(-1) : static_cast<T>(1);
          const T sign_y =
              (y % 2 == 0) ? static_cast<T>(-1) : static_cast<T>(1);

          fine_values(linear_index) =
              u_center + sign_x * grad_x + sign_y * grad_y;
        }
      });
  ExecSpace().fence();
}

} // namespace csr
} // namespace subsetix

