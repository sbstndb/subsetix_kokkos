#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include <subsetix/csr_field.hpp>
#include <subsetix/csr_interval_set.hpp>

namespace subsetix {
namespace csr {

/**
 * @brief Convert a field to an interval set by thresholding.
 *
 * Selects all cells where |value| > epsilon.
 *
 * @tparam T Field value type.
 * @param field The input field.
 * @param epsilon The threshold value (exclusive).
 * @return IntervalSet2DDevice containing intervals where the condition is met.
 */
template <typename T>
IntervalSet2DDevice threshold_field(const IntervalField2DDevice<T>& field,
                                    double epsilon) {
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  IntervalSet2DDevice result_set;

  if (field.num_rows == 0) {
    return result_set;
  }

  const std::size_t num_rows = field.num_rows;
  result_set.num_rows = num_rows;

  // Allocate result row structures
  using SetRowKeyView = typename IntervalSet2DDevice::RowKeyView;
  using SetIndexView = typename IntervalSet2DDevice::IndexView;

  SetRowKeyView row_keys("subsetix_threshold_row_keys", num_rows);
  SetIndexView row_ptr("subsetix_threshold_row_ptr", num_rows + 1);
  
  // Copy row keys directly as geometry rows are preserved (though some might become empty)
  Kokkos::deep_copy(row_keys, field.row_keys);

  // Temporary storage for per-row interval counts
  Kokkos::View<std::size_t*, DeviceMemorySpace> row_counts(
      "subsetix_threshold_row_counts", num_rows);

  auto field_row_ptr = field.row_ptr;
  auto field_intervals = field.intervals;
  auto field_values = field.values;

  // -------------------------------------------------------------------------
  // Pass 1: Count resulting intervals per row
  // -------------------------------------------------------------------------
  Kokkos::parallel_for(
      "subsetix_threshold_count",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows),
      KOKKOS_LAMBDA(const std::size_t i) {
        const std::size_t row_begin = field_row_ptr(i);
        const std::size_t row_end = field_row_ptr(i + 1);

        std::size_t count = 0;
        bool current_segment_active = false;
        Coord last_processed_end = 0; // Track end of last processed field interval

        for (std::size_t k = row_begin; k < row_end; ++k) {
          const auto iv = field_intervals(k);
          const Coord x_begin = iv.begin;
          const Coord x_end = iv.end;
          const std::size_t val_offset = iv.value_offset;

          // Check for gap between field intervals
          if (current_segment_active && x_begin > last_processed_end) {
            // Gap breaks the current segment
            count++;
            current_segment_active = false;
          }

          for (Coord x = x_begin; x < x_end; ++x) {
            const T val = field_values(val_offset + (x - x_begin));
            // Compute magnitude. Handle floating point and integer types safely.
            const double mag = static_cast<double>(val > T(0) ? val : -val);
            
            const bool pass = (mag > epsilon);

            if (pass) {
              if (!current_segment_active) {
                current_segment_active = true;
              }
            } else {
              if (current_segment_active) {
                current_segment_active = false;
                count++;
              }
            }
          }
          last_processed_end = x_end;
        }

        if (current_segment_active) {
          count++;
        }

        row_counts(i) = count;
      });

  // -------------------------------------------------------------------------
  // Exclusive scan to compute row pointers
  // -------------------------------------------------------------------------
  Kokkos::View<std::size_t, DeviceMemorySpace> total_intervals_view(
      "subsetix_threshold_total_intervals");

  Kokkos::parallel_scan(
      "subsetix_threshold_scan",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows),
      KOKKOS_LAMBDA(const std::size_t i, std::size_t& update,
                    const bool final_pass) {
        const std::size_t c = row_counts(i);
        if (final_pass) {
          row_ptr(i) = update;
          if (i + 1 == num_rows) {
            row_ptr(num_rows) = update + c;
            total_intervals_view() = update + c;
          }
        }
        update += c;
      });
  
  ExecSpace().fence();

  std::size_t num_intervals = 0;
  Kokkos::deep_copy(num_intervals, total_intervals_view);
  result_set.num_intervals = num_intervals;

  // -------------------------------------------------------------------------
  // Pass 2: Fill intervals
  // -------------------------------------------------------------------------
  typename IntervalSet2DDevice::IntervalView intervals(
      "subsetix_threshold_intervals", num_intervals);

  if (num_intervals > 0) {
    Kokkos::parallel_for(
        "subsetix_threshold_fill",
        Kokkos::RangePolicy<ExecSpace>(0, num_rows),
        KOKKOS_LAMBDA(const std::size_t i) {
          const std::size_t write_start = row_ptr(i);
          // If no intervals to write, skip
          if (row_counts(i) == 0) return;

          std::size_t current_write_idx = write_start;
          
          const std::size_t row_begin = field_row_ptr(i);
          const std::size_t row_end = field_row_ptr(i + 1);

          bool current_segment_active = false;
          Coord current_segment_start = 0;
          Coord last_processed_end = 0;

          for (std::size_t k = row_begin; k < row_end; ++k) {
            const auto iv = field_intervals(k);
            const Coord x_begin = iv.begin;
            const Coord x_end = iv.end;
            const std::size_t val_offset = iv.value_offset;

            if (current_segment_active && x_begin > last_processed_end) {
              // Write interval ending at last_processed_end
              intervals(current_write_idx) = Interval{current_segment_start, last_processed_end};
              current_write_idx++;
              current_segment_active = false;
            }

            for (Coord x = x_begin; x < x_end; ++x) {
              const T val = field_values(val_offset + (x - x_begin));
              const double mag = static_cast<double>(val > T(0) ? val : -val);
              const bool pass = (mag > epsilon);

              if (pass) {
                if (!current_segment_active) {
                  current_segment_active = true;
                  current_segment_start = x;
                }
              } else {
                if (current_segment_active) {
                  intervals(current_write_idx) = Interval{current_segment_start, x};
                  current_write_idx++;
                  current_segment_active = false;
                }
              }
            }
            last_processed_end = x_end;
          }

          if (current_segment_active) {
            intervals(current_write_idx) = Interval{current_segment_start, last_processed_end};
          }
        });
  }

  ExecSpace().fence();

  result_set.row_keys = row_keys;
  result_set.row_ptr = row_ptr;
  result_set.intervals = intervals;

  return result_set;
}

} // namespace csr
} // namespace subsetix

