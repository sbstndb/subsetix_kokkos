#pragma once

#include <cstddef>

#include <Kokkos_Core.hpp>

#include <subsetix/csr_interval_set.hpp>

namespace subsetix {
namespace csr {

using ExecSpace = Kokkos::DefaultExecutionSpace;

namespace detail {

/**
 * @brief Count the number of intervals in the union of two sorted,
 *        non-overlapping interval lists on a single row.
 *
 * Intervals from both lists are merged in order of begin, and overlapping
 * or touching intervals are fused into a single output interval.
 */
template <class IntervalView>
KOKKOS_INLINE_FUNCTION
std::size_t row_union_count(const IntervalView& intervals_a,
                            std::size_t begin_a,
                            std::size_t end_a,
                            const IntervalView& intervals_b,
                            std::size_t begin_b,
                            std::size_t end_b) {
  std::size_t ia = begin_a;
  std::size_t ib = begin_b;

  bool have_current = false;
  Coord current_begin = 0;
  Coord current_end = 0;
  std::size_t count = 0;

  while (ia < end_a || ib < end_b) {
    Coord b = 0;
    Coord e = 0;

    const bool take_a =
        (ib >= end_b) ||
        (ia < end_a &&
         intervals_a(ia).begin <= intervals_b(ib).begin);

    if (take_a) {
      const auto iv = intervals_a(ia);
      b = iv.begin;
      e = iv.end;
      ++ia;
    } else {
      const auto iv = intervals_b(ib);
      b = iv.begin;
      e = iv.end;
      ++ib;
    }

    if (!have_current) {
      current_begin = b;
      current_end = e;
      have_current = true;
    } else {
      if (b <= current_end) {
        current_end = (e > current_end) ? e : current_end;
      } else {
        ++count;
        current_begin = b;
        current_end = e;
      }
    }
  }

  if (have_current) {
    ++count;
  }

  return count;
}

/**
 * @brief Fill the union intervals for a single row into an output view.
 *
 * The layout and merge rules are identical to row_union_count; this
 * function must be kept in sync with it.
 */
template <class IntervalViewIn, class IntervalViewOut>
KOKKOS_INLINE_FUNCTION
void row_union_fill(const IntervalViewIn& intervals_a,
                    std::size_t begin_a,
                    std::size_t end_a,
                    const IntervalViewIn& intervals_b,
                    std::size_t begin_b,
                    std::size_t end_b,
                    const IntervalViewOut& intervals_out,
                    std::size_t out_offset) {
  std::size_t ia = begin_a;
  std::size_t ib = begin_b;

  bool have_current = false;
  Coord current_begin = 0;
  Coord current_end = 0;
  std::size_t write_idx = 0;

  while (ia < end_a || ib < end_b) {
    Coord b = 0;
    Coord e = 0;

    const bool take_a =
        (ib >= end_b) ||
        (ia < end_a &&
         intervals_a(ia).begin <= intervals_b(ib).begin);

    if (take_a) {
      const auto iv = intervals_a(ia);
      b = iv.begin;
      e = iv.end;
      ++ia;
    } else {
      const auto iv = intervals_b(ib);
      b = iv.begin;
      e = iv.end;
      ++ib;
    }

    if (!have_current) {
      current_begin = b;
      current_end = e;
      have_current = true;
    } else {
      if (b <= current_end) {
        current_end = (e > current_end) ? e : current_end;
      } else {
        intervals_out(out_offset + write_idx) = Interval{current_begin,
                                                         current_end};
        ++write_idx;
        current_begin = b;
        current_end = e;
      }
    }
  }

  if (have_current) {
    intervals_out(out_offset + write_idx) = Interval{current_begin,
                                                     current_end};
  }
}

struct RowMergeResult {
  IntervalSet2DDevice::RowKeyView row_keys;
  Kokkos::View<int*, DeviceMemorySpace> row_index_a;
  Kokkos::View<int*, DeviceMemorySpace> row_index_b;
  std::size_t num_rows = 0;
};

/**
 * @brief Build the union of row keys between two IntervalSet2DDevice sets.
 *
 * The result contains sorted unique row keys, plus mapping arrays that
 * tell, for each output row, which row index it corresponds to in A and B
 * (or -1 if the row is absent in that input).
 */
inline RowMergeResult
build_row_union_mapping(const IntervalSet2DDevice& A,
                        const IntervalSet2DDevice& B) {
  RowMergeResult result;

  const std::size_t num_rows_a = A.num_rows;
  const std::size_t num_rows_b = B.num_rows;

  if (num_rows_a == 0 && num_rows_b == 0) {
    return result;
  }

  // Temporary storage large enough for all rows from A and B.
  IntervalSet2DDevice::RowKeyView tmp_rows(
      "subsetix_csr_union_tmp_rows", num_rows_a + num_rows_b);
  Kokkos::View<int*, DeviceMemorySpace> tmp_idx_a(
      "subsetix_csr_union_tmp_idx_a", num_rows_a + num_rows_b);
  Kokkos::View<int*, DeviceMemorySpace> tmp_idx_b(
      "subsetix_csr_union_tmp_idx_b", num_rows_a + num_rows_b);

  Kokkos::View<std::size_t, DeviceMemorySpace> d_num_rows(
      "subsetix_csr_union_num_rows");

  auto rows_a = A.row_keys;
  auto rows_b = B.row_keys;

  Kokkos::parallel_for(
      "subsetix_csr_union_row_merge",
      Kokkos::RangePolicy<ExecSpace>(0, 1),
      KOKKOS_LAMBDA(const int) {
        std::size_t ia = 0;
        std::size_t ib = 0;
        std::size_t out = 0;

        while (ia < num_rows_a || ib < num_rows_b) {
          const bool has_a = ia < num_rows_a;
          const bool has_b = ib < num_rows_b;

          if (has_a && (!has_b || rows_a(ia).y < rows_b(ib).y)) {
            tmp_rows(out) = rows_a(ia);
            tmp_idx_a(out) = static_cast<int>(ia);
            tmp_idx_b(out) = -1;
            ++ia;
            ++out;
          } else if (has_b &&
                     (!has_a || rows_b(ib).y < rows_a(ia).y)) {
            tmp_rows(out) = rows_b(ib);
            tmp_idx_a(out) = -1;
            tmp_idx_b(out) = static_cast<int>(ib);
            ++ib;
            ++out;
          } else {
            // Same Y value in A and B.
            tmp_rows(out) = rows_a(ia);
            tmp_idx_a(out) = static_cast<int>(ia);
            tmp_idx_b(out) = static_cast<int>(ib);
            ++ia;
            ++ib;
            ++out;
          }
        }

        d_num_rows() = out;
      });

  ExecSpace().fence();

  std::size_t num_rows_out = 0;
  Kokkos::deep_copy(num_rows_out, d_num_rows);

  result.num_rows = num_rows_out;
  if (num_rows_out == 0) {
    return result;
  }

  result.row_keys = IntervalSet2DDevice::RowKeyView(
      "subsetix_csr_union_row_keys", num_rows_out);
  result.row_index_a = Kokkos::View<int*, DeviceMemorySpace>(
      "subsetix_csr_union_row_index_a", num_rows_out);
  result.row_index_b = Kokkos::View<int*, DeviceMemorySpace>(
      "subsetix_csr_union_row_index_b", num_rows_out);

  auto out_rows = result.row_keys;
  auto out_idx_a = result.row_index_a;
  auto out_idx_b = result.row_index_b;

  Kokkos::parallel_for(
      "subsetix_csr_union_row_merge_copy",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        out_rows(i) = tmp_rows(i);
        out_idx_a(i) = tmp_idx_a(i);
        out_idx_b(i) = tmp_idx_b(i);
      });

  ExecSpace().fence();

  return result;
}

} // namespace detail

/**
 * @brief Compute the set union of two 2D CSR interval sets on device.
 *
 * Both inputs are assumed to satisfy the usual invariants:
 *  - row_keys sorted by Y, strictly increasing
 *  - row_ptr.size() == num_rows + 1
 *  - intervals sorted by X within each row, non-overlapping
 *
 * The output satisfies the same invariants and is built entirely via
 * device-side Kokkos operations (no host-side interval loops).
 */
inline IntervalSet2DDevice
set_union_device(const IntervalSet2DDevice& A,
                 const IntervalSet2DDevice& B) {
  IntervalSet2DDevice out;

  const std::size_t num_rows_a = A.num_rows;
  const std::size_t num_rows_b = B.num_rows;

  if (num_rows_a == 0 && num_rows_b == 0) {
    return out;
  }

  // 1) Merge row keys and build mapping A/B -> output rows.
  detail::RowMergeResult merge = detail::build_row_union_mapping(A, B);
  const std::size_t num_rows_out = merge.num_rows;
  if (num_rows_out == 0) {
    return out;
  }

  auto row_keys_out = merge.row_keys;
  auto row_index_a = merge.row_index_a;
  auto row_index_b = merge.row_index_b;

  // 2) For each row, count how many intervals will be in the union.
  Kokkos::View<std::size_t*, DeviceMemorySpace> row_counts(
      "subsetix_csr_union_row_counts", num_rows_out);

  auto row_ptr_a = A.row_ptr;
  auto row_ptr_b = B.row_ptr;
  auto intervals_a = A.intervals;
  auto intervals_b = B.intervals;

  Kokkos::parallel_for(
      "subsetix_csr_union_count",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        const int ia = row_index_a(i);
        const int ib = row_index_b(i);

        std::size_t begin_a = 0;
        std::size_t end_a = 0;
        std::size_t begin_b = 0;
        std::size_t end_b = 0;

        if (ia >= 0) {
          const std::size_t row_a = static_cast<std::size_t>(ia);
          begin_a = row_ptr_a(row_a);
          end_a = row_ptr_a(row_a + 1);
        }
        if (ib >= 0) {
          const std::size_t row_b = static_cast<std::size_t>(ib);
          begin_b = row_ptr_b(row_b);
          end_b = row_ptr_b(row_b + 1);
        }

        if (begin_a == end_a && begin_b == end_b) {
          row_counts(i) = 0;
          return;
        }

        row_counts(i) = detail::row_union_count(
            intervals_a, begin_a, end_a,
            intervals_b, begin_b, end_b);
      });

  ExecSpace().fence();

  // 3) Exclusive scan on row_counts to build row_ptr for the union and
  //    obtain the total number of intervals.
  IntervalSet2DDevice::IndexView row_ptr_out(
      "subsetix_csr_union_row_ptr", num_rows_out + 1);
  Kokkos::View<std::size_t, DeviceMemorySpace> total_intervals(
      "subsetix_csr_union_total_intervals");

  Kokkos::parallel_scan(
      "subsetix_csr_union_scan",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i,
                    std::size_t& update,
                    const bool final_pass) {
        const std::size_t c = row_counts(i);
        if (final_pass) {
          row_ptr_out(i) = update;
          if (i + 1 == num_rows_out) {
            row_ptr_out(num_rows_out) = update + c;
            total_intervals() = update + c;
          }
        }
        update += c;
      });

  ExecSpace().fence();

  std::size_t num_intervals_out = 0;
  Kokkos::deep_copy(num_intervals_out, total_intervals);

  out.num_rows = num_rows_out;
  out.num_intervals = num_intervals_out;
  out.row_keys = row_keys_out;
  out.row_ptr = row_ptr_out;

  if (num_intervals_out == 0) {
    out.intervals = IntervalSet2DDevice::IntervalView();
    return out;
  }

  IntervalSet2DDevice::IntervalView intervals_out(
      "subsetix_csr_union_intervals", num_intervals_out);

  // 4) Fill intervals for each row using the offsets defined by row_ptr_out.
  Kokkos::parallel_for(
      "subsetix_csr_union_fill",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        const int ia = row_index_a(i);
        const int ib = row_index_b(i);

        std::size_t begin_a = 0;
        std::size_t end_a = 0;
        std::size_t begin_b = 0;
        std::size_t end_b = 0;

        if (ia >= 0) {
          const std::size_t row_a = static_cast<std::size_t>(ia);
          begin_a = row_ptr_a(row_a);
          end_a = row_ptr_a(row_a + 1);
        }
        if (ib >= 0) {
          const std::size_t row_b = static_cast<std::size_t>(ib);
          begin_b = row_ptr_b(row_b);
          end_b = row_ptr_b(row_b + 1);
        }

        if (begin_a == end_a && begin_b == end_b) {
          return;
        }

        const std::size_t out_offset = row_ptr_out(i);
        detail::row_union_fill(intervals_a, begin_a, end_a,
                               intervals_b, begin_b, end_b,
                               intervals_out, out_offset);
      });

  ExecSpace().fence();

  out.intervals = intervals_out;

  return out;
}

} // namespace csr
} // namespace subsetix

