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
        (ib >= end_b) || (ia < end_a &&
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

/**
 * @brief Count the number of intervals in the intersection of two sorted,
 *        non-overlapping interval lists on a single row.
 */
template <class IntervalView>
KOKKOS_INLINE_FUNCTION
std::size_t row_intersection_count(const IntervalView& intervals_a,
                                   std::size_t begin_a,
                                   std::size_t end_a,
                                   const IntervalView& intervals_b,
                                   std::size_t begin_b,
                                   std::size_t end_b) {
  std::size_t ia = begin_a;
  std::size_t ib = begin_b;
  std::size_t count = 0;

  while (ia < end_a && ib < end_b) {
    const auto a = intervals_a(ia);
    const auto b = intervals_b(ib);

    const Coord start =
        (a.begin > b.begin) ? a.begin : b.begin;
    const Coord end =
        (a.end < b.end) ? a.end : b.end;

    if (start < end) {
      ++count;
    }

    if (a.end < b.end) {
      ++ia;
    } else if (b.end < a.end) {
      ++ib;
    } else {
      ++ia;
      ++ib;
    }
  }

  return count;
}

/**
 * @brief Fill the intersection intervals for a single row into an
 *        output view.
 *
 * Must stay in sync with row_intersection_count.
 */
template <class IntervalViewIn, class IntervalViewOut>
KOKKOS_INLINE_FUNCTION
void row_intersection_fill(const IntervalViewIn& intervals_a,
                           std::size_t begin_a,
                           std::size_t end_a,
                           const IntervalViewIn& intervals_b,
                           std::size_t begin_b,
                           std::size_t end_b,
                           const IntervalViewOut& intervals_out,
                           std::size_t out_offset) {
  std::size_t ia = begin_a;
  std::size_t ib = begin_b;
  std::size_t write_idx = 0;

  while (ia < end_a && ib < end_b) {
    const auto a = intervals_a(ia);
    const auto b = intervals_b(ib);

    const Coord start =
        (a.begin > b.begin) ? a.begin : b.begin;
    const Coord end =
        (a.end < b.end) ? a.end : b.end;

    if (start < end) {
      intervals_out(out_offset + write_idx) = Interval{start, end};
      ++write_idx;
    }

    if (a.end < b.end) {
      ++ia;
    } else if (b.end < a.end) {
      ++ib;
    } else {
      ++ia;
      ++ib;
    }
  }
}

/**
 * @brief Count the number of intervals in the difference A \ B on a
 *        single row.
 *
 * Intervals are assumed sorted and non-overlapping within each list.
 */
template <class IntervalView>
KOKKOS_INLINE_FUNCTION
std::size_t row_difference_count(const IntervalView& intervals_a,
                                 std::size_t begin_a,
                                 std::size_t end_a,
                                 const IntervalView& intervals_b,
                                 std::size_t begin_b,
                                 std::size_t end_b) {
  if (begin_a == end_a) {
    return 0;
  }
  if (begin_b == end_b) {
    return end_a - begin_a;
  }

  std::size_t ia = begin_a;
  std::size_t ib = begin_b;
  std::size_t count = 0;

  while (ia < end_a) {
    const auto a = intervals_a(ia);
    Coord cur = a.begin;

    // Skip B intervals that end before the current A interval.
    while (ib < end_b && intervals_b(ib).end <= a.begin) {
      ++ib;
    }

    while (ib < end_b && intervals_b(ib).begin < a.end) {
      const auto b = intervals_b(ib);

      if (b.begin > cur) {
        ++count;
      }

      if (b.end >= a.end) {
        cur = a.end;
        break;
      } else {
        cur = b.end;
        ++ib;
      }
    }

    if (cur < a.end) {
      ++count;
    }

    ++ia;
  }

  return count;
}

/**
 * @brief Fill the difference intervals A \ B for a single row into an
 *        output view.
 *
 * Must stay in sync with row_difference_count.
 */
template <class IntervalViewIn, class IntervalViewOut>
KOKKOS_INLINE_FUNCTION
void row_difference_fill(const IntervalViewIn& intervals_a,
                         std::size_t begin_a,
                         std::size_t end_a,
                         const IntervalViewIn& intervals_b,
                         std::size_t begin_b,
                         std::size_t end_b,
                         const IntervalViewOut& intervals_out,
                         std::size_t out_offset) {
  if (begin_a == end_a) {
    return;
  }

  if (begin_b == end_b) {
    // Fast path: just copy A.
    std::size_t write_idx = 0;
    for (std::size_t ia = begin_a; ia < end_a; ++ia) {
      intervals_out(out_offset + write_idx) = intervals_a(ia);
      ++write_idx;
    }
    return;
  }

  std::size_t ia = begin_a;
  std::size_t ib = begin_b;
  std::size_t write_idx = 0;

  while (ia < end_a) {
    const auto a = intervals_a(ia);
    Coord cur = a.begin;

    // Skip B intervals that end before the current A interval.
    while (ib < end_b && intervals_b(ib).end <= a.begin) {
      ++ib;
    }

    while (ib < end_b && intervals_b(ib).begin < a.end) {
      const auto b = intervals_b(ib);

      if (b.begin > cur) {
        intervals_out(out_offset + write_idx) =
            Interval{cur, b.begin};
        ++write_idx;
      }

      if (b.end >= a.end) {
        cur = a.end;
        break;
      } else {
        cur = b.end;
        ++ib;
      }
    }

    if (cur < a.end) {
      intervals_out(out_offset + write_idx) =
          Interval{cur, a.end};
      ++write_idx;
    }

    ++ia;
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

/**
 * @brief Build the intersection of row keys between two
 *        IntervalSet2DDevice sets.
 *
 * The result contains sorted row keys which appear in both A and B,
 * plus mapping arrays giving the corresponding row indices.
 */
inline RowMergeResult
build_row_intersection_mapping(const IntervalSet2DDevice& A,
                               const IntervalSet2DDevice& B) {
  RowMergeResult result;

  const std::size_t num_rows_a = A.num_rows;
  const std::size_t num_rows_b = B.num_rows;

  if (num_rows_a == 0 || num_rows_b == 0) {
    return result;
  }

  const std::size_t max_rows_out =
      (num_rows_a < num_rows_b) ? num_rows_a : num_rows_b;

  IntervalSet2DDevice::RowKeyView tmp_rows(
      "subsetix_csr_intersection_tmp_rows", max_rows_out);
  Kokkos::View<int*, DeviceMemorySpace> tmp_idx_a(
      "subsetix_csr_intersection_tmp_idx_a", max_rows_out);
  Kokkos::View<int*, DeviceMemorySpace> tmp_idx_b(
      "subsetix_csr_intersection_tmp_idx_b", max_rows_out);

  Kokkos::View<std::size_t, DeviceMemorySpace> d_num_rows(
      "subsetix_csr_intersection_num_rows");

  auto rows_a = A.row_keys;
  auto rows_b = B.row_keys;

  Kokkos::parallel_for(
      "subsetix_csr_intersection_row_merge",
      Kokkos::RangePolicy<ExecSpace>(0, 1),
      KOKKOS_LAMBDA(const int) {
        std::size_t ia = 0;
        std::size_t ib = 0;
        std::size_t out = 0;

        while (ia < num_rows_a && ib < num_rows_b) {
          const Coord ya = rows_a(ia).y;
          const Coord yb = rows_b(ib).y;

          if (ya < yb) {
            ++ia;
          } else if (yb < ya) {
            ++ib;
          } else {
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
      "subsetix_csr_intersection_row_keys", num_rows_out);
  result.row_index_a = Kokkos::View<int*, DeviceMemorySpace>(
      "subsetix_csr_intersection_row_index_a", num_rows_out);
  result.row_index_b = Kokkos::View<int*, DeviceMemorySpace>(
      "subsetix_csr_intersection_row_index_b", num_rows_out);

  auto out_rows = result.row_keys;
  auto out_idx_a = result.row_index_a;
  auto out_idx_b = result.row_index_b;

  Kokkos::parallel_for(
      "subsetix_csr_intersection_row_merge_copy",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        out_rows(i) = tmp_rows(i);
        out_idx_a(i) = tmp_idx_a(i);
        out_idx_b(i) = tmp_idx_b(i);
      });

  ExecSpace().fence();

  return result;
}

struct RowDifferenceResult {
  IntervalSet2DDevice::RowKeyView row_keys;
  Kokkos::View<int*, DeviceMemorySpace> row_index_b;
  std::size_t num_rows = 0;
};

/**
 * @brief Build a mapping from rows of A to matching rows in B for
 *        computing A \ B.
 *
 * The result has the same row keys as A.
 */
inline RowDifferenceResult
build_row_difference_mapping(const IntervalSet2DDevice& A,
                             const IntervalSet2DDevice& B) {
  RowDifferenceResult result;

  const std::size_t num_rows_a = A.num_rows;
  const std::size_t num_rows_b = B.num_rows;

  if (num_rows_a == 0) {
    return result;
  }

  result.num_rows = num_rows_a;
  result.row_keys = IntervalSet2DDevice::RowKeyView(
      "subsetix_csr_difference_row_keys", num_rows_a);
  result.row_index_b = Kokkos::View<int*, DeviceMemorySpace>(
      "subsetix_csr_difference_row_index_b", num_rows_a);

  auto rows_a = A.row_keys;
  auto rows_b = B.row_keys;
  auto out_rows = result.row_keys;
  auto out_idx_b = result.row_index_b;

  // Copy row keys from A.
  Kokkos::parallel_for(
      "subsetix_csr_difference_row_copy_keys",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
      KOKKOS_LAMBDA(const std::size_t i) {
        out_rows(i) = rows_a(i);
      });

  ExecSpace().fence();

  // Build mapping A.rows -> B.rows.
  Kokkos::parallel_for(
      "subsetix_csr_difference_row_mapping",
      Kokkos::RangePolicy<ExecSpace>(0, 1),
      KOKKOS_LAMBDA(const int) {
        std::size_t ia = 0;
        std::size_t ib = 0;

        while (ia < num_rows_a) {
          const Coord ya = rows_a(ia).y;

          while (ib < num_rows_b && rows_b(ib).y < ya) {
            ++ib;
          }

          if (ib < num_rows_b && rows_b(ib).y == ya) {
            out_idx_b(ia) = static_cast<int>(ib);
          } else {
            out_idx_b(ia) = -1;
          }

          ++ia;
        }
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

/**
 * @brief Compute the set intersection of two 2D CSR interval sets on
 *        device.
 */
inline IntervalSet2DDevice
set_intersection_device(const IntervalSet2DDevice& A,
                        const IntervalSet2DDevice& B) {
  IntervalSet2DDevice out;

  const std::size_t num_rows_a = A.num_rows;
  const std::size_t num_rows_b = B.num_rows;

  if (num_rows_a == 0 || num_rows_b == 0) {
    return out;
  }

  detail::RowMergeResult merge =
      detail::build_row_intersection_mapping(A, B);
  const std::size_t num_rows_out = merge.num_rows;
  if (num_rows_out == 0) {
    return out;
  }

  auto row_keys_out = merge.row_keys;
  auto row_index_a = merge.row_index_a;
  auto row_index_b = merge.row_index_b;

  Kokkos::View<std::size_t*, DeviceMemorySpace> row_counts(
      "subsetix_csr_intersection_row_counts", num_rows_out);

  auto row_ptr_a = A.row_ptr;
  auto row_ptr_b = B.row_ptr;
  auto intervals_a = A.intervals;
  auto intervals_b = B.intervals;

  Kokkos::parallel_for(
      "subsetix_csr_intersection_count",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        const int ia = row_index_a(i);
        const int ib = row_index_b(i);

        if (ia < 0 || ib < 0) {
          row_counts(i) = 0;
          return;
        }

        const std::size_t row_a = static_cast<std::size_t>(ia);
        const std::size_t row_b = static_cast<std::size_t>(ib);

        const std::size_t begin_a = row_ptr_a(row_a);
        const std::size_t end_a = row_ptr_a(row_a + 1);
        const std::size_t begin_b = row_ptr_b(row_b);
        const std::size_t end_b = row_ptr_b(row_b + 1);

        if (begin_a == end_a || begin_b == end_b) {
          row_counts(i) = 0;
          return;
        }

        row_counts(i) = detail::row_intersection_count(
            intervals_a, begin_a, end_a,
            intervals_b, begin_b, end_b);
      });

  ExecSpace().fence();

  IntervalSet2DDevice::IndexView row_ptr_out(
      "subsetix_csr_intersection_row_ptr", num_rows_out + 1);
  Kokkos::View<std::size_t, DeviceMemorySpace> total_intervals(
      "subsetix_csr_intersection_total_intervals");

  Kokkos::parallel_scan(
      "subsetix_csr_intersection_scan",
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
      "subsetix_csr_intersection_intervals", num_intervals_out);

  Kokkos::parallel_for(
      "subsetix_csr_intersection_fill",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        const int ia = row_index_a(i);
        const int ib = row_index_b(i);

        if (ia < 0 || ib < 0) {
          return;
        }

        const std::size_t row_a = static_cast<std::size_t>(ia);
        const std::size_t row_b = static_cast<std::size_t>(ib);

        const std::size_t begin_a = row_ptr_a(row_a);
        const std::size_t end_a = row_ptr_a(row_a + 1);
        const std::size_t begin_b = row_ptr_b(row_b);
        const std::size_t end_b = row_ptr_b(row_b + 1);

        if (begin_a == end_a || begin_b == end_b) {
          return;
        }

        const std::size_t out_offset = row_ptr_out(i);
        detail::row_intersection_fill(
            intervals_a, begin_a, end_a,
            intervals_b, begin_b, end_b,
            intervals_out, out_offset);
      });

  ExecSpace().fence();

  out.intervals = intervals_out;

  return out;
}

/**
 * @brief Compute the set difference A \ B of two 2D CSR interval sets
 *        on device.
 */
inline IntervalSet2DDevice
set_difference_device(const IntervalSet2DDevice& A,
                      const IntervalSet2DDevice& B) {
  IntervalSet2DDevice out;

  const std::size_t num_rows_a = A.num_rows;
  if (num_rows_a == 0) {
    return out;
  }

  detail::RowDifferenceResult diff_rows =
      detail::build_row_difference_mapping(A, B);
  const std::size_t num_rows_out = diff_rows.num_rows;

  if (num_rows_out == 0) {
    return out;
  }

  auto row_keys_out = diff_rows.row_keys;
  auto row_index_b = diff_rows.row_index_b;

  Kokkos::View<std::size_t*, DeviceMemorySpace> row_counts(
      "subsetix_csr_difference_row_counts", num_rows_out);

  auto row_ptr_a = A.row_ptr;
  auto row_ptr_b = B.row_ptr;
  auto intervals_a = A.intervals;
  auto intervals_b = B.intervals;

  Kokkos::parallel_for(
      "subsetix_csr_difference_count",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        const std::size_t row_a = i;
        const std::size_t begin_a = row_ptr_a(row_a);
        const std::size_t end_a = row_ptr_a(row_a + 1);

        if (begin_a == end_a) {
          row_counts(i) = 0;
          return;
        }

        const int ib = row_index_b(i);
        std::size_t begin_b = 0;
        std::size_t end_b = 0;

        if (ib >= 0) {
          const std::size_t row_b = static_cast<std::size_t>(ib);
          begin_b = row_ptr_b(row_b);
          end_b = row_ptr_b(row_b + 1);
        }

        row_counts(i) = detail::row_difference_count(
            intervals_a, begin_a, end_a,
            intervals_b, begin_b, end_b);
      });

  ExecSpace().fence();

  IntervalSet2DDevice::IndexView row_ptr_out(
      "subsetix_csr_difference_row_ptr", num_rows_out + 1);
  Kokkos::View<std::size_t, DeviceMemorySpace> total_intervals(
      "subsetix_csr_difference_total_intervals");

  Kokkos::parallel_scan(
      "subsetix_csr_difference_scan",
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
      "subsetix_csr_difference_intervals", num_intervals_out);

  Kokkos::parallel_for(
      "subsetix_csr_difference_fill",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        const std::size_t row_a = i;
        const std::size_t begin_a = row_ptr_a(row_a);
        const std::size_t end_a = row_ptr_a(row_a + 1);

        if (begin_a == end_a) {
          return;
        }

        const int ib = row_index_b(i);
        std::size_t begin_b = 0;
        std::size_t end_b = 0;

        if (ib >= 0) {
          const std::size_t row_b = static_cast<std::size_t>(ib);
          begin_b = row_ptr_b(row_b);
          end_b = row_ptr_b(row_b + 1);
        }

        const std::size_t out_offset = row_ptr_out(i);
        detail::row_difference_fill(
            intervals_a, begin_a, end_a,
            intervals_b, begin_b, end_b,
            intervals_out, out_offset);
      });

  ExecSpace().fence();

  out.intervals = intervals_out;

  return out;
}

/**
 * @brief Translate all intervals of a CSR interval set by a constant
 *        offset along X.
 *
 * Row structure (row_keys, row_ptr) is preserved; only the begin/end
 * coordinates of intervals are shifted by dx.
 */
inline IntervalSet2DDevice
translate_x_device(const IntervalSet2DDevice& in,
                   Coord dx) {
  IntervalSet2DDevice out;

  out.num_rows = in.num_rows;
  out.num_intervals = in.num_intervals;
  out.row_keys = in.row_keys;
  out.row_ptr = in.row_ptr;

  if (in.num_intervals == 0) {
    out.intervals = IntervalSet2DDevice::IntervalView();
    return out;
  }

  if (dx == 0) {
    out.intervals = in.intervals;
    return out;
  }

  IntervalSet2DDevice::IntervalView intervals_out(
      "subsetix_csr_translate_x_intervals", in.num_intervals);

  auto intervals_in = in.intervals;

  Kokkos::parallel_for(
      "subsetix_csr_translate_x",
      Kokkos::RangePolicy<ExecSpace>(0, in.num_intervals),
      KOKKOS_LAMBDA(const std::size_t i) {
        Interval iv = intervals_in(i);
        iv.begin = static_cast<Coord>(iv.begin + dx);
        iv.end = static_cast<Coord>(iv.end + dx);
        intervals_out(i) = iv;
      });

  ExecSpace().fence();

  out.intervals = intervals_out;

  return out;
}

} // namespace csr
} // namespace subsetix
