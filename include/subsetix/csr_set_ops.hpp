#pragma once

#include <cstddef>
#include <stdexcept>

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include <subsetix/csr_interval_set.hpp>

namespace subsetix {
namespace csr {

using ExecSpace = Kokkos::DefaultExecutionSpace;

namespace detail {

/**
 * @brief Workspace for row-intersection mapping kernels.
 *
 * Buffers are grown on demand and reused across calls to avoid paying
 * device allocations on every MapRows operation.
 */
struct RowIntersectionWorkspace {
  Kokkos::View<int*, DeviceMemorySpace> flags;
  Kokkos::View<int*, DeviceMemorySpace> tmp_idx_a;
  Kokkos::View<int*, DeviceMemorySpace> tmp_idx_b;
  Kokkos::View<std::size_t*, DeviceMemorySpace> positions;
  Kokkos::View<std::size_t, DeviceMemorySpace> d_num_rows;
  std::size_t capacity_small = 0;

  void ensure_capacity(std::size_t n_small) {
    if (n_small <= capacity_small) {
      return;
    }

    flags = Kokkos::View<int*, DeviceMemorySpace>(
        "subsetix_csr_intersection_flags", n_small);
    tmp_idx_a = Kokkos::View<int*, DeviceMemorySpace>(
        "subsetix_csr_intersection_tmp_idx_a", n_small);
    tmp_idx_b = Kokkos::View<int*, DeviceMemorySpace>(
        "subsetix_csr_intersection_tmp_idx_b", n_small);
    positions = Kokkos::View<std::size_t*, DeviceMemorySpace>(
        "subsetix_csr_intersection_row_positions", n_small);

    if (!d_num_rows.data()) {
      d_num_rows = Kokkos::View<std::size_t, DeviceMemorySpace>(
          "subsetix_csr_intersection_num_rows");
    }

    capacity_small = n_small;
  }
};

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

struct RowUnionElement {
  RowKey2D key;
  int src; // 0 = A, 1 = B
  int idx;
};

struct RowUnionLess {
  KOKKOS_INLINE_FUNCTION
  bool operator()(const RowUnionElement& a,
                  const RowUnionElement& b) const {
    return a.key.y < b.key.y;
  }
};

struct RowUnionWorkspace {
  Kokkos::View<RowUnionElement*, DeviceMemorySpace> elems;
  Kokkos::View<int*, DeviceMemorySpace> is_head;
  Kokkos::View<std::size_t*, DeviceMemorySpace> head_pos;
  Kokkos::View<std::size_t, DeviceMemorySpace> d_num_rows;
  std::size_t capacity_total = 0;

  void ensure_capacity(std::size_t total) {
    if (total <= capacity_total) {
      return;
    }

    elems = Kokkos::View<RowUnionElement*, DeviceMemorySpace>(
        "subsetix_csr_union_elems", total);
    is_head = Kokkos::View<int*, DeviceMemorySpace>(
        "subsetix_csr_union_is_head", total);
    head_pos = Kokkos::View<std::size_t*, DeviceMemorySpace>(
        "subsetix_csr_union_head_pos", total);

    if (!d_num_rows.data()) {
      d_num_rows = Kokkos::View<std::size_t, DeviceMemorySpace>(
          "subsetix_csr_union_num_rows");
    }

    capacity_total = total;
  }
};

/**
 * @brief Build the union of row keys between two IntervalSet2DDevice sets.
 *
 * The result contains sorted unique row keys, plus mapping arrays that
 * tell, for each output row, which row index it corresponds to in A and B
 * (or -1 if the row is absent in that input). The workspace version
 * reuses device buffers across calls.
 */
inline RowMergeResult
build_row_union_mapping(const IntervalSet2DDevice& A,
                        const IntervalSet2DDevice& B,
                        RowUnionWorkspace& workspace) {
  RowMergeResult result;

  const std::size_t num_rows_a = A.num_rows;
  const std::size_t num_rows_b = B.num_rows;

  if (num_rows_a == 0 && num_rows_b == 0) {
    return result;
  }

  auto rows_a = A.row_keys;
  auto rows_b = B.row_keys;

  const std::size_t total = num_rows_a + num_rows_b;

  workspace.ensure_capacity(total);

  // Pack all row keys from A and B into a single device array with
  // source tags, then sort by Y and compact uniques.
  auto elems = workspace.elems;

  // Fill from A.
  Kokkos::parallel_for(
      "subsetix_csr_union_pack_a",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
      KOKKOS_LAMBDA(const std::size_t i) {
        elems(i).key = rows_a(i);
        elems(i).src = 0;
        elems(i).idx = static_cast<int>(i);
      });

  // Fill from B.
  Kokkos::parallel_for(
      "subsetix_csr_union_pack_b",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_b),
      KOKKOS_LAMBDA(const std::size_t j) {
        const std::size_t pos = num_rows_a + j;
        elems(pos).key = rows_b(j);
        elems(pos).src = 1;
        elems(pos).idx = static_cast<int>(j);
      });

  ExecSpace().fence();

  ExecSpace exec;
  Kokkos::sort(exec, elems, RowUnionLess{});
  exec.fence();

  if (total == 0) {
    return result;
  }

  // Mark the first occurrence of each distinct row key.
  auto is_head = workspace.is_head;

  Kokkos::parallel_for(
      "subsetix_csr_union_mark_heads",
      Kokkos::RangePolicy<ExecSpace>(0, total),
      KOKKOS_LAMBDA(const std::size_t i) {
        if (i == 0) {
          is_head(i) = 1;
        } else {
          const Coord y_curr = elems(i).key.y;
          const Coord y_prev = elems(i - 1).key.y;
          is_head(i) = (y_curr != y_prev) ? 1 : 0;
        }
      });

  ExecSpace().fence();

  // Exclusive scan of heads to get output positions and count.
  auto head_pos = workspace.head_pos;
  auto d_num_rows = workspace.d_num_rows;

  Kokkos::parallel_scan(
      "subsetix_csr_union_head_scan",
      Kokkos::RangePolicy<ExecSpace>(0, total),
      KOKKOS_LAMBDA(const std::size_t i,
                    std::size_t& update,
                    const bool final_pass) {
        const std::size_t flag =
            static_cast<std::size_t>(is_head(i));
        if (final_pass) {
          head_pos(i) = update;
          if (i + 1 == total) {
            d_num_rows() = update + flag;
          }
        }
        update += flag;
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

  // Compact grouped entries (at most two per Y: one from A, one from B).
  Kokkos::parallel_for(
      "subsetix_csr_union_compact",
      Kokkos::RangePolicy<ExecSpace>(0, total),
      KOKKOS_LAMBDA(const std::size_t i) {
        if (!is_head(i)) {
          return;
        }

        const std::size_t pos = head_pos(i);
        const RowUnionElement e0 = elems(i);

        int ia = -1;
        int ib = -1;

        if (e0.src == 0) {
          ia = e0.idx;
        } else {
          ib = e0.idx;
        }

        if (i + 1 < total &&
            elems(i + 1).key.y == e0.key.y) {
          const RowUnionElement e1 = elems(i + 1);
          if (e1.src == 0 && ia < 0) {
            ia = e1.idx;
          } else if (e1.src == 1 && ib < 0) {
            ib = e1.idx;
          }
        }

        out_rows(pos) = e0.key;
        out_idx_a(pos) = ia;
        out_idx_b(pos) = ib;
      });

  ExecSpace().fence();

  return result;
}

/**
 * @brief Convenience overload that uses a local workspace. Kept for
 * tests and small utilities.
 */
inline RowMergeResult
build_row_union_mapping(const IntervalSet2DDevice& A,
                        const IntervalSet2DDevice& B) {
  RowUnionWorkspace workspace;
  return build_row_union_mapping(A, B, workspace);
}

/**
 * @brief Build the intersection of row keys between two
 *        IntervalSet2DDevice sets.
 *
 * The result contains sorted row keys which appear in both A and B,
 * plus mapping arrays giving the corresponding row indices. The
 * workspace version reuses device buffers across calls.
 */
inline RowMergeResult
build_row_intersection_mapping(const IntervalSet2DDevice& A,
                               const IntervalSet2DDevice& B,
                               RowIntersectionWorkspace& workspace) {
  RowMergeResult result;
  const std::size_t num_rows_a = A.num_rows;
  const std::size_t num_rows_b = B.num_rows;

  if (num_rows_a == 0 || num_rows_b == 0) {
    return result;
  }

  auto rows_a = A.row_keys;
  auto rows_b = B.row_keys;

  const bool small_is_a = (num_rows_a <= num_rows_b);
  const std::size_t n_small =
      small_is_a ? num_rows_a : num_rows_b;
  const std::size_t n_big =
      small_is_a ? num_rows_b : num_rows_a;

  auto rows_small = small_is_a ? rows_a : rows_b;
  auto rows_big = small_is_a ? rows_b : rows_a;

  workspace.ensure_capacity(n_small);

  // 1) For each row of the smaller set, binary-search its Y in the
  //    larger set and record matches.
  auto flags = workspace.flags;
  auto tmp_idx_a = workspace.tmp_idx_a;
  auto tmp_idx_b = workspace.tmp_idx_b;

  Kokkos::parallel_for(
      "subsetix_csr_intersection_row_map_binary",
      Kokkos::RangePolicy<ExecSpace>(0, n_small),
      KOKKOS_LAMBDA(const std::size_t i) {
        const Coord y = rows_small(i).y;

        std::size_t lo = 0;
        std::size_t hi = n_big;
        while (lo < hi) {
          const std::size_t mid = (lo + hi) / 2;
          const Coord ymid = rows_big(mid).y;
          if (ymid < y) {
            lo = mid + 1;
          } else {
            hi = mid;
          }
        }

        if (lo < n_big && rows_big(lo).y == y) {
          flags(i) = 1;
          const int ia =
              small_is_a ? static_cast<int>(i)
                         : static_cast<int>(lo);
          const int ib =
              small_is_a ? static_cast<int>(lo)
                         : static_cast<int>(i);
          tmp_idx_a(i) = ia;
          tmp_idx_b(i) = ib;
        } else {
          flags(i) = 0;
          tmp_idx_a(i) = -1;
          tmp_idx_b(i) = -1;
        }
      });

  ExecSpace().fence();

  // 2) Exclusive scan on flags to compute positions and number of
  //    intersection rows.
  auto positions = workspace.positions;
  auto d_num_rows = workspace.d_num_rows;

  Kokkos::parallel_scan(
      "subsetix_csr_intersection_row_scan",
      Kokkos::RangePolicy<ExecSpace>(0, n_small),
      KOKKOS_LAMBDA(const std::size_t i,
                    std::size_t& update,
                    const bool final_pass) {
        const std::size_t flag =
            static_cast<std::size_t>(flags(i));
        if (final_pass) {
          positions(i) = update;
          if (i + 1 == n_small) {
            d_num_rows() = update + flag;
          }
        }
        update += flag;
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

  // 3) Compact matching rows into the output views.
  Kokkos::parallel_for(
      "subsetix_csr_intersection_row_compact",
      Kokkos::RangePolicy<ExecSpace>(0, n_small),
      KOKKOS_LAMBDA(const std::size_t i) {
        if (!flags(i)) {
          return;
        }
        const std::size_t pos = positions(i);
        out_rows(pos) = rows_small(i);
        out_idx_a(pos) = tmp_idx_a(i);
        out_idx_b(pos) = tmp_idx_b(i);
      });

  ExecSpace().fence();

  return result;
}

/**
 * @brief Convenience overload that uses a static workspace. This keeps
 * the existing API for tests/benchmarks while avoiding per-call
 * allocations in typical usage.
 */
inline RowMergeResult
build_row_intersection_mapping(const IntervalSet2DDevice& A,
                               const IntervalSet2DDevice& B) {
  RowIntersectionWorkspace workspace;
  return build_row_intersection_mapping(A, B, workspace);
}

struct RowDifferenceResult {
  IntervalSet2DDevice::RowKeyView row_keys;
  Kokkos::View<int*, DeviceMemorySpace> row_index_b;
  std::size_t num_rows = 0;
};

/**
 * @brief Shared workspace for row-wise counts and scans.
 *
 * This is reused across set operations (union/intersection/difference)
 * to avoid allocating separate row_counts/total_intervals buffers in
 * each algorithm.
 */
struct RowScanWorkspace {
  Kokkos::View<std::size_t*, DeviceMemorySpace> row_counts;
  Kokkos::View<std::size_t, DeviceMemorySpace> total_intervals;
  std::size_t capacity_rows = 0;

  void ensure_capacity(std::size_t rows) {
    if (rows <= capacity_rows) {
      return;
    }

    row_counts = Kokkos::View<std::size_t*, DeviceMemorySpace>(
        "subsetix_csr_row_counts_ws", rows);

    if (!total_intervals.data()) {
      total_intervals = Kokkos::View<std::size_t, DeviceMemorySpace>(
          "subsetix_csr_total_intervals_ws");
    }

    capacity_rows = rows;
  }
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

  // Build mapping A.rows -> B.rows using binary search on B.
  Kokkos::parallel_for(
      "subsetix_csr_difference_row_mapping",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
      KOKKOS_LAMBDA(const std::size_t ia) {
        const Coord ya = rows_a(ia).y;

        std::size_t lo = 0;
        std::size_t hi = num_rows_b;
        while (lo < hi) {
          const std::size_t mid = (lo + hi) / 2;
          const Coord yb = rows_b(mid).y;
          if (yb < ya) {
            lo = mid + 1;
          } else {
            hi = mid;
          }
        }

        if (lo < num_rows_b && rows_b(lo).y == ya) {
          out_idx_b(ia) = static_cast<int>(lo);
        } else {
          out_idx_b(ia) = -1;
        }
      });

  ExecSpace().fence();

  return result;
}

template <class Transform>
struct RowKeyTransformFunctor {
  IntervalSet2DDevice::RowKeyView row_keys_out;
  IntervalSet2DDevice::RowKeyView row_keys_in;
  Transform transform;

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t i) const {
    row_keys_out(i) = transform(row_keys_in(i));
  }
};

/**
 * @brief Apply a row-key transform on device, preserving row_ptr and
 *        intervals.
 *
 * The transform is a functor with signature:
 *   KOKKOS_INLINE_FUNCTION RowKey2D operator()(RowKey2D) const;
 *
 * This building block is used to implement translations in Y and can
 * be reused for other Y-only transforms.
 */
template <class Transform>
inline IntervalSet2DDevice
apply_row_key_transform_device(const IntervalSet2DDevice& in,
                               Transform transform) {
  IntervalSet2DDevice out;

  out.num_rows = in.num_rows;
  out.num_intervals = in.num_intervals;

  if (in.num_rows == 0) {
    out.row_keys = IntervalSet2DDevice::RowKeyView();
    out.row_ptr = in.row_ptr;
    out.intervals = in.intervals;
    return out;
  }

  IntervalSet2DDevice::RowKeyView row_keys_out(
      "subsetix_csr_row_key_transform", in.num_rows);

  RowKeyTransformFunctor<Transform> functor;
  functor.row_keys_out = row_keys_out;
  functor.row_keys_in = in.row_keys;
  functor.transform = transform;

  Kokkos::parallel_for(
      "subsetix_csr_row_key_transform_kernel",
      Kokkos::RangePolicy<ExecSpace>(0, in.num_rows),
      functor);

  ExecSpace().fence();

  out.row_keys = row_keys_out;
  out.row_ptr = in.row_ptr;
  out.intervals = in.intervals;

  return out;
}

KOKKOS_INLINE_FUNCTION
Coord floor_div2(Coord x) {
  // Floor division by 2 for both positive and negative coordinates.
  return (x >= 0) ? (x / 2) : ((x - 1) / 2);
}

KOKKOS_INLINE_FUNCTION
Coord ceil_div2(Coord x) {
  // Ceil division by 2 for both positive and negative coordinates.
  return (x >= 0) ? ((x + 1) / 2) : (x / 2);
}

template <class IntervalView>
struct CoarsenIntervalView {
  IntervalView base;

  KOKKOS_INLINE_FUNCTION
  Interval operator()(const std::size_t i) const {
    const Interval iv = base(i);
    const Coord b = floor_div2(iv.begin);
    const Coord e = ceil_div2(iv.end);
    return Interval{b, e};
  }
};

template <class Transform>
struct IntervalTransformFunctor {
  IntervalSet2DDevice::IntervalView intervals_out;
  IntervalSet2DDevice::IntervalView intervals_in;
  Transform transform;

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t i) const {
    intervals_out(i) = transform(intervals_in(i));
  }
};

/**
 * @brief Apply an interval transform on device, preserving row structure.
 *
 * The transform is a functor with signature:
 *   KOKKOS_INLINE_FUNCTION Interval operator()(Interval) const;
 */
template <class Transform>
inline IntervalSet2DDevice
apply_interval_transform_device(const IntervalSet2DDevice& in,
                                Transform transform) {
  IntervalSet2DDevice out;

  out.num_rows = in.num_rows;
  out.num_intervals = in.num_intervals;
  out.row_keys = in.row_keys;
  out.row_ptr = in.row_ptr;

  if (in.num_intervals == 0) {
    out.intervals = IntervalSet2DDevice::IntervalView();
    return out;
  }

  IntervalSet2DDevice::IntervalView intervals_out(
      "subsetix_csr_interval_transform", in.num_intervals);

  IntervalTransformFunctor<Transform> functor;
  functor.intervals_out = intervals_out;
  functor.intervals_in = in.intervals;
  functor.transform = transform;

  Kokkos::parallel_for(
      "subsetix_csr_interval_transform_kernel",
      Kokkos::RangePolicy<ExecSpace>(0, in.num_intervals),
      functor);

  ExecSpace().fence();

  out.intervals = intervals_out;
  return out;
}

struct TranslateXTransform {
  Coord dx;

  KOKKOS_INLINE_FUNCTION
  Interval operator()(const Interval& iv) const {
    Interval out_iv = iv;
    out_iv.begin = static_cast<Coord>(out_iv.begin + dx);
    out_iv.end = static_cast<Coord>(out_iv.end + dx);
    return out_iv;
  }
};

struct TranslateYTransform {
  Coord dy;

  KOKKOS_INLINE_FUNCTION
  RowKey2D operator()(const RowKey2D& key) const {
    RowKey2D out_key = key;
    out_key.y = static_cast<Coord>(out_key.y + dy);
    return out_key;
  }
};

struct RowCoarsenResult {
  IntervalSet2DDevice::RowKeyView row_keys;
  Kokkos::View<int*, DeviceMemorySpace> row_index_first;
  Kokkos::View<int*, DeviceMemorySpace> row_index_second;
  std::size_t num_rows = 0;
};

/**
 * @brief Build coarse-row mapping for a level-down projection (fine -> coarse).
 *
 * Each coarse row corresponds to either one or two fine rows whose Y-coordinates
 * map to the same coarse Y via floor_div2.
 */
inline RowCoarsenResult
build_row_coarsen_mapping(const IntervalSet2DDevice& fine) {
  RowCoarsenResult result;

  const std::size_t num_rows_fine = fine.num_rows;
  if (num_rows_fine == 0) {
    return result;
  }

  IntervalSet2DDevice::RowKeyView tmp_rows(
      "subsetix_csr_coarsen_tmp_rows", num_rows_fine);
  Kokkos::View<int*, DeviceMemorySpace> tmp_first(
      "subsetix_csr_coarsen_tmp_first", num_rows_fine);
  Kokkos::View<int*, DeviceMemorySpace> tmp_second(
      "subsetix_csr_coarsen_tmp_second", num_rows_fine);

  Kokkos::View<std::size_t, DeviceMemorySpace> d_num_rows(
      "subsetix_csr_coarsen_num_rows");

  auto rows_in = fine.row_keys;

  Kokkos::parallel_for(
      "subsetix_csr_coarsen_row_merge",
      Kokkos::RangePolicy<ExecSpace>(0, 1),
      KOKKOS_LAMBDA(const int) {
        std::size_t i = 0;
        std::size_t out = 0;

        while (i < num_rows_fine) {
          const Coord y_f0 = rows_in(i).y;
          const Coord y_c = floor_div2(y_f0);

          tmp_rows(out) = RowKey2D{y_c};
          tmp_first(out) = static_cast<int>(i);
          tmp_second(out) = -1;
          ++i;

          if (i < num_rows_fine) {
            const Coord y_f1 = rows_in(i).y;
            const Coord y_c1 = floor_div2(y_f1);
            if (y_c1 == y_c) {
              tmp_second(out) = static_cast<int>(i);
              ++i;
            }
          }

          ++out;
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
      "subsetix_csr_coarsen_row_keys", num_rows_out);
  result.row_index_first = Kokkos::View<int*, DeviceMemorySpace>(
      "subsetix_csr_coarsen_row_index_first", num_rows_out);
  result.row_index_second = Kokkos::View<int*, DeviceMemorySpace>(
      "subsetix_csr_coarsen_row_index_second", num_rows_out);

  auto out_rows = result.row_keys;
  auto out_first = result.row_index_first;
  auto out_second = result.row_index_second;

  Kokkos::parallel_for(
      "subsetix_csr_coarsen_row_merge_copy",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        out_rows(i) = tmp_rows(i);
        out_first(i) = tmp_first(i);
        out_second(i) = tmp_second(i);
      });

  ExecSpace().fence();

  return result;
}

} // namespace detail

/**
 * @brief Reset a preallocated IntervalSet2DDevice buffer.
 *
 * For now we simply fill the full capacity with zeros for all views.
 * TODO: consider restricting the zeroing to the portions actually used.
 */
inline void reset_preallocated_interval_set(IntervalSet2DDevice& out) {
  out.num_rows = 0;
  out.num_intervals = 0;

  const std::size_t rows_cap = out.row_keys.extent(0);
  const std::size_t row_ptr_cap = out.row_ptr.extent(0);
  const std::size_t intervals_cap = out.intervals.extent(0);

  if (rows_cap > 0) {
    RowKey2D zero_key{};
    Kokkos::deep_copy(out.row_keys, zero_key);
  }
  if (row_ptr_cap > 0) {
    Kokkos::deep_copy(out.row_ptr, std::size_t(0));
  }
  if (intervals_cap > 0) {
    Interval zero_interval{};
    Kokkos::deep_copy(out.intervals, zero_interval);
  }
}

/**
 * @brief Context object carrying reusable workspaces for CSR set
 *        algebra operations.
 *
 * This allows applications to chain many set operations on device
 * (union, intersection, difference, AMR ops) without paying device
 * allocations for scratch buffers on every call.
 */
struct CsrSetAlgebraContext {
  detail::RowIntersectionWorkspace intersection_workspace;
  detail::RowUnionWorkspace union_workspace;
  detail::RowScanWorkspace scan_workspace;
};

/**
 * @brief Compute the set union into a preallocated CSR buffer on device.
 *
 * The views in @p out must already be allocated with sufficient capacity:
/**
 * @brief Compute the set difference A \ B into a preallocated CSR buffer
 *        on device.
 *
 * The views in @p out must already be allocated with sufficient capacity:
 *  - out.row_keys.extent(0)   >= num_rows_out,
 *  - out.row_ptr.extent(0)    >= num_rows_out + 1,
 *  - out.intervals.extent(0)  >= num_intervals_out.
 *
 * If the capacity is insufficient, this function throws std::runtime_error.
 * The buffer is reset to zeros before writing into it.
 */
inline void
set_union_device(const IntervalSet2DDevice& A,
                 const IntervalSet2DDevice& B,
                 IntervalSet2DDevice& out,
                 CsrSetAlgebraContext& ctx) {
  reset_preallocated_interval_set(out);

  const std::size_t num_rows_a = A.num_rows;
  const std::size_t num_rows_b = B.num_rows;

  if (num_rows_a == 0 && num_rows_b == 0) {
    return;
  }

  detail::RowMergeResult merge =
      detail::build_row_union_mapping(A, B, ctx.union_workspace);

  const std::size_t num_rows_out = merge.num_rows;
  if (num_rows_out == 0) {
    return;
  }

  const std::size_t rows_cap = out.row_keys.extent(0);
  const std::size_t row_ptr_cap = out.row_ptr.extent(0);

  if (num_rows_out > rows_cap || num_rows_out + 1 > row_ptr_cap) {
    throw std::runtime_error(
        "subsetix::csr::set_union_device (preallocated): "
        "insufficient row capacity in output IntervalSet2DDevice");
  }

  auto row_keys_out_tmp = merge.row_keys;
  auto row_index_a = merge.row_index_a;
  auto row_index_b = merge.row_index_b;

  auto row_keys_out = out.row_keys;

  Kokkos::parallel_for(
      "subsetix_csr_union_copy_row_keys",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        row_keys_out(i) = row_keys_out_tmp(i);
      });

  ExecSpace().fence();

  ctx.scan_workspace.ensure_capacity(num_rows_out);
  auto row_counts = ctx.scan_workspace.row_counts;

  auto row_ptr_a = A.row_ptr;
  auto row_ptr_b = B.row_ptr;
  auto intervals_a = A.intervals;
  auto intervals_b = B.intervals;

  Kokkos::parallel_for(
      "subsetix_csr_union_count_prealloc",
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

  auto total_intervals = ctx.scan_workspace.total_intervals;

  auto row_ptr_out = out.row_ptr;

  Kokkos::parallel_scan(
      "subsetix_csr_union_scan_prealloc",
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

  const std::size_t intervals_cap = out.intervals.extent(0);
  if (num_intervals_out > intervals_cap) {
    throw std::runtime_error(
        "subsetix::csr::set_union_device (preallocated): "
        "insufficient interval capacity in output IntervalSet2DDevice");
  }

  out.num_rows = num_rows_out;
  out.num_intervals = num_intervals_out;

  if (num_intervals_out == 0) {
    return;
  }

  auto intervals_out = out.intervals;

  Kokkos::parallel_for(
      "subsetix_csr_union_fill_prealloc",
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
}

/**
 * @brief Compute the set intersection into a preallocated CSR buffer on
 *        device.
 *
 * The views in @p out must already be allocated with sufficient capacity:
 *  - out.row_keys.extent(0)   >= num_rows_out,
 *  - out.row_ptr.extent(0)    >= num_rows_out + 1,
 *  - out.intervals.extent(0)  >= num_intervals_out.
 *
 * If the capacity is insufficient, this function throws std::runtime_error.
 * The buffer is reset to zeros before writing into it.
 */
inline void
set_intersection_device(const IntervalSet2DDevice& A,
                        const IntervalSet2DDevice& B,
                        IntervalSet2DDevice& out,
                        CsrSetAlgebraContext& ctx) {
  reset_preallocated_interval_set(out);

  const std::size_t num_rows_a = A.num_rows;
  const std::size_t num_rows_b = B.num_rows;

  if (num_rows_a == 0 || num_rows_b == 0) {
    return;
  }

  detail::RowMergeResult merge =
      detail::build_row_intersection_mapping(
          A, B, ctx.intersection_workspace);
  const std::size_t num_rows_out = merge.num_rows;
  if (num_rows_out == 0) {
    return;
  }

  const std::size_t rows_cap = out.row_keys.extent(0);
  const std::size_t row_ptr_cap = out.row_ptr.extent(0);

  if (num_rows_out > rows_cap || num_rows_out + 1 > row_ptr_cap) {
    throw std::runtime_error(
        "subsetix::csr::set_intersection_device (preallocated): "
        "insufficient row capacity in output IntervalSet2DDevice");
  }

  ctx.scan_workspace.ensure_capacity(num_rows_out);
  auto row_counts = ctx.scan_workspace.row_counts;

  auto row_keys_out_tmp = merge.row_keys;
  auto row_index_a = merge.row_index_a;
  auto row_index_b = merge.row_index_b;
  auto row_keys_out = out.row_keys;

  auto row_ptr_a = A.row_ptr;
  auto row_ptr_b = B.row_ptr;
  auto intervals_a = A.intervals;
  auto intervals_b = B.intervals;

  Kokkos::parallel_for(
      "subsetix_csr_intersection_copy_and_count_prealloc",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        row_keys_out(i) = row_keys_out_tmp(i);

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

  auto total_intervals = ctx.scan_workspace.total_intervals;

  auto row_ptr_out = out.row_ptr;

  Kokkos::parallel_scan(
      "subsetix_csr_intersection_scan_prealloc",
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

  const std::size_t intervals_cap = out.intervals.extent(0);
  if (num_intervals_out > intervals_cap) {
    throw std::runtime_error(
        "subsetix::csr::set_intersection_device (preallocated): "
        "insufficient interval capacity in output IntervalSet2DDevice");
  }

  out.num_rows = num_rows_out;
  out.num_intervals = num_intervals_out;

  if (num_intervals_out == 0) {
    return;
  }

  auto intervals_out = out.intervals;

  Kokkos::parallel_for(
      "subsetix_csr_intersection_fill_prealloc",
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

        const std::size_t out_offset = row_ptr_out(i);
        detail::row_intersection_fill(
            intervals_a, begin_a, end_a,
            intervals_b, begin_b, end_b,
            intervals_out, out_offset);
      });

  ExecSpace().fence();
}

/**
 * @brief Compute the set difference A \ B into a preallocated CSR buffer
 *        on device.
 *
 * The views in @p out must already be allocated with sufficient capacity:
 *  - out.row_keys.extent(0)   >= num_rows_out,
 *  - out.row_ptr.extent(0)    >= num_rows_out + 1,
 *  - out.intervals.extent(0)  >= num_intervals_out.
 *
 * If the capacity is insufficient, this function throws std::runtime_error.
 * The buffer is reset to zeros before writing into it.
 */
inline void
set_difference_device(const IntervalSet2DDevice& A,
                      const IntervalSet2DDevice& B,
                      IntervalSet2DDevice& out,
                      CsrSetAlgebraContext& ctx) {
  reset_preallocated_interval_set(out);

  const std::size_t num_rows_a = A.num_rows;
  if (num_rows_a == 0) {
    return;
  }

  detail::RowDifferenceResult diff_rows =
      detail::build_row_difference_mapping(A, B);
  const std::size_t num_rows_out = diff_rows.num_rows;

  if (num_rows_out == 0) {
    return;
  }

  const std::size_t rows_cap = out.row_keys.extent(0);
  const std::size_t row_ptr_cap = out.row_ptr.extent(0);

  if (num_rows_out > rows_cap || num_rows_out + 1 > row_ptr_cap) {
    throw std::runtime_error(
        "subsetix::csr::set_difference_device (preallocated): "
        "insufficient row capacity in output IntervalSet2DDevice");
  }

  auto row_keys_out_tmp = diff_rows.row_keys;
  auto row_index_b = diff_rows.row_index_b;
  auto row_keys_out = out.row_keys;

  Kokkos::parallel_for(
      "subsetix_csr_difference_copy_row_keys",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
      KOKKOS_LAMBDA(const std::size_t i) {
        row_keys_out(i) = row_keys_out_tmp(i);
      });

  ExecSpace().fence();

  ctx.scan_workspace.ensure_capacity(num_rows_out);
  auto row_counts = ctx.scan_workspace.row_counts;

  auto row_ptr_a = A.row_ptr;
  auto row_ptr_b = B.row_ptr;
  auto intervals_a = A.intervals;
  auto intervals_b = B.intervals;

  Kokkos::parallel_for(
      "subsetix_csr_difference_count_prealloc",
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

  auto total_intervals = ctx.scan_workspace.total_intervals;
  auto row_ptr_out = out.row_ptr;

  Kokkos::parallel_scan(
      "subsetix_csr_difference_scan_prealloc",
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

  const std::size_t intervals_cap = out.intervals.extent(0);
  if (num_intervals_out > intervals_cap) {
    throw std::runtime_error(
        "subsetix::csr::set_difference_device (preallocated): "
        "insufficient interval capacity in output IntervalSet2DDevice");
  }

  out.num_rows = num_rows_out;
  out.num_intervals = num_intervals_out;

  if (num_intervals_out == 0) {
    return;
  }

  auto intervals_out = out.intervals;

  Kokkos::parallel_for(
      "subsetix_csr_difference_fill_prealloc",
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
}

/**
 * @brief Refine a CSR interval set to the next finer level (×2 in X and Y).
 *
 * Coordinates are scaled by 2 on both axes. This is a purely geometric
 * operation; row structure is preserved.
 */
inline IntervalSet2DDevice
refine_level_up_device(const IntervalSet2DDevice& in) {
  IntervalSet2DDevice out;
  const std::size_t num_rows_in = in.num_rows;
  const std::size_t num_intervals_in = in.num_intervals;

  if (num_rows_in == 0) {
    return out;
  }

  const std::size_t num_rows_out = num_rows_in * 2;
  const std::size_t num_intervals_out = num_intervals_in * 2;

  IntervalSet2DDevice::RowKeyView row_keys_out(
      "subsetix_csr_refine_row_keys", num_rows_out);
  IntervalSet2DDevice::IndexView row_ptr_out(
      "subsetix_csr_refine_row_ptr", num_rows_out + 1);

  IntervalSet2DDevice::IntervalView intervals_out;
  if (num_intervals_out > 0) {
    intervals_out = IntervalSet2DDevice::IntervalView(
        "subsetix_csr_refine_intervals", num_intervals_out);
  }

  auto row_keys_in = in.row_keys;
  auto row_ptr_in = in.row_ptr;
  auto intervals_in = in.intervals;

  Kokkos::parallel_for(
      "subsetix_csr_refine_row_keys",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_in),
      KOKKOS_LAMBDA(const std::size_t i) {
        const Coord y = row_keys_in(i).y;
        row_keys_out(2 * i) = RowKey2D{
            static_cast<Coord>(2 * y)};
        row_keys_out(2 * i + 1) = RowKey2D{
            static_cast<Coord>(2 * y + 1)};
      });

  Kokkos::parallel_for(
      "subsetix_csr_refine_row_ptr",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_in),
      KOKKOS_LAMBDA(const std::size_t i) {
        const std::size_t start = row_ptr_in(i);
        const std::size_t end = row_ptr_in(i + 1);
        const std::size_t count = end - start;
        row_ptr_out(2 * i) = 2 * start;
        row_ptr_out(2 * i + 1) = 2 * start + count;
        if (i + 1 == num_rows_in) {
          row_ptr_out(2 * num_rows_in) =
              2 * row_ptr_in(num_rows_in);
        }
      });

  if (num_intervals_out > 0) {
    Kokkos::parallel_for(
        "subsetix_csr_refine_intervals",
        Kokkos::RangePolicy<ExecSpace>(0, num_rows_in),
        KOKKOS_LAMBDA(const std::size_t i) {
          const std::size_t start = row_ptr_in(i);
          const std::size_t end = row_ptr_in(i + 1);
          const std::size_t count = end - start;
          const std::size_t out_even = 2 * start;
          const std::size_t out_odd = out_even + count;
          for (std::size_t k = 0; k < count; ++k) {
            const Interval iv = intervals_in(start + k);
            const Interval refined{
                static_cast<Coord>(iv.begin * 2),
                static_cast<Coord>(iv.end * 2)};
            intervals_out(out_even + k) = refined;
            intervals_out(out_odd + k) = refined;
          }
        });
  }

  out.num_rows = num_rows_out;
  out.num_intervals = num_intervals_out;
  out.row_keys = row_keys_out;
  out.row_ptr = row_ptr_out;
  out.intervals = intervals_out;

  return out;
}

/**
 * @brief Project a CSR interval set to the next coarser level (÷2 in X and Y),
 *        covering the fine geometry.
 *
 * Semantics (1D example):
 *   level1 [0,1) -> level0 [0,1)
 *   level1 [0,3) -> level0 [0,2)
 */
inline IntervalSet2DDevice
project_level_down_device(const IntervalSet2DDevice& fine) {
  IntervalSet2DDevice out;

  const std::size_t num_rows_fine = fine.num_rows;
  const std::size_t num_intervals_fine = fine.num_intervals;

  if (num_rows_fine == 0 || num_intervals_fine == 0) {
    return out;
  }

  detail::RowCoarsenResult rows =
      detail::build_row_coarsen_mapping(fine);
  const std::size_t num_rows_coarse = rows.num_rows;
  if (num_rows_coarse == 0) {
    return out;
  }

  auto row_keys_coarse = rows.row_keys;
  auto row_index_first = rows.row_index_first;
  auto row_index_second = rows.row_index_second;

  auto row_ptr_fine = fine.row_ptr;
  auto intervals_fine = fine.intervals;

  using CoarsenView =
      detail::CoarsenIntervalView<IntervalSet2DDevice::IntervalView>;
  CoarsenView coarsen_view{intervals_fine};

  Kokkos::View<std::size_t*, DeviceMemorySpace> row_counts(
      "subsetix_csr_project_row_counts", num_rows_coarse);

  Kokkos::parallel_for(
      "subsetix_csr_project_count",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_coarse),
      KOKKOS_LAMBDA(const std::size_t i) {
        const int ia = row_index_first(i);
        const int ib = row_index_second(i);

        std::size_t begin_a = 0;
        std::size_t end_a = 0;
        std::size_t begin_b = 0;
        std::size_t end_b = 0;

        if (ia >= 0) {
          const std::size_t row_a = static_cast<std::size_t>(ia);
          begin_a = row_ptr_fine(row_a);
          end_a = row_ptr_fine(row_a + 1);
        }
        if (ib >= 0) {
          const std::size_t row_b = static_cast<std::size_t>(ib);
          begin_b = row_ptr_fine(row_b);
          end_b = row_ptr_fine(row_b + 1);
        }

        if (begin_a == end_a && begin_b == end_b) {
          row_counts(i) = 0;
          return;
        }

        row_counts(i) = detail::row_union_count(
            coarsen_view, begin_a, end_a,
            coarsen_view, begin_b, end_b);
      });

  ExecSpace().fence();

  IntervalSet2DDevice::IndexView row_ptr_out(
      "subsetix_csr_project_row_ptr", num_rows_coarse + 1);
  Kokkos::View<std::size_t, DeviceMemorySpace> total_intervals(
      "subsetix_csr_project_total_intervals");

  Kokkos::parallel_scan(
      "subsetix_csr_project_scan",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_coarse),
      KOKKOS_LAMBDA(const std::size_t i,
                    std::size_t& update,
                    const bool final_pass) {
        const std::size_t c = row_counts(i);
        if (final_pass) {
          row_ptr_out(i) = update;
          if (i + 1 == num_rows_coarse) {
            row_ptr_out(num_rows_coarse) = update + c;
            total_intervals() = update + c;
          }
        }
        update += c;
      });

  ExecSpace().fence();

  std::size_t num_intervals_coarse = 0;
  Kokkos::deep_copy(num_intervals_coarse, total_intervals);

  out.num_rows = num_rows_coarse;
  out.num_intervals = num_intervals_coarse;
  out.row_keys = row_keys_coarse;
  out.row_ptr = row_ptr_out;

  if (num_intervals_coarse == 0) {
    out.intervals = IntervalSet2DDevice::IntervalView();
    return out;
  }

  IntervalSet2DDevice::IntervalView intervals_out(
      "subsetix_csr_project_intervals", num_intervals_coarse);

  Kokkos::parallel_for(
      "subsetix_csr_project_fill",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_coarse),
      KOKKOS_LAMBDA(const std::size_t i) {
        const int ia = row_index_first(i);
        const int ib = row_index_second(i);

        std::size_t begin_a = 0;
        std::size_t end_a = 0;
        std::size_t begin_b = 0;
        std::size_t end_b = 0;

        if (ia >= 0) {
          const std::size_t row_a = static_cast<std::size_t>(ia);
          begin_a = row_ptr_fine(row_a);
          end_a = row_ptr_fine(row_a + 1);
        }
        if (ib >= 0) {
          const std::size_t row_b = static_cast<std::size_t>(ib);
          begin_b = row_ptr_fine(row_b);
          end_b = row_ptr_fine(row_b + 1);
        }

        if (begin_a == end_a && begin_b == end_b) {
          return;
        }

        const std::size_t out_offset = row_ptr_out(i);
        detail::row_union_fill(coarsen_view, begin_a, end_a,
                               coarsen_view, begin_b, end_b,
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
  if (dx == 0) {
    return in;
  }
  detail::TranslateXTransform transform{dx};
  return detail::apply_interval_transform_device(in, transform);
}

/**
 * @brief Translate all rows of a CSR interval set by a constant offset
 *        along Y.
 *
 * Interval structure and row_ptr are preserved; only row_keys(i).y are
 * shifted by dy.
 */
inline IntervalSet2DDevice
translate_y_device(const IntervalSet2DDevice& in,
                   Coord dy) {
  if (dy == 0 || in.num_rows == 0) {
    return in;
  }
  detail::TranslateYTransform transform{dy};
  return detail::apply_row_key_transform_device(in, transform);
}

} // namespace csr
} // namespace subsetix
