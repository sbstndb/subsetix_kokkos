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
  IntervalSet2DDevice::RowKeyView row_keys;
  Kokkos::View<int*, DeviceMemorySpace> row_index_a;
  Kokkos::View<int*, DeviceMemorySpace> row_index_b;
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

    row_keys = IntervalSet2DDevice::RowKeyView(
        "subsetix_csr_intersection_row_keys_ws", n_small);
    row_index_a = Kokkos::View<int*, DeviceMemorySpace>(
        "subsetix_csr_intersection_row_index_a_ws", n_small);
    row_index_b = Kokkos::View<int*, DeviceMemorySpace>(
        "subsetix_csr_intersection_row_index_b_ws", n_small);

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

/**
 * @brief Count the number of intervals in the symmetric difference
 *        (A XOR B) = (A \ B) U (B \ A) on a single row.
 */
template <class IntervalView>
KOKKOS_INLINE_FUNCTION
std::size_t row_symmetric_difference_count(const IntervalView& intervals_a,
                                           std::size_t begin_a,
                                           std::size_t end_a,
                                           const IntervalView& intervals_b,
                                           std::size_t begin_b,
                                           std::size_t end_b) {
  std::size_t ia = begin_a;
  std::size_t ib = begin_b;
  std::size_t count = 0;

  bool in_a = false;
  bool in_b = false;
  bool xor_active = false;

  // Loop until both interval lists are fully exhausted
  while (ia < end_a || ib < end_b || in_a || in_b) {
    // Determine next event coordinate.
    // Use a large value for exhausted streams.
    // Be careful with potential overflow if using MAX, but coordinates are int32.
    // We can use conditions.

    Coord next_a_pos = 0;
    bool has_next_a = false;
    if (in_a) {
      next_a_pos = intervals_a(ia).end; // current A end
      has_next_a = true;
    } else if (ia < end_a) {
      next_a_pos = intervals_a(ia).begin; // next A start
      has_next_a = true;
    }

    Coord next_b_pos = 0;
    bool has_next_b = false;
    if (in_b) {
      next_b_pos = intervals_b(ib).end; // current B end
      has_next_b = true;
    } else if (ib < end_b) {
      next_b_pos = intervals_b(ib).begin; // next B start
      has_next_b = true;
    }

    if (!has_next_a && !has_next_b) {
      break;
    }

    Coord p = 0;
    if (has_next_a && has_next_b) {
      p = (next_a_pos < next_b_pos) ? next_a_pos : next_b_pos;
    } else if (has_next_a) {
      p = next_a_pos;
    } else {
      p = next_b_pos;
    }

    // Process events at p
    if (has_next_a && next_a_pos == p) {
      if (in_a) {
        in_a = false;
        ++ia; // Consumed this interval
      } else {
        in_a = true;
      }
    }

    if (has_next_b && next_b_pos == p) {
      if (in_b) {
        in_b = false;
        ++ib; // Consumed this interval
      } else {
        in_b = true;
      }
    }

    // Check new state
    const bool new_xor = (in_a != in_b);
    if (xor_active != new_xor) {
      if (!new_xor) {
        // XOR region ended
        ++count;
      }
      xor_active = new_xor;
    }
  }

  return count;
}

/**
 * @brief Fill the symmetric difference intervals for a single row.
 *        Must be kept in sync with row_symmetric_difference_count.
 */
template <class IntervalViewIn, class IntervalViewOut>
KOKKOS_INLINE_FUNCTION
void row_symmetric_difference_fill(const IntervalViewIn& intervals_a,
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

  bool in_a = false;
  bool in_b = false;
  bool xor_active = false;
  Coord start_pos = 0;

  while (ia < end_a || ib < end_b || in_a || in_b) {
    Coord next_a_pos = 0;
    bool has_next_a = false;
    if (in_a) {
      next_a_pos = intervals_a(ia).end;
      has_next_a = true;
    } else if (ia < end_a) {
      next_a_pos = intervals_a(ia).begin;
      has_next_a = true;
    }

    Coord next_b_pos = 0;
    bool has_next_b = false;
    if (in_b) {
      next_b_pos = intervals_b(ib).end;
      has_next_b = true;
    } else if (ib < end_b) {
      next_b_pos = intervals_b(ib).begin;
      has_next_b = true;
    }

    if (!has_next_a && !has_next_b) {
      break;
    }

    Coord p = 0;
    if (has_next_a && has_next_b) {
      p = (next_a_pos < next_b_pos) ? next_a_pos : next_b_pos;
    } else if (has_next_a) {
      p = next_a_pos;
    } else {
      p = next_b_pos;
    }

    if (has_next_a && next_a_pos == p) {
      if (in_a) {
        in_a = false;
        ++ia;
      } else {
        in_a = true;
      }
    }

    if (has_next_b && next_b_pos == p) {
      if (in_b) {
        in_b = false;
        ++ib;
      } else {
        in_b = true;
      }
    }

    const bool new_xor = (in_a != in_b);
    if (xor_active != new_xor) {
      if (new_xor) {
        start_pos = p;
      } else {
        intervals_out(out_offset + write_idx) = Interval{start_pos, p};
        ++write_idx;
      }
      xor_active = new_xor;
    }
  }
}

struct RowMergeResult {
  IntervalSet2DDevice::RowKeyView row_keys;
  Kokkos::View<int*, DeviceMemorySpace> row_index_a;
  Kokkos::View<int*, DeviceMemorySpace> row_index_b;
  std::size_t num_rows = 0;
};

struct RowUnionWorkspace {
  Kokkos::View<int*, DeviceMemorySpace> map_a_to_b;
  Kokkos::View<int*, DeviceMemorySpace> map_b_to_a;
  Kokkos::View<int*, DeviceMemorySpace> b_only_flags;
  Kokkos::View<std::size_t*, DeviceMemorySpace> b_only_positions;
  Kokkos::View<int*, DeviceMemorySpace> b_only_indices;
   IntervalSet2DDevice::RowKeyView row_keys;
   Kokkos::View<int*, DeviceMemorySpace> row_index_a;
   Kokkos::View<int*, DeviceMemorySpace> row_index_b;
  Kokkos::View<std::size_t, DeviceMemorySpace> d_num_b_only;
  std::size_t capacity_a = 0;
  std::size_t capacity_b = 0;
   std::size_t capacity_rows_out = 0;

  void ensure_capacity(std::size_t num_rows_a,
                       std::size_t num_rows_b) {
    const std::size_t needed_rows_out =
        num_rows_a + num_rows_b;

    if (num_rows_a > capacity_a) {
      map_a_to_b = Kokkos::View<int*, DeviceMemorySpace>(
          "subsetix_csr_union_map_a_to_b", num_rows_a);
      capacity_a = num_rows_a;
    }

    if (num_rows_b > capacity_b) {
      map_b_to_a = Kokkos::View<int*, DeviceMemorySpace>(
          "subsetix_csr_union_map_b_to_a", num_rows_b);
      b_only_flags = Kokkos::View<int*, DeviceMemorySpace>(
          "subsetix_csr_union_b_only_flags", num_rows_b);
      b_only_positions =
          Kokkos::View<std::size_t*, DeviceMemorySpace>(
              "subsetix_csr_union_b_only_positions",
              num_rows_b);
      b_only_indices = Kokkos::View<int*, DeviceMemorySpace>(
          "subsetix_csr_union_b_only_indices",
          num_rows_b);
      capacity_b = num_rows_b;
    }

    if (needed_rows_out > capacity_rows_out) {
      row_keys = IntervalSet2DDevice::RowKeyView(
          "subsetix_csr_union_row_keys_ws",
          needed_rows_out);
      row_index_a = Kokkos::View<int*, DeviceMemorySpace>(
          "subsetix_csr_union_row_index_a_ws",
          needed_rows_out);
      row_index_b = Kokkos::View<int*, DeviceMemorySpace>(
          "subsetix_csr_union_row_index_b_ws",
          needed_rows_out);
      capacity_rows_out = needed_rows_out;
    }

    if (!d_num_b_only.data()) {
      d_num_b_only = Kokkos::View<std::size_t, DeviceMemorySpace>(
          "subsetix_csr_union_num_b_only");
    }
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

  // Fast paths for empty inputs.
  if (num_rows_a == 0) {
    workspace.ensure_capacity(num_rows_a, num_rows_b);

    result.num_rows = num_rows_b;
    if (num_rows_b == 0) {
      return result;
    }

    result.row_keys = workspace.row_keys;
    result.row_index_a = workspace.row_index_a;
    result.row_index_b = workspace.row_index_b;

    auto out_rows = workspace.row_keys;
    auto out_idx_a = workspace.row_index_a;
    auto out_idx_b = workspace.row_index_b;

    Kokkos::parallel_for(
        "subsetix_csr_union_rows_b_only",
        Kokkos::RangePolicy<ExecSpace>(0, num_rows_b),
        KOKKOS_LAMBDA(const std::size_t j) {
          out_rows(j) = rows_b(j);
          out_idx_a(j) = -1;
          out_idx_b(j) = static_cast<int>(j);
        });

    ExecSpace().fence();
    return result;
  }

  if (num_rows_b == 0) {
    workspace.ensure_capacity(num_rows_a, num_rows_b);

    result.num_rows = num_rows_a;
    result.row_keys = workspace.row_keys;
    result.row_index_a = workspace.row_index_a;
    result.row_index_b = workspace.row_index_b;

    auto out_rows = workspace.row_keys;
    auto out_idx_a = workspace.row_index_a;
    auto out_idx_b = workspace.row_index_b;

    Kokkos::parallel_for(
        "subsetix_csr_union_rows_a_only",
        Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
        KOKKOS_LAMBDA(const std::size_t i) {
          out_rows(i) = rows_a(i);
          out_idx_a(i) = static_cast<int>(i);
          out_idx_b(i) = -1;
        });

    ExecSpace().fence();
    return result;
  }

  // General case: A and B both non-empty. We exploit that row_keys in
  // A and B are individually sorted by Y to build the union mapping
  // with parallel binary searches and scans, without a global sort.
  workspace.ensure_capacity(num_rows_a, num_rows_b);

  auto map_a_to_b = workspace.map_a_to_b;
  auto map_b_to_a = workspace.map_b_to_a;
  auto b_only_flags = workspace.b_only_flags;
  auto b_only_positions = workspace.b_only_positions;
  auto b_only_indices = workspace.b_only_indices;
  auto d_num_b_only = workspace.d_num_b_only;

  // 1) For each row of A, binary-search its Y in B.
  Kokkos::parallel_for(
      "subsetix_csr_union_map_a_to_b",
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
          map_a_to_b(ia) = static_cast<int>(lo);
        } else {
          map_a_to_b(ia) = -1;
        }
      });

  ExecSpace().fence();

  // 2) For each row of B, binary-search its Y in A.
  Kokkos::parallel_for(
      "subsetix_csr_union_map_b_to_a",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_b),
      KOKKOS_LAMBDA(const std::size_t ib) {
        const Coord yb = rows_b(ib).y;

        std::size_t lo = 0;
        std::size_t hi = num_rows_a;
        while (lo < hi) {
          const std::size_t mid = (lo + hi) / 2;
          const Coord ya = rows_a(mid).y;
          if (ya < yb) {
            lo = mid + 1;
          } else {
            hi = mid;
          }
        }

        if (lo < num_rows_a && rows_a(lo).y == yb) {
          map_b_to_a(ib) = static_cast<int>(lo);
        } else {
          map_b_to_a(ib) = -1;
        }
      });

  ExecSpace().fence();

  // 3) Extract rows of B that are not present in A.
  Kokkos::parallel_for(
      "subsetix_csr_union_mark_b_only",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_b),
      KOKKOS_LAMBDA(const std::size_t ib) {
        b_only_flags(ib) = (map_b_to_a(ib) < 0) ? 1 : 0;
      });

  ExecSpace().fence();

  Kokkos::parallel_scan(
      "subsetix_csr_union_b_only_scan",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_b),
      KOKKOS_LAMBDA(const std::size_t ib,
                    std::size_t& update,
                    const bool final_pass) {
        const std::size_t flag =
            static_cast<std::size_t>(b_only_flags(ib));
        if (final_pass) {
          b_only_positions(ib) = update;
          if (ib + 1 == num_rows_b) {
            d_num_b_only() = update + flag;
          }
        }
        update += flag;
      });

  ExecSpace().fence();

  std::size_t num_b_only = 0;
  Kokkos::deep_copy(num_b_only, d_num_b_only);

  if (num_b_only > 0) {
    Kokkos::parallel_for(
        "subsetix_csr_union_compact_b_only",
        Kokkos::RangePolicy<ExecSpace>(0, num_rows_b),
        KOKKOS_LAMBDA(const std::size_t ib) {
          if (!b_only_flags(ib)) {
            return;
          }
          const std::size_t pos = b_only_positions(ib);
          b_only_indices(pos) = static_cast<int>(ib);
        });

    ExecSpace().fence();
  }

  const std::size_t num_rows_out =
      num_rows_a + num_b_only;

  result.num_rows = num_rows_out;
  if (num_rows_out == 0) {
    return result;
  }

  result.row_keys = workspace.row_keys;
  result.row_index_a = workspace.row_index_a;
  result.row_index_b = workspace.row_index_b;

  auto out_rows = workspace.row_keys;
  auto out_idx_a = workspace.row_index_a;
  auto out_idx_b = workspace.row_index_b;

  // 4) Write rows coming from A. For each row in A we count how many
  // B-only rows precede it, using a binary search in the compacted
  // B-only list, and place it at its union rank.
  Kokkos::parallel_for(
      "subsetix_csr_union_write_from_a",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_a),
      KOKKOS_LAMBDA(const std::size_t ia) {
        const Coord ya = rows_a(ia).y;

        std::size_t lo = 0;
        std::size_t hi = num_b_only;
        while (lo < hi) {
          const std::size_t mid = (lo + hi) / 2;
          const int jb = b_only_indices(mid);
          const Coord yb = rows_b(jb).y;
          if (yb < ya) {
            lo = mid + 1;
          } else {
            hi = mid;
          }
        }

        const std::size_t cnt_b_before = lo;
        const std::size_t pos =
            ia + cnt_b_before;

        out_rows(pos) = rows_a(ia);
        out_idx_a(pos) = static_cast<int>(ia);
        out_idx_b(pos) = map_a_to_b(ia);
      });

  ExecSpace().fence();

  // 5) Write rows that exist only in B. For each such row we count how
  // many A rows precede it using a binary search in A, and place it at
  // its union rank.
  if (num_b_only > 0) {
    Kokkos::parallel_for(
        "subsetix_csr_union_write_from_b_only",
        Kokkos::RangePolicy<ExecSpace>(0, num_b_only),
        KOKKOS_LAMBDA(const std::size_t k) {
          const int jb = b_only_indices(k);
          const Coord yb = rows_b(jb).y;

          std::size_t lo = 0;
          std::size_t hi = num_rows_a;
          while (lo < hi) {
            const std::size_t mid = (lo + hi) / 2;
            const Coord ya = rows_a(mid).y;
            if (ya < yb) {
              lo = mid + 1;
            } else {
              hi = mid;
            }
          }

          const std::size_t cnt_a_before = lo;
          const std::size_t pos =
              cnt_a_before + k;

          out_rows(pos) = rows_b(jb);
          out_idx_a(pos) = -1;
          out_idx_b(pos) = jb;
        });

    ExecSpace().fence();
  }

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

  // Reuse preallocated mapping buffers from the workspace; only the
  // first num_rows_out entries are written and used by callers.
  result.row_keys = workspace.row_keys;
  result.row_index_a = workspace.row_index_a;
  result.row_index_b = workspace.row_index_b;

  auto out_rows = workspace.row_keys;
  auto out_idx_a = workspace.row_index_a;
  auto out_idx_b = workspace.row_index_b;

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
 * @brief Workspace for row-difference mapping (A rows -> B rows).
 *
 * Reuses mapping buffers across calls to avoid per-call allocations.
 */
struct RowDifferenceWorkspace {
  IntervalSet2DDevice::RowKeyView row_keys;
  Kokkos::View<int*, DeviceMemorySpace> row_index_b;
  std::size_t capacity_rows = 0;

  void ensure_capacity(std::size_t rows) {
    if (rows <= capacity_rows) {
      return;
    }

    row_keys = IntervalSet2DDevice::RowKeyView(
        "subsetix_csr_difference_row_keys_ws", rows);
    row_index_b = Kokkos::View<int*, DeviceMemorySpace>(
        "subsetix_csr_difference_row_index_b_ws", rows);

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
                             const IntervalSet2DDevice& B,
                             RowDifferenceWorkspace& workspace) {
  RowDifferenceResult result;

  const std::size_t num_rows_a = A.num_rows;
  const std::size_t num_rows_b = B.num_rows;

  if (num_rows_a == 0) {
    return result;
  }

  workspace.ensure_capacity(num_rows_a);

  result.num_rows = num_rows_a;
  result.row_keys = workspace.row_keys;
  result.row_index_b = workspace.row_index_b;

  auto rows_a = A.row_keys;
  auto rows_b = B.row_keys;
  auto out_rows = workspace.row_keys;
  auto out_idx_b = workspace.row_index_b;

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

/**
 * @brief Convenience overload using a local workspace, for tests and
 * small utilities.
 */
inline RowDifferenceResult
build_row_difference_mapping(const IntervalSet2DDevice& A,
                             const IntervalSet2DDevice& B) {
  RowDifferenceWorkspace workspace;
  return build_row_difference_mapping(A, B, workspace);
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
 * @brief Workspace for translation operations.
 *
 * Reuses buffers across calls to avoid per-call allocations.
 */
struct TranslationWorkspace {
  IntervalSet2DDevice::RowKeyView row_keys_out;
  IntervalSet2DDevice::IntervalView intervals_out;
  std::size_t capacity_rows = 0;
  std::size_t capacity_intervals = 0;

  void ensure_capacity_rows(std::size_t rows) {
    if (rows <= capacity_rows) {
      return;
    }
    row_keys_out = IntervalSet2DDevice::RowKeyView(
        "subsetix_csr_translation_row_keys_ws", rows);
    capacity_rows = rows;
  }

  void ensure_capacity_intervals(std::size_t intervals) {
    if (intervals <= capacity_intervals) {
      return;
    }
    intervals_out = IntervalSet2DDevice::IntervalView(
        "subsetix_csr_translation_intervals_ws", intervals);
    capacity_intervals = intervals;
  }
};

/**
 * @brief Workspace for project (coarsening) operations to avoid per-call allocations.
 */
struct ProjectWorkspace {
  Kokkos::View<std::size_t*, DeviceMemorySpace> row_counts;
  Kokkos::View<std::size_t, DeviceMemorySpace> total_intervals;
  std::size_t capacity_rows = 0;

  void ensure_capacity_coarse(std::size_t num_rows_coarse) {
    if (num_rows_coarse <= capacity_rows) {
      return;
    }
    row_counts = Kokkos::View<std::size_t*, DeviceMemorySpace>(
        "subsetix_csr_project_row_counts_ws", num_rows_coarse);
    capacity_rows = num_rows_coarse;
  }
  
  void ensure_total_intervals() {
    if (!total_intervals.data()) {
      total_intervals = Kokkos::View<std::size_t, DeviceMemorySpace>(
          "subsetix_csr_project_total_intervals_ws");
    }
  }
};

/**
 * @brief Workspace for row coarsening operations to avoid per-call allocations.
 */
struct CoarsenWorkspace {
  IntervalSet2DDevice::RowKeyView tmp_rows;
  Kokkos::View<int*, DeviceMemorySpace> tmp_first;
  Kokkos::View<int*, DeviceMemorySpace> tmp_second;
  Kokkos::View<std::size_t, DeviceMemorySpace> d_num_rows;
  std::size_t capacity_rows = 0;

  void ensure_capacity(std::size_t num_rows_fine) {
    if (num_rows_fine <= capacity_rows) {
      return;
    }
    tmp_rows = IntervalSet2DDevice::RowKeyView(
        "subsetix_csr_coarsen_tmp_rows_ws", num_rows_fine);
    tmp_first = Kokkos::View<int*, DeviceMemorySpace>(
        "subsetix_csr_coarsen_tmp_first_ws", num_rows_fine);
    tmp_second = Kokkos::View<int*, DeviceMemorySpace>(
        "subsetix_csr_coarsen_tmp_second_ws", num_rows_fine);
    capacity_rows = num_rows_fine;
  }
  
  void ensure_d_num_rows() {
    if (!d_num_rows.data()) {
      d_num_rows = Kokkos::View<std::size_t, DeviceMemorySpace>(
          "subsetix_csr_coarsen_num_rows_ws");
    }
  }
};

/**
 * @brief Apply a row-key transform on device with workspace,
 *        preserving row_ptr and intervals.
 *
 * The transform is a functor with signature:
 *   KOKKOS_INLINE_FUNCTION RowKey2D operator()(RowKey2D) const;
 *
 * This building block is used to implement translations in Y and can
 * be reused for other Y-only transforms.
 */
template <class Transform>
inline void
apply_row_key_transform_device(const IntervalSet2DDevice& in,
                               Transform transform,
                               IntervalSet2DDevice& out,
                               TranslationWorkspace& workspace) {
  out.num_rows = in.num_rows;
  out.num_intervals = in.num_intervals;
  out.row_ptr = in.row_ptr;
  out.intervals = in.intervals;

  if (in.num_rows == 0) {
    out.row_keys = IntervalSet2DDevice::RowKeyView();
    return;
  }

  // Allocate independent output buffer for row_keys
  if (out.row_keys.extent(0) < in.num_rows) {
    out.row_keys = IntervalSet2DDevice::RowKeyView(
        "subsetix_csr_row_key_transform_out", in.num_rows);
  }

  RowKeyTransformFunctor<Transform> functor;
  functor.row_keys_out = out.row_keys;
  functor.row_keys_in = in.row_keys;
  functor.transform = transform;

  Kokkos::parallel_for(
      "subsetix_csr_row_key_transform_kernel",
      Kokkos::RangePolicy<ExecSpace>(0, in.num_rows),
      functor);

  ExecSpace().fence();
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
 * @brief Apply an interval transform on device with workspace,
 *        preserving row structure.
 *
 * This version reuses buffers from the workspace to avoid per-call
 * allocations. The transformed intervals are written into the provided
 * output buffer, avoiding the semantic regression of returning subviews.
 */
template <class Transform>
inline void
apply_interval_transform_device(const IntervalSet2DDevice& in,
                                Transform transform,
                                IntervalSet2DDevice& out,
                                TranslationWorkspace& workspace) {
  out.num_rows = in.num_rows;
  out.num_intervals = in.num_intervals;
  out.row_keys = in.row_keys;
  out.row_ptr = in.row_ptr;

  if (in.num_intervals == 0) {
    out.intervals = IntervalSet2DDevice::IntervalView();
    return;
  }

  // Allocate independent output buffer for intervals
  if (out.intervals.extent(0) < in.num_intervals) {
    out.intervals = IntervalSet2DDevice::IntervalView(
        "subsetix_csr_interval_transform_out", in.num_intervals);
  }

  IntervalTransformFunctor<Transform> functor;
  functor.intervals_out = out.intervals;
  functor.intervals_in = in.intervals;
  functor.transform = transform;

  Kokkos::parallel_for(
      "subsetix_csr_interval_transform_kernel",
      Kokkos::RangePolicy<ExecSpace>(0, in.num_intervals),
      functor);

  ExecSpace().fence();
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

/**
 * @brief Build coarse-row mapping with workspace for buffer reuse.
 *
 * This variant reuses buffers from the workspace to avoid repeated
 * allocations on GPU, which are very expensive.
 */
inline RowCoarsenResult
build_row_coarsen_mapping(const IntervalSet2DDevice& fine,
                          CoarsenWorkspace& workspace) {
  RowCoarsenResult result;

  const std::size_t num_rows_fine = fine.num_rows;
  if (num_rows_fine == 0) {
    return result;
  }

  // Use workspace buffers instead of allocating
  workspace.ensure_capacity(num_rows_fine);
  workspace.ensure_d_num_rows();

  auto tmp_rows = workspace.tmp_rows;
  auto tmp_first = workspace.tmp_first;
  auto tmp_second = workspace.tmp_second;
  auto d_num_rows = workspace.d_num_rows;

  auto rows_in = fine.row_keys;

  // Parallel: compute coarse Y for each fine row and mark boundaries
  Kokkos::View<int*, DeviceMemorySpace> is_boundary(
      "subsetix_coarsen_is_boundary", num_rows_fine);
  Kokkos::View<Coord*, DeviceMemorySpace> coarse_y(
      "subsetix_coarsen_y_vals", num_rows_fine);

  Kokkos::parallel_for(
      "subsetix_csr_coarsen_compute_boundaries",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_fine),
      KOKKOS_LAMBDA(const std::size_t i) {
        const Coord y_f = rows_in(i).y;
        const Coord y_c = floor_div2(y_f);
        coarse_y(i) = y_c;
        
        // Mark boundary if first row or coarse Y changes
        if (i == 0) {
          is_boundary(i) = 1;
        } else {
          const Coord y_c_prev = floor_div2(rows_in(i - 1).y);
          is_boundary(i) = (y_c != y_c_prev) ? 1 : 0;
        }
      });

  ExecSpace().fence();

  // Scan to get output indices for boundaries
  Kokkos::View<int*, DeviceMemorySpace> boundary_scan(
      "subsetix_coarsen_boundary_scan", num_rows_fine);
  Kokkos::View<int, DeviceMemorySpace> d_total_boundaries(
      "subsetix_coarsen_total_boundaries");

  Kokkos::parallel_scan(
      "subsetix_csr_coarsen_scan_boundaries",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_fine),
      KOKKOS_LAMBDA(const std::size_t i, int& update, const bool final_pass) {
        const int val = is_boundary(i);
        if (final_pass) {
          boundary_scan(i) = update;
          if (i + 1 == num_rows_fine) {
            d_total_boundaries() = update + val;
          }
        }
        update += val;
      });

  ExecSpace().fence();

  int num_rows_out = 0;
  Kokkos::deep_copy(num_rows_out, d_total_boundaries);

  result.num_rows = static_cast<std::size_t>(num_rows_out);
  if (num_rows_out == 0) {
    return result;
  }

  // Fill output arrays in parallel
  Kokkos::parallel_for(
      "subsetix_csr_coarsen_fill_output",
      Kokkos::RangePolicy<ExecSpace>(0, num_rows_fine),
      KOKKOS_LAMBDA(const std::size_t i) {
        if (is_boundary(i)) {
          const int out_idx = boundary_scan(i);
          const Coord y_c = coarse_y(i);
          
          tmp_rows(out_idx) = RowKey2D{y_c};
          tmp_first(out_idx) = static_cast<int>(i);
          
          // Check if next row has same coarse Y
          if (i + 1 < num_rows_fine && coarse_y(i + 1) == y_c) {
            tmp_second(out_idx) = static_cast<int>(i + 1);
          } else {
            tmp_second(out_idx) = -1;
          }
        }
      });

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



/**
 * @brief Workspace for morphology operations (Shrink/Expand).
 */
struct MorphologyWorkspace {
  // For Expand: Map each output row to a range of input rows [start_idx, end_idx)
  // For Shrink: Map each output row to the central input row index, plus validity flag
  
  IntervalSet2DDevice::RowKeyView row_keys;
  Kokkos::View<int*, DeviceMemorySpace> row_map_start; // Expand: start index, Shrink: central index
  Kokkos::View<int*, DeviceMemorySpace> row_map_end;   // Expand: end index, Shrink: unused
  
  // For Expand row generation (Union of intervals)
  Kokkos::View<int*, DeviceMemorySpace> input_row_exists; 
  Kokkos::View<int*, DeviceMemorySpace> input_row_to_output_start;
  
  Kokkos::View<std::size_t, DeviceMemorySpace> d_num_rows;
  
  // Temporary buffers for build_*_row_mapping to avoid allocation
  Kokkos::View<std::size_t*, DeviceMemorySpace> row_block_counts;
  Kokkos::View<std::size_t*, DeviceMemorySpace> scan_offsets;
  Kokkos::View<std::size_t, DeviceMemorySpace> d_total_rows;
  
  std::size_t capacity_rows = 0;
  std::size_t capacity_input_map = 0;
  std::size_t capacity_temps = 0;

  void ensure_capacity_rows(std::size_t n_rows) {
    if (n_rows <= capacity_rows) return;
    
    row_keys = IntervalSet2DDevice::RowKeyView("subsetix_morph_row_keys", n_rows);
    row_map_start = Kokkos::View<int*, DeviceMemorySpace>("subsetix_morph_map_start", n_rows);
    row_map_end = Kokkos::View<int*, DeviceMemorySpace>("subsetix_morph_map_end", n_rows);
    
    if (!d_num_rows.data()) {
        d_num_rows = Kokkos::View<std::size_t, DeviceMemorySpace>("subsetix_morph_num_rows");
    }
    capacity_rows = n_rows;
  }

  void ensure_capacity_input_map(std::size_t n_in) {
      if (n_in <= capacity_input_map) return;
      input_row_exists = Kokkos::View<int*, DeviceMemorySpace>("subsetix_morph_in_exists", n_in);
      input_row_to_output_start = Kokkos::View<int*, DeviceMemorySpace>("subsetix_morph_in_to_out", n_in);
      capacity_input_map = n_in;
  }
  
  void ensure_capacity_temps(std::size_t n_in) {
      if (n_in <= capacity_temps) return;
      row_block_counts = Kokkos::View<std::size_t*, DeviceMemorySpace>("subsetix_morph_temp_counts", n_in);
      scan_offsets = Kokkos::View<std::size_t*, DeviceMemorySpace>("subsetix_morph_temp_offsets", n_in);
      
      if (!d_total_rows.data()) {
          d_total_rows = Kokkos::View<std::size_t, DeviceMemorySpace>("subsetix_morph_temp_total");
      }
      capacity_temps = n_in;
  }
};

// Max expected N for N-way operations (radius 3 -> N=7). 
// We use a small static buffer in kernels.
constexpr int MAX_MORPH_N = 16;

/**
 * @brief Count intervals in the union of N expanded rows.
 * 
 * Each row i contributes intervals [begin - rx, end + rx).
 */
template <class IntervalView>
KOKKOS_INLINE_FUNCTION
std::size_t row_n_way_union_count(const IntervalView& intervals,
                                  const std::size_t* row_ptrs, // Array of current ptrs
                                  const std::size_t* row_ends, // Array of end ptrs
                                  int n_rows,
                                  Coord rx) {
  std::size_t ptrs[MAX_MORPH_N];
  for(int i=0; i<n_rows; ++i) ptrs[i] = row_ptrs[i];

  std::size_t count = 0;
  bool active = true;

  // Current merged interval
  Coord cur_begin = 0;
  Coord cur_end = 0;
  bool have_current = false;

  while(active) {
    active = false;
    Coord min_start = 0;
    bool first = true;
    int best_idx = -1;

    // Find the interval that starts earliest among all rows
    for(int i=0; i<n_rows; ++i) {
      if(ptrs[i] < row_ends[i]) {
        active = true;
        const auto iv = intervals(ptrs[i]);
        const Coord b = iv.begin - rx;
        if(first || b < min_start) {
          min_start = b;
          best_idx = i;
          first = false;
        }
      }
    }

    if(!active) break;

    // Process the best interval
    const auto iv = intervals(ptrs[best_idx]);
    const Coord b = iv.begin - rx;
    const Coord e = iv.end + rx;
    
    // Advance the pointer
    ptrs[best_idx]++;

    if(!have_current) {
      cur_begin = b;
      cur_end = e;
      have_current = true;
    } else {
      // Overlap or adjacent?
      if(b <= cur_end) {
        // Merge
        if(e > cur_end) cur_end = e;
      } else {
        // Emit current
        count++;
        cur_begin = b;
        cur_end = e;
      }
    }
  }

  if(have_current) count++;
  return count;
}

template <class IntervalView, class IntervalViewOut>
KOKKOS_INLINE_FUNCTION
void row_n_way_union_fill(const IntervalView& intervals,
                          const std::size_t* row_ptrs,
                          const std::size_t* row_ends,
                          int n_rows,
                          Coord rx,
                          const IntervalViewOut& intervals_out,
                          std::size_t out_offset) {
  std::size_t ptrs[MAX_MORPH_N];
  for(int i=0; i<n_rows; ++i) ptrs[i] = row_ptrs[i];

  std::size_t write_idx = 0;
  bool active = true;
  Coord cur_begin = 0;
  Coord cur_end = 0;
  bool have_current = false;

  while(active) {
    active = false;
    Coord min_start = 0;
    bool first = true;
    int best_idx = -1;

    for(int i=0; i<n_rows; ++i) {
      if(ptrs[i] < row_ends[i]) {
        active = true;
        const auto iv = intervals(ptrs[i]);
        const Coord b = iv.begin - rx;
        if(first || b < min_start) {
          min_start = b;
          best_idx = i;
          first = false;
        }
      }
    }

    if(!active) break;

    const auto iv = intervals(ptrs[best_idx]);
    const Coord b = iv.begin - rx;
    const Coord e = iv.end + rx;
    ptrs[best_idx]++;

    if(!have_current) {
      cur_begin = b;
      cur_end = e;
      have_current = true;
    } else {
      if(b <= cur_end) {
        if(e > cur_end) cur_end = e;
      } else {
        intervals_out(out_offset + write_idx) = Interval{cur_begin, cur_end};
        write_idx++;
        cur_begin = b;
        cur_end = e;
      }
    }
  }

  if(have_current) {
    intervals_out(out_offset + write_idx) = Interval{cur_begin, cur_end};
  }
}

/**
 * @brief Count intervals in the intersection of N shrunk rows.
 * 
 * Each row i contributes intervals [begin + rx, end - rx).
 * If begin + rx >= end - rx, the interval is ignored.
 */
template <class IntervalView>
KOKKOS_INLINE_FUNCTION
std::size_t row_n_way_intersection_count(const IntervalView& intervals,
                                         const std::size_t* row_ptrs,
                                         const std::size_t* row_ends,
                                         int n_rows,
                                         Coord rx) {
  if(n_rows == 0) return 0;

  std::size_t ptrs[MAX_MORPH_N];
  for(int i=0; i<n_rows; ++i) ptrs[i] = row_ptrs[i];

  std::size_t count = 0;
  
  // Skip empty/invalid intervals initially
  for(int i=0; i<n_rows; ++i) {
    while(ptrs[i] < row_ends[i]) {
      const auto iv = intervals(ptrs[i]);
      const Coord b = iv.begin + rx;
      const Coord e = iv.end - rx;
      if(b < e) break; // Valid
      ptrs[i]++;
    }
    if(ptrs[i] == row_ends[i]) return 0; // One row empty -> intersection empty
  }

  while(true) {
    // Find max start and min end of current intervals
    Coord max_start = -2147483648; 
    Coord min_end = 2147483647; 
    
    // We also need to identify which interval ends earliest to advance it
    int earliest_end_idx = -1;
    Coord earliest_end_val = 2147483647;

    for(int i=0; i<n_rows; ++i) {
      const auto iv = intervals(ptrs[i]);
      const Coord b = iv.begin + rx;
      const Coord e = iv.end - rx;

      if(i==0 || b > max_start) max_start = b;
      if(i==0 || e < min_end) min_end = e;

      if(e < earliest_end_val) {
        earliest_end_val = e;
        earliest_end_idx = i;
      }
    }

    if(max_start < min_end) {
      count++;
    }

    // Advance the one that ends earliest
    int idx = earliest_end_idx;
    ptrs[idx]++;
    
    // Skip invalid intervals for this row
    while(ptrs[idx] < row_ends[idx]) {
      const auto iv = intervals(ptrs[idx]);
      const Coord b = iv.begin + rx;
      const Coord e = iv.end - rx;
      if(b < e) break;
      ptrs[idx]++;
    }

    if(ptrs[idx] == row_ends[idx]) break; // Exhausted one row
  }

  return count;
}

template <class IntervalView, class IntervalViewOut>
KOKKOS_INLINE_FUNCTION
void row_n_way_intersection_fill(const IntervalView& intervals,
                                 const std::size_t* row_ptrs,
                                 const std::size_t* row_ends,
                                 int n_rows,
                                 Coord rx,
                                 const IntervalViewOut& intervals_out,
                                 std::size_t out_offset) {
  if(n_rows == 0) return;

  std::size_t ptrs[MAX_MORPH_N];
  for(int i=0; i<n_rows; ++i) ptrs[i] = row_ptrs[i];

  std::size_t write_idx = 0;

  for(int i=0; i<n_rows; ++i) {
    while(ptrs[i] < row_ends[i]) {
      const auto iv = intervals(ptrs[i]);
      const Coord b = iv.begin + rx;
      const Coord e = iv.end - rx;
      if(b < e) break;
      ptrs[i]++;
    }
    if(ptrs[i] == row_ends[i]) return;
  }

  while(true) {
    Coord max_start = -2147483648;
    Coord min_end = 2147483647;
    int earliest_end_idx = -1;
    Coord earliest_end_val = 2147483647;

    for(int i=0; i<n_rows; ++i) {
      const auto iv = intervals(ptrs[i]);
      const Coord b = iv.begin + rx;
      const Coord e = iv.end - rx;

      if(i==0 || b > max_start) max_start = b;
      if(i==0 || e < min_end) min_end = e;

      if(e < earliest_end_val) {
        earliest_end_val = e;
        earliest_end_idx = i;
      }
    }

    if(max_start < min_end) {
      intervals_out(out_offset + write_idx) = Interval{max_start, min_end};
      write_idx++;
    }

    int idx = earliest_end_idx;
    ptrs[idx]++;
    while(ptrs[idx] < row_ends[idx]) {
      const auto iv = intervals(ptrs[idx]);
      const Coord b = iv.begin + rx;
      const Coord e = iv.end - rx;
      if(b < e) break;
      ptrs[idx]++;
    }

    if(ptrs[idx] == row_ends[idx]) break;
  }
}

struct RowMorphologyResult {
    IntervalSet2DDevice::RowKeyView row_keys;
    Kokkos::View<int*, DeviceMemorySpace> map_start;
    Kokkos::View<int*, DeviceMemorySpace> map_end;
    std::size_t num_rows = 0;
};

/**
 * @brief Build row mapping for Expand (Dilation).
 * Output rows are Union of [y - ry, y + ry] for all input rows y.
 */
inline RowMorphologyResult
build_expand_row_mapping(const IntervalSet2DDevice& in,
                         Coord ry,
                         MorphologyWorkspace& workspace) {
    RowMorphologyResult result;
    std::size_t num_rows_in = in.num_rows;
    if (num_rows_in == 0) return result;

    workspace.ensure_capacity_input_map(num_rows_in);
    workspace.ensure_capacity_temps(num_rows_in);
    
    auto in_exists = workspace.input_row_exists;
    auto row_keys_in = in.row_keys;
    
    // 1. Identify where overlaps break
    auto row_block_counts = workspace.row_block_counts;

    Kokkos::parallel_for("morph_expand_counts", Kokkos::RangePolicy<ExecSpace>(0, num_rows_in),
        KOKKOS_LAMBDA(const std::size_t i) {
            const Coord y = row_keys_in(i).y;
            const Coord start = y - ry;
            const Coord end = y + ry + 1;
            
            bool is_start = true;
            Coord prev_end = 0;
            
            if (i > 0) {
                const Coord prev_y = row_keys_in(i-1).y;
                prev_end = prev_y + ry + 1;
                if (start < prev_end) {
                    is_start = false;
                }
            }
            
            Coord actual_start = start;
            if (!is_start) {
                actual_start = prev_end;
            }
            
            if (actual_start < end) {
                row_block_counts(i) = (end - actual_start);
            } else {
                row_block_counts(i) = 0;
            }
            
            in_exists(i) = is_start ? 1 : 0;
    });
    
    ExecSpace().fence();
    
    auto d_total = workspace.d_total_rows;
    auto scan_offsets = workspace.scan_offsets;
    
    Kokkos::parallel_scan("morph_expand_scan", Kokkos::RangePolicy<ExecSpace>(0, num_rows_in),
        KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
            const std::size_t c = row_block_counts(i);
            if (final_pass) {
                scan_offsets(i) = update;
                if (i + 1 == num_rows_in) d_total() = update + c;
            }
            update += c;
    });
    
    ExecSpace().fence();
    
    std::size_t total_rows_out = 0;
    Kokkos::deep_copy(total_rows_out, d_total);
    
    result.num_rows = total_rows_out;
    if (total_rows_out == 0) return result;
    
    workspace.ensure_capacity_rows(total_rows_out);
    result.row_keys = workspace.row_keys;
    result.map_start = workspace.row_map_start;
    result.map_end = workspace.row_map_end;
    
    auto out_rows = result.row_keys;
    auto map_start = result.map_start;
    auto map_end = result.map_end;
    
    // 2. Fill output rows
    Kokkos::parallel_for("morph_expand_fill_rows", Kokkos::RangePolicy<ExecSpace>(0, num_rows_in),
        KOKKOS_LAMBDA(const std::size_t i) {
            const std::size_t count = row_block_counts(i);
            if (count == 0) return;
            
            const Coord y_in = row_keys_in(i).y;
            const std::size_t offset = scan_offsets(i);
            const bool is_start = (in_exists(i) == 1);
            
            Coord start_y = y_in - ry;
            if (!is_start) {
                 const Coord prev_y = row_keys_in(i-1).y;
                 start_y = prev_y + ry + 1;
            }
            
            for(std::size_t k=0; k<count; ++k) {
                out_rows(offset + k) = RowKey2D{static_cast<Coord>(start_y + k)};
            }
    });
    
    ExecSpace().fence();
    
    // 3. Compute map_start and map_end
    Kokkos::parallel_for("morph_expand_map", Kokkos::RangePolicy<ExecSpace>(0, total_rows_out),
        KOKKOS_LAMBDA(const std::size_t j) {
            const Coord y = out_rows(j).y;
            
            std::size_t start_idx = num_rows_in;
            {
                std::size_t l = 0, r = num_rows_in;
                while(l < r) {
                    std::size_t m = l + (r - l)/2;
                    if(row_keys_in(m).y >= y - ry) r = m;
                    else l = m + 1;
                }
                start_idx = l;
            }
            
            std::size_t end_idx = num_rows_in;
            {
                std::size_t l = 0, r = num_rows_in;
                while(l < r) {
                    std::size_t m = l + (r - l)/2;
                    if(row_keys_in(m).y > y + ry) r = m;
                    else l = m + 1;
                }
                end_idx = l;
            }
            
            map_start(j) = static_cast<int>(start_idx);
            map_end(j) = static_cast<int>(end_idx);
    });
    
    ExecSpace().fence();
    
    return result;
}

/**
 * @brief Build row mapping for Shrink (Erosion).
 * Output rows are subset of input rows.
 */
inline RowMorphologyResult
build_shrink_row_mapping(const IntervalSet2DDevice& in,
                         Coord ry,
                         MorphologyWorkspace& workspace) {
    RowMorphologyResult result;
    std::size_t num_rows_in = in.num_rows;
    if (num_rows_in == 0) return result;
    
    workspace.ensure_capacity_rows(num_rows_in);
    workspace.ensure_capacity_temps(num_rows_in);
    
    auto valid_flags = workspace.row_map_end;
    // Actually, we can use `row_map_start` as scan_offsets later if we are careful, 
    // but here we need valid_flags AND offsets. 
    // In original code: valid_flags was row_map_end, scan_offsets was row_map_start. 
    // But wait, build_shrink_row_mapping used row_map_start for offsets in the original implementation? 
    // Let's check: 
    // "auto scan_offsets = workspace.row_map_start;"
    // And then later: "result.map_start = out_map_start;"
    
    // We need temporary scan offsets during scan, then we fill `out_map_start` in the final loop.
    // In the original code, `offsets` variable was allocated locally:
    // Kokkos::View<std::size_t*, DeviceMemorySpace> offsets("morph_shrink_offsets", num_rows_in);
    // Let's use workspace.scan_offsets for this.
    
    auto offsets = workspace.scan_offsets;
    // row_map_end is used as valid_flags temporary buffer?
    // In original code: "auto valid_flags = workspace.row_map_end;"
    // This is fine as row_map_end is not part of the output for Shrink (only map_start is used).
    
    auto row_keys_in = in.row_keys;
    
    // Check validity of each row
    Kokkos::parallel_for("morph_shrink_check", Kokkos::RangePolicy<ExecSpace>(0, num_rows_in),
        KOKKOS_LAMBDA(const std::size_t i) {
            const Coord y = row_keys_in(i).y;
            bool possible = true;
            
            std::size_t idx_L = 0;
            {
                std::size_t l = 0, r = num_rows_in;
                while(l < r) {
                    std::size_t m = l + (r - l)/2;
                    if(row_keys_in(m).y >= y - ry) r = m;
                    else l = m + 1;
                }
                idx_L = l;
            }
            
            if (idx_L >= num_rows_in || row_keys_in(idx_L).y != y - ry) {
                possible = false;
            }
            
            if (possible) {
                std::size_t idx_R_candidate = idx_L + 2*ry;
                if (idx_R_candidate < num_rows_in && row_keys_in(idx_R_candidate).y == y + ry) {
                    possible = true;
                } else {
                    possible = false;
                }
            }
            
            valid_flags(i) = possible ? 1 : 0;
    });
    
    ExecSpace().fence();
    
    auto d_total = workspace.d_total_rows;
    
    Kokkos::parallel_scan("morph_shrink_scan", Kokkos::RangePolicy<ExecSpace>(0, num_rows_in),
        KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
            const int val = valid_flags(i);
            if (final_pass) {
                offsets(i) = update;
                if (i + 1 == num_rows_in) d_total() = update + val;
            }
            update += val;
    });
    
    ExecSpace().fence();
    
    std::size_t total_rows_out = 0;
    Kokkos::deep_copy(total_rows_out, d_total);
    
    result.num_rows = total_rows_out;
    if (total_rows_out == 0) return result;
    
    auto out_keys = workspace.row_keys;
    auto out_map_start = workspace.row_map_start;
    
    Kokkos::parallel_for("morph_shrink_fill", Kokkos::RangePolicy<ExecSpace>(0, num_rows_in),
        KOKKOS_LAMBDA(const std::size_t i) {
            if (valid_flags(i)) {
                const std::size_t off = offsets(i);
                out_keys(off) = row_keys_in(i);
                out_map_start(off) = static_cast<int>(i);
            }
    });
    
    ExecSpace().fence();
    
    result.row_keys = out_keys;
    result.map_start = out_map_start;
    
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
  detail::RowDifferenceWorkspace difference_workspace;
  detail::RowScanWorkspace scan_workspace;
  detail::TranslationWorkspace translation_workspace;
  detail::ProjectWorkspace project_workspace;
  detail::CoarsenWorkspace coarsen_workspace;
  detail::MorphologyWorkspace morphology_workspace;
};

/**
 * @brief Translate all intervals of a CSR interval set by a constant
 *        offset along X.
 *
 * Row structure (row_keys, row_ptr) is preserved; only the begin/end
 * coordinates of intervals are shifted by dx. Requires a context for
 * workspace buffer reuse.
 */
inline void
translate_x_device(const IntervalSet2DDevice& in,
                   Coord dx,
                   IntervalSet2DDevice& out,
                   CsrSetAlgebraContext& ctx) {
  if (dx == 0) {
    out = in;
    return;
  }
  detail::TranslateXTransform transform{dx};
  detail::apply_interval_transform_device(in, transform, out,
                                          ctx.translation_workspace);
}

/**
 * @brief Translate all rows of a CSR interval set by a constant offset
 *        along Y.
 *
 * Interval structure and row_ptr are preserved; only row_keys(i).y are
 * shifted by dy. Requires a context for workspace buffer reuse.
 */
inline void
translate_y_device(const IntervalSet2DDevice& in,
                   Coord dy,
                   IntervalSet2DDevice& out,
                   CsrSetAlgebraContext& ctx) {
  if (dy == 0 || in.num_rows == 0) {
    out = in;
    return;
  }
  detail::TranslateYTransform transform{dy};
  detail::apply_row_key_transform_device(in, transform, out,
                                         ctx.translation_workspace);
}

/**
 * @brief Compute the set union into a preallocated CSR buffer on device.
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
      detail::build_row_difference_mapping(
          A, B, ctx.difference_workspace);
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
 * @brief Compute the symmetric difference A XOR B = (A \ B) U (B \ A)
 *        into a preallocated CSR buffer on device.
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
set_symmetric_difference_device(const IntervalSet2DDevice& A,
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
        "subsetix::csr::set_symmetric_difference_device (preallocated): "
        "insufficient row capacity in output IntervalSet2DDevice");
  }

  auto row_keys_out_tmp = merge.row_keys;
  auto row_index_a = merge.row_index_a;
  auto row_index_b = merge.row_index_b;

  auto row_keys_out = out.row_keys;

  Kokkos::parallel_for(
      "subsetix_csr_xor_copy_row_keys",
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
      "subsetix_csr_xor_count_prealloc",
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

        row_counts(i) = detail::row_symmetric_difference_count(
            intervals_a, begin_a, end_a,
            intervals_b, begin_b, end_b);
      });

  ExecSpace().fence();

  auto total_intervals = ctx.scan_workspace.total_intervals;
  auto row_ptr_out = out.row_ptr;

  Kokkos::parallel_scan(
      "subsetix_csr_xor_scan_prealloc",
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
        "subsetix::csr::set_symmetric_difference_device (preallocated): "
        "insufficient interval capacity in output IntervalSet2DDevice");
  }

  out.num_rows = num_rows_out;
  out.num_intervals = num_intervals_out;

  if (num_intervals_out == 0) {
    return;
  }

  auto intervals_out = out.intervals;

  Kokkos::parallel_for(
      "subsetix_csr_xor_fill_prealloc",
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
        detail::row_symmetric_difference_fill(
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
 * operation; row structure is preserved. Requires a context for workspace
 * buffer reuse.
 */
inline void
refine_level_up_device(const IntervalSet2DDevice& in,
                       IntervalSet2DDevice& out,
                       CsrSetAlgebraContext& ctx) {
  const std::size_t num_rows_in = in.num_rows;
  const std::size_t num_intervals_in = in.num_intervals;

  if (num_rows_in == 0) {
    out.num_rows = 0;
    out.num_intervals = 0;
    return;
  }

  const std::size_t num_rows_out = num_rows_in * 2;
  const std::size_t num_intervals_out = num_intervals_in * 2;

  // Allocate output buffers if needed
  if (out.row_keys.extent(0) < num_rows_out) {
    out.row_keys = IntervalSet2DDevice::RowKeyView(
        "subsetix_csr_refine_row_keys_out", num_rows_out);
  }
  if (out.row_ptr.extent(0) < num_rows_out + 1) {
    out.row_ptr = IntervalSet2DDevice::IndexView(
        "subsetix_csr_refine_row_ptr_out", num_rows_out + 1);
  }
  if (num_intervals_out > 0 && out.intervals.extent(0) < num_intervals_out) {
    out.intervals = IntervalSet2DDevice::IntervalView(
        "subsetix_csr_refine_intervals_out", num_intervals_out);
  }

  auto row_keys_in = in.row_keys;
  auto row_ptr_in = in.row_ptr;
  auto intervals_in = in.intervals;
  auto row_keys_out = out.row_keys;
  auto row_ptr_out = out.row_ptr;
  auto intervals_out = out.intervals;

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

  ExecSpace().fence();

  out.num_rows = num_rows_out;
  out.num_intervals = num_intervals_out;
}

/**
 * @brief Project a CSR interval set to the next coarser level (÷2 in X and Y),
 *        covering the fine geometry.
 *
 * Semantics (1D example):
 *   level1 [0,1) -> level0 [0,1)
 *   level1 [0,3) -> level0 [0,2)
 *
 * Requires a context for workspace buffer reuse.
 */
inline void
project_level_down_device(const IntervalSet2DDevice& fine,
                           IntervalSet2DDevice& out,
                           CsrSetAlgebraContext& ctx) {
  const std::size_t num_rows_fine = fine.num_rows;
  const std::size_t num_intervals_fine = fine.num_intervals;

  if (num_rows_fine == 0 || num_intervals_fine == 0) {
    out.num_rows = 0;
    out.num_intervals = 0;
    return;
  }

  detail::RowCoarsenResult rows =
      detail::build_row_coarsen_mapping(fine, ctx.coarsen_workspace);
  const std::size_t num_rows_coarse = rows.num_rows;
  if (num_rows_coarse == 0) {
    out.num_rows = 0;
    out.num_intervals = 0;
    return;
  }

  auto row_keys_coarse = rows.row_keys;
  auto row_index_first = rows.row_index_first;
  auto row_index_second = rows.row_index_second;

  auto row_ptr_fine = fine.row_ptr;
  auto intervals_fine = fine.intervals;

  using CoarsenView =
      detail::CoarsenIntervalView<IntervalSet2DDevice::IntervalView>;
  CoarsenView coarsen_view{intervals_fine};

  // Use workspace buffers to avoid allocation
  ctx.project_workspace.ensure_capacity_coarse(num_rows_coarse);
  ctx.project_workspace.ensure_total_intervals();
  
  auto row_counts = ctx.project_workspace.row_counts;
  auto total_intervals = ctx.project_workspace.total_intervals;

  // Count intervals needed for each coarse row
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

  // Allocate output row_ptr buffer if needed
  if (out.row_ptr.extent(0) < num_rows_coarse + 1) {
    out.row_ptr = IntervalSet2DDevice::IndexView(
        "subsetix_csr_project_row_ptr_out", num_rows_coarse + 1);
  }
  auto row_ptr_out = out.row_ptr;

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

  if (num_intervals_coarse == 0) {
    out.intervals = IntervalSet2DDevice::IntervalView();
    return;
  }

  if (out.intervals.extent(0) < num_intervals_coarse) {
    out.intervals = IntervalSet2DDevice::IntervalView(
        "subsetix_csr_project_intervals_out", num_intervals_coarse);
  }

  auto intervals_out = out.intervals;

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
}




/**
 * @brief Expand (dilate) a CSR set by rx in X and ry in Y.
 *
 * Each interval [b, e) in row y effectively becomes:
 *   Union over dy in [-ry, ry] of intervals [b - rx, e + rx) in row y+dy.
 *
 * Requires a context for workspace buffer reuse.
 */
inline void
expand_device(const IntervalSet2DDevice& in,
              Coord rx,
              Coord ry,
              IntervalSet2DDevice& out,
              CsrSetAlgebraContext& ctx) {
  reset_preallocated_interval_set(out);
  if (in.num_rows == 0) return;

  if (rx < 0) rx = 0;
  if (ry < 0) ry = 0;

  // 1. Determine output rows and mapping
  detail::RowMorphologyResult map_result =
      detail::build_expand_row_mapping(in, ry, ctx.morphology_workspace);
  
  const std::size_t num_rows_out = map_result.num_rows;
  if (num_rows_out == 0) return;
  
  // Allocate rows
  if (out.row_keys.extent(0) < num_rows_out) {
    out.row_keys = IntervalSet2DDevice::RowKeyView("subsetix_expand_rows", num_rows_out);
  }
  
  auto row_keys_out = out.row_keys;
  auto map_keys = map_result.row_keys;
  
  // Copy valid row keys to output buffer. 
  // Using parallel_for instead of deep_copy+subview to match codebase style 
  // and handle preallocated buffer size mismatch.
  Kokkos::parallel_for("subsetix_expand_copy_keys", Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
    KOKKOS_LAMBDA(const std::size_t i) {
      row_keys_out(i) = map_keys(i);
  });
  
  // 2. Count intervals
  ctx.scan_workspace.ensure_capacity(num_rows_out);
  auto row_counts = ctx.scan_workspace.row_counts;
  
  auto map_start = map_result.map_start;
  auto map_end = map_result.map_end;
  auto row_ptr_in = in.row_ptr;
  auto intervals_in = in.intervals;
  
  Kokkos::parallel_for("subsetix_expand_count", Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
    KOKKOS_LAMBDA(const std::size_t i) {
      const int start_idx = map_start(i);
      const int end_idx = map_end(i);
      const int n_rows = end_idx - start_idx;
      
      if (n_rows <= 0) {
        row_counts(i) = 0;
        return;
      }
      
      std::size_t ptrs[detail::MAX_MORPH_N];
      std::size_t ends[detail::MAX_MORPH_N];
      
      for(int k=0; k<n_rows; ++k) {
        const std::size_t r = static_cast<std::size_t>(start_idx + k);
        ptrs[k] = row_ptr_in(r);
        ends[k] = row_ptr_in(r+1);
      }
      
      row_counts(i) = detail::row_n_way_union_count(intervals_in, ptrs, ends, n_rows, rx);
  });
  
  ExecSpace().fence();
  
  // 3. Scan
  if (out.row_ptr.extent(0) < num_rows_out + 1) {
      out.row_ptr = IntervalSet2DDevice::IndexView("subsetix_expand_row_ptr", num_rows_out + 1);
  }
  auto row_ptr_out = out.row_ptr;
  auto total_intervals = ctx.scan_workspace.total_intervals;
  
  Kokkos::parallel_scan("subsetix_expand_scan", Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
    KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
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
  
  if (num_intervals_out == 0) return;
  
  if (out.intervals.extent(0) < num_intervals_out) {
      out.intervals = IntervalSet2DDevice::IntervalView("subsetix_expand_intervals", num_intervals_out);
  }
  auto intervals_out = out.intervals;
  
  // 4. Fill
  Kokkos::parallel_for("subsetix_expand_fill", Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
    KOKKOS_LAMBDA(const std::size_t i) {
      const int start_idx = map_start(i);
      const int end_idx = map_end(i);
      const int n_rows = end_idx - start_idx;
      
      if (n_rows <= 0) return;
      
      std::size_t ptrs[detail::MAX_MORPH_N];
      std::size_t ends[detail::MAX_MORPH_N];
      
      for(int k=0; k<n_rows; ++k) {
        const std::size_t r = static_cast<std::size_t>(start_idx + k);
        ptrs[k] = row_ptr_in(r);
        ends[k] = row_ptr_in(r+1);
      }
      
      const std::size_t offset = row_ptr_out(i);
      detail::row_n_way_union_fill(intervals_in, ptrs, ends, n_rows, rx, intervals_out, offset);
  });
  
  ExecSpace().fence();
}

/**
 * @brief Shrink (erode) a CSR set by rx in X and ry in Y.
 *
 * A point (x,y) is in the output iff the box [x-rx, x+rx] x [y-ry, y+ry] is fully contained in the input.
 * This is equivalent to:
 *   Intersection over dy in [-ry, ry] of intervals [b + rx, e - rx) in row y+dy.
 *
 * Requires a context for workspace buffer reuse.
 */
inline void
shrink_device(const IntervalSet2DDevice& in,
              Coord rx,
              Coord ry,
              IntervalSet2DDevice& out,
              CsrSetAlgebraContext& ctx) {
  reset_preallocated_interval_set(out);
  if (in.num_rows == 0) return;

  if (rx < 0) rx = 0;
  if (ry < 0) ry = 0;

  // 1. Determine output rows (valid subset of input rows)
  detail::RowMorphologyResult map_result =
      detail::build_shrink_row_mapping(in, ry, ctx.morphology_workspace);
  
  const std::size_t num_rows_out = map_result.num_rows;
  if (num_rows_out == 0) return;
  
  if (out.row_keys.extent(0) < num_rows_out) {
      out.row_keys = IntervalSet2DDevice::RowKeyView("subsetix_shrink_rows", num_rows_out);
  }
  auto row_keys_out = out.row_keys;
  auto map_keys = map_result.row_keys;
  
  Kokkos::parallel_for("subsetix_shrink_copy_keys", Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
    KOKKOS_LAMBDA(const std::size_t i) {
      row_keys_out(i) = map_keys(i);
  });
  
  // 2. Count intervals
  ctx.scan_workspace.ensure_capacity(num_rows_out);
  auto row_counts = ctx.scan_workspace.row_counts;
  
  auto map_start = map_result.map_start; // Contains index of central row
  auto row_ptr_in = in.row_ptr;
  auto intervals_in = in.intervals;
  
  Kokkos::parallel_for("subsetix_shrink_count", Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
    KOKKOS_LAMBDA(const std::size_t i) {
      const int central_idx = map_start(i);
      
      // The neighbors are strictly [central_idx - ry ... central_idx + ry] 
      // because we already verified they exist and are contiguous in build_shrink_row_mapping.
      
      // Construct pointers for the window
      // Window size = 2*ry + 1
      int n_rows = 2 * static_cast<int>(ry) + 1;
      if (n_rows > detail::MAX_MORPH_N) n_rows = detail::MAX_MORPH_N; // Should not happen given limits
      
      std::size_t ptrs[detail::MAX_MORPH_N];
      std::size_t ends[detail::MAX_MORPH_N];
      
      // The first row in the window is at index: central_idx - ry
      // We need to be careful about indices, but build_shrink_row_mapping guarantees
      // that if row y exists, then y-ry ... y+ry exist at contiguous indices relative to y's index.
      // Actually, `build_shrink_row_mapping` checks if keys are correct.
      // But it doesn't guarantee indices are contiguous in the input array if there are gaps in Y keys.
      // Let's re-read `build_shrink_row_mapping`. It checks `idx_L` (for y-ry) and `idx_R_candidate` (for y+ry).
      // It checks `idx_R_candidate = idx_L + 2*ry`. This IMPLIES that all rows between them exist and are contiguous!
      // Because indices are integers.
      
      // So we can safely assume indices are [start_idx, start_idx + n_rows)
      // where start_idx corresponds to y - ry.
      
      // Wait, we need to find the index of y-ry.
      // map_start(i) gives us index of row y (central).
      // So index of y-ry is central_idx - ry.
      
      int start_row_idx = central_idx - static_cast<int>(ry);
      
      for(int k=0; k<n_rows; ++k) {
        std::size_t r = static_cast<std::size_t>(start_row_idx + k);
        ptrs[k] = row_ptr_in(r);
        ends[k] = row_ptr_in(r+1);
      }
      
      row_counts(i) = detail::row_n_way_intersection_count(intervals_in, ptrs, ends, n_rows, rx);
  });
  
  ExecSpace().fence();
  
  // 3. Scan
  if (out.row_ptr.extent(0) < num_rows_out + 1) {
      out.row_ptr = IntervalSet2DDevice::IndexView("subsetix_shrink_row_ptr", num_rows_out + 1);
  }
  auto row_ptr_out = out.row_ptr;
  auto total_intervals = ctx.scan_workspace.total_intervals;
  
  Kokkos::parallel_scan("subsetix_shrink_scan", Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
    KOKKOS_LAMBDA(const std::size_t i, std::size_t& update, const bool final_pass) {
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
  
  if (num_intervals_out == 0) return;
  
  if (out.intervals.extent(0) < num_intervals_out) {
      out.intervals = IntervalSet2DDevice::IntervalView("subsetix_shrink_intervals", num_intervals_out);
  }
  auto intervals_out = out.intervals;
  
  // 4. Fill
  Kokkos::parallel_for("subsetix_shrink_fill", Kokkos::RangePolicy<ExecSpace>(0, num_rows_out),
    KOKKOS_LAMBDA(const std::size_t i) {
      const int central_idx = map_start(i);
      int n_rows = 2 * static_cast<int>(ry) + 1;
      if (n_rows > detail::MAX_MORPH_N) n_rows = detail::MAX_MORPH_N;
      
      int start_row_idx = central_idx - static_cast<int>(ry);
      
      std::size_t ptrs[detail::MAX_MORPH_N];
      std::size_t ends[detail::MAX_MORPH_N];
      
      for(int k=0; k<n_rows; ++k) {
        std::size_t r = static_cast<std::size_t>(start_row_idx + k);
        ptrs[k] = row_ptr_in(r);
        ends[k] = row_ptr_in(r+1);
      }
      
      const std::size_t offset = row_ptr_out(i);
      detail::row_n_way_intersection_fill(intervals_in, ptrs, ends, n_rows, rx, intervals_out, offset);
  });
  
  ExecSpace().fence();
}

} // namespace csr
} // namespace subsetix
