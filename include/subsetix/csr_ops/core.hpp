#pragma once

#include <Kokkos_Core.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/csr_ops/workspace.hpp>
#include <subsetix/detail/csr_utils.hpp>
#include <algorithm> // for std::min/max if needed, though we use custom if possible
#include <stdexcept>

// Forward declaration
namespace subsetix { namespace csr { namespace detail { struct RowMergeResult; struct RowDifferenceResult; } } }

namespace subsetix {
namespace csr {
namespace detail {

struct RowMergeResult {
  IntervalSet2DDevice::RowKeyView row_keys;
  Kokkos::View<int*, DeviceMemorySpace> row_index_a;
  Kokkos::View<int*, DeviceMemorySpace> row_index_b;
  std::size_t num_rows = 0;
};

// ============================================================================
// Unified row operations: count and fill combined via CountOnly template param
// ============================================================================

/**
 * @brief Unified implementation for row union operation.
 *
 * When CountOnly=true, only counts intervals without writing.
 * When CountOnly=false, writes intervals to intervals_out.
 *
 * @return Number of intervals in the union.
 */
template <bool CountOnly, class IntervalViewIn, class IntervalViewOut>
KOKKOS_INLINE_FUNCTION
std::size_t row_union_impl(const IntervalViewIn& intervals_a,
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
  std::size_t count = 0;

  while (ia < end_a || ib < end_b) {
    Coord b = 0;
    Coord e = 0;

    const bool take_a =
        (ib >= end_b) ||
        (ia < end_a && intervals_a(ia).begin <= intervals_b(ib).begin);

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
        if constexpr (!CountOnly) {
          intervals_out(out_offset + count) = Interval{current_begin, current_end};
        }
        ++count;
        current_begin = b;
        current_end = e;
      }
    }
  }

  if (have_current) {
    if constexpr (!CountOnly) {
      intervals_out(out_offset + count) = Interval{current_begin, current_end};
    }
    ++count;
  }

  return count;
}

/**
 * @brief Unified implementation for row intersection operation.
 *
 * When CountOnly=true, only counts intervals without writing.
 * When CountOnly=false, writes intervals to intervals_out.
 *
 * @return Number of intervals in the intersection.
 */
template <bool CountOnly, class IntervalViewIn, class IntervalViewOut>
KOKKOS_INLINE_FUNCTION
std::size_t row_intersection_impl(const IntervalViewIn& intervals_a,
                                  std::size_t begin_a,
                                  std::size_t end_a,
                                  const IntervalViewIn& intervals_b,
                                  std::size_t begin_b,
                                  std::size_t end_b,
                                  const IntervalViewOut& intervals_out,
                                  std::size_t out_offset) {
  std::size_t ia = begin_a;
  std::size_t ib = begin_b;
  std::size_t count = 0;

  while (ia < end_a && ib < end_b) {
    const auto a = intervals_a(ia);
    const auto b = intervals_b(ib);

    const Coord start = (a.begin > b.begin) ? a.begin : b.begin;
    const Coord end = (a.end < b.end) ? a.end : b.end;

    if (start < end) {
      if constexpr (!CountOnly) {
        intervals_out(out_offset + count) = Interval{start, end};
      }
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
 * @brief Unified implementation for row difference (A \ B) operation.
 *
 * When CountOnly=true, only counts intervals without writing.
 * When CountOnly=false, writes intervals to intervals_out.
 *
 * @return Number of intervals in the difference.
 */
template <bool CountOnly, class IntervalViewIn, class IntervalViewOut>
KOKKOS_INLINE_FUNCTION
std::size_t row_difference_impl(const IntervalViewIn& intervals_a,
                                std::size_t begin_a,
                                std::size_t end_a,
                                const IntervalViewIn& intervals_b,
                                std::size_t begin_b,
                                std::size_t end_b,
                                const IntervalViewOut& intervals_out,
                                std::size_t out_offset) {
  if (begin_a == end_a) {
    return 0;
  }

  if (begin_b == end_b) {
    // Fast path: A unchanged (no B to subtract)
    const std::size_t n = end_a - begin_a;
    if constexpr (!CountOnly) {
      for (std::size_t i = 0; i < n; ++i) {
        intervals_out(out_offset + i) = intervals_a(begin_a + i);
      }
    }
    return n;
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
        if constexpr (!CountOnly) {
          intervals_out(out_offset + count) = Interval{cur, b.begin};
        }
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
      if constexpr (!CountOnly) {
        intervals_out(out_offset + count) = Interval{cur, a.end};
      }
      ++count;
    }

    ++ia;
  }

  return count;
}

/**
 * @brief Unified implementation for row symmetric difference (A XOR B).
 *
 * When CountOnly=true, only counts intervals without writing.
 * When CountOnly=false, writes intervals to intervals_out.
 *
 * @return Number of intervals in the symmetric difference.
 */
template <bool CountOnly, class IntervalViewIn, class IntervalViewOut>
KOKKOS_INLINE_FUNCTION
std::size_t row_symmetric_difference_impl(const IntervalViewIn& intervals_a,
                                          std::size_t begin_a,
                                          std::size_t end_a,
                                          const IntervalViewIn& intervals_b,
                                          std::size_t begin_b,
                                          std::size_t end_b,
                                          const IntervalViewOut& intervals_out,
                                          std::size_t out_offset) {
  std::size_t ia = begin_a;
  std::size_t ib = begin_b;
  std::size_t count = 0;

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
        if constexpr (!CountOnly) {
          intervals_out(out_offset + count) = Interval{start_pos, p};
        }
        ++count;
      }
      xor_active = new_xor;
    }
  }

  return count;
}

// ============================================================================
// Backward-compatible wrappers: row_*_count and row_*_fill
// ============================================================================

// Dummy view type for count-only operations
struct NullIntervalView {
  KOKKOS_INLINE_FUNCTION Interval& operator()(std::size_t) const {
    // This should never be called in CountOnly=true mode
    static Interval dummy{0, 0};
    return dummy;
  }
};

template <class IntervalView>
KOKKOS_INLINE_FUNCTION
std::size_t row_union_count(const IntervalView& intervals_a,
                            std::size_t begin_a,
                            std::size_t end_a,
                            const IntervalView& intervals_b,
                            std::size_t begin_b,
                            std::size_t end_b) {
  return row_union_impl<true>(intervals_a, begin_a, end_a,
                              intervals_b, begin_b, end_b,
                              NullIntervalView{}, 0);
}

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
  row_union_impl<false>(intervals_a, begin_a, end_a,
                        intervals_b, begin_b, end_b,
                        intervals_out, out_offset);
}

template <class IntervalView>
KOKKOS_INLINE_FUNCTION
std::size_t row_intersection_count(const IntervalView& intervals_a,
                                   std::size_t begin_a,
                                   std::size_t end_a,
                                   const IntervalView& intervals_b,
                                   std::size_t begin_b,
                                   std::size_t end_b) {
  return row_intersection_impl<true>(intervals_a, begin_a, end_a,
                                     intervals_b, begin_b, end_b,
                                     NullIntervalView{}, 0);
}

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
  row_intersection_impl<false>(intervals_a, begin_a, end_a,
                               intervals_b, begin_b, end_b,
                               intervals_out, out_offset);
}

template <class IntervalView>
KOKKOS_INLINE_FUNCTION
std::size_t row_difference_count(const IntervalView& intervals_a,
                                 std::size_t begin_a,
                                 std::size_t end_a,
                                 const IntervalView& intervals_b,
                                 std::size_t begin_b,
                                 std::size_t end_b) {
  return row_difference_impl<true>(intervals_a, begin_a, end_a,
                                   intervals_b, begin_b, end_b,
                                   NullIntervalView{}, 0);
}

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
  row_difference_impl<false>(intervals_a, begin_a, end_a,
                             intervals_b, begin_b, end_b,
                             intervals_out, out_offset);
}

template <class IntervalView>
KOKKOS_INLINE_FUNCTION
std::size_t row_symmetric_difference_count(const IntervalView& intervals_a,
                                           std::size_t begin_a,
                                           std::size_t end_a,
                                           const IntervalView& intervals_b,
                                           std::size_t begin_b,
                                           std::size_t end_b) {
  return row_symmetric_difference_impl<true>(intervals_a, begin_a, end_a,
                                             intervals_b, begin_b, end_b,
                                             NullIntervalView{}, 0);
}

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
  row_symmetric_difference_impl<false>(intervals_a, begin_a, end_a,
                                       intervals_b, begin_b, end_b,
                                       intervals_out, out_offset);
}

/**
 * @brief Build the union of row keys between two IntervalSet2DDevice sets.
 */
inline RowMergeResult
build_row_union_mapping(const IntervalSet2DDevice& A,
                        const IntervalSet2DDevice& B,
                        UnifiedCsrWorkspace& workspace) {
  RowMergeResult result;
  const std::size_t num_rows_a = A.num_rows;
  const std::size_t num_rows_b = B.num_rows;

  if (num_rows_a == 0 && num_rows_b == 0) {
    return result;
  }

  auto rows_a = A.row_keys;
  auto rows_b = B.row_keys;

  // Checkouts from unified workspace
  // We need map_a_to_b [num_rows_a] -> int_buf_0
  // We need map_b_to_a [num_rows_b] -> int_buf_1
  // We need b_only_flags [num_rows_b] -> int_buf_2
  // We need b_only_positions [num_rows_b] -> size_t_buf_0
  // We need b_only_indices [num_rows_b] -> int_buf_3
  
  // We need row_keys [num_rows_a + num_rows_b] -> row_key_buf_0
  // We need row_index_a [num_rows_a + num_rows_b] -> int_buf_0 (reused?)
  // We need row_index_b [num_rows_a + num_rows_b] -> int_buf_1 (reused?)
  
  // Be careful with reuse. map_a_to_b is needed until step 4.
  // map_b_to_a is needed until step 3.
  // b_only_flags needed until step 3/4 (compaction).
  // b_only_indices needed until step 4.
  
  // Output buffers:
  // out_rows: need allocation or workspace?
  // The result struct returned by this function typically points to
  // workspace memory that is then copied to the final output or used directly.
  // In the original code: result.row_keys = workspace.row_keys;
  
  // So we need to allocate enough space in the workspace buffers.
  
  const std::size_t needed_rows_out = num_rows_a + num_rows_b;
  
  // Checkout buffers
  // Note: ensure_capacity handles the size check and reallocation
  
  auto out_rows = workspace.get_row_key_buf_0(needed_rows_out);
  auto out_idx_a = workspace.get_int_buf_0(needed_rows_out);
  auto out_idx_b = workspace.get_int_buf_1(needed_rows_out);

  // Fast paths for empty inputs.
  if (num_rows_a == 0) {
    result.num_rows = num_rows_b;
    if (num_rows_b == 0) {
      return result;
    }

    result.row_keys = out_rows;
    result.row_index_a = out_idx_a;
    result.row_index_b = out_idx_b;

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
    result.num_rows = num_rows_a;
    result.row_keys = out_rows;
    result.row_index_a = out_idx_a;
    result.row_index_b = out_idx_b;

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

  // General case
  // We can't use out_idx_a/out_idx_b for temporary maps because they are needed at the end.
  // But wait, map_a_to_b is size A, out_idx_b is size A+B.
  // Can we use the end of out_idx_b? Or just use separate buffers.
  // Unified workspace has multiple int buffers.
  
  auto map_a_to_b = workspace.get_int_buf_2(num_rows_a);
  auto map_b_to_a = workspace.get_int_buf_3(num_rows_b);
  auto b_only_flags = workspace.get_int_buf_4(num_rows_b);
  auto b_only_positions = workspace.get_size_t_buf_0(num_rows_b);
  // We need another int buffer for b_only_indices.
  // We have int_buf_0..4.
  // int_buf_0 -> out_idx_a
  // int_buf_1 -> out_idx_b
  // int_buf_2 -> map_a_to_b
  // int_buf_3 -> map_b_to_a
  // int_buf_4 -> b_only_flags
  // We need one more for b_only_indices?
  // Actually b_only_indices is used after b_only_flags is consumed (mostly).
  // But we can reuse b_only_flags if we are careful?
  // Or we can reuse map_b_to_a after step 3?
  // Step 3 computes b_only_flags from map_b_to_a.
  // Then scans b_only_flags.
  // Then compacts into b_only_indices.
  // So map_b_to_a is not needed after step 3 start.
  // b_only_flags is needed for scan and compaction.
  // b_only_indices is output of compaction.
  
  // Let's reuse map_b_to_a for b_only_indices.
  auto b_only_indices = map_b_to_a; 
  
  auto d_num_b_only = workspace.get_scalar_size_t_buf_0();

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

  result.row_keys = out_rows;
  result.row_index_a = out_idx_a;
  result.row_index_b = out_idx_b;

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
 * @brief Convenience overload that uses a local workspace.
 */
inline RowMergeResult
build_row_union_mapping(const IntervalSet2DDevice& A,
                        const IntervalSet2DDevice& B) {
  UnifiedCsrWorkspace workspace;
  return build_row_union_mapping(A, B, workspace);
}

/**
 * @brief Build the intersection of row keys between two
 *        IntervalSet2DDevice sets.
 */
inline RowMergeResult
build_row_intersection_mapping(const IntervalSet2DDevice& A,
                               const IntervalSet2DDevice& B,
                               UnifiedCsrWorkspace& workspace) {
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

  // Checkout buffers from unified workspace
  // flags: int_buf_0 [n_small]
  // tmp_idx_a: int_buf_1 [n_small]
  // tmp_idx_b: int_buf_2 [n_small]
  // positions: size_t_buf_0 [n_small]
  // row_keys: row_key_buf_0 [n_small]
  // row_index_a: int_buf_3 [n_small]
  // row_index_b: int_buf_4 [n_small]
  // d_num_rows: scalar_size_t_buf_0
  
  auto flags = workspace.get_int_buf_0(n_small);
  auto tmp_idx_a = workspace.get_int_buf_1(n_small);
  auto tmp_idx_b = workspace.get_int_buf_2(n_small);
  auto positions = workspace.get_size_t_buf_0(n_small);
  auto d_num_rows = workspace.get_scalar_size_t_buf_0();

  // 1) For each row of the smaller set, binary-search its Y in the
  //    larger set and record matches.

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

  // Reuse preallocated mapping buffers from the workspace
  // Only the first num_rows_out entries are written and used by callers.
  auto out_rows = workspace.get_row_key_buf_0(num_rows_out);
  auto out_idx_a = workspace.get_int_buf_3(num_rows_out);
  auto out_idx_b = workspace.get_int_buf_4(num_rows_out);
  
  result.row_keys = out_rows;
  result.row_index_a = out_idx_a;
  result.row_index_b = out_idx_b;

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
 * @brief Convenience overload that uses a local workspace.
 */
inline RowMergeResult
build_row_intersection_mapping(const IntervalSet2DDevice& A,
                               const IntervalSet2DDevice& B) {
  UnifiedCsrWorkspace workspace;
  return build_row_intersection_mapping(A, B, workspace);
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
                             const IntervalSet2DDevice& B,
                             UnifiedCsrWorkspace& workspace) {
  RowDifferenceResult result;

  const std::size_t num_rows_a = A.num_rows;
  const std::size_t num_rows_b = B.num_rows;

  if (num_rows_a == 0) {
    return result;
  }

  // Checkout buffers
  // row_keys: row_key_buf_0 [num_rows_a]
  // row_index_b: int_buf_0 [num_rows_a]
  
  auto out_rows = workspace.get_row_key_buf_0(num_rows_a);
  auto out_idx_b = workspace.get_int_buf_0(num_rows_a);

  result.num_rows = num_rows_a;
  result.row_keys = out_rows;
  result.row_index_b = out_idx_b;

  auto rows_a = A.row_keys;
  auto rows_b = B.row_keys;

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
 * @brief Convenience overload that uses a local workspace.
 */
inline RowDifferenceResult
build_row_difference_mapping(const IntervalSet2DDevice& A,
                             const IntervalSet2DDevice& B) {
  UnifiedCsrWorkspace workspace;
  return build_row_difference_mapping(A, B, workspace);
}

} // namespace detail

/**
 * @brief Compute the set union into a preallocated CSR buffer on device.
 */
inline void
set_union_device(const IntervalSet2DDevice& A,
                 const IntervalSet2DDevice& B,
                 IntervalSet2DDevice& out,
                 CsrSetAlgebraContext& ctx) {
  detail::reset_preallocated_interval_set(out);

  const std::size_t num_rows_a = A.num_rows;
  const std::size_t num_rows_b = B.num_rows;

  if (num_rows_a == 0 && num_rows_b == 0) {
    return;
  }

  detail::RowMergeResult merge =
      detail::build_row_union_mapping(A, B, ctx.workspace);

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

  auto row_counts = ctx.workspace.get_size_t_buf_0(num_rows_out);

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

  auto total_intervals = ctx.workspace.get_scalar_size_t_buf_0();

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
 */
inline void
set_intersection_device(const IntervalSet2DDevice& A,
                        const IntervalSet2DDevice& B,
                        IntervalSet2DDevice& out,
                        CsrSetAlgebraContext& ctx) {
  detail::reset_preallocated_interval_set(out);

  const std::size_t num_rows_a = A.num_rows;
  const std::size_t num_rows_b = B.num_rows;

  if (num_rows_a == 0 || num_rows_b == 0) {
    return;
  }

  detail::RowMergeResult merge =
      detail::build_row_intersection_mapping(
          A, B, ctx.workspace);
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

  auto row_counts = ctx.workspace.get_size_t_buf_0(num_rows_out);

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

  auto total_intervals = ctx.workspace.get_scalar_size_t_buf_0();

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
 */
inline void
set_difference_device(const IntervalSet2DDevice& A,
                      const IntervalSet2DDevice& B,
                      IntervalSet2DDevice& out,
                      CsrSetAlgebraContext& ctx) {
  detail::reset_preallocated_interval_set(out);

  const std::size_t num_rows_a = A.num_rows;
  if (num_rows_a == 0) {
    return;
  }

  detail::RowDifferenceResult diff_rows =
      detail::build_row_difference_mapping(
          A, B, ctx.workspace);
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

  auto row_counts = ctx.workspace.get_size_t_buf_0(num_rows_out);

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

  auto total_intervals = ctx.workspace.get_scalar_size_t_buf_0();
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
 */
inline void
set_symmetric_difference_device(const IntervalSet2DDevice& A,
                                const IntervalSet2DDevice& B,
                                IntervalSet2DDevice& out,
                                CsrSetAlgebraContext& ctx) {
  detail::reset_preallocated_interval_set(out);

  const std::size_t num_rows_a = A.num_rows;
  const std::size_t num_rows_b = B.num_rows;

  if (num_rows_a == 0 && num_rows_b == 0) {
    return;
  }

  detail::RowMergeResult merge =
      detail::build_row_union_mapping(A, B, ctx.workspace);

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

  auto row_counts = ctx.workspace.get_size_t_buf_0(num_rows_out);

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

  auto total_intervals = ctx.workspace.get_scalar_size_t_buf_0();
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

} // namespace csr
} // namespace subsetix
