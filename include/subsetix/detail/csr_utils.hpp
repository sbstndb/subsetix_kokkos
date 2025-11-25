#pragma once

#include <Kokkos_Core.hpp>
#include <subsetix/geometry/csr_interval_set.hpp>
#include <subsetix/detail/scan_utils.hpp>

namespace subsetix {
namespace csr {
namespace detail {

// ============================================================================
// Row range extraction helper for binary CSR operations
// ============================================================================

/**
 * @brief Holds the interval ranges for two rows in a binary CSR operation.
 */
struct RowRanges {
  std::size_t begin_a = 0;
  std::size_t end_a = 0;
  std::size_t begin_b = 0;
  std::size_t end_b = 0;

  KOKKOS_INLINE_FUNCTION
  bool both_empty() const {
    return begin_a == end_a && begin_b == end_b;
  }

  KOKKOS_INLINE_FUNCTION
  bool a_empty() const {
    return begin_a == end_a;
  }

  KOKKOS_INLINE_FUNCTION
  bool b_empty() const {
    return begin_b == end_b;
  }
};

/**
 * @brief Extract interval ranges for two rows given their indices.
 *
 * This helper eliminates the repetitive boilerplate for extracting
 * begin/end indices from CSR row_ptr arrays in binary set operations.
 *
 * @tparam RowPtrViewA Row pointer view type for set A
 * @tparam RowPtrViewB Row pointer view type for set B
 * @param ia Row index in A (-1 if row doesn't exist in A)
 * @param ib Row index in B (-1 if row doesn't exist in B)
 * @param row_ptr_a Row pointer array for set A
 * @param row_ptr_b Row pointer array for set B
 * @return RowRanges struct with begin/end indices for both rows
 */
template <class RowPtrViewA, class RowPtrViewB>
KOKKOS_FORCEINLINE_FUNCTION
RowRanges extract_row_ranges(int ia, int ib,
                              const RowPtrViewA& row_ptr_a,
                              const RowPtrViewB& row_ptr_b) {
  RowRanges r;
  if (ia >= 0) {
    const std::size_t row_a = static_cast<std::size_t>(ia);
    r.begin_a = row_ptr_a(row_a);
    r.end_a = row_ptr_a(row_a + 1);
  }
  if (ib >= 0) {
    const std::size_t row_b = static_cast<std::size_t>(ib);
    r.begin_b = row_ptr_b(row_b);
    r.end_b = row_ptr_b(row_b + 1);
  }
  return r;
}

/**
 * @brief Overload for same row_ptr type (common case).
 */
template <class RowPtrView>
KOKKOS_FORCEINLINE_FUNCTION
RowRanges extract_row_ranges(int ia, int ib,
                              const RowPtrView& row_ptr_a,
                              const RowPtrView& row_ptr_b) {
  RowRanges r;
  if (ia >= 0) {
    const std::size_t row_a = static_cast<std::size_t>(ia);
    r.begin_a = row_ptr_a(row_a);
    r.end_a = row_ptr_a(row_a + 1);
  }
  if (ib >= 0) {
    const std::size_t row_b = static_cast<std::size_t>(ib);
    r.begin_b = row_ptr_b(row_b);
    r.end_b = row_ptr_b(row_b + 1);
  }
  return r;
}

// ============================================================================
// Coordinate utilities
// ============================================================================

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

/**
 * @brief Find a row index by y-coordinate using binary search.
 *
 * @tparam RowKeyView A Kokkos view of RowKey elements (must have .y member)
 * @param rows The view of row keys (sorted by y)
 * @param num_rows Number of rows in the view
 * @param y The y-coordinate to search for
 * @return The index of the row if found, -1 otherwise
 */
template <class RowKeyView>
KOKKOS_INLINE_FUNCTION
int find_row_by_y(const RowKeyView& rows, std::size_t num_rows, Coord y) {
  std::size_t lo = 0;
  std::size_t hi = num_rows;
  while (lo < hi) {
    const std::size_t mid = lo + (hi - lo) / 2;
    if (rows(mid).y < y) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo < num_rows && rows(lo).y == y) {
    return static_cast<int>(lo);
  }
  return -1;
}

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

} // namespace detail
} // namespace csr
} // namespace subsetix
