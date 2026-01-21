// SPDX-FileCopyrightText: 2025 Subsetix Kokkos Contributors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <experimental/subsetix/csr/types.hpp>
#include <string>

namespace experimental::subsetix::csr::detail {

// ============================================================================
// Memory utilities
// ============================================================================

/**
 * @brief Ensure a Kokkos View has at least the required capacity.
 *
 * If the current capacity is less than required_size, the view is
 * reallocated. Content is NOT preserved (for scratch buffers).
 */
template <class ViewType>
inline void ensure_view_capacity(ViewType& view,
                                 std::size_t required_size,
                                 const std::string& label) {
  if (view.extent(0) < required_size) {
    view = ViewType(label, required_size);
  }
}

// ============================================================================
// Binary search utilities (2D and 3D)
// ============================================================================

/**
 * @brief Find a row index by y-coordinate using binary search (2D).
 *
 * @tparam RowKeyView Type of the row keys view (must support .y member)
 * @tparam CoordType The coordinate type (deduced from RowKeyView)
 *
 * @param rows View of row keys (sorted)
 * @param num_rows Number of rows in the view
 * @param y Y-coordinate to search for
 * @return Row index if found, -1 otherwise
 */
template <class RowKeyView>
KOKKOS_INLINE_FUNCTION
auto find_row_by_y(const RowKeyView& rows, std::size_t num_rows,
                   const auto& y) -> int {
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
 * @brief Find a row index by (y,z) coordinates using binary search (3D).
 *
 * Uses lexicographic ordering: first by y, then by z.
 *
 * @tparam RowKeyView Type of the row keys view (must support .y and .z members)
 * @tparam CoordType The coordinate type (deduced from RowKeyView)
 *
 * @param rows View of row keys (sorted)
 * @param num_rows Number of rows in the view
 * @param y Y-coordinate to search for
 * @param z Z-coordinate to search for
 * @return Row index if found, -1 otherwise
 */
template <class RowKeyView>
KOKKOS_INLINE_FUNCTION
auto find_row_by_yz(const RowKeyView& rows, std::size_t num_rows,
                    const auto& y, const auto& z) -> int {
  std::size_t lo = 0;
  std::size_t hi = num_rows;

  while (lo < hi) {
    const std::size_t mid = lo + (hi - lo) / 2;
    const auto key = rows(mid);

    if (key.y < y || (key.y == y && key.z < z)) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  if (lo < num_rows) {
    const auto key = rows(lo);
    if (key.y == y && key.z == z) {
      return static_cast<int>(lo);
    }
  }

  return -1;
}

// ============================================================================
// Row range extraction helper
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
 * If a row index is negative (not found), the corresponding range is empty.
 *
 * @tparam RowPtrViewA Type of row_ptr view for mesh A
 * @tparam RowPtrViewB Type of row_ptr view for mesh B
 *
 * @param ia Row index in mesh A (-1 if not found)
 * @param ib Row index in mesh B (-1 if not found)
 * @param row_ptr_a CSR row pointers for mesh A
 * @param row_ptr_b CSR row pointers for mesh B
 * @return RowRanges structure with the interval ranges
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

} // namespace experimental::subsetix::csr::detail
